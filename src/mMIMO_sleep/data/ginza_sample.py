"""从一次 Ginza PathSolver 求解构造一个 UE 的 normal/sleep 样本。

设计约束：

* 每个 UE 只运行一次 PathSolver，normal 和 sleep 共享 paths 与 CFR。
* 所有 1024 个 codebook beam 都参与 PMI 选择，valid mask 只记录 metadata。
* 先将项目码字映射到 Sionna TX 端口顺序，再计算 CFR、PDP 和接收功率。
* sleep mask 不重新归一化，也不额外重复乘一个 1/2。
* Sionna、Mitsuba、PyTorch 均延迟导入，使纯数值辅助函数可以在 CPU 上测试。
"""

from __future__ import annotations

import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mMIMO_sleep.data.ginza_config import GinzaPipelineConfig
from mMIMO_sleep.data.schema import DATASET_SCHEMA_VERSION, SampleRecord

BOLTZMANN_CONSTANT = 1.380649e-23
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


class NoValidPathsError(RuntimeError):
    """当前 UE 没有可用于构造样本的传播路径。"""


@dataclass(frozen=True, slots=True)
class DelayStatistics:
    total_path_count: int
    valid_path_count: int
    usable_path_count: int
    first_path_delay_s: float
    strongest_path_delay_s: float
    power_weighted_delay_s: float
    rms_delay_spread_s: float
    last_path_delay_s: float
    total_beamformed_path_power: float


@dataclass(frozen=True, slots=True)
class LinkBudget:
    normal_rx_power_dbm: float
    sleep_rx_power_dbm: float
    normal_snr_db: float
    sleep_snr_db: float
    normal_mean_subcarrier_rx_power_dbm: float
    sleep_mean_subcarrier_rx_power_dbm: float
    normal_mean_subcarrier_snr_db: float
    sleep_mean_subcarrier_snr_db: float
    normal_snr_p5_db: float
    normal_snr_p50_db: float
    normal_snr_p95_db: float
    sleep_snr_p5_db: float
    sleep_snr_p50_db: float
    sleep_snr_p95_db: float
    rx_power_loss_db: float
    snr_loss_db: float
    mean_subcarrier_loss_db: float
    normal_tx_power_w: float
    sleep_tx_power_w: float
    per_subcarrier_tx_power_w: float
    per_subcarrier_noise_power_w: float


def _finite_vector(values: Any, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return vector


def extract_path_arrays(
    paths: Any,
    *,
    num_rx_ports: int,
    num_tx_ports: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """显式提取 ``A[rx,tx,path]``、``tau[path]`` 与 ``valid[path]``。

    不使用无条件 squeeze：当只有一个 path 时 squeeze 会误删 path 维度。
    """

    a = paths.a
    if isinstance(a, tuple):
        if len(a) == 2:
            real = np.asarray(a[0])
            imag = np.asarray(a[1])
            if real.shape != imag.shape:
                raise ValueError(
                    "paths.a[0] and paths.a[1] must have the same shape; "
                    f"got {real.shape} and {imag.shape}."
                )
            raw_coefficients = real + 1j * imag
        elif len(a) == 1:
            # Future Sionna versions may return a single complex tensor.
            raw_coefficients = np.asarray(a[0])
        else:
            raise TypeError(
                f"paths.a must be a 1- or 2-tuple, got length {len(a)}."
            )
    else:
        # Future versions may directly expose a complex array.
        raw_coefficients = np.asarray(a)

    if raw_coefficients.ndim == 5:
        expected_prefix = (1, num_rx_ports, 1, num_tx_ports)
        if raw_coefficients.shape[:4] != expected_prefix:
            raise ValueError(
                "Unexpected paths.a shape: "
                f"{raw_coefficients.shape}; expected prefix {expected_prefix}."
            )
        coefficients = raw_coefficients[0, :, 0, :, :]
    elif raw_coefficients.ndim == 3:
        coefficients = raw_coefficients
    else:
        raise ValueError(
            "paths.a must have shape [1,rx,1,tx,path] or [rx,tx,path], "
            f"got {raw_coefficients.shape}."
        )

    if coefficients.shape[:2] != (num_rx_ports, num_tx_ports):
        raise ValueError(
            "Path coefficient RX/TX axes do not match configured port counts: "
            f"{coefficients.shape}."
        )
    if not np.iscomplexobj(coefficients):
        raise TypeError("Path coefficients must contain complex channel values.")
    num_paths = coefficients.shape[-1]
    if num_paths == 0:
        raise NoValidPathsError("PathSolver returned zero path entries.")

    def read_path_vector(raw: Any, name: str) -> np.ndarray:
        array = np.asarray(raw)
        if array.ndim == 0 or array.shape[-1] != num_paths:
            raise ValueError(
                f"{name} must preserve a final path dimension of {num_paths}; "
                f"got {array.shape}."
            )
        if int(np.prod(array.shape[:-1], dtype=np.int64)) != 1:
            raise ValueError(
                f"{name} must describe exactly one TX/RX device pair; "
                f"got {array.shape}."
            )
        return array.reshape(num_paths)

    delays_s = read_path_vector(paths.tau, "paths.tau").astype(np.float64)
    valid_mask = read_path_vector(paths.valid, "paths.valid").astype(bool)
    return coefficients, delays_s, valid_mask


def compute_beamformed_path_powers(
    path_coefficients: np.ndarray,
    weights_sionna_order: np.ndarray,
) -> np.ndarray:
    """计算 ``P_p = sum_rx |A_p @ w|²``，不跨路径相干求和。"""

    coefficients = np.asarray(path_coefficients)
    weights = np.asarray(weights_sionna_order)
    if coefficients.ndim != 3:
        raise ValueError("path_coefficients must have shape [rx, tx, path].")
    if weights.ndim != 1 or weights.shape[0] != coefficients.shape[1]:
        raise ValueError(
            "weights_sionna_order must contain one weight for every TX port."
        )
    if not np.iscomplexobj(coefficients) or not np.iscomplexobj(weights):
        raise TypeError("Path coefficients and TX weights must both be complex.")
    effective = np.einsum("rtp,t->rp", coefficients, weights, optimize=True)
    return np.sum(np.abs(effective) ** 2, axis=0, dtype=np.float64)


def compute_delay_statistics(
    path_delays_s: np.ndarray,
    path_powers: np.ndarray,
    valid_paths_mask: np.ndarray,
) -> DelayStatistics:
    """从 normal-mode beamformed PDP 计算加权时延和 RMS delay spread。"""

    delays = _finite_vector(path_delays_s, "path_delays_s").astype(np.float64)
    powers = _finite_vector(path_powers, "path_powers").astype(np.float64)
    valid = _finite_vector(valid_paths_mask, "valid_paths_mask").astype(bool)
    if not (delays.shape == powers.shape == valid.shape):
        raise ValueError(
            "Path delay, power, and validity vectors must have equal shapes."
        )

    usable = (
        valid
        & np.isfinite(delays)
        & (delays >= 0.0)
        & np.isfinite(powers)
        & (powers >= 0.0)
    )
    if not np.any(usable):
        raise NoValidPathsError("No valid finite non-negative propagation paths.")
    usable_delays = delays[usable]
    usable_powers = powers[usable]
    total_power = float(np.sum(usable_powers, dtype=np.float64))
    if not math.isfinite(total_power) or total_power <= 0.0:
        raise NoValidPathsError("The usable beamformed path power is zero or invalid.")

    weighted_delay = float(np.sum(usable_powers * usable_delays) / total_power)
    spread = float(
        np.sqrt(
            np.sum(usable_powers * (usable_delays - weighted_delay) ** 2)
            / total_power
        )
    )
    strongest = int(np.argmax(usable_powers))
    return DelayStatistics(
        total_path_count=int(delays.size),
        valid_path_count=int(np.count_nonzero(valid)),
        usable_path_count=int(np.count_nonzero(usable)),
        first_path_delay_s=float(np.min(usable_delays)),
        strongest_path_delay_s=float(usable_delays[strongest]),
        power_weighted_delay_s=weighted_delay,
        rms_delay_spread_s=spread,
        last_path_delay_s=float(np.max(usable_delays)),
        total_beamformed_path_power=total_power,
    )


def infer_los_from_delay(
    valid_path_delays_s: np.ndarray,
    distance_3d_m: float,
    *,
    tolerance_s: float,
) -> bool:
    """用几何传播时延判断是否存在 LOS 候选，结果是 metadata 而非筛选条件。

    Sionna RT 1.2.2 没有稳定暴露每条路径类型的公开接口，因此这里判断
    是否存在与直线距离/c 一致的有效路径。结果应视为明确记录的启发式。
    """

    if distance_3d_m < 0 or tolerance_s < 0:
        raise ValueError("Distance and LOS delay tolerance must be non-negative.")
    delays = _finite_vector(valid_path_delays_s, "valid_path_delays_s")
    geometric_delay = distance_3d_m / SPEED_OF_LIGHT_M_PER_S
    return bool(np.any(np.abs(delays - geometric_delay) <= tolerance_s))


def _linear_to_db(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise ValueError(f"{name} must be finite and strictly positive before log10.")
    return 10.0 * np.log10(values)


def compute_link_budget(
    normal_channel_gains: np.ndarray,
    sleep_channel_gains: np.ndarray,
    *,
    tx_power_dbm: float,
    temperature_k: float,
    noise_figure_db: float,
    subcarrier_spacing_hz: float,
    sleep_weight_norm_squared: float,
) -> LinkBudget:
    """计算子载波级功率和 SNR，并明确区分线性平均与 dB 平均。"""

    normal = _finite_vector(normal_channel_gains, "normal_channel_gains").astype(
        np.float64
    )
    sleep = _finite_vector(sleep_channel_gains, "sleep_channel_gains").astype(
        np.float64
    )
    if normal.shape != sleep.shape:
        raise ValueError("Normal and sleep channel gains must have the same shape.")
    if not math.isfinite(sleep_weight_norm_squared) or sleep_weight_norm_squared < 0:
        raise ValueError("sleep_weight_norm_squared must be finite and non-negative.")

    normal_tx_power_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    sleep_tx_power_w = normal_tx_power_w * sleep_weight_norm_squared
    per_subcarrier_power_w = normal_tx_power_w / normal.size
    noise_factor = 10.0 ** (noise_figure_db / 10.0)
    noise_power_w = (
        BOLTZMANN_CONSTANT
        * temperature_k
        * subcarrier_spacing_hz
        * noise_factor
    )
    if noise_power_w <= 0.0 or not math.isfinite(noise_power_w):
        raise ValueError("Per-subcarrier noise power must be finite and positive.")

    normal_rx_w = per_subcarrier_power_w * normal
    sleep_rx_w = per_subcarrier_power_w * sleep
    normal_snr = normal_rx_w / noise_power_w
    sleep_snr = sleep_rx_w / noise_power_w
    normal_rx_dbm = 30.0 + _linear_to_db(normal_rx_w, "normal_rx_w")
    sleep_rx_dbm = 30.0 + _linear_to_db(sleep_rx_w, "sleep_rx_w")
    normal_snr_db = _linear_to_db(normal_snr, "normal_snr")
    sleep_snr_db = _linear_to_db(sleep_snr, "sleep_snr")
    normal_average_rx_dbm = 30.0 + 10.0 * math.log10(float(np.mean(normal_rx_w)))
    sleep_average_rx_dbm = 30.0 + 10.0 * math.log10(float(np.mean(sleep_rx_w)))
    normal_average_snr_db = 10.0 * math.log10(float(np.mean(normal_snr)))
    sleep_average_snr_db = 10.0 * math.log10(float(np.mean(sleep_snr)))

    return LinkBudget(
        normal_rx_power_dbm=normal_average_rx_dbm,
        sleep_rx_power_dbm=sleep_average_rx_dbm,
        normal_snr_db=normal_average_snr_db,
        sleep_snr_db=sleep_average_snr_db,
        normal_mean_subcarrier_rx_power_dbm=float(np.mean(normal_rx_dbm)),
        sleep_mean_subcarrier_rx_power_dbm=float(np.mean(sleep_rx_dbm)),
        normal_mean_subcarrier_snr_db=float(np.mean(normal_snr_db)),
        sleep_mean_subcarrier_snr_db=float(np.mean(sleep_snr_db)),
        normal_snr_p5_db=float(np.percentile(normal_snr_db, 5)),
        normal_snr_p50_db=float(np.percentile(normal_snr_db, 50)),
        normal_snr_p95_db=float(np.percentile(normal_snr_db, 95)),
        sleep_snr_p5_db=float(np.percentile(sleep_snr_db, 5)),
        sleep_snr_p50_db=float(np.percentile(sleep_snr_db, 50)),
        sleep_snr_p95_db=float(np.percentile(sleep_snr_db, 95)),
        rx_power_loss_db=normal_average_rx_dbm - sleep_average_rx_dbm,
        snr_loss_db=normal_average_snr_db - sleep_average_snr_db,
        mean_subcarrier_loss_db=float(np.mean(normal_snr_db - sleep_snr_db)),
        normal_tx_power_w=normal_tx_power_w,
        sleep_tx_power_w=sleep_tx_power_w,
        per_subcarrier_tx_power_w=per_subcarrier_power_w,
        per_subcarrier_noise_power_w=noise_power_w,
    )


def load_ue_positions(config: GinzaPipelineConfig) -> list[np.ndarray]:
    """读取可信参考 notebook 生成的 UE 坐标 pickle。

    pickle 具有执行代码的能力，因此该文件必须来自用户信任的本地数据源。
    """

    source = Path(config.scene.ue_positions_path)
    with source.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, dict) or config.scene.ue_positions_key not in loaded:
        raise ValueError(
            f"UE pickle must contain key {config.scene.ue_positions_key!r}."
        )
    positions = []
    for index, position in enumerate(loaded[config.scene.ue_positions_key]):
        point = np.asarray(position, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid UE position at index {index}: {point!r}.")
        positions.append(point)
    if not positions:
        raise ValueError("The UE position file contains no usable coordinates.")
    return positions


class GinzaSimulation:
    """复用一个场景、solver、完整码本和 PMI metadata mask 处理多个 UE。"""

    def __init__(self, config: GinzaPipelineConfig, *, git_commit: str | None = None):
        self.config = config
        self.git_commit = git_commit
        try:
            import mitsuba as mi
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "GinzaSimulation requires the project's RunPod environment with "
                "Mitsuba, Sionna RT, and PyTorch installed."
            ) from exc
        if mi.variant() != config.mitsuba_variant:
            mi.set_variant(config.mitsuba_variant)
        try:
            import sionna.rt as rt
        except ImportError as exc:
            raise RuntimeError(
                "Sionna RT is unavailable in this Python environment."
            ) from exc

        self.mi = mi
        self.torch = torch
        self.rt = rt
        self.scene = self._prepare_scene()
        self.solver = rt.PathSolver()
        self._prepare_codebook()
        self._prepare_frequency_grid()

    def _prepare_scene(self) -> Any:
        from mMIMO_sleep.array_config import ArrayConfig
        from mMIMO_sleep.simulation.sionna_array import array_config_to_planar_array

        cfg = self.config
        mi = self.mi
        rt = self.rt
        scene = rt.load_scene(cfg.scene.scene_xml_path)
        scene.frequency = mi.Float(cfg.scene.carrier_frequency_hz)
        scene.bandwidth = mi.Float(cfg.scene.bandwidth_hz)
        scene.temperature = mi.Float(cfg.scene.temperature_k)

        concrete = rt.ITURadioMaterial(
            name=cfg.scene.material_name,
            itu_type=cfg.scene.material_itu_type,
            thickness=mi.Float(cfg.scene.material_thickness_m),
            scattering_coefficient=cfg.scene.material_scattering_coefficient,
            xpd_coefficient=cfg.scene.material_xpd_coefficient,
        )
        replacements = 0
        for obj in scene.objects.values():
            if obj.radio_material.name in {"itu_concrete", "concrete"}:
                obj.radio_material = concrete
                replacements += 1
        if cfg.scene.require_concrete_override and replacements == 0:
            raise RuntimeError("No default concrete material was found to override.")

        self.array_config = ArrayConfig(
            num_subarray_rows=cfg.antenna.num_subarray_rows,
            num_horizontal=cfg.antenna.num_horizontal,
            elements_per_subarray=cfg.antenna.elements_per_subarray,
            num_polarizations=cfg.antenna.num_polarizations,
        )
        scene.tx_array = array_config_to_planar_array(
            self.array_config,
            pattern=cfg.antenna.tx_pattern,
            polarization=cfg.antenna.tx_polarization,
            vertical_spacing=cfg.antenna.vertical_spacing_wavelengths,
            horizontal_spacing=cfg.antenna.horizontal_spacing_wavelengths,
        )
        scene.rx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            pattern=cfg.antenna.rx_pattern,
            polarization=cfg.antenna.rx_polarization,
        )
        scene.add(
            rt.Transmitter(
                name=cfg.scene.tx_name,
                position=cfg.scene.tx_position_m,
                orientation=cfg.scene.tx_orientation_rad,
            )
        )
        self.num_rx_ports = int(scene.rx_array.num_ant)
        self.num_tx_ports = int(scene.tx_array.num_ant)
        if self.num_tx_ports != self.array_config.num_physical_ports:
            raise RuntimeError("Sionna TX port count disagrees with ArrayConfig.")
        return scene

    def _prepare_codebook(self) -> None:
        from mMIMO_sleep.codebook.dft import generate_dft_codebook
        from mMIMO_sleep.codebook.muting import create_right_half_mask
        from mMIMO_sleep.codebook.pmi_mask import create_total_loss_pmi_mask
        from mMIMO_sleep.simulation.sionna_array import weights_to_sionna_precoding

        cfg = self.config
        self.codebook_project = generate_dft_codebook(
            self.array_config,
            cfg.codebook.num_vertical_beams,
            cfg.codebook.num_horizontal_beams,
            cfg.codebook.num_i2,
            device=cfg.torch_device,
        )
        expected_shape = (cfg.codebook.total_beams, self.num_tx_ports)
        if tuple(self.codebook_project.shape) != expected_shape:
            raise RuntimeError(
                f"Expected complete codebook shape {expected_shape}, "
                f"got {tuple(self.codebook_project.shape)}."
            )
        real, imag = weights_to_sionna_precoding(
            self.codebook_project,
            self.array_config,
        )
        self.codebook_sionna = self.torch.complex(real, imag)
        self.valid_pmi_mask = create_total_loss_pmi_mask(
            self.array_config,
            cfg.codebook.num_vertical_beams,
            cfg.codebook.num_horizontal_beams,
            cfg.codebook.num_i2,
            target_loss_db=cfg.codebook.validity_target_loss_db,
            tolerance_db=cfg.codebook.validity_tolerance_db,
            grid_points=cfg.codebook.validity_grid_points,
            device=cfg.torch_device,
        )
        self.right_half_mask = create_right_half_mask(
            self.array_config,
            device=cfg.torch_device,
        )

    def _prepare_frequency_grid(self) -> None:
        cfg = self.config.ofdm
        all_frequencies = self.rt.subcarrier_frequencies(
            cfg.fft_size,
            cfg.subcarrier_spacing_hz,
        )
        start = cfg.left_guard_subcarriers
        stop = start + cfg.num_active_subcarriers
        self.active_frequencies_hz = np.asarray(
            all_frequencies[start:stop],
            dtype=np.float64,
        )
        if self.active_frequencies_hz.size != cfg.num_active_subcarriers:
            raise RuntimeError("Unexpected number of active OFDM subcarriers.")
        self.center_subcarrier_index = int(
            np.argmin(np.abs(self.active_frequencies_hz))
        )
        if self.active_frequencies_hz[self.center_subcarrier_index] != 0.0:
            raise RuntimeError("The active frequency grid does not contain DC.")

    def _add_receiver(self, position: np.ndarray) -> None:
        receiver_name = self.config.scene.rx_name
        if receiver_name in self.scene.receivers:
            self.scene.remove(receiver_name)
        self.scene.add(
            self.rt.Receiver(name=receiver_name, position=position.tolist())
        )

    def _compute_channels(self, paths: Any) -> tuple[Any, Any, float]:
        started = time.perf_counter()
        center_raw = np.asarray(
            paths.cfr(
                frequencies=[0.0],
                normalize=False,
                normalize_delays=False,
                out_type="numpy",
            )
        )
        expected_center = (1, self.num_rx_ports, 1, self.num_tx_ports, 1, 1)
        if center_raw.shape != expected_center:
            raise ValueError(
                f"Unexpected center CFR shape {center_raw.shape}; "
                f"expected {expected_center}."
            )
        center = center_raw[0, :, 0, :, 0, 0]
        num_paths = int(np.asarray(paths.tau).shape[-1])
        broadcast_base = self.num_rx_ports * self.num_tx_ports * max(1, num_paths)
        chunk_size = max(1, self.config.ofdm.max_cfr_chunk_entries // broadcast_base)
        chunk_size = min(
            chunk_size,
            self.config.ofdm.max_cfr_chunk_subcarriers,
            self.config.ofdm.num_active_subcarriers,
        )
        chunks = []
        for start in range(0, self.active_frequencies_hz.size, chunk_size):
            frequencies = self.active_frequencies_hz[start : start + chunk_size]
            chunk = np.asarray(
                paths.cfr(
                    frequencies=frequencies.tolist(),
                    normalize=False,
                    normalize_delays=False,
                    out_type="numpy",
                )
            )
            expected_prefix = (1, self.num_rx_ports, 1, self.num_tx_ports, 1)
            if chunk.ndim != 6 or chunk.shape[:5] != expected_prefix:
                raise ValueError(f"Unexpected wideband CFR chunk shape: {chunk.shape}.")
            chunks.append(chunk[0, :, 0, :, 0, :])
        wideband_rx_tx_frequency = np.concatenate(chunks, axis=-1)
        wideband = np.transpose(wideband_rx_tx_frequency, (2, 0, 1))
        if not np.isfinite(center).all() or not np.isfinite(wideband).all():
            raise ValueError("The center or wideband CFR contains non-finite values.")

        torch = self.torch
        center_tensor = torch.as_tensor(
            center,
            dtype=torch.complex64,
            device=self.config.torch_device,
        )
        wideband_tensor = torch.as_tensor(
            wideband,
            dtype=torch.complex64,
            device=self.config.torch_device,
        )
        # wideband_tensor[center] and center_tensor both describe the same
        # 0 Hz (center) subcarrier from the same `paths` object. They are
        # materialized through slightly different CFR code paths on the GPU,
        # which can introduce small complex64 floating-point differences.
        # rtol=1e-4 is a consistency-check tolerance, not a physical-model
        # tolerance.
        torch.testing.assert_close(
            wideband_tensor[self.center_subcarrier_index],
            center_tensor,
            rtol=1e-4,
            atol=1e-12,
        )
        return center_tensor, wideband_tensor, time.perf_counter() - started

    def _select_beams(
        self,
        center_channel: Any,
        wideband_channel: Any,
    ) -> tuple[int, int]:
        torch = self.torch
        center_effective = center_channel @ self.codebook_sionna.T
        center_scores = torch.sum(torch.abs(center_effective) ** 2, dim=0)
        center_index = int(torch.argmax(center_scores).item())

        num_beams = self.config.codebook.total_beams
        scores = torch.empty(
            num_beams,
            dtype=torch.float32,
            device=self.config.torch_device,
        )
        chunk_size = self.config.codebook.beam_chunk_size
        for start in range(0, num_beams, chunk_size):
            stop = min(start + chunk_size, num_beams)
            weights = self.codebook_sionna[start:stop]
            effective = torch.einsum("krt,bt->kbr", wideband_channel, weights)
            scores[start:stop] = torch.sum(torch.abs(effective) ** 2, dim=-1).mean(
                dim=0
            )
        wideband_index = int(torch.argmax(scores).item())
        return center_index, wideband_index

    def simulate(self, ue_index: int, position: np.ndarray) -> SampleRecord:
        """完整仿真指定 UE；NLOS 和 invalid PMI 均正常生成样本。"""

        from mMIMO_sleep.codebook.muting import apply_muting_mask
        from mMIMO_sleep.codebook.pmi import beam_index_to_pmi
        from mMIMO_sleep.simulation.sionna_array import weights_to_sionna_precoding

        cfg = self.config
        torch = self.torch
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Invalid UE position: {position!r}.")
        started = time.perf_counter()
        self._add_receiver(position)
        try:
            solver_started = time.perf_counter()
            paths = self.solver(self.scene, **cfg.solver.solver_kwargs())
            coefficients, delays, valid = extract_path_arrays(
                paths,
                num_rx_ports=self.num_rx_ports,
                num_tx_ports=self.num_tx_ports,
            )
            solver_runtime = time.perf_counter() - solver_started
            if not np.any(valid):
                raise NoValidPathsError(f"UE index {ue_index} has no valid paths.")

            center_h, wideband_h, cfr_runtime = self._compute_channels(paths)
            center_index, wideband_index = self._select_beams(center_h, wideband_h)
            pmi_args = {
                "num_horizontal_beams": cfg.codebook.num_horizontal_beams,
                "num_vertical_beams": cfg.codebook.num_vertical_beams,
                "num_i2": cfg.codebook.num_i2,
            }
            center_pmi = beam_index_to_pmi(center_index, **pmi_args)
            selected_pmi = beam_index_to_pmi(wideband_index, **pmi_args)

            normal_project = self.codebook_project[wideband_index]
            sleep_project = apply_muting_mask(normal_project, self.right_half_mask)
            normal_real, normal_imag = weights_to_sionna_precoding(
                normal_project,
                self.array_config,
            )
            sleep_real, sleep_imag = weights_to_sionna_precoding(
                sleep_project,
                self.array_config,
            )
            normal_sionna = torch.complex(normal_real, normal_imag)
            sleep_sionna = torch.complex(sleep_real, sleep_imag)
            normal_norm = float(torch.sum(torch.abs(normal_project) ** 2).item())
            sleep_norm = float(torch.sum(torch.abs(sleep_project) ** 2).item())
            if not math.isclose(normal_norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(f"Normal codeword is not unit norm: {normal_norm}.")
            if not math.isclose(sleep_norm, 0.5, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(
                    "Right-half muting unexpectedly changed the sleep norm: "
                    f"{sleep_norm}."
                )

            normal_effective = wideband_h @ normal_sionna
            sleep_effective = wideband_h @ sleep_sionna
            normal_gains = (
                torch.sum(torch.abs(normal_effective) ** 2, dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            sleep_gains = (
                torch.sum(torch.abs(sleep_effective) ** 2, dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            link = compute_link_budget(
                normal_gains,
                sleep_gains,
                tx_power_dbm=cfg.ofdm.tx_power_dbm,
                temperature_k=cfg.scene.temperature_k,
                noise_figure_db=cfg.ofdm.noise_figure_db,
                subcarrier_spacing_hz=cfg.ofdm.subcarrier_spacing_hz,
                sleep_weight_norm_squared=sleep_norm,
            )
            path_powers = compute_beamformed_path_powers(
                coefficients,
                normal_sionna.detach().cpu().numpy(),
            )
            delay = compute_delay_statistics(delays, path_powers, valid)
            tx_position = np.asarray(cfg.scene.tx_position_m, dtype=np.float64)
            distance_2d = float(np.linalg.norm(position[:2] - tx_position[:2]))
            distance_3d = float(np.linalg.norm(position - tx_position))
            finite_valid_delays = delays[valid & np.isfinite(delays) & (delays >= 0.0)]
            has_los = infer_los_from_delay(
                finite_valid_delays,
                distance_3d,
                tolerance_s=cfg.scene.los_delay_tolerance_s,
            )
            selected_valid = bool(
                self.valid_pmi_mask[selected_pmi.i12, selected_pmi.i11].item()
            )
            center_valid = bool(
                self.valid_pmi_mask[center_pmi.i12, center_pmi.i11].item()
            )

            return SampleRecord(
                schema_version=DATASET_SCHEMA_VERSION,
                sample_id=f"{cfg.scene.scene_id}_ue_{ue_index:06d}",
                ue_index=ue_index,
                ue_x_m=float(position[0]),
                ue_y_m=float(position[1]),
                ue_z_m=float(position[2]),
                tx_ue_distance_2d_m=distance_2d,
                tx_ue_distance_3d_m=distance_3d,
                seed=cfg.solver.seed,
                scene_id=cfg.scene.scene_id,
                config_hash=cfg.config_hash(),
                git_commit=self.git_commit,
                status="ok",
                has_los=has_los,
                los_classification_method="delay_matches_geometric_distance",
                total_path_count=delay.total_path_count,
                valid_path_count=delay.valid_path_count,
                usable_path_count=delay.usable_path_count,
                pmi_i11=selected_pmi.i11,
                pmi_i12=selected_pmi.i12,
                pmi_i2=selected_pmi.i2,
                beam_index=wideband_index,
                selected_pmi_valid=selected_valid,
                center_pmi_i11=center_pmi.i11,
                center_pmi_i12=center_pmi.i12,
                center_pmi_i2=center_pmi.i2,
                center_beam_index=center_index,
                center_selected_pmi_valid=center_valid,
                center_and_wideband_pmi_differ=center_index != wideband_index,
                center_beam_sweep_candidate_count=cfg.codebook.total_beams,
                wideband_beam_sweep_candidate_count=cfg.codebook.total_beams,
                normal_rx_power_dbm=link.normal_rx_power_dbm,
                sleep_rx_power_dbm=link.sleep_rx_power_dbm,
                normal_snr_db=link.normal_snr_db,
                sleep_snr_db=link.sleep_snr_db,
                normal_mean_subcarrier_rx_power_dbm=(
                    link.normal_mean_subcarrier_rx_power_dbm
                ),
                sleep_mean_subcarrier_rx_power_dbm=(
                    link.sleep_mean_subcarrier_rx_power_dbm
                ),
                normal_mean_subcarrier_snr_db=link.normal_mean_subcarrier_snr_db,
                sleep_mean_subcarrier_snr_db=link.sleep_mean_subcarrier_snr_db,
                normal_snr_p5_db=link.normal_snr_p5_db,
                normal_snr_p50_db=link.normal_snr_p50_db,
                normal_snr_p95_db=link.normal_snr_p95_db,
                sleep_snr_p5_db=link.sleep_snr_p5_db,
                sleep_snr_p50_db=link.sleep_snr_p50_db,
                sleep_snr_p95_db=link.sleep_snr_p95_db,
                rx_power_loss_db=link.rx_power_loss_db,
                snr_loss_db=link.snr_loss_db,
                mean_subcarrier_loss_db=link.mean_subcarrier_loss_db,
                power_weighted_delay_s=delay.power_weighted_delay_s,
                rms_delay_spread_s=delay.rms_delay_spread_s,
                first_path_delay_s=delay.first_path_delay_s,
                strongest_path_delay_s=delay.strongest_path_delay_s,
                last_path_delay_s=delay.last_path_delay_s,
                total_beamformed_path_power=delay.total_beamformed_path_power,
                normal_tx_power_w=link.normal_tx_power_w,
                sleep_tx_power_w=link.sleep_tx_power_w,
                per_subcarrier_tx_power_w=link.per_subcarrier_tx_power_w,
                per_subcarrier_noise_power_w=link.per_subcarrier_noise_power_w,
                num_active_subcarriers=cfg.ofdm.num_active_subcarriers,
                solver_runtime_s=solver_runtime,
                cfr_runtime_s=cfr_runtime,
                total_runtime_s=time.perf_counter() - started,
            )
        finally:
            if cfg.scene.rx_name in self.scene.receivers:
                self.scene.remove(cfg.scene.rx_name)
