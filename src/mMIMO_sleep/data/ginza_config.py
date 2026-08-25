"""Ginza 数据流水线配置。

配置文件采用 JSON 兼容的 YAML：安装 PyYAML 时按 YAML 读取，未安装时
仍可由 Python 标准库 json 解析，不强制改变现有项目依赖。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SceneConfig:
    scene_xml_path: str = (
        "/workspace/Study_Sionna/Projects/TAP_TCB_Resource/"
        "Ginza_012/ginza_1/ginza_1.xml"
    )
    ue_positions_path: str = (
        "/workspace/Study_Sionna/Projects/TAP_TCB_Resource/"
        "Ginza_012/Results/rx_1000.pkl"
    )
    ue_positions_key: str = "all_rx_positions"
    scene_id: str = "ginza_1"
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    temperature_k: float = 290.0
    tx_position_m: tuple[float, float, float] = (-122.0, -108.5, 41.0)
    tx_orientation_rad: tuple[float, float, float] = (
        0.0,
        15.0 / 360.0 * math.pi,
        0.0,
    )
    tx_name: str = "tx"
    rx_name: str = "rx"
    material_name: str = "my_concrete"
    material_itu_type: str = "concrete"
    material_thickness_m: float = 0.5
    material_scattering_coefficient: float = 0.3
    material_xpd_coefficient: float = 0.3
    require_concrete_override: bool = True
    los_delay_tolerance_s: float = 5e-9


@dataclass(frozen=True, slots=True)
class AntennaConfig:
    num_subarray_rows: int = 4
    num_horizontal: int = 8
    elements_per_subarray: int = 2
    num_polarizations: int = 2
    tx_pattern: str = "tr38901"
    tx_polarization: str = "cross"
    rx_pattern: str = "iso"
    rx_polarization: str = "VH"
    vertical_spacing_wavelengths: float = 0.5
    horizontal_spacing_wavelengths: float = 0.5


@dataclass(frozen=True, slots=True)
class CodebookConfig:
    num_vertical_beams: int = 8
    num_horizontal_beams: int = 32
    num_i2: int = 4
    validity_target_loss_db: float = 10.0 * math.log10(4.0)
    validity_tolerance_db: float = 1.0
    validity_grid_points: int = 121
    beam_chunk_size: int = 128

    @property
    def total_beams(self) -> int:
        return (
            self.num_vertical_beams
            * self.num_horizontal_beams
            * self.num_i2
        )


@dataclass(frozen=True, slots=True)
class OfdmConfig:
    fft_size: int = 4096
    cp_length_samples: int = 288
    subcarrier_spacing_hz: float = 30e3
    num_resource_blocks: int = 273
    subcarriers_per_resource_block: int = 12
    left_guard_subcarriers: int = 410
    right_guard_subcarriers: int = 410
    max_cfr_chunk_subcarriers: int = 512
    max_cfr_chunk_entries: int = 2**30
    tx_power_dbm: float = 51.13
    noise_figure_db: float = 5.0

    @property
    def num_active_subcarriers(self) -> int:
        return self.num_resource_blocks * self.subcarriers_per_resource_block


@dataclass(frozen=True, slots=True)
class SolverConfig:
    max_depth: int = 15
    samples_per_src: int = 1_000_000
    max_num_paths_per_src: int = 1_000_000
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = False
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = False
    seed: int = 36

    def solver_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: str = "outputs/ginza_dataset_v1"
    table_format: str = "auto"
    start_index: int = 0
    stop_index: int | None = None
    resume: bool = False
    retry_failed: bool = False
    fail_fast: bool = False


@dataclass(frozen=True, slots=True)
class GinzaPipelineConfig:
    scene: SceneConfig = field(default_factory=SceneConfig)
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
    ofdm: OfdmConfig = field(default_factory=OfdmConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mitsuba_variant: str = "cuda_ad_mono_polarized"
    torch_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.antenna.tx_polarization != "cross":
            raise ValueError("The validated Ginza TX polarization must be 'cross'.")
        if self.antenna.num_polarizations != 2:
            raise ValueError("This pipeline requires two TX polarizations.")
        if self.codebook.total_beams <= 0:
            raise ValueError("The codebook must contain at least one beam.")
        if self.ofdm.num_active_subcarriers != (
            self.ofdm.fft_size
            - self.ofdm.left_guard_subcarriers
            - self.ofdm.right_guard_subcarriers
        ):
            raise ValueError("OFDM active-subcarrier and guard definitions disagree.")
        if self.output.start_index < 0:
            raise ValueError("output.start_index must be non-negative.")
        if self.output.stop_index is not None and (
            self.output.stop_index < self.output.start_index
        ):
            raise ValueError("output.stop_index cannot precede output.start_index.")
        if self.output.table_format not in {"auto", "jsonl", "csv", "parquet"}:
            raise ValueError("table_format must be auto, jsonl, csv, or parquet.")
        if (
            not isinstance(self.scene.los_delay_tolerance_s, (int, float))
            or not math.isfinite(self.scene.los_delay_tolerance_s)
            or self.scene.los_delay_tolerance_s < 0
        ):
            raise ValueError("los_delay_tolerance_s must be finite and non-negative.")
        for name, value in (
            ("carrier_frequency_hz", self.scene.carrier_frequency_hz),
            ("bandwidth_hz", self.scene.bandwidth_hz),
            ("temperature_k", self.scene.temperature_k),
            ("subcarrier_spacing_hz", self.ofdm.subcarrier_spacing_hz),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    def physics_dict(self) -> dict[str, Any]:
        """只保留影响物理结果的配置，不包含 output 路径和 resume 选项。"""

        values = asdict(self)
        values.pop("output")
        return values

    def config_hash(self) -> str:
        encoded = json.dumps(
            self.physics_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataclass_from_mapping(cls: type[Any], values: Any) -> Any:
    if values is None:
        return cls()
    if not isinstance(values, dict):
        raise TypeError(f"{cls.__name__} configuration must be a mapping.")
    allowed = {item.name for item in fields(cls)}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(f"Unknown {cls.__name__} configuration keys: {unexpected}.")
    normalized = dict(values)
    for key in ("tx_position_m", "tx_orientation_rad"):
        if key in normalized:
            coordinates = normalized[key]
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
                raise ValueError(f"{key} must contain exactly three values.")
            normalized[key] = tuple(float(value) for value in coordinates)
    return cls(**normalized)


def config_from_mapping(values: dict[str, Any]) -> GinzaPipelineConfig:
    """将读取后的配置字典转换为带严格 key 校验的数据类。"""

    if not isinstance(values, dict):
        raise TypeError("Pipeline configuration must be a mapping.")
    sections = {
        "scene": SceneConfig,
        "antenna": AntennaConfig,
        "codebook": CodebookConfig,
        "ofdm": OfdmConfig,
        "solver": SolverConfig,
        "output": OutputConfig,
    }
    allowed = set(sections) | {"mitsuba_variant", "torch_device"}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(f"Unknown pipeline configuration sections: {unexpected}.")
    arguments = {
        name: _dataclass_from_mapping(cls, values.get(name))
        for name, cls in sections.items()
    }
    for name in ("mitsuba_variant", "torch_device"):
        if name in values:
            arguments[name] = values[name]
    return GinzaPipelineConfig(**arguments)


def load_config(path: str | Path) -> GinzaPipelineConfig:
    """读取 YAML 配置；缺少 PyYAML 时支持 JSON 兼容的 YAML 文件。"""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        try:
            values = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is unavailable and the configuration is not JSON-compatible "
                "YAML. Install PyYAML or use the supplied JSON-compatible config."
            ) from exc
    else:
        values = yaml.safe_load(text)
    return config_from_mapping(values)
