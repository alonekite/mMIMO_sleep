"""Ginza normal/sleep 数据集的字段约定与轻量验证。

每条记录表示一个 UE 位置及其对应的 normal/sleep 仿真结果。MCS 不属于
Sionna RT 仿真输出，因此有意不包含在第一版 schema 中。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

DATASET_SCHEMA_VERSION = "ginza-normal-sleep-v1"


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """一个 UE 的结构化仿真结果。

    ``normal_rx_power_dbm`` 和 ``sleep_rx_power_dbm`` 表示：先对各有效
    子载波的接收功率在线性域求平均，再转换为 dBm。它们不等于各子载波
    dBm 值的算术平均；后者保存在独立的 ``*_mean_subcarrier_*`` 字段中。

    ``selected_pmi_valid`` 只是 metadata，不得用于过滤正常模式波束扫描。
    """

    schema_version: str
    sample_id: str
    ue_index: int
    ue_x_m: float
    ue_y_m: float
    ue_z_m: float
    tx_ue_distance_2d_m: float
    tx_ue_distance_3d_m: float
    seed: int
    scene_id: str
    config_hash: str
    git_commit: str | None
    status: str
    error_message: str | None = None

    has_los: bool | None = None
    los_classification_method: str | None = None
    total_path_count: int | None = None
    valid_path_count: int | None = None
    usable_path_count: int | None = None

    pmi_i11: int | None = None
    pmi_i12: int | None = None
    pmi_i2: int | None = None
    beam_index: int | None = None
    selected_pmi_valid: bool | None = None

    center_pmi_i11: int | None = None
    center_pmi_i12: int | None = None
    center_pmi_i2: int | None = None
    center_beam_index: int | None = None
    center_selected_pmi_valid: bool | None = None
    center_and_wideband_pmi_differ: bool | None = None
    center_beam_sweep_candidate_count: int | None = None
    wideband_beam_sweep_candidate_count: int | None = None

    normal_rx_power_dbm: float | None = None
    sleep_rx_power_dbm: float | None = None
    normal_snr_db: float | None = None
    sleep_snr_db: float | None = None
    normal_mean_subcarrier_rx_power_dbm: float | None = None
    sleep_mean_subcarrier_rx_power_dbm: float | None = None
    normal_mean_subcarrier_snr_db: float | None = None
    sleep_mean_subcarrier_snr_db: float | None = None
    normal_snr_p5_db: float | None = None
    normal_snr_p50_db: float | None = None
    normal_snr_p95_db: float | None = None
    sleep_snr_p5_db: float | None = None
    sleep_snr_p50_db: float | None = None
    sleep_snr_p95_db: float | None = None
    rx_power_loss_db: float | None = None
    snr_loss_db: float | None = None
    mean_subcarrier_loss_db: float | None = None

    power_weighted_delay_s: float | None = None
    rms_delay_spread_s: float | None = None
    first_path_delay_s: float | None = None
    strongest_path_delay_s: float | None = None
    last_path_delay_s: float | None = None
    total_beamformed_path_power: float | None = None

    normal_tx_power_w: float | None = None
    sleep_tx_power_w: float | None = None
    per_subcarrier_tx_power_w: float | None = None
    per_subcarrier_noise_power_w: float | None = None
    num_active_subcarriers: int | None = None
    solver_runtime_s: float | None = None
    cfr_runtime_s: float | None = None
    total_runtime_s: float | None = None

    ALLOWED_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"ok", "no_valid_paths", "error"}
    )

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version!r}; "
                f"expected {DATASET_SCHEMA_VERSION!r}."
            )
        if not isinstance(self.ue_index, int) or self.ue_index < 0:
            raise ValueError(f"ue_index must be non-negative, got {self.ue_index!r}.")
        if self.status not in self.ALLOWED_STATUSES:
            raise ValueError(f"Unsupported sample status: {self.status!r}.")
        for name in (
            "ue_x_m",
            "ue_y_m",
            "ue_z_m",
            "tx_ue_distance_2d_m",
            "tx_ue_distance_3d_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")
        if self.status == "ok":
            self._validate_success()
        elif not self.error_message:
            raise ValueError("Unsuccessful records must include error_message.")

    def _validate_success(self) -> None:
        required_names = (
            "pmi_i11",
            "pmi_i12",
            "pmi_i2",
            "beam_index",
            "selected_pmi_valid",
            "normal_rx_power_dbm",
            "sleep_rx_power_dbm",
            "normal_snr_db",
            "sleep_snr_db",
            "power_weighted_delay_s",
            "rms_delay_spread_s",
            "valid_path_count",
            "usable_path_count",
        )
        missing = [name for name in required_names if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Successful sample is missing fields: {missing}.")
        if self.valid_path_count is not None and self.valid_path_count <= 0:
            raise ValueError("Successful samples must have at least one valid path.")
        if self.usable_path_count is not None and self.usable_path_count <= 0:
            raise ValueError("Successful samples must have at least one usable path.")
        if self.rms_delay_spread_s is not None and self.rms_delay_spread_s < 0:
            raise ValueError("rms_delay_spread_s cannot be negative.")
        for name in (
            "normal_rx_power_dbm",
            "sleep_rx_power_dbm",
            "normal_snr_db",
            "sleep_snr_db",
            "power_weighted_delay_s",
            "rms_delay_spread_s",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")

    def to_dict(self) -> dict[str, Any]:
        """返回能够直接写入 JSON、CSV 或 Parquet 的普通 Python 值。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SampleRecord:
        """从 checkpoint 分片恢复并重新执行 schema 校验。"""

        return cls(**values)

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        """按照稳定顺序返回所有主数据表字段。"""

        return tuple(item.name for item in fields(cls))


ML_INPUT_FIELDS: tuple[str, ...] = (
    "normal_rx_power_dbm",
    "normal_snr_db",
    "pmi_i11",
    "pmi_i12",
    "pmi_i2",
    "power_weighted_delay_s",
    "rms_delay_spread_s",
)

LABEL_FIELDS: tuple[str, ...] = (
    "sleep_rx_power_dbm",
    "sleep_snr_db",
    "rx_power_loss_db",
    "snr_loss_db",
)

METADATA_FIELDS: tuple[str, ...] = (
    "sample_id",
    "ue_index",
    "selected_pmi_valid",
    "has_los",
    "config_hash",
    "git_commit",
    "status",
)
