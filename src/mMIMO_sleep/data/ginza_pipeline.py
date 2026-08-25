"""可断点续跑的 Ginza 多 UE normal/sleep 数据流水线。"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ginza_config import GinzaPipelineConfig
from .ginza_sample import GinzaSimulation, NoValidPathsError, load_ue_positions
from .schema import DATASET_SCHEMA_VERSION, SampleRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """一次执行的处理数量和最终合并数据表位置。"""

    requested_count: int
    processed_count: int
    reused_count: int
    dataset_count: int
    success_count: int
    failure_count: int
    output_directory: str
    table_path: str
    config_hash: str


def resolve_git_commit(project_root: str | Path | None = None) -> str | None:
    """尽量记录代码版本；目录不是 Git 仓库时返回 None。"""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _atomic_write_json(path: Path, values: dict[str, Any]) -> None:
    """先写入同目录临时文件，再原子替换，避免留下半条 checkpoint。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(values, temporary, ensure_ascii=False, allow_nan=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


class CheckpointStore:
    """每个 UE 一个 JSON 分片；manifest 阻止不同物理配置混写。"""

    def __init__(self, config: GinzaPipelineConfig):
        self.config = config
        self.directory = Path(config.output.directory)
        self.parts_directory = self.directory / "parts"
        self.manifest_path = self.directory / "manifest.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parts_directory.mkdir(parents=True, exist_ok=True)
        self._check_or_create_manifest()

    def _check_or_create_manifest(self) -> None:
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("schema_version") != DATASET_SCHEMA_VERSION:
                raise ValueError("Existing dataset has an incompatible schema version.")
            if existing.get("config_hash") != self.config.config_hash():
                raise ValueError(
                    "Existing dataset uses a different physical configuration; "
                    "choose another output directory instead of mixing samples."
                )
            return
        if any(self.parts_directory.glob("ue_*.json")):
            raise ValueError("Checkpoint parts exist but manifest.json is missing.")
        _atomic_write_json(
            self.manifest_path,
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "config_hash": self.config.config_hash(),
                "physical_configuration": self.config.physics_dict(),
                "columns": list(SampleRecord.column_names()),
            },
        )

    def sample_path(self, ue_index: int) -> Path:
        return self.parts_directory / f"ue_{ue_index:07d}.json"

    def load_sample(self, ue_index: int) -> SampleRecord | None:
        path = self.sample_path(ue_index)
        if not path.exists():
            return None
        record = SampleRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if record.ue_index != ue_index:
            raise ValueError(f"Checkpoint index does not match its filename: {path}")
        if record.config_hash != self.config.config_hash():
            raise ValueError(f"Checkpoint uses a different physical config: {path}")
        return record

    def write_sample(self, record: SampleRecord) -> None:
        if record.config_hash != self.config.config_hash():
            raise ValueError("Sample configuration hash does not match this dataset.")
        _atomic_write_json(self.sample_path(record.ue_index), record.to_dict())

    def iter_records(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        for path in sorted(self.parts_directory.glob("ue_*.json")):
            record = SampleRecord.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if record.config_hash != self.config.config_hash():
                raise ValueError(f"Checkpoint uses a different physical config: {path}")
            records.append(record)
        return records


def _resolve_table_format(requested: str) -> str:
    if requested == "auto":
        return "parquet" if importlib.util.find_spec("pyarrow") else "jsonl"
    if requested == "parquet" and importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError(
            "Parquet output requires pyarrow. Choose --format jsonl/csv or install "
            "pyarrow explicitly after reviewing your environment."
        )
    return requested


def _temporary_table_path(final_path: Path) -> Path:
    handle, raw_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    )
    os.close(handle)
    return Path(raw_path)


def consolidate_dataset(
    records: Sequence[SampleRecord],
    directory: str | Path,
    table_format: str,
) -> Path:
    """按 UE index 合并 checkpoint；支持 JSONL、CSV 和可选 Parquet。"""

    resolved_format = _resolve_table_format(table_format)
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / f"ginza_dataset.{resolved_format}"
    temporary_path = _temporary_table_path(final_path)
    ordered = sorted(records, key=lambda record: record.ue_index)
    try:
        if resolved_format == "jsonl":
            with temporary_path.open("w", encoding="utf-8") as output:
                for record in ordered:
                    output.write(
                        json.dumps(
                            record.to_dict(),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    )
                    output.write("\n")
        elif resolved_format == "csv":
            with temporary_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=SampleRecord.column_names())
                writer.writeheader()
                writer.writerows(record.to_dict() for record in ordered)
        elif resolved_format == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pylist([record.to_dict() for record in ordered])
            pq.write_table(table, temporary_path)
        else:
            raise ValueError(f"Unsupported table format: {resolved_format!r}")
        os.replace(temporary_path, final_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return final_path


def _failure_record(
    config: GinzaPipelineConfig,
    ue_index: int,
    position: np.ndarray,
    exception: BaseException,
    git_commit: str | None,
) -> SampleRecord:
    coordinates = np.asarray(position, dtype=np.float64).reshape(3)
    displacement = coordinates - np.asarray(config.scene.tx_position_m)
    status = "no_valid_paths" if isinstance(exception, NoValidPathsError) else "error"
    return SampleRecord(
        schema_version=DATASET_SCHEMA_VERSION,
        sample_id=f"{config.scene.scene_id}_ue_{ue_index:06d}",
        ue_index=ue_index,
        ue_x_m=float(coordinates[0]),
        ue_y_m=float(coordinates[1]),
        ue_z_m=float(coordinates[2]),
        tx_ue_distance_2d_m=float(np.linalg.norm(displacement[:2])),
        tx_ue_distance_3d_m=float(np.linalg.norm(displacement)),
        seed=config.solver.seed,
        scene_id=config.scene.scene_id,
        config_hash=config.config_hash(),
        git_commit=git_commit,
        status=status,
        error_message=f"{type(exception).__name__}: {exception}",
    )


def run_pipeline(
    config: GinzaPipelineConfig,
    *,
    positions: Sequence[np.ndarray] | None = None,
    sample_builder: Callable[[int, np.ndarray], SampleRecord] | None = None,
    project_root: str | Path | None = None,
) -> PipelineSummary:
    """处理指定 UE 区间，单 UE 失败不会使其他已完成样本丢失。

    ``sample_builder`` 仅用于离线单元测试或替换仿真实现；真实运行默认
    使用 ``GinzaSimulation``，并在第一个需要处理的 UE 出现时惰性初始化。
    """

    loaded_positions = (
        list(positions) if positions is not None else load_ue_positions(config)
    )
    start = config.output.start_index
    requested_stop = config.output.stop_index
    stop = len(loaded_positions) if requested_stop is None else requested_stop
    if start > len(loaded_positions) or stop > len(loaded_positions):
        raise ValueError(
            f"Requested UE range [{start}, {stop}) exceeds "
            f"the available {len(loaded_positions)} positions."
        )
    store = CheckpointStore(config)
    git_commit = resolve_git_commit(project_root)
    simulation: GinzaSimulation | None = None
    processed_count = 0
    reused_count = 0

    for ue_index in range(start, stop):
        previous = store.load_sample(ue_index)
        if previous is not None:
            if not config.output.resume:
                raise FileExistsError(
                    f"Checkpoint already exists for UE {ue_index}; use --resume "
                    "or choose a different output directory."
                )
            if previous.status == "ok" or not config.output.retry_failed:
                reused_count += 1
                LOGGER.info("Reusing UE %s checkpoint (%s)", ue_index, previous.status)
                continue

        position = np.asarray(loaded_positions[ue_index], dtype=np.float64)
        try:
            if sample_builder is None:
                if simulation is None:
                    simulation = GinzaSimulation(config, git_commit=git_commit)
                record = simulation.simulate(ue_index, position)
            else:
                record = sample_builder(ue_index, position)
            if record.ue_index != ue_index:
                raise ValueError("Sample builder returned an unexpected UE index.")
        except Exception as exc:
            record = _failure_record(config, ue_index, position, exc, git_commit)
            store.write_sample(record)
            processed_count += 1
            LOGGER.exception("UE %s failed; checkpoint retained", ue_index)
            if config.output.fail_fast:
                raise
        else:
            store.write_sample(record)
            processed_count += 1
            LOGGER.info(
                "UE %s complete: PMI=(%s,%s,%s), valid=%s, normal=%.3f dBm, "
                "sleep=%.3f dBm",
                ue_index,
                record.pmi_i11,
                record.pmi_i12,
                record.pmi_i2,
                record.selected_pmi_valid,
                record.normal_rx_power_dbm,
                record.sleep_rx_power_dbm,
            )

    records = store.iter_records()
    table_path = consolidate_dataset(
        records,
        store.directory,
        config.output.table_format,
    )
    success_count = sum(record.status == "ok" for record in records)
    summary = PipelineSummary(
        requested_count=stop - start,
        processed_count=processed_count,
        reused_count=reused_count,
        dataset_count=len(records),
        success_count=success_count,
        failure_count=len(records) - success_count,
        output_directory=str(store.directory.resolve()),
        table_path=str(table_path.resolve()),
        config_hash=config.config_hash(),
    )
    _atomic_write_json(store.directory / "run_summary.json", asdict(summary))
    return summary
