"""checkpoint、失败重试及物理配置隔离的无 GPU 测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mMIMO_sleep.data.ginza_config import GinzaPipelineConfig
from mMIMO_sleep.data.ginza_pipeline import run_pipeline
from mMIMO_sleep.data.schema import DATASET_SCHEMA_VERSION, SampleRecord


def success_record(config: GinzaPipelineConfig, ue_index: int) -> SampleRecord:
    return SampleRecord(
        schema_version=DATASET_SCHEMA_VERSION,
        sample_id=f"{config.scene.scene_id}_ue_{ue_index:06d}",
        ue_index=ue_index,
        ue_x_m=float(ue_index),
        ue_y_m=0.0,
        ue_z_m=1.5,
        tx_ue_distance_2d_m=10.0,
        tx_ue_distance_3d_m=11.0,
        seed=config.solver.seed,
        scene_id=config.scene.scene_id,
        config_hash=config.config_hash(),
        git_commit=None,
        status="ok",
        valid_path_count=1,
        usable_path_count=1,
        pmi_i11=ue_index,
        pmi_i12=0,
        pmi_i2=0,
        beam_index=ue_index * 4,
        selected_pmi_valid=ue_index != 1,
        normal_rx_power_dbm=-107.0,
        sleep_rx_power_dbm=-112.0,
        normal_snr_db=17.0,
        sleep_snr_db=12.0,
        power_weighted_delay_s=1e-6,
        rms_delay_spread_s=0.0,
    )


class PipelineResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        base = GinzaPipelineConfig()
        output = replace(
            base.output,
            directory=self.temporary_directory.name,
            table_format="jsonl",
        )
        self.config = replace(base, output=output)
        self.positions = [np.array([float(index), 0.0, 1.5]) for index in range(3)]

    def test_resume_only_processes_missing_positions(self) -> None:
        first_config = replace(
            self.config,
            output=replace(self.config.output, stop_index=2),
        )
        first = run_pipeline(
            first_config,
            positions=self.positions,
            sample_builder=lambda index, _: success_record(first_config, index),
        )
        self.assertEqual(first.processed_count, 2)

        resume_config = replace(
            self.config,
            output=replace(self.config.output, resume=True),
        )
        processed: list[int] = []

        def builder(index: int, _: np.ndarray) -> SampleRecord:
            processed.append(index)
            return success_record(resume_config, index)

        resumed = run_pipeline(
            resume_config,
            positions=self.positions,
            sample_builder=builder,
        )
        self.assertEqual(processed, [2])
        self.assertEqual(resumed.reused_count, 2)
        self.assertEqual(resumed.dataset_count, 3)
        rows = Path(resumed.table_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 3)
        self.assertFalse(json.loads(rows[1])["selected_pmi_valid"])

    def test_existing_sample_requires_explicit_resume(self) -> None:
        config = replace(
            self.config,
            output=replace(self.config.output, stop_index=1),
        )
        run_pipeline(
            config,
            positions=self.positions,
            sample_builder=lambda index, _: success_record(config, index),
        )
        with self.assertRaises(FileExistsError):
            run_pipeline(
                config,
                positions=self.positions,
                sample_builder=lambda index, _: success_record(config, index),
            )

    def test_failed_sample_is_checkpointed_and_can_be_retried(self) -> None:
        config = replace(
            self.config,
            output=replace(self.config.output, stop_index=2),
        )

        def first_builder(index: int, _: np.ndarray) -> SampleRecord:
            if index == 1:
                raise RuntimeError("synthetic solver failure")
            return success_record(config, index)

        with self.assertLogs("mMIMO_sleep.data.ginza_pipeline", level="ERROR"):
            first = run_pipeline(
                config,
                positions=self.positions,
                sample_builder=first_builder,
            )
        self.assertEqual(first.success_count, 1)
        self.assertEqual(first.failure_count, 1)

        retry_config = replace(
            config,
            output=replace(config.output, resume=True, retry_failed=True),
        )
        retried = run_pipeline(
            retry_config,
            positions=self.positions,
            sample_builder=lambda index, _: success_record(retry_config, index),
        )
        self.assertEqual(retried.reused_count, 1)
        self.assertEqual(retried.processed_count, 1)
        self.assertEqual(retried.failure_count, 0)

    def test_configuration_hash_prevents_mixed_physics(self) -> None:
        config = replace(
            self.config,
            output=replace(self.config.output, stop_index=1),
        )
        run_pipeline(
            config,
            positions=self.positions,
            sample_builder=lambda index, _: success_record(config, index),
        )
        incompatible = replace(
            config,
            solver=replace(config.solver, seed=config.solver.seed + 1),
            output=replace(config.output, resume=True),
        )
        with self.assertRaisesRegex(ValueError, "different physical configuration"):
            run_pipeline(
                incompatible,
                positions=self.positions,
                sample_builder=lambda index, _: success_record(incompatible, index),
            )

    def test_requested_range_cannot_exceed_positions(self) -> None:
        config = replace(
            self.config,
            output=replace(self.config.output, stop_index=10),
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            run_pipeline(config, positions=self.positions)


if __name__ == "__main__":
    unittest.main()
