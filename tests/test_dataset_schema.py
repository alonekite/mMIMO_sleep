"""Ginza 数据表 schema 和 ML 特征边界的离线测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from mMIMO_sleep.data.ginza_config import (
    GinzaPipelineConfig,
    config_from_mapping,
    load_config,
)
from mMIMO_sleep.data.schema import (
    DATASET_SCHEMA_VERSION,
    LABEL_FIELDS,
    METADATA_FIELDS,
    ML_INPUT_FIELDS,
    SampleRecord,
)


def make_success_record(
    config: GinzaPipelineConfig,
    ue_index: int = 0,
    *,
    selected_pmi_valid: bool = True,
) -> SampleRecord:
    """构造不依赖 GPU 和 Sionna 的最小合法成功样本。"""

    return SampleRecord(
        schema_version=DATASET_SCHEMA_VERSION,
        sample_id=f"{config.scene.scene_id}_ue_{ue_index:06d}",
        ue_index=ue_index,
        ue_x_m=float(ue_index),
        ue_y_m=2.0,
        ue_z_m=1.5,
        tx_ue_distance_2d_m=10.0,
        tx_ue_distance_3d_m=12.0,
        seed=config.solver.seed,
        scene_id=config.scene.scene_id,
        config_hash=config.config_hash(),
        git_commit="test-commit",
        status="ok",
        valid_path_count=2,
        usable_path_count=2,
        pmi_i11=23,
        pmi_i12=1,
        pmi_i2=0,
        beam_index=220,
        selected_pmi_valid=selected_pmi_valid,
        normal_rx_power_dbm=-107.0,
        sleep_rx_power_dbm=-112.0,
        normal_snr_db=17.0,
        sleep_snr_db=12.0,
        power_weighted_delay_s=1.45e-6,
        rms_delay_spread_s=145e-9,
    )


class DatasetSchemaTest(unittest.TestCase):
    def test_success_record_round_trip(self) -> None:
        record = make_success_record(GinzaPipelineConfig())
        self.assertEqual(SampleRecord.from_dict(record.to_dict()), record)
        self.assertEqual(tuple(record.to_dict()), SampleRecord.column_names())

    def test_invalid_pmi_is_metadata_not_rejection(self) -> None:
        record = make_success_record(
            GinzaPipelineConfig(),
            selected_pmi_valid=False,
        )
        self.assertEqual(record.status, "ok")
        self.assertFalse(record.selected_pmi_valid)

    def test_success_requires_complete_ml_fields(self) -> None:
        values = make_success_record(GinzaPipelineConfig()).to_dict()
        values["normal_snr_db"] = None
        with self.assertRaisesRegex(ValueError, "missing fields"):
            SampleRecord.from_dict(values)

    def test_failure_requires_error_message(self) -> None:
        values = make_success_record(GinzaPipelineConfig()).to_dict()
        values["status"] = "error"
        values["error_message"] = None
        with self.assertRaisesRegex(ValueError, "error_message"):
            SampleRecord.from_dict(values)

    def test_feature_and_label_boundaries(self) -> None:
        self.assertIn("power_weighted_delay_s", ML_INPUT_FIELDS)
        self.assertIn("rms_delay_spread_s", ML_INPUT_FIELDS)
        self.assertIn("selected_pmi_valid", METADATA_FIELDS)
        self.assertTrue(all(not name.startswith("sleep_") for name in ML_INPUT_FIELDS))
        self.assertTrue(set(ML_INPUT_FIELDS).isdisjoint(LABEL_FIELDS))
        self.assertFalse(
            any("mcs" in name.lower() for name in SampleRecord.column_names())
        )

    def test_output_configuration_does_not_change_physics_hash(self) -> None:
        original = GinzaPipelineConfig()
        alternative = config_from_mapping(
            {"output": {"directory": "elsewhere", "resume": True}}
        )
        self.assertEqual(original.config_hash(), alternative.config_hash())

    def test_physics_change_changes_hash(self) -> None:
        original = GinzaPipelineConfig()
        changed = config_from_mapping({"solver": {"seed": 37}})
        self.assertNotEqual(original.config_hash(), changed.config_hash())

    def test_unknown_configuration_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown"):
            config_from_mapping({"solver": {"unexpected_parameter": 1}})

    def test_supplied_yaml_retains_numeric_los_tolerance(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "ginza_dataset_v1.yaml"
        )
        config = load_config(config_path)
        self.assertIsInstance(config.scene.los_delay_tolerance_s, float)
        self.assertAlmostEqual(config.scene.los_delay_tolerance_s, 5e-9)


if __name__ == "__main__":
    unittest.main()
