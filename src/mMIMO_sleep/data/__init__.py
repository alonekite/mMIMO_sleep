"""Ginza 数据集生成模块。

这里故意不导入 Sionna RT、Mitsuba 或 PyTorch：schema 和断点续跑逻辑
应该能够在没有 GPU 的本地开发环境中被导入、检查与单元测试。
"""

from mMIMO_sleep.data.ginza_config import GinzaPipelineConfig, load_config
from mMIMO_sleep.data.schema import DATASET_SCHEMA_VERSION, SampleRecord

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "GinzaPipelineConfig",
    "SampleRecord",
    "load_config",
]
