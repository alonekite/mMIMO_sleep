#!/usr/bin/env python3
"""生成 Ginza normal/sleep 数据集，支持断点续跑与区间采样。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mMIMO_sleep.data.ginza_config import load_config  # noqa: E402
from mMIMO_sleep.data.ginza_pipeline import run_pipeline  # noqa: E402
from mMIMO_sleep.data.schema import SampleRecord  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成支持断点续跑的 Ginza normal/sleep 宽带数据集。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ginza_dataset_v1.yaml",
        help="JSON-compatible YAML 配置文件路径。",
    )
    parser.add_argument("--start-index", type=int, help="起始 UE 索引，包含。")
    parser.add_argument("--stop-index", type=int, help="结束 UE 索引，不包含。")
    parser.add_argument("--max-samples", type=int, help="本次最多处理的 UE 数。")
    parser.add_argument("--output-dir", type=str, help="输出目录。")
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "csv", "parquet"),
        help="最终主数据表格式；checkpoint 始终是 JSON。",
    )
    parser.add_argument("--resume", action="store_true", help="跳过已有样本。")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="断点续跑时重新处理失败样本。",
    )
    parser.add_argument("--fail-fast", action="store_true", help="失败后立即停止。")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="仅打印解析后的配置及 config hash，不加载 GPU。",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="仅打印主数据表字段，不运行仿真。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.show_schema:
        for name in SampleRecord.column_names():
            print(name)
        return 0

    config = load_config(arguments.config)
    start = (
        arguments.start_index
        if arguments.start_index is not None
        else config.output.start_index
    )
    if arguments.stop_index is not None and arguments.max_samples is not None:
        parser.error("--stop-index and --max-samples cannot be supplied together.")
    if arguments.max_samples is not None:
        if arguments.max_samples <= 0:
            parser.error("--max-samples must be positive.")
        stop = start + arguments.max_samples
    else:
        stop = (
            arguments.stop_index
            if arguments.stop_index is not None
            else config.output.stop_index
        )
    if arguments.retry_failed and not (
        arguments.resume or config.output.resume
    ):
        parser.error("--retry-failed requires --resume.")
    output = replace(
        config.output,
        directory=arguments.output_dir or config.output.directory,
        table_format=arguments.format or config.output.table_format,
        start_index=start,
        stop_index=stop,
        resume=arguments.resume or config.output.resume,
        retry_failed=arguments.retry_failed or config.output.retry_failed,
        fail_fast=arguments.fail_fast or config.output.fail_fast,
    )
    config = replace(config, output=output)
    if arguments.print_config:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        print(f"config_hash: {config.config_hash()}")
        return 0

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = run_pipeline(config, project_root=PROJECT_ROOT)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0 if summary.failure_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
