"""从标准行情独立核对指定日期的100ms基础表。"""

import argparse
import json
from pathlib import Path

from mm_research.config import load_pipeline_config
from mm_research.data.storage import file_locks, scan_series
from mm_research.features.base import BASE_COLUMNS, KEY_COLUMNS
from mm_research.features.validation import validate_feature_intermediates


# 【非核心】使用与生产一致的配置定位输入，独立参考计算由质量模块完成。
def main() -> None:
    """输入研究配置、日期和报告路径，比较标准行情与基础采样。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Features/Targets配置，例如config/features/build_30d.yaml")
    parser.add_argument("date")
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_pipeline_config(arguments.config, "features")
    folder = root / config["feature_root"] / config["feature_version"]
    columns = [column for column in BASE_COLUMNS if column not in KEY_COLUMNS]
    with file_locks([folder / f"{name}.parquet" for name in columns], shared=True):
        scan, _ = scan_series(folder, columns, [arguments.date])
        base = scan.collect(engine="streaming")
    standard = root / config["standard_root"] / config["standard_version"]
    report = validate_feature_intermediates(
        base, standard, arguments.date, config["market_data_latency_ns"], config["interval_ns"],
        config["lot_size"], config["max_book_age_ns"],
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
