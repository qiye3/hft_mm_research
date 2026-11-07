"""读取Features/Targets配置，从标准行情生成分组特征和独立多周期标签。"""

import argparse
from pathlib import Path

from mm_research.data.dataset import run_dataset_production


# 【非核心】从命令行选择生产配置，允许显式重建已有日期。
def main() -> None:
    """输入配置路径和重建标志，执行按日数据生产并返回进程状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Features/Targets配置，例如config/features/build_30d.yaml")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--only", choices=["all", "features", "labels"], default="all",
        help="生成范围，features为特征，labels为预测标签，all为两者",
    )
    arguments = parser.parse_args()
    run_dataset_production(arguments.config, arguments.rebuild, arguments.only)


if __name__ == "__main__":
    main()
