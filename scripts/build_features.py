"""读取Features配置，从标准行情生成100ms分组特征。"""

import argparse
from pathlib import Path

from mm_research.data.dataset import run_dataset_production


# 【非核心】从命令行选择生产配置，允许显式重建已有日期。
def main() -> None:
    """输入配置路径和重建标志，执行按日数据生产并返回进程状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Features配置，例如config/features/build_30d.yaml")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="有界基础采样进程数，线程数由POLARS_MAX_THREADS控制")
    arguments = parser.parse_args()
    run_dataset_production(arguments.config, arguments.rebuild, arguments.workers)


if __name__ == "__main__":
    main()
