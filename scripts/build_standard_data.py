"""读取Data标准化配置，将本地原始成交ZIP和25档快照转换为每日两份Parquet。"""

import argparse
from pathlib import Path

from mm_research.data.standard import run_standard_production


# 【非核心】读取Data阶段的命令行配置，将生产任务交给标准化模块。
def main() -> None:
    """输入Data配置路径，生成或复用对应版本的标准盘口和逐笔文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path,
        help="Data标准化配置，例如config/data/standardize_30d.yaml",
    )
    arguments = parser.parse_args()
    run_standard_production(arguments.config)


if __name__ == "__main__":
    main()
