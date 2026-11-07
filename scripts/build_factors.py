"""使用Factors配置，从全日期单特征文件构造指定因子。"""

import argparse
from pathlib import Path

from mm_research.factors.dataset import run_factor_production


# 【非核心】将配置及显式重建参数交给因子生产入口。
def main() -> None:
    """命令行输入配置路径和可选重建标志，执行因子计算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="例如config/factors/gen001.yaml")
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    run_factor_production(arguments.config, arguments.rebuild)


if __name__ == "__main__":
    main()
