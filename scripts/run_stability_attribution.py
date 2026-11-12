"""汇总本轮策略的收益环境与方向归因。"""

import argparse
from pathlib import Path

from mm_research.strategies.stability_attribution import attribute_stability


# 【非核心】显式读取冻结策略配置，只分析已有交易账本。
def main() -> None:
    """输出与原案例名称一一对应的分组归因。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    arguments = parser.parse_args()
    attribute_stability(arguments.config)


if __name__ == '__main__':
    main()
