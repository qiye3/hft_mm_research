"""解释冻结费用分类策略的波动环境与尾部风险。"""

import argparse
from pathlib import Path

from mm_research.strategies.cost_attribution import attribute_cost_trades, compare_native_execution, compare_trade_directions


# 【非核心】读取归因配置，复用已完成的逐笔账本，不重新选择模型或交易参数。
def main() -> None:
    """将现有成交账本和秒级盘口交给归因模块。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument('--native-output', type=Path)
    comparison.add_argument('--direction-comparison', action='store_true')
    arguments = parser.parse_args()
    if arguments.direction_comparison:
        compare_trade_directions(arguments.config)
    elif arguments.native_output is None:
        attribute_cost_trades(arguments.config)
    else:
        compare_native_execution(arguments.config, arguments.native_output)


if __name__ == '__main__':
    main()
