"""市场状态条件交易价值实验入口。"""

import argparse
from pathlib import Path

from mm_research.models.state_value import prepare_state_volatility, build_state_value


# 【非核心】状态缓存准备与逐种子价值映射独立调度，不混入策略选择。
def main() -> None:
    """读取明确配置后计算指定阶段，现有GPU预测直接复用。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['prepare', 'values'])
    parser.add_argument('--seed', type=int, choices=[17, 42, 73])
    arguments = parser.parse_args()
    if arguments.stage == 'prepare':
        prepare_state_volatility(arguments.config)
    else:
        if arguments.seed is None:
            parser.error('values需要--seed')
        build_state_value(arguments.config, arguments.seed)


if __name__ == '__main__':
    main()
