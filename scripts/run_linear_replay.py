"""在最终测试日运行冻结压力模型的C++实时推理与做市对照。"""

import argparse
from pathlib import Path

from mm_research.models.replay import run_linear_replay


# 【非核心】接收LinearReplay配置，调用真实数据、引擎、推理和账本核对链路。
def main() -> None:
    """输出逐时刻特征、订单成交权益和原生回测对照报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    run_linear_replay(arguments.config)


if __name__ == "__main__":
    main()
