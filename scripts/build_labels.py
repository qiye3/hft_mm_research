"""从Targets配置生成五个秒级周期的跨日预测标签。"""

import argparse
from pathlib import Path

from mm_research.targets.dataset import run_target_production


# 【非核心】接收标签配置和显式重建标志，执行独立Targets链路。
def main() -> None:
    """输入config/targets配置，输出按日期和周期保存的整秒标签。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()
    run_target_production(arguments.config, arguments.rebuild)


if __name__ == "__main__":
    main()
