"""Gen041独立研究生产入口，目前先冻结可复用的小单基础列阈值。"""

import argparse
from pathlib import Path

from mm_research.models.gen041 import prepare_small_threshold


# 【非核心】配置明确标准逐笔、旧Features及本轮输出，不修改旧生产版本。
def main() -> None:
    """读取研究配置并执行训练期小单阈值准备。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("stage", choices=["threshold"])
    arguments = parser.parse_args()
    prepare_small_threshold(arguments.config)


if __name__ == "__main__":
    main()
