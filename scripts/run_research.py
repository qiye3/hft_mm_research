"""按所属配置运行Factors、Evaluation、Models与C++验证流水线。"""

import argparse
from pathlib import Path

from mm_research.pipeline import run_research


# 【非核心】接收研究配置，核心流程位于研究包。
def main() -> None:
    """输入研究总配置，依次调用各所属阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    run_research(arguments.config)


if __name__ == "__main__":
    main()
