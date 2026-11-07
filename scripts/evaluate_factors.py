"""从Evaluation配置执行分布、IC及分组收益评估。"""

import argparse
from pathlib import Path

from mm_research.evaluation.report import run_evaluation


# 【非核心】读取评估配置并调用独立评估模块。
def main() -> None:
    """命令行输入Evaluation配置，生成结果表格和图表报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="例如config/evaluation/gen001.yaml")
    arguments = parser.parse_args()
    result = run_evaluation(arguments.config)
    print(f"评估耗时 {result['elapsed_seconds']:.3f}s")


if __name__ == "__main__":
    main()
