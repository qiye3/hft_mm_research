"""输出各模型分布、时间特征与实际交易失败场景。"""

import argparse
from pathlib import Path
from mm_research.evaluation.model_failures import evaluate_model_failures, evaluate_entry_regimes


# 【非核心】配置明确模型、同规则交易账本及独立诊断输出目录。
def main() -> None:
    """仅分析开发期，不触发历史复核选型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--regimes-only', action='store_true')
    arguments = parser.parse_args()
    if arguments.regimes_only:
        evaluate_entry_regimes(arguments.config)
    else:
        evaluate_model_failures(arguments.config)


if __name__ == '__main__':
    main()
