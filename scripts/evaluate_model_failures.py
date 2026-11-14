"""输出各模型分布、时间特征与实际交易失败场景。"""

import argparse
from pathlib import Path
from mm_research.evaluation.model_failures import evaluate_model_failures


# 【非核心】配置明确模型、同规则交易账本及独立诊断输出目录。
def main() -> None:
    """仅分析开发期，不触发历史复核选型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    evaluate_model_failures(parser.parse_args().config)


if __name__ == '__main__':
    main()
