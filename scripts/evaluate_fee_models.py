"""按固定开发期共同样本评价费用分类模型的新增信息。"""

import argparse
from pathlib import Path

from mm_research.evaluation.fee_models import evaluate_fee_models


# 【非核心】读取评价配置，预测与标签来源、分区和基线均由配置明确指定。
def main() -> None:
    """生成总体、每日、概率桶及按周配对误差区间。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    evaluate_fee_models(arguments.config)


if __name__ == "__main__":
    main()
