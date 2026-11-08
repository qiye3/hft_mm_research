"""训练并评价时间隔离的OLS、Ridge和Lasso基线。"""

import argparse
from pathlib import Path

from mm_research.models.runner import run_linear_baseline


# 【非核心】读取Models配置，实际训练与测试由研究包负责。
def main() -> None:
    """输入配置路径，输出冻结模型、指标与逐日测试预测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    run_linear_baseline(arguments.config)


if __name__ == "__main__":
    main()
