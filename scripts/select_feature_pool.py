"""按显式配置选择扩展因子池中的输入列，不重新复制全量训练矩阵。"""

import argparse
from pathlib import Path

from mm_research.models.feature_pool import select_feature_pool


# 【非核心】命令只接收Models阶段配置，输入宽因子矩阵，输出选列索引。
def main() -> None:
    """执行已预定的家族覆盖或相关性筛选。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    select_feature_pool(parser.parse_args().config)


if __name__ == '__main__':
    main()
