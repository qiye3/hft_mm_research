"""使用所属Features配置，为Gen023准备100ms逐笔规模分布列。"""
import argparse
from pathlib import Path
from mm_research.config import load_pipeline_config
from mm_research.features.trade_distribution import build_distribution_features


# 【非核心】选择既有Features配置，只增量计算逐笔分布组。
def main() -> None:
    """输入Features YAML和并发数，输出仍是同目录每列一个全日期Parquet。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build_distribution_features(root, load_pipeline_config(arguments.config, 'features'), arguments.workers)


if __name__ == '__main__':
    main()
