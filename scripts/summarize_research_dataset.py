"""汇总当前分组特征和独立标签的逐日样本量与分布。"""

import argparse
import json
from pathlib import Path

from mm_research.data.dataset import summarize_dataset


# 【非核心】输出数据集统计文件，便于研究前检查样本规模与异常值。
def main() -> None:
    """输入数据集目录和 JSON 路径，执行基础统计，不训练模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    arguments = parser.parse_args()
    report = summarize_dataset(arguments.dataset, arguments.labels, arguments.output)
    overview = [{"date": day["date"], "rows": day["rows"], "usable_rows": day["usable_rows"]}
                for day in report["days"]]
    print(json.dumps(overview, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
