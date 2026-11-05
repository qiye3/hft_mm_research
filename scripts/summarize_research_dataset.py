"""读取按日研究 Parquet，生成可用于 Python 研究的 X、y 和基础统计。

load_research_xy 只读取清单声明的特征与有效标签，summarize_dataset 输出按日数量及列统计。
输入是 M2-A 数据集根目录，输出为 NumPy 矩阵或 JSON 报告，不进行模型训练。
"""

import argparse
import json
from pathlib import Path

import polars as pl


# 按清单选择已完成日期，过滤预热、缺口和无未来标签的样本。
def load_research_xy(dataset_root: Path):
    """输入数据集根目录，返回 X、y 及含日期和时间戳的样本索引。

    列顺序固定来自 manifest，标签和未来时刻列不会进入 X。
    多日数据按日期和样本时间顺序组合，只读取研究所需列。
    """
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    if not manifest["validated"]:
        raise ValueError("研究数据集尚未完成校验")
    paths = [dataset_root / f"date={day['date']}" / "research.parquet" for day in manifest["days"]]
    features = manifest["feature_columns"]
    label = manifest["label_column"]
    table = pl.scan_parquet(paths).filter(pl.col("feature_valid") & pl.col("label_valid")).select(
        "date", "timestamp_ns", *features, label,
    ).collect(engine="streaming").sort(["date", "timestamp_ns"])
    return table.select(features).to_numpy(), table[label].to_numpy(), table.select("date", "timestamp_ns")


# 通过列式扫描统计多日可用量和特征分布，不物化整张基础盘口表。
def summarize_dataset(dataset_root: Path, output: Path) -> dict:
    """输入数据集目录与报告文件，返回按日计数、标签和特征分布。

    正式统计仅使用同时具备有效特征和标签的样本，保留每个日期的剔除数量。
    """
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    if not manifest["validated"]:
        raise ValueError("研究数据集尚未完成校验")
    paths = [dataset_root / f"date={day['date']}" / "research.parquet" for day in manifest["days"]]
    scan = pl.scan_parquet(paths)
    counts = scan.group_by("date").agg(
        pl.len().alias("rows"),
        (pl.col("feature_valid") & pl.col("label_valid")).sum().alias("usable_rows"),
    ).sort("date").collect().to_dicts()
    valid = scan.filter(pl.col("feature_valid") & pl.col("label_valid"))
    distributions = []
    for column in [*manifest["feature_columns"], manifest["label_column"]]:
        statistics = valid.select(
            pl.col(column).mean().alias("mean"), pl.col(column).std().alias("std"),
            pl.col(column).min().alias("min"), pl.col(column).max().alias("max"),
            pl.col(column).null_count().alias("null_count"),
        ).collect().to_dicts()[0]
        distributions.append({"column": column, **statistics})
    report = {"days": counts, "feature_columns": manifest["feature_columns"], "statistics": distributions}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


# 输出数据集统计文件，便于研究前检查样本规模与异常值。
def main() -> None:
    """输入数据集目录和 JSON 路径，执行基础统计，不训练模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    report = summarize_dataset(arguments.dataset, arguments.output)
    print(json.dumps(report["days"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
