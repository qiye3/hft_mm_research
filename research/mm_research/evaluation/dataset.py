"""按日期读取同源预测标签，供因子评价与模型数据集共同使用。"""

from pathlib import Path
import datetime as dt

import polars as pl

from mm_research.data.storage import file_locks, read_metadata
from mm_research.targets.midprice import horizon_name


# 【核心逻辑115】在共享锁内读取指定周期标签并核对共同输入定义。
def load_evaluation_targets(labels: Path, date: str, horizon_ns: int, dataset: dict) -> pl.DataFrame:
    """返回timestamp_ns、label_valid和bps target，未通过质量规则的标签仍保留。"""
    suffix = horizon_name(horizon_ns)
    filename = labels / f"horizon={suffix}" / f"{date}.parquet"
    with file_locks([filename], shared=True):
        metadata = read_metadata(filename)
        if metadata["dataset"] != dataset or metadata["horizon_ns"] != horizon_ns:
            raise ValueError("标签与输入数据定义或周期不一致")
        return pl.read_parquet(filename, columns=["timestamp_ns", "label_valid", "label_end_timestamp_ns", f"y_mid_{suffix}_bps"]).rename(
            {f"y_mid_{suffix}_bps": "target"},
        )


# 【核心逻辑116】按绝对时间隔离训练与验证末端，检查目标时刻未跨入下一分区。
def partition_sample_mask(date: str, config: dict) -> pl.Expr:
    """分界前统一剔除300s，测试期由标签可用性决定样本范围。"""
    if date <= config["train_end"]:
        final_date = config["train_end"]
    elif date <= config["validation_end"]:
        final_date = config["validation_end"]
    else:
        return pl.lit(True)
    calendar = dt.date.fromisoformat(final_date) + dt.timedelta(days=1)
    boundary = int(dt.datetime.combine(calendar, dt.time(), tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    return (pl.col("timestamp_ns") < boundary - config["purge_ns"]) & (pl.col("label_end_timestamp_ns") < boundary)
