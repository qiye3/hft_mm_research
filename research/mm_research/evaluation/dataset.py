"""在共享锁内按日期读取因子和标签，以时间键关联评估样本。"""

from pathlib import Path

import polars as pl

from mm_research.data.storage import file_locks, read_metadata
from mm_research.targets.midprice import horizon_name


# 【核心逻辑30】读取单日因子和指定周期标签，保留全部因子样本以统计损耗。
def load_evaluation_day(factor: Path, labels: Path, date: str, horizon_ns: int) -> pl.DataFrame:
    """输入因子文件、标签版本目录和日期周期，返回value、target及标签有效性。

    因子与标签通过时间戳关联。缺失因子或标签均保留，指标计算阶段分别选样本。
    """
    label_file = labels / f"horizon={horizon_name(horizon_ns)}" / f"{date}.parquet"
    with file_locks([factor, label_file], shared=True):
        factor_metadata = read_metadata(factor)
        label_metadata = read_metadata(label_file)
        if (factor_metadata["dataset"] != label_metadata["dataset"]
                or label_metadata["horizon_ns"] != horizon_ns):
            raise ValueError("因子与标签来源或周期不一致")
        values = pl.scan_parquet(factor).filter(pl.col("date") == date).select(
            "timestamp_ns", pl.col(factor.stem).alias("value"),
        )
        targets = pl.scan_parquet(label_file).select(
            "timestamp_ns", "label_valid", pl.col(f"y_mid_{horizon_name(horizon_ns)}_bps").alias("target"),
        )
        return values.join(targets, on="timestamp_ns", how="left", validate="1:1").collect(engine="streaming")
