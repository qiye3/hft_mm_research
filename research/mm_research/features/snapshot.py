"""快照基础特征计算与独立公式示例。

build_snapshot_grid 从标准快照计算100ms盘口、Mid与Spread。
add_snapshot_notional_features 是独立的挂单金额示例，接收带book_valid的网格。
返回双侧深度金额和金额失衡，可作为新增快照特征模块的参考。
"""

from pathlib import Path

import polars as pl

from mm_research.data.sources import scan_standard_history

from mm_research.features.expressions import make_day_grid, normalized_imbalance

SNAPSHOT_EXAMPLE_COLUMNS = ["snapshot_bid_notional_10", "snapshot_ask_notional_10", "snapshot_notional_imbalance_10"]


# ==================== 标准快照的基础特征计算 ====================
# 【核心逻辑14】按固定延迟对齐最近可见快照，计算Mid、Spread和盘口连续区段。
def build_snapshot_grid(folder: Path, date: str, interval_ns: int,
                        max_snapshot_age_ns: int, market_data_latency_ns: int) -> pl.DataFrame:
    """输入标准版本目录与日期和时间参数，返回前十档盘口网格。

    同时刻快照按源行号取最后状态，时间倒序先按交易所时间稳定排序。
    缺口恢复行记录盘口区段供标签使用，滚动因子通过null与min_samples处理缺失。
    """
    columns = [f"{side}_{field}_{level}" for level in range(1, 11)
               for side in ["bid", "ask"] for field in ["price", "quantity"]]
    source_columns = ["exchange_timestamp_ns", "quality_flags", *columns]
    snapshots = scan_standard_history(folder, "snapshots", date, source_columns).select(
        (pl.col("exchange_timestamp_ns") + market_data_latency_ns).alias("last_book_timestamp_ns"),
        pl.col("exchange_timestamp_ns").alias("book_exchange_ns"),
        *columns,
        (pl.col("quality_flags").list.contains("gap_before_confirmed")
         | pl.col("quality_flags").list.contains("gap_before_suspected")).cast(pl.Int64).alias("source_gap"),
    ).collect(engine="streaming")
    if snapshots["last_book_timestamp_ns"].null_count():
        raise ValueError("快照时间缺失，无法确定状态可见时刻")
    snapshots = snapshots.with_columns(pl.col("source_gap").cum_sum().alias("source_segment_id"))
    snapshots = snapshots.unique(subset="last_book_timestamp_ns", keep="last", maintain_order=True)
    base = make_day_grid(date, interval_ns).join_asof(
        snapshots, left_on="timestamp_ns", right_on="last_book_timestamp_ns", strategy="backward",
    ).with_columns(
        (pl.col("timestamp_ns") - pl.col("last_book_timestamp_ns")).alias("snapshot_age_ns"),
        ((pl.col("bid_price_1") + pl.col("ask_price_1")) * 0.5).alias("mid_price"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
        pl.lit(date).alias("date"), pl.lit("BTCUSDT").alias("symbol"),
    )
    valid_levels = [pl.col(column).is_finite() & (pl.col(column) > 0) for column in columns]
    for level in range(1, 10):
        valid_levels.extend([
            pl.col(f"bid_price_{level}") > pl.col(f"bid_price_{level + 1}"),
            pl.col(f"ask_price_{level}") < pl.col(f"ask_price_{level + 1}"),
        ])
    base = base.with_columns(
        (pl.all_horizontal(valid_levels) & (pl.col("spread") > 0)
         & (pl.col("snapshot_age_ns") <= max_snapshot_age_ns)).fill_null(False).alias("book_valid"),
    )
    recovery = pl.col("book_valid") & ~pl.col("book_valid").shift(1).fill_null(False)
    source_change = pl.col("source_segment_id").diff().fill_null(0) != 0
    return base.with_columns((recovery | source_change).cast(pl.Int64).cum_sum().alias("book_segment_id"))


# ==================== 快照特征模块示例 ====================
# 【核心逻辑184】计算前十档双侧挂单金额及金额失衡，作为快照特征示例。
def add_snapshot_notional_features(table: pl.DataFrame) -> pl.DataFrame:
    """输入包含十档价格数量及 book_valid 的网格，返回新增示例特征的表。

    双侧金额单位 USDT，失衡无量纲，正值表示买侧金额较大。
    无效盘口的示例特征置空，且所有公式只依赖当前快照。
    """
    table = table.with_columns([
        pl.sum_horizontal([
            pl.col(f"{side}_price_{level}") * pl.col(f"{side}_quantity_{level}")
            for level in range(1, 11)
        ]).alias(f"snapshot_{side}_notional_10")
        for side in ["bid", "ask"]
    ])
    table = table.with_columns(
        normalized_imbalance(
            pl.col("snapshot_bid_notional_10"), pl.col("snapshot_ask_notional_10"), empty_value=None,
        ).alias("snapshot_notional_imbalance_10"),
    )
    return table.with_columns([
        pl.when(pl.col("book_valid")).then(pl.col(column)).otherwise(None).alias(column)
        for column in SNAPSHOT_EXAMPLE_COLUMNS
    ])
