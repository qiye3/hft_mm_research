"""独立的 Polars 逐笔成交特征示例，按接收时间研究成交净额与强度。

build_trade_event_features 输入 build_trade_bins 的原始成交汇总和完整 UTC 网格。
输出 100ms 净主动成交金额、1s 净额和每秒成交笔数，可作为新增逐笔特征模块的参考。
"""

import polars as pl

from mm_research.features.expressions import rolling_sum_in_segment

TRADE_EVENT_EXAMPLE_COLUMNS = ["trade_net_notional_100ms", "trade_net_notional_1s", "trade_intensity_1s"]


# 补齐无成交窗口，按首次可见的成交跳号切段后计算净额与强度。
def build_trade_event_features(bins: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """输入有效源文件的成交分桶及连续 100ms 网格，返回独立成交示例特征表。

    bins 的数量、方向、编号与接收时序已由 build_trade_bins 校验。
    已知空窗补零，跳号检测时刻切段并重新预热五秒，金额单位 USDT，强度单位笔/秒。
    """
    table = grid.join(bins, on="timestamp_ns", how="left").with_columns(
        pl.col("trade_count").fill_null(0),
        pl.col("buy_notional", "sell_notional").fill_null(0.0),
        pl.col("trade_sequence_gap").fill_null(False),
    )
    table = table.with_columns(
        pl.col("trade_sequence_gap").cast(pl.Int64).cum_sum().alias("segment_id"),
        (pl.col("buy_notional") - pl.col("sell_notional")).alias("trade_net_notional_100ms"),
    )
    table = table.with_columns(
        pl.col("timestamp_ns").min().over("segment_id").alias("segment_start_ns"),
        rolling_sum_in_segment(pl.col("trade_net_notional_100ms"), 10).alias("trade_net_notional_1s"),
        rolling_sum_in_segment(pl.col("trade_count"), 10).cast(pl.Float64).alias("trade_intensity_1s"),
        rolling_sum_in_segment((~pl.col("trade_sequence_gap")).cast(pl.Int32), 50).alias("valid_bins_5s"),
    )
    table = table.with_columns(
        ((pl.col("timestamp_ns") - pl.col("segment_start_ns") >= 5_000_000_000)
         & (pl.col("valid_bins_5s") == 50)).fill_null(False).alias("trade_feature_valid"),
    )
    return table.with_columns([
        pl.when(pl.col("trade_feature_valid")).then(pl.col(column)).otherwise(None).alias(column)
        for column in TRADE_EVENT_EXAMPLE_COLUMNS
    ]).drop("valid_bins_5s")
