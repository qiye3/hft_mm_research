"""逐笔特征模块示例，计算主动净成交金额及一秒成交强度。"""

import polars as pl

TRADE_EVENT_EXAMPLE_COLUMNS = ["trade_net_notional_100ms", "trade_net_notional_1s", "trade_intensity_1s"]


# 【核心逻辑185】在完整时间网格上计算成交净额和强度，以空值表达异常窗口。
def build_trade_event_features(bins: pl.DataFrame, grid: pl.DataFrame, min_samples: int = 8) -> pl.DataFrame:
    """输入成交分桶、100ms网格及一秒窗口最少样本数，返回独立示例特征。

    正常无成交窗口补零，异常窗口保持null，滚动计算跨过缺失值继续累计。
    """
    table = grid.join(bins, on="timestamp_ns", how="left").with_columns(
        pl.col("trade_count").fill_null(0), pl.col("buy_notional", "sell_notional").fill_null(0.0),
        pl.col("trade_valid").fill_null(True),
    ).with_columns(
        pl.when(pl.col("trade_valid")).then(pl.col("buy_notional") - pl.col("sell_notional"))
        .otherwise(None).alias("trade_net_notional_100ms"),
        pl.when(pl.col("trade_valid")).then(pl.col("trade_count")).otherwise(None).alias("valid_count"),
    )
    table = table.with_columns(
        pl.col("trade_net_notional_100ms").rolling_sum(10, min_samples=min_samples).alias("trade_net_notional_1s"),
        pl.col("valid_count").rolling_sum(10, min_samples=min_samples).cast(pl.Float64).alias("trade_intensity_1s"),
    )
    return table.with_columns(
        pl.all_horizontal([pl.col(name).is_not_null() for name in TRADE_EVENT_EXAMPLE_COLUMNS]).alias("trade_feature_valid"),
    ).drop("valid_count")
