"""Gen041长周期微观结构压力及短长背离，严格按真实秒数滚动。"""

import math

import polars as pl

from mm_research.factors.gen012 import execution_vwap

VERSION = "seconds_v1"
SECOND = 1_000_000_000
BOOK_COLUMNS = [f"{side}_{field}_{level}" for side in ["bid", "ask"]
                for field in ["price", "quantity"] for level in range(1, 11)]
TRADE_COLUMNS = ["buy_volume", "sell_volume", "trade_small_buy_qty_100ms", "trade_small_sell_qty_100ms"]
FAMILIES = {
    "book_obi_l1_mean": ("obi", "mean", [300, 900, 1800]),
    "book_net_depth_direction_signed_ratio": ("net_delta", "signed_ratio", [300, 900, 1800]),
    "book_bid_depth_down_share": ("bid_delta", "down_share", [900, 1800, 3600]),
    "book_ask_depth_down_share": ("ask_delta", "down_share", [900, 1800, 3600]),
    "trade_aggr_qtyimb": ("buy_volume", "quantity", [300, 900, 1800]),
    "trade_small_qtyimb": ("trade_small_buy_qty_100ms", "quantity", [300, 900, 1800]),
    "cross_trade_book_sign_agree_mean": ("agreement", "mean", [300, 900, 1800]),
    "book_exec_cost_gap_25k_mean": ("cost_gap", "mean", [900, 1800, 3600]),
    "book_depth_notional_p10": ("notional_depth", "p10", [900, 1800, 3600]),
}


# 【核心逻辑395】聚合右闭合整秒，盘口取秒末状态、OBI均值至少八格、成交必须十格完整。
def second_statistics(raw: pl.DataFrame) -> pl.DataFrame:
    """输入完整100ms网格和冻结小单列；返回九类因子的秒级统计及质量敏感差分。"""
    bid = pl.col("bid_quantity_1")
    ask = pl.col("ask_quantity_1")
    raw = raw.with_columns(pl.when(bid + ask > 0).then((bid - ask) / (bid + ask)).alias("obi_100ms"))
    aggregations = [pl.col(column).last() for column in [*BOOK_COLUMNS, "mid_price", "book_segment_id"]]
    aggregations.append(pl.when(pl.col("obi_100ms").count() >= 8).then(pl.col("obi_100ms").mean()).alias("obi"))
    aggregations.extend([
        pl.when(pl.col(column).count() == 10).then(pl.col(column).sum()).alias(column)
        for column in TRADE_COLUMNS
    ])
    seconds = raw.group_by(((pl.col("timestamp_ns") + SECOND - 1) // SECOND * SECOND).alias("timestamp_ns"))
    frame = seconds.agg(*aggregations).sort("timestamp_ns").with_columns(
        pl.col("timestamp_ns").cast(pl.Datetime("ns")).dt.strftime("%Y-%m-%d").alias("date"),
        pl.sum_horizontal([pl.col(f"bid_quantity_{level}") for level in range(1, 11)], ignore_nulls=False).alias("bid_depth"),
        pl.sum_horizontal([pl.col(f"ask_quantity_{level}") for level in range(1, 11)], ignore_nulls=False).alias("ask_depth"),
    )
    same_day = pl.col("date") == pl.col("date").shift(1)
    segment_change = same_day & (pl.col("book_segment_id") != pl.col("book_segment_id").shift(1))
    source_switch = (pl.col("date") == "2025-10-01") & ~same_day
    continuous = ((pl.col("timestamp_ns").diff() == SECOND)
                  & ~segment_change.fill_null(False) & ~source_switch.fill_null(False))
    mid = pl.col("mid_price")
    buy_cost = (execution_vwap("ask", 25000 / mid) / mid - 1) * 10000
    sell_cost = (1 - execution_vwap("bid", 25000 / mid) / mid) * 10000
    return frame.select(
        "timestamp_ns", "date", "obi", *TRADE_COLUMNS,
        pl.when(continuous).then((pl.col("bid_depth") - pl.col("ask_depth")).diff()).alias("net_delta"),
        pl.when(continuous).then(pl.col("bid_depth").diff()).alias("bid_delta"),
        pl.when(continuous).then(pl.col("ask_depth").diff()).alias("ask_delta"),
        ((pl.col("buy_volume") - pl.col("sell_volume")).sign() * pl.col("obi").sign()).alias("agreement"),
        (buy_cost - sell_cost).alias("cost_gap"),
        pl.sum_horizontal([pl.col(f"{side}_price_{level}") * pl.col(f"{side}_quantity_{level}")
                           for side in ["bid", "ask"] for level in range(1, 11)], ignore_nulls=False).alias("notional_depth"),
    )


# 【非核心】比值先汇总数量或变动量，已知零活动为零，观测不足的分母仍为空。
def window_statistic(column: str, operation: str, window: int, min_fraction: float) -> tuple[pl.Expr, pl.Expr]:
    """返回窗口值和零活动标志；分位数采用Polars线性插值，绝不平均分段分位数。"""
    minimum = math.ceil(window * min_fraction)
    value = pl.col(column).fill_nan(None)
    inactive = pl.lit(False)
    if operation == "mean":
        return value.rolling_mean(window, min_samples=minimum), inactive
    if operation == "p10":
        return value.rolling_quantile(.1, interpolation="linear", window_size=window, min_samples=minimum), inactive
    if operation == "quantity":
        sell_column = "sell_volume" if column == "buy_volume" else "trade_small_sell_qty_100ms"
        sell = pl.col(sell_column).fill_nan(None)
        numerator = (value - sell).rolling_sum(window, min_samples=minimum)
        denominator = (value + sell).rolling_sum(window, min_samples=minimum)
    else:
        numerator_value = value if operation == "signed_ratio" else (-value).clip(lower_bound=0)
        numerator = numerator_value.rolling_sum(window, min_samples=minimum)
        denominator = value.abs().rolling_sum(window, min_samples=minimum)
    result = pl.when(denominator > 0).then(numerator / denominator).when(denominator == 0).then(0.)
    return result, denominator == 0


# 【核心逻辑396】展开27个长期统计和9个60秒短长背离，缺失位置继续占据真实时间窗口。
def factor_expressions(min_fraction: float = .8) -> tuple[dict[str, pl.Expr], dict[str, pl.Expr]]:
    """返回36个Float64因子及独立零活动日志表达式，不把质量标记混入模型因子。"""
    factors = {}
    quality = {}
    for name, (column, operation, windows) in FAMILIES.items():
        statistics = {}
        for window in [60, *windows]:
            value, inactive = window_statistic(column, operation, window, min_fraction)
            statistics[window] = value
            if operation in ["signed_ratio", "down_share", "quantity"]:
                quality[f"{name}_{window}s_zero_activity"] = inactive.fill_null(False)
            if window != 60:
                factors[f"{name}_{window}s"] = value.cast(pl.Float64)
        longest = windows[-1]
        if operation == "p10":
            factors[f"{name}_ratio_60s_{longest}s"] = pl.when(statistics[longest] > 0).then(
                statistics[60] / statistics[longest] - 1,
            ).cast(pl.Float64)
        else:
            factors[f"{name}_diff_60s_{longest}s"] = (statistics[60] - statistics[longest]).cast(pl.Float64)
    return factors, quality
