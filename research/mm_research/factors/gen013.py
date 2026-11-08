"""Gen013成交到达节律，只使用当前与历史100ms笔数。"""
import math
import polars as pl
from mm_research.factors.operators import rolling_factors

VERSION = "seconds_v1"


# 【核心逻辑67】用已知成交笔数构造活动、跳变与历史突发比例。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """上一秒均值排除当前桶，所有跳变要求前后桶有效，输出四档均值。"""
    count = pl.when(pl.col("flow_valid")).then(pl.col("trade_count").cast(pl.Float64)).otherwise(None)
    previous = count.shift(1)
    change = count - previous
    sources = {
        "trade_active_flag_100ms": (count > 0).cast(pl.Float64),
        "trade_multi_flag_100ms": (count >= 3).cast(pl.Float64),
        "trade_dense_flag_100ms": (count >= 10).cast(pl.Float64),
        "trade_count_burst_ratio_100ms": count / (1 + previous.rolling_mean(10, min_samples=math.ceil(10 * config["min_fraction"]))),
        "trade_count_jump_abs_100ms": change.abs(),
        "trade_count_jump_up_100ms": change.clip(lower_bound=0),
        "trade_count_jump_down_100ms": (-change).clip(lower_bound=0),
        "trade_idle_to_active_flag_100ms": pl.when(previous.is_not_null() & count.is_not_null()).then(((count > 0) & (previous == 0)).cast(pl.Float64)),
        "trade_active_to_idle_flag_100ms": pl.when(previous.is_not_null() & count.is_not_null()).then(((count == 0) & (previous > 0)).cast(pl.Float64)),
    }
    return rolling_factors(sources, (5, 10, 30, 100), "mean", config["min_fraction"])
