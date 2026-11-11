"""Gen018滞后成交压力与已发生100ms收益的滚动Pearson相关，所有输入为历史。"""
import math
import polars as pl
from mm_research.factors.gen017 import flow_capacity_sources
from mm_research.factors.operators import book_transition_allowed

VERSION = "seconds_v2"


# 【核心逻辑75】以双有效配对计数计算三种滞后与四档窗口的历史相关。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """零方差和不足80%配对均置空，lag按100ms位置计数。"""
    sources = {
        "trade_aggr_imb_100ms": pl.col("trade_imbalance_100ms"),
        "trade_aggr_imb_1s": pl.col("trade_imbalance_1s"),
        "trade_net_flow_rel_depth_l5_100ms": flow_capacity_sources()["trade_net_flow_rel_depth_l5_100ms"],
    }
    returns = pl.col("return_100ms_bps")
    boundaries = (~book_transition_allowed()).cast(pl.Int64).cum_sum()
    expressions = {}
    for name, values in sources.items():
        for lag in [1, 5, 10]:
            previous = values.shift(lag)
            paired = previous.is_finite() & returns.is_finite() & (boundaries == boundaries.shift(lag))
            left = pl.when(paired).then(previous)
            right = pl.when(paired).then(returns)
            for seconds in config["aggregation_windows_s"]:
                window = seconds * 10
                minimum = math.ceil(window * config["min_fraction"])
                correlation = pl.rolling_corr(left, right, window_size=window, min_samples=minimum)
                variable = (left.rolling_var(window, min_samples=minimum) > 0) & (right.rolling_var(window, min_samples=minimum) > 0)
                expressions[f"lead_{name}_ret_lag{lag}_corr_ret_{seconds}s"] = pl.when(variable & correlation.is_finite()).then(correlation).otherwise(None).cast(pl.Float64)
    return expressions
