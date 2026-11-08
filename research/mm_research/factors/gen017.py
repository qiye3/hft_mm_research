"""Gen017主动成交量相对桶末挂单深度，保留原始压力方向。"""
import polars as pl
from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = "seconds_v1"


# 【核心逻辑74】分别以对手侧深度和双侧深度归一化主动流量。
def flow_capacity_sources() -> dict[str, pl.Expr]:
    """返回三个深度的买卖与净流量比例，零深度为空，供Gen018复用。"""
    sources = {}
    for depth in [1, 5, 10]:
        bid, ask = depth_quantity("bid", depth), depth_quantity("ask", depth)
        sources[f"trade_buy_flow_rel_depth_l{depth}_100ms"] = pl.col("buy_volume") / pl.when(ask > 0).then(ask)
        sources[f"trade_sell_flow_rel_depth_l{depth}_100ms"] = pl.col("sell_volume") / pl.when(bid > 0).then(bid)
        sources[f"trade_net_flow_rel_depth_l{depth}_100ms"] = (pl.col("buy_volume") - pl.col("sell_volume")) / pl.when(bid + ask > 0).then(bid + ask)
    return sources


# 【核心逻辑73】展开流量容量比例的四档秒级历史均值。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入有效比例，输出三种深度下九个压力序列的窗口均值。"""
    return rolling_factors(flow_capacity_sources(), (5, 10, 30, 100), "mean", config["min_fraction"])
