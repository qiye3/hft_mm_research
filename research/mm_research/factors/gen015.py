"""Gen015买卖报价的幅度与协同变化，以相邻有效100ms快照构造。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = "seconds_v2"


# 【核心逻辑71】比较相邻买卖一报价，构造双侧调整与单边方向。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入Tick与有效比例，返回九类报价调整的四档秒级均值。"""
    valid_pair = book_transition_allowed()
    bid = pl.when(valid_pair).then((pl.col("bid_price_1").diff() / config["tick_size"]).round(8))
    ask = pl.when(valid_pair).then((pl.col("ask_price_1").diff() / config["tick_size"]).round(8))
    sources = {
        "quote_bid_up_ticks_100ms": bid.clip(lower_bound=0),
        "quote_bid_down_ticks_100ms": (-bid).clip(lower_bound=0),
        "quote_ask_up_ticks_100ms": ask.clip(lower_bound=0),
        "quote_ask_down_ticks_100ms": (-ask).clip(lower_bound=0),
        "quote_bid_delta_ticks_100ms": bid,
        "quote_ask_delta_ticks_100ms": ask,
        "quote_bid_only_dir_100ms": bid.sign() * (ask == 0).cast(pl.Float64),
        "quote_ask_only_dir_100ms": ask.sign() * (bid == 0).cast(pl.Float64),
        "quote_bid_ask_comove_100ms": bid.sign() * ask.sign(),
    }
    return rolling_factors(sources, (5, 10, 30, 100), "mean", config["min_fraction"])
