"""Gen011价差水平、变化幅度与方向切换，输入100ms盘口，整秒输出。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = "seconds_v2"


# 【核心逻辑62】构造价差水平和变化，缺失保持为空，外层窗口连续滑动。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入Tick与有效比例，输出九种价差状态的四档秒级均值表达式。"""
    spread = ((pl.col("ask_price_1") - pl.col("bid_price_1")) / config["tick_size"]).round(8)
    change = pl.when(book_transition_allowed()).then(spread.diff())
    sources = {
        "spread_level_ticks": spread,
        "spread_wide_1_flag": (spread > 1).cast(pl.Float64),
        "spread_wide_2_flag": (spread > 2).cast(pl.Float64),
        "spread_chg_ticks_100ms": change,
        "spread_expand_ticks_100ms": change.clip(lower_bound=0),
        "spread_contract_ticks_100ms": (-change).clip(lower_bound=0),
        "spread_expand_flag_100ms": (change > 0).cast(pl.Float64),
        "spread_contract_flag_100ms": (change < 0).cast(pl.Float64),
        "spread_flip_flag_100ms": (change.sign() * change.shift(1).sign() == -1).cast(pl.Float64),
    }
    return rolling_factors(sources, config["aggregation_windows_s"], "mean", config["min_fraction"])
