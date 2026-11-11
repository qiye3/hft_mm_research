"""Gen019固定0.6阈值的压力状态和进入退出事件，外层求和保留方向。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = "seconds_v2"


# 【核心逻辑76】相邻有效压力之间识别极端状态及有方向的进入退出。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """符号翻转同时形成旧状态退出和新状态进入，缺失不当作零。"""
    sources = {}
    for name, column in [("book_obi_l1", "imbalance_1"), ("book_obi_l5", "imbalance_5"), ("trade_aggr_imb_1s", "trade_imbalance_1s")]:
        current = pl.col(column)
        previous = current.shift(1)
        valid = current.is_finite() & previous.is_finite()
        if name.startswith("book_"):
            valid = valid & book_transition_allowed()
        extreme = current.abs() >= 0.6
        old_extreme = previous.abs() >= 0.6
        flip = current.sign() != previous.sign()
        sources[f"{name}_extreme_state_thr06"] = pl.when(valid).then(current.sign() * extreme.cast(pl.Float64))
        sources[f"{name}_extreme_entry_thr06"] = pl.when(valid).then(current.sign() * (extreme & (~old_extreme | flip)).cast(pl.Float64))
        sources[f"{name}_extreme_exit_thr06"] = pl.when(valid).then(previous.sign() * (old_extreme & (~extreme | flip)).cast(pl.Float64))
    return rolling_factors(sources, config["aggregation_windows_s"], "sum", config["min_fraction"])
