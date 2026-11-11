"""Gen014测量100ms采样状态停留时间，连续区段重建后计算外层均值。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = "seconds_v2"


# 【核心逻辑70】在连续有效区段内向前携带最近触发时间，区段初始化前为空。
def elapsed_since_event(values: pl.Expr, event: pl.Expr, initial_event_ns: int | None = None) -> pl.Expr:
    """输入有效性与当前触发条件，缺失重置里层年龄，外层窗口仍按80%规则滑动。"""
    timestamp = pl.col("timestamp_ns")
    last_event = pl.when(event & values).then(timestamp).forward_fill().fill_null(pl.lit(initial_event_ns, dtype=pl.Int64))
    last_invalid = pl.when(~values).then(timestamp).forward_fill().fill_null(-1)
    return pl.when(values & (last_event > last_invalid)).then((timestamp - last_event) / 1e9).otherwise(None)


# 【核心逻辑69】分别计算报价、数量变化及最近成交桶的可见年龄。
def source_expressions(config: dict, initial_events: dict | None = None) -> dict[str, pl.Expr]:
    """返回九类100ms年龄序列，数值单位为秒；跨块携带真实事件锚点。"""
    initial_events = {} if initial_events is None else initial_events
    sources = {}
    columns = {
        "book_bid_px": pl.col("bid_price_1"), "book_ask_px": pl.col("ask_price_1"),
        "book_mid_px": pl.col("mid_price"),
        "book_spread": pl.col("ask_price_1") - pl.col("bid_price_1"),
        "book_bid_q1": pl.col("bid_quantity_1"), "book_ask_q1": pl.col("ask_quantity_1"),
    }
    for name, column in columns.items():
        reset = ~book_transition_allowed() & pl.col("timestamp_ns").shift(1).is_not_null()
        valid = column.is_not_null() & ~reset
        source_name = f"{name}_age_sec_100ms"
        sources[source_name] = elapsed_since_event(
            valid, (column != column.shift(1)).fill_null(False), initial_events.get(source_name),
        )
    for name, column in [("any", "trade_count"), ("buy", "buy_volume"), ("sell", "sell_volume")]:
        valid = pl.col("flow_valid") & pl.col(column).is_not_null()
        source_name = f"trade_{name}_age_sec_100ms"
        sources[source_name] = elapsed_since_event(valid, pl.col(column) > 0, initial_events.get(source_name))
    return sources


# 【核心逻辑68】共享已生成的年龄状态，在最终输出附近计算秒级窗口。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入预处理的九个年龄列，避免为短输出区间重复滚动全历史。"""
    sources = {name: pl.col(name) for name in source_expressions(config)}
    return rolling_factors(sources, config["aggregation_windows_s"], "mean", config["min_fraction"])
