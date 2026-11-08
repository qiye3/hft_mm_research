"""Gen025跟踪已发生流动性冲击，恢复到80%或满100秒结束，新冲击取代旧冲击。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = 'seconds_v1'


# 【核心逻辑89】从最近触发前后状态和已观察恢复判断活动冲击，不读取未来。
def shock_states(values: pl.Expr, spread: bool) -> dict[str, pl.Expr]:
    """返回活动标记、恢复比例与秒数，失效清空里层状态，外层窗口继续滑动。"""
    timestamp = pl.col('timestamp_ns')
    previous = values.shift(1)
    allowed = book_transition_allowed() & values.is_finite() & previous.is_finite()
    if spread:
        trigger = allowed & (values - previous >= 2)
    else:
        trigger = allowed & (previous > 0) & ((previous - values) / previous >= .2)
    start = pl.when(trigger).then(timestamp).forward_fill()
    before = pl.when(trigger).then(previous).forward_fill()
    after = pl.when(trigger).then(values).forward_fill()
    recovery = (after - values) / (after - before) if spread else (values - after) / (before - after)
    recovery = recovery.clip(0, 1)
    age = (timestamp - start) / 1e9
    invalid = ~values.is_finite() | (~allowed & timestamp.shift(1).is_not_null())
    finished = pl.when(invalid | (recovery >= .8) | (age >= 100)).then(timestamp).forward_fill()
    active = start.is_not_null() & (start > finished.fill_null(-1))
    known = start.is_not_null() | finished.is_not_null() | (timestamp - timestamp.first() >= 100_000_000_000)
    valid = values.is_finite() & known & ~invalid
    return {
        'active': pl.when(valid).then(active.cast(pl.Float64)),
        'recovery': pl.when(valid).then(pl.when(active).then(recovery).otherwise(0)),
        'age_sec': pl.when(valid).then(pl.when(active).then(age).otherwise(0)),
    }


# 【核心逻辑88】同时跟踪价差扩张与双侧十档深度骤降三种冲击。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """价差阈值2Tick，深度阈值20%，输出九个状态的四档秒级均值。"""
    inputs = {'spread': ((pl.col('ask_price_1') - pl.col('bid_price_1')) / config['tick_size']).round(8),
              'bid_depth': depth_quantity('bid', 10), 'ask_depth': depth_quantity('ask', 10)}
    sources = {}
    for name, values in inputs.items():
        for metric, expression in shock_states(values, name == 'spread').items():
            sources[f'liq_shock_{name}_{metric}_100ms'] = expression
    return sources


# 【核心逻辑87】共享已计算的里层状态，再计算各秒级窗口以控制中间内存。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """每个里层序列只生成一次，返回四档窗口的Float64表达式。"""
    sources = {name: pl.col(name) for name in source_expressions(config)}
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
