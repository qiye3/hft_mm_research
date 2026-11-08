"""Gen032中间价与双侧最优报价的连续、反转和有方向变价频率。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = 'seconds_v1'


# 【核心逻辑101】保留零变价位置，只有两次连续有效变化才能形成转移。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输出每个100ms位置的出现频率，未对非零变价进行条件抽样。"""
    sources = {}
    for name, column in [('mid', 'mid_price'), ('bid', 'bid_price_1'), ('ask', 'ask_price_1')]:
        direction = pl.when(book_transition_allowed()).then(pl.col(column).diff().sign())
        previous = direction.shift(1)
        valid = direction.is_finite() & previous.is_finite()
        sources[f'px_{name}_transition_cont_100ms'] = pl.when(valid).then(((direction == previous) & (direction != 0)).cast(pl.Float64))
        sources[f'px_{name}_transition_flip_100ms'] = pl.when(valid).then(((direction == -previous) & (direction != 0)).cast(pl.Float64))
        sources[f'px_{name}_transition_signed_100ms'] = pl.when(valid).then(direction * ((direction != 0) & (previous != 0)).cast(pl.Float64))
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
