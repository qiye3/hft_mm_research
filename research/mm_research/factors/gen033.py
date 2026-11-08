"""Gen033历史报价收益乘积与反向抵消幅度，单位分别为bps²和bps。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = 'seconds_v1'


# 【核心逻辑102】组合当前已发生收益和两个历史滞后，识别报价回弹。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """连续性检查只使受影响配对为空，外层窗口继续按时间位置滑动。"""
    sources = {}
    for name, column in [('mid', 'mid_price'), ('bid', 'bid_price_1'), ('ask', 'ask_price_1')]:
        price = pl.col(column)
        returns = pl.when(book_transition_allowed()).then((price / price.shift(1) - 1) * 10000)
        previous = returns.shift(1)
        sources[f'noise_{name}_ret_lag1_prod_100ms'] = returns * previous
        sources[f'noise_{name}_ret_lag2_prod_100ms'] = pl.when(previous.is_finite()).then(returns * returns.shift(2))
        valid = returns.is_finite() & previous.is_finite()
        sources[f'noise_{name}_reversal_overlap_100ms'] = pl.when(valid).then(
            pl.when(returns * previous < 0).then(pl.min_horizontal(returns.abs(), previous.abs())).otherwise(0),
        )
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
