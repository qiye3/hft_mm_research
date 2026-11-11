"""Gen024相邻快照的深度毛变化，保留双侧同向变化规模。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = 'seconds_v1'


# 【核心逻辑86】在有效接续处计算双侧深度绝对变动及毛净差。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """只比较排行档位总量，不推断事件级新增与撤单，输出秒窗口均值。"""
    sources = {}
    for depth in [1, 5, 10]:
        bid = pl.when(book_transition_allowed()).then(depth_quantity('bid', depth).diff())
        ask = pl.when(book_transition_allowed()).then(depth_quantity('ask', depth).diff())
        sources[f'book_depth_bid_abs_l{depth}_100ms'] = bid.abs()
        sources[f'book_depth_ask_abs_l{depth}_100ms'] = ask.abs()
        sources[f'book_depth_gross_minus_net_l{depth}_100ms'] = bid.abs() + ask.abs() - (bid - ask).abs()
    return rolling_factors(sources, config["aggregation_windows_s"], 'mean', config['min_fraction'])
