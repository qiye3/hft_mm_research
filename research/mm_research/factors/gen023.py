"""Gen023基于官方逐笔重建的桶内规模分布，基础列位于Features交易分布组。"""
import polars as pl
from mm_research.factors.operators import rolling_factors

VERSION = 'seconds_v1'


# 【核心逻辑85】对买、卖、全部成交的中位数、头部占比与HHI计算历史均值。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """输入已发布的九个100ms分布特征，输出四档秒窗口均值。"""
    sources = {f'trade_size_{side}_{metric}_100ms': pl.col(f'trade_size_{side}_{metric}_100ms')
               for side in ['buy', 'sell', 'all'] for metric in ['median_qty', 'topdecile_share', 'hhi']}
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
