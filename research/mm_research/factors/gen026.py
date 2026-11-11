"""Gen026历史收益强度与当前价差、薄弱深度、显示执行成本的交互。"""
import polars as pl
from mm_research.factors.gen012 import execution_vwap
from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = 'seconds_v1'


# 【核心逻辑91】组合三种已发生波动强度和三类可见流动性状态。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """执行成本复用25k USDT买入公式，无法吃完十档时为空。"""
    returns = pl.col('return_100ms_bps')
    mid = pl.col('mid_price')
    depth = depth_quantity('bid', 10) + depth_quantity('ask', 10)
    spread = (pl.col('ask_price_1') - pl.col('bid_price_1')) / config['tick_size']
    volatility = {'abs': returns.abs(), 'down': (-returns).clip(lower_bound=0), 'square': returns.pow(2)}
    liquidity = {'spread': spread.log1p(), 'thin': 1 / pl.when(depth > 0).then(depth),
                 'exec': (execution_vwap('ask', 25000 / mid) / mid - 1) * 10000}
    sources = {f'cross_vol_{name}_x_{state}_100ms': strength * measure
               for name, strength in volatility.items() for state, measure in liquidity.items()}
    return sources


# 【核心逻辑90】共享已计算的里层状态，再计算各秒级窗口以控制中间内存。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """每个里层序列只生成一次，返回四档窗口的Float64表达式。"""
    sources = {name: pl.col(name) for name in source_expressions(config)}
    return rolling_factors(sources, config["aggregation_windows_s"], 'mean', config['min_fraction'])
