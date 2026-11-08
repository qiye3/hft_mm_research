"""Gen031三种显示盘口隐含价格中心，相对当前Mid以Tick表达。"""
import polars as pl
from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = 'seconds_v1'


# 【核心逻辑100】以不同深度的价格和数量构造交叉权重、平方根权重及方向价格。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """隐含中心是已显示盘口的启发式参考，所有使用档位必须完整有效。"""
    mid = pl.col('mid_price')
    sources = {}
    for depth in [1, 5, 10]:
        bid = depth_quantity('bid', depth)
        ask = depth_quantity('ask', depth)
        bid_price = pl.sum_horizontal([pl.col(f'bid_price_{level}') * pl.col(f'bid_quantity_{level}') for level in range(1, depth + 1)], ignore_nulls=False) / bid
        ask_price = pl.sum_horizontal([pl.col(f'ask_price_{level}') * pl.col(f'ask_quantity_{level}') for level in range(1, depth + 1)], ignore_nulls=False) / ask
        prices = {
            'crossqty': (bid_price * ask + ask_price * bid) / (bid + ask),
            'sqrtqty': (bid_price * ask.sqrt() + ask_price * bid.sqrt()) / (bid.sqrt() + ask.sqrt()),
            'touchsign': pl.when(bid > ask).then(pl.col('ask_price_1')).when(bid < ask).then(pl.col('bid_price_1')).otherwise(mid),
        }
        for name, price in prices.items():
            sources[f'book_micro_l{depth}_{name}_delta_ticks_100ms'] = pl.when((bid > 0) & (ask > 0)).then((price - mid) / config['tick_size'])
    return sources


# 【核心逻辑99】共享隐含价格源，展开四档秒级均值。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """九个隐含价格偏移各生成5、10、30、100秒均值。"""
    return rolling_factors({name: pl.col(name) for name in source_expressions(config)}, (5, 10, 30, 100), 'mean', config['min_fraction'])
