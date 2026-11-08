"""Gen022显示挂单墙与相邻价格空洞，输入有效前十档。"""
import polars as pl
from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = 'seconds_v1'


# 【核心逻辑84】取最大挂单比例、最近最大墙位置与相邻档位最大间隔。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """并列最大数量优先选择靠近最优报价的档位，单位分别为比例与Tick。"""
    sources = {}
    for side in ['bid', 'ask']:
        quantities = [pl.col(f'{side}_quantity_{level}') for level in range(1, 11)]
        prices = [pl.col(f'{side}_price_{level}') for level in range(1, 11)]
        total = depth_quantity(side, 10)
        valid = pl.all_horizontal([quantity > 0 for quantity in quantities])
        maximum = pl.max_horizontal(quantities)
        wall_price = pl.coalesce([pl.when(quantity == maximum).then(price) for quantity, price in zip(quantities, prices)])
        sources[f'book_{side}_max_share_l10'] = pl.when(valid & (total > 0)).then(maximum / total)
        sources[f'book_{side}_wall_offset_ticks_l10'] = pl.when(valid).then((wall_price - prices[0]).abs() / config['tick_size'])
        sources[f'book_{side}_max_gap_ticks_l10'] = pl.when(valid).then(
            pl.max_horizontal([(prices[index] - prices[index - 1]).abs() for index in range(1, 10)]) / config['tick_size'],
        )
    for metric in ['max_share', 'wall_offset_ticks', 'max_gap_ticks']:
        sources[f'book_bid_ask_{metric}_gap_l10'] = sources[f'book_bid_{metric}_l10'] - sources[f'book_ask_{metric}_l10']
    return sources


# 【非核心】通过通用窗口接口展开共享盘口墙状态，保持已有因子名和公式。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """计算入口先物化九个盘口状态，四档窗口直接复用。"""
    sources = {name: pl.col(name) for name in source_expressions(config)}
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
