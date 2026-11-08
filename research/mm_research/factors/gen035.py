"""Gen035固定价格段的显示数量密度与双侧不对称，单位BTC/Tick。"""
import polars as pl
from mm_research.factors.operators import rolling_factors

VERSION = 'seconds_v1'


# 【核心逻辑106】在固定三段完整档位上测量数量相对价格跨度的密度。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """价格跨度最小1Tick，正数量与档位完整性由当前有效盘口保证。"""
    sources = {}
    for first, last in [(1, 2), (3, 5), (6, 10)]:
        densities = {}
        for side in ['bid', 'ask']:
            quantities = [pl.col(f'{side}_quantity_{level}') for level in range(first, last + 1)]
            total = pl.sum_horizontal(quantities, ignore_nulls=False)
            span = (pl.col(f'{side}_price_{first}') - pl.col(f'{side}_price_{last}')).abs() / config['tick_size']
            density = pl.when(pl.all_horizontal([quantity > 0 for quantity in quantities])).then(total / span.clip(lower_bound=1))
            densities[side] = density
            sources[f'book_{side}_marginal_density_l{first}_{last}'] = density
        bid, ask = densities['bid'], densities['ask']
        sources[f'book_marginal_density_asym_l{first}_{last}'] = (bid - ask) / pl.when(bid + ask > 0).then(bid + ask)
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
