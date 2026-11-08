"""Gen030现货与永续跨市场公式，生产需先取得并对齐现货盘口，当前不生成数据。"""
import polars as pl
from mm_research.factors.operators import rolling_factors

VERSION = 'seconds_v1'


# 【核心逻辑98】对已经按可见时钟对齐的现货和永续计算基差及历史收益差。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """spot列须由独立有效现货源提供，缺少输入时由生产入口报错。"""
    spot = (pl.col('spot_bid_price_1') + pl.col('spot_ask_price_1')) / 2
    perp = pl.col('mid_price')
    basis = (perp - spot) / spot * 10000
    delta = basis.diff()
    spot_return = (spot / spot.shift(1) - 1) * 10000
    perp_return = pl.col('return_100ms_bps')
    sources = {
        'cross_spot_perp_mid_basis_bps_100ms': basis,
        'cross_spot_perp_pos_basis_bps_100ms': basis.clip(lower_bound=0),
        'cross_spot_perp_neg_basis_bps_100ms': (-basis).clip(lower_bound=0),
        'cross_spot_perp_basis_delta_bps_100ms': delta,
        'cross_spot_perp_basis_absdelta_bps_100ms': delta.abs(),
        'cross_spot_perp_ret_gap_bps_100ms': spot_return - perp_return,
        'cross_spot_lag1_perp_ret_gap_bps_100ms': spot_return.shift(1) - perp_return,
        'cross_buyperp_sellspot_edge_bps_100ms': (pl.col('spot_bid_price_1') - pl.col('ask_price_1')) / spot * 10000,
        'cross_buyspot_sellperp_edge_bps_100ms': (pl.col('bid_price_1') - pl.col('spot_ask_price_1')) / spot * 10000,
    }
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
