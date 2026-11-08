"""Gen027由过去1s累计金额和数量构造VWAP锚点，禁止平均各桶VWAP。"""
import polars as pl
from mm_research.factors.operators import rolling_factors

VERSION = 'seconds_v1'


# 【核心逻辑92】构造买卖及全部成交的1s均价锚点偏离、移动和相对位置。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """1s里层至少8个有效桶且成交量为正，外层按四档秒窗口均值计算。"""
    mid = pl.col('mid_price')
    sources = {}
    for side in ['buy', 'sell', 'all']:
        if side == 'all':
            notional = pl.col('buy_notional') + pl.col('sell_notional')
            quantity = pl.col('buy_volume') + pl.col('sell_volume')
        else:
            notional = pl.col(f'{side}_notional')
            quantity = pl.col(f'{side}_volume')
        volume = quantity.rolling_sum(10, min_samples=8)
        anchor = notional.rolling_sum(10, min_samples=8) / pl.when(volume > 0).then(volume)
        previous = anchor.shift(10)
        sources[f'trade_{side}_vwap1s_mid_gap_bps'] = (mid - anchor) / mid * 10000
        sources[f'trade_{side}_vwap1s_mom_bps'] = (anchor / pl.when(previous > 0).then(previous) - 1) * 10000
        sources[f'trade_{side}_vwap1s_above_flag'] = (mid > anchor).cast(pl.Float64)
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
