"""现货成交与价格联动的首轮20列因子，输入均为决策时已经可见的秒数据。"""

import polars as pl


# 【核心逻辑347】在已对齐的整秒序列上计算现货压力、价格联动和分钟溢价变量。
def cross_market_factors(frame: pl.DataFrame, windows: list[int]) -> pl.DataFrame:
    """输入现货秒线、过去永续Mid和已结束分钟溢价，返回时间键及20列因子。"""
    frame = frame.with_columns(pl.col('perp_mid').fill_nan(None))
    spot = pl.col('spot_close')
    perpetual = pl.col('perp_mid')
    gap = (spot / perpetual).log() * 10000
    expressions = []
    for window in windows:
        minimum = int(window * 0.8)
        quantity = pl.col('volume').rolling_sum(window, min_samples=minimum)
        buys = pl.col('buy_volume').rolling_sum(window, min_samples=minimum)
        spot_return = (spot / spot.shift(window)).log() * 10000
        perpetual_return = (perpetual / perpetual.shift(window)).log() * 10000
        expressions.extend([
            pl.when(quantity > 0).then((2 * buys - quantity) / quantity).otherwise(None)
            .alias(f'spot_imbalance_{window}s'),
            spot_return.alias(f'spot_return_{window}s'),
            gap.rolling_mean(window, min_samples=minimum).alias(f'spot_close_perp_mid_gap_{window}s'),
            (spot_return - perpetual_return).alias(f'spot_perp_return_gap_{window}s'),
        ])
    premium = pl.col('premium_bps')
    expressions.append(premium.alias('premium_level_bps'))
    for window in [60, 180, 600]:
        expressions.append((premium - premium.shift(window)).alias(f'premium_change_{window}s'))
    return frame.select('timestamp_ns', *expressions)
