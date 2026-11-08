"""Gen029比较最近1s与排除当前桶的过去5s状态，全部统计只使用历史。"""
import polars as pl
from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = 'seconds_v1'


# 【核心逻辑97】计算均值、方差及标准化累计偏离的状态改变。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """里层1s与5s分别要求8与40个有效观测，零历史方差为空。"""
    inputs = {'spread': ((pl.col('ask_price_1') - pl.col('bid_price_1')) / config['tick_size']).round(0),
              'activity': pl.col('trade_count').cast(pl.Float64),
              'depth': depth_quantity('bid', 10) + depth_quantity('ask', 10)}
    sources = {}
    for name, values in inputs.items():
        previous = values.shift(1)
        mean = previous.rolling_mean(50, min_samples=40)
        variance = previous.rolling_var(50, min_samples=40)
        standard = pl.when(variance > 0).then(variance.sqrt())
        recent_mean = values.rolling_mean(10, min_samples=8)
        recent_variance = values.rolling_var(10, min_samples=8)
        count = values.is_not_null().cast(pl.Int64).rolling_sum(10, min_samples=10)
        sources[f'regime_{name}_mean_shift_1s5s'] = (recent_mean - mean) / standard
        sources[f'regime_{name}_var_shift_1s5s'] = recent_variance / pl.when(variance > 0).then(variance) - 1
        sources[f'regime_{name}_cusum_shift_1s5s'] = (values.rolling_sum(10, min_samples=8) - count * mean) / (standard * count.sqrt())
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
