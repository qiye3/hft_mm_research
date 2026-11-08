"""Gen036过去已完成收益对上一桶压力的历史局部线性响应。"""
import math
import polars as pl
from mm_research.factors.operators import book_transition_allowed

VERSION = 'seconds_v1'


# 【核心逻辑108】在明确样本数和非零方差下计算局部OLS斜率。
def local_slope(left: pl.Expr, right: pl.Expr, window: int, minimum: int) -> pl.Expr:
    """配对列具有相同缺失位置，结果为cov(X,R)/var(X)，无需未来标签。"""
    variance = left.rolling_var(window, min_samples=minimum)
    covariance = pl.rolling_cov(left, right, window_size=window, min_samples=minimum)
    return pl.when(variance > 0).then(covariance / variance)


# 【核心逻辑107】比较局部斜率、相关与正负压力响应差，全部使用历史配对。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """总配对至少80%，正负子组各至少8条，零方差结果为空。"""
    returns = pl.col('return_100ms_bps')
    sources = {'book_obi_l5': pl.col('imbalance_5'), 'trade_aggr_imb_100ms': pl.col('trade_imbalance_100ms'),
               'trade_aggr_imb_1s': pl.col('trade_imbalance_1s')}
    expressions = {}
    for name, pressure in sources.items():
        previous = pressure.shift(1)
        valid = book_transition_allowed() & previous.is_finite() & returns.is_finite()
        left, right = pl.when(valid).then(previous), pl.when(valid).then(returns)
        for seconds in [5, 10, 30, 100]:
            window = seconds * 10
            minimum = math.ceil(window * config['min_fraction'])
            enough = valid.fill_null(False).cast(pl.Int64).rolling_sum(window, min_samples=1) >= minimum
            beta = local_slope(left, right, window, minimum)
            covariance = pl.rolling_cov(left, right, window_size=window, min_samples=minimum)
            variance_x = left.rolling_var(window, min_samples=minimum)
            variance_y = right.rolling_var(window, min_samples=minimum)
            correlation = pl.when((variance_x > 0) & (variance_y > 0)).then(covariance / (variance_x * variance_y).sqrt())
            positive = left > 0
            negative = left < 0
            beta_positive = local_slope(pl.when(positive).then(left), pl.when(positive).then(right), window, 8)
            beta_negative = local_slope(pl.when(negative).then(left), pl.when(negative).then(right), window, 8)
            expressions[f'{name}_lag_beta_{seconds}s'] = beta.cast(pl.Float64)
            expressions[f'{name}_lag_corr_{seconds}s'] = correlation.cast(pl.Float64)
            expressions[f'{name}_lag_asym_beta_{seconds}s'] = pl.when(enough).then(beta_positive - beta_negative).cast(pl.Float64)
    return expressions
