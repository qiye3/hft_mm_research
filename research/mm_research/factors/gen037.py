"""Gen037先计算已完成UTC秒的十个100ms位置结构，再在整秒序列滚动。"""
import math
import polars as pl

VERSION = 'seconds_v1'


# 【核心逻辑110】保留秒内时间位置，构造早晚活动差、重心和集中度。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """十个位置至少八个有效，全部已知且无活动时为零，未知零总量为空。"""
    inputs = {'count': pl.col('trade_count').cast(pl.Float64),
              'qty': pl.col('buy_volume') + pl.col('sell_volume'),
              'absret': pl.col('return_100ms_bps').abs()}
    sources = {}
    for name, values in inputs.items():
        total = values.rolling_sum(10, min_samples=8)
        count = values.is_finite().fill_null(False).cast(pl.Int64).rolling_sum(10, min_samples=1)
        late = pl.sum_horizontal([values.shift(lag) for lag in range(3)])
        early = pl.sum_horizontal([values.shift(lag) for lag in range(7, 10)])
        weighted = pl.sum_horizontal([values.shift(lag) * (9.5 - lag) / 10 for lag in range(10)])
        metrics = {'late_early': (late - early) / total,
                   'centroid': weighted / total - .5,
                   'hhi': values.pow(2).rolling_sum(10, min_samples=8) / total.pow(2)}
        for metric, expression in metrics.items():
            sources[f'intrasec_{name}_{metric}_1s'] = pl.when((total == 0) & (count == 10)).then(0.0).when((total > 0) & (count >= 8)).then(expression)
    return sources


# 【核心逻辑109】在已完成整秒基础量上滚动，窗口5秒对应五个整秒位置。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """调用方先从完整100ms信息生成内层，再筛UTC整秒，外层保持80%有效比例。"""
    expressions = {}
    for name in source_expressions(config):
        for seconds in [5, 10, 30, 100]:
            expressions[f'{name}_sec_mean_{seconds}s'] = pl.col(name).rolling_mean(seconds, min_samples=math.ceil(seconds * config['min_fraction'])).cast(pl.Float64)
    return expressions
