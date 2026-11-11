"""Gen034离散市场状态的信息熵、条件熵与互信息，配对不跨窗口左边界。"""
import math
from itertools import product
import polars as pl
from mm_research.factors.operators import book_transition_allowed

VERSION = 'seconds_v1'
STATES = [-1, 0, 1]


# 【核心逻辑105】从经验计数计算bit熵，明确采用0log2(0)=0。
def count_entropy(counts: list[pl.Expr]) -> pl.Expr:
    """输入同一分布的类别计数，空分布由调用方有效比例条件排除。"""
    total = pl.sum_horizontal(counts)
    terms = []
    for count in counts:
        probability = count / total
        terms.append(pl.when(count > 0).then(-probability * probability.log(2)).otherwise(0.0))
    return pl.sum_horizontal(terms, ignore_nulls=False)


# 【核心逻辑104】共享三种状态的单点及滞后配对计数，每个滚动计数只物化一次。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """保留100ms窗口位置，配对不越过窗口左端，缺失位置不压缩。"""
    imbalance = pl.col('imbalance_5')
    book = pl.when(imbalance.is_finite()).then(pl.when(imbalance > .1).then(1).when(imbalance < -.1).then(-1).otherwise(0))
    sources = {
        'info_mid_move_sign_100ms': pl.when(book_transition_allowed()).then(pl.col('mid_price').diff().sign()),
        'info_trade_aggr_sign_100ms': (pl.col('buy_volume') - pl.col('sell_volume')).sign(),
        'info_book_obi_l5_sign_100ms': book,
    }
    counts = {}
    for (name, state), seconds in product(sources.items(), config["aggregation_windows_s"]):
        window = seconds * 10
        for index, value in enumerate(STATES):
            indicator = (state == value).fill_null(False).cast(pl.Int64)
            counts[f'{name}_count_{seconds}s_{index}'] = indicator.rolling_sum(window, min_samples=1)
        for lag in [1, 2]:
            valid = pl.all_horizontal([state.shift(offset).is_finite() for offset in range(lag + 1)])
            for index, (old, current) in enumerate(product(STATES, repeat=2)):
                indicator = ((state.shift(lag) == old) & (state == current) & valid).fill_null(False).cast(pl.Int64)
                counts[f'{name}_pair_{seconds}s_lag{lag}_{index}'] = indicator.rolling_sum(window - lag, min_samples=1)
    return counts


# 【核心逻辑103】从已物化的共享计数生成bit信息量，避免嵌套表达式反复展开滚动图。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """只有计数阶段使用100ms位置，整秒位置上的熵计算与逐事件计算结果一致。"""
    expressions = {}
    sources = ['info_mid_move_sign_100ms', 'info_trade_aggr_sign_100ms', 'info_book_obi_l5_sign_100ms']
    for name, seconds in product(sources, config["aggregation_windows_s"]):
        window = seconds * 10
        counts = [pl.col(f'{name}_count_{seconds}s_{index}') for index in range(3)]
        enough = pl.sum_horizontal(counts) >= math.ceil(window * config['min_fraction'])
        expressions[f'{name}_entropy_{seconds}s'] = pl.when(enough).then(count_entropy(counts)).cast(pl.Float64)
        for lag in [1, 2]:
            pairs = [pl.col(f'{name}_pair_{seconds}s_lag{lag}_{index}') for index in range(9)]
            rows = [pl.sum_horizontal(pairs[index * 3:index * 3 + 3]) for index in range(3)]
            acceptable = enough & (pl.sum_horizontal(pairs) >= math.ceil((window - lag) * config['min_fraction']))
            joint = count_entropy(pairs)
            if lag == 1:
                value = joint - count_entropy(rows)
                result_name = f'{name}_cond_entropy_lag1_{seconds}s'
            else:
                columns = [pl.sum_horizontal([pairs[index + shift * 3] for shift in range(3)]) for index in range(3)]
                value = count_entropy(rows) + count_entropy(columns) - joint
                result_name = f'{name}_mi_lag2_{seconds}s'
            expressions[result_name] = pl.when(acceptable).then(value.clip(lower_bound=0)).cast(pl.Float64)
    return expressions
