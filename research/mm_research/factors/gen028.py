"""Gen028过去5s突破与已确认的2s失败突破，状态转换用Numba计算有界历史。"""
import polars as pl
from mm_research.factors.operators import book_transition_allowed, rolling_factors

VERSION = 'seconds_v1'
COLUMNS = [f'px_range_{name}_100ms' for name in [
    'up_flag', 'down_flag', 'up_down_flag_gap', 'up_dist_ticks', 'down_dist_ticks',
    'up_down_dist_ticks_gap', 'up_fail_flag', 'down_fail_flag', 'up_down_fail_flag_gap',
]]


# 【核心逻辑95】在有序100ms列上执行有界历史状态，输出九列共同状态。
def breakout_series(series: list[pl.Series], tick_size: float) -> pl.Series:
    """生产环境要求Numba，使用明确依赖，返回与输入等长的Struct列。"""
    from mm_research.factors.event_states import breakout_states
    result = breakout_states(series[0].to_numpy(), series[1].to_numpy(), tick_size)
    return pl.DataFrame({name: result[:, index] for index, name in enumerate(COLUMNS)}).to_struct()


# 【核心逻辑94】共享一次突破状态计算，再展开四档秒级历史均值。
def source_expressions(config: dict) -> dict[str, pl.Expr]:
    """状态只读取当前及之前价格，前日历史覆盖5s参考、2s确认和100s外层。"""
    state = pl.map_batches(
        [pl.col('mid_price'), book_transition_allowed()],
        lambda series: breakout_series(series, config['tick_size']),
        return_dtype=pl.Struct({name: pl.Float64 for name in COLUMNS}), is_elementwise=False,
    )
    return {'_breakout_state': state}


# 【核心逻辑93】在一次Struct状态计算之后共享九种基础量的四档窗口。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """Numba状态整段只计算一次，外层滚动从已物化的状态列读取。"""
    sources = {name: pl.col('_breakout_state').struct.field(name) for name in COLUMNS}
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
