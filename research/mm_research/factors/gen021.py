"""Gen021聚合成交桶的方向记忆，零方向或缺失中断里层连续长度。"""
import polars as pl
from mm_research.factors.operators import rolling_factors

VERSION = 'seconds_v1'


# 【核心逻辑83】从主动净成交方向计算连续长度、方向保持和反转标记。
def source_expressions(config: dict, initial_start_ns: int | None = None) -> dict[str, pl.Expr]:
    """连续长度按100ms位置计数，外层均值跨缺失滑动，缺失方向不当作零。"""
    direction = (pl.col('buy_volume') - pl.col('sell_volume')).sign()
    previous = direction.shift(1)
    reset = (direction != previous).fill_null(True) | (direction == 0).fill_null(True)
    reset_time = pl.col('timestamp_ns')
    if initial_start_ns is not None:
        # 前置的上一位置只提供方向比较，连续长度沿用其真实起点。
        reset_time = pl.when(pl.col('timestamp_ns') == pl.col('timestamp_ns').first()).then(initial_start_ns).otherwise(reset_time)
    last_reset = pl.when(reset).then(reset_time).forward_fill()
    length = (pl.col('timestamp_ns') - last_reset) / 100_000_000 + 1
    signed = pl.when(direction.is_not_null()).then(pl.when(direction != 0).then(direction * length).otherwise(0))
    valid_pair = direction.is_not_null() & previous.is_not_null()
    sources = {
        'trade_dir_buy_run_100ms': signed.clip(lower_bound=0),
        'trade_dir_sell_run_100ms': (-signed).clip(lower_bound=0),
        'trade_dir_signed_run_100ms': signed,
        'trade_dir_buy_flag_100ms': (direction == 1).cast(pl.Float64),
        'trade_dir_sell_flag_100ms': (direction == -1).cast(pl.Float64),
        'trade_dir_switch_flag_100ms': pl.when(valid_pair).then((direction * previous == -1).cast(pl.Float64)),
        'trade_dir_buy_to_sell_100ms': pl.when(valid_pair).then(((previous == 1) & (direction == -1)).cast(pl.Float64)),
        'trade_dir_sell_to_buy_100ms': pl.when(valid_pair).then(((previous == -1) & (direction == 1)).cast(pl.Float64)),
        'trade_dir_same_nonzero_100ms': pl.when(valid_pair).then(((previous == direction) & (direction != 0)).cast(pl.Float64)),
    }
    return sources


# 【核心逻辑82】把连续方向状态与窗口计算分开，整段状态只计算一次。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """九种里层状态共享输入，输出四档秒级均值。"""
    sources = {name: pl.col(name) for name in source_expressions(config)}
    return rolling_factors(sources, (5, 10, 30, 100), 'mean', config['min_fraction'])
