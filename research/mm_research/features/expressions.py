"""跨特征类别复用的 Polars 基础运算。

make_day_grid 建立UTC网格，normalized_imbalance 计算失衡，rolling_sum_in_segment 计算分段窗口。
网格供快照与逐笔共享，表达式供各类特征定义业务字段与单位。
"""

import datetime as dt

import numpy as np
import polars as pl

from mm_research.data.sources import DAY_NS


# 【核心逻辑20】计算买卖双方的归一化失衡，区分零量窗口与未知值。
def normalized_imbalance(buy: pl.Expr, sell: pl.Expr, *, empty_value: float | None) -> pl.Expr:
    """输入双方非负数量表达式及空窗值，输出归一化差值表达式。

    总量为零时返回 empty_value，源缺失继续传播为空，用于盘口与成交失衡。
    """
    total = buy + sell
    return (
        pl.when(total > 0).then((buy - sell) / total)
        .when(total == 0).then(empty_value).otherwise(None)
    )


# 【核心逻辑19】在连续区段内累计完整滚动窗口，数量可按整数Lot精确求和。
def rolling_sum_in_segment(value: pl.Expr, samples: int, *, quantity_step: float | None = None) -> pl.Expr:
    """输入数值表达式、窗口点数与可选数量步长，输出分段滚动表达式。

    调用方保证网格连续且时间有序，segment_id 标识质量区段。
    数量步长已在源层检查，整数累计保持空窗精确零值，未预热窗口为空。
    """
    if quantity_step is not None:
        value = (value / quantity_step).round().cast(pl.Int64)
    rolling = value.rolling_sum(samples, min_samples=samples).over("segment_id")
    if quantity_step is not None:
        rolling = rolling * quantity_step
    return rolling


# 【核心逻辑14】建立完整UTC日采样网格，统一盘口与成交的窗口结束时刻。
def make_day_grid(date: str, interval_ns: int) -> pl.DataFrame:
    """输入 UTC 日期与正的采样间隔，返回覆盖 [日初,次日日初) 的时间表。

    网格必须整除一天，标签和交易窗口使用相同的右端点时间。
    """
    if interval_ns <= 0 or DAY_NS % interval_ns:
        raise ValueError("采样间隔必须为一天的正整数约数")
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    return pl.DataFrame({"timestamp_ns": np.arange(start, start + DAY_NS, interval_ns, dtype=np.int64)})
