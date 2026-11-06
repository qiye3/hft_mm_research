"""跨特征类别复用的 Polars 基础运算。

normalized_imbalance 计算双方数量失衡，rolling_sum_in_segment 计算分段完整窗口。
输入列表达式，返回表达式，供具体特征模块定义业务字段与单位。
"""

import polars as pl


# 区分已知空窗与未知数量，使盘口和成交各自选择空窗口径。
def normalized_imbalance(buy: pl.Expr, sell: pl.Expr, *, empty_value: float | None) -> pl.Expr:
    """输入双方非负数量表达式及空窗值，输出归一化差值表达式。

    总量为零时返回 empty_value，源缺失继续传播为空，用于盘口与成交失衡。
    """
    total = buy + sell
    return (
        pl.when(total > 0).then((buy - sell) / total)
        .when(total == 0).then(empty_value).otherwise(None)
    )


# 按质量区段累计完整窗口，数量字段可转为整数 Lot 后求和。
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
