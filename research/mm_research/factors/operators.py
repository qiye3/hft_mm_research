"""把具名里层公式展开为原生Polars窗口，统一缺失样本口径。"""

import math

import polars as pl


# 【核心逻辑63】识别相邻有效盘口的质量接续，逐日区段重置不单独切断午夜。
def book_transition_allowed() -> pl.Expr:
    """返回100ms相邻位置是否可差分，日内区段变化及已知来源切换使该次差分失效。"""
    same_day = pl.col("date") == pl.col("date").shift(1)
    segment_change = same_day & (pl.col("book_segment_id") != pl.col("book_segment_id").shift(1))
    source_switch = (pl.col("date") == "2025-10-01") & ~same_day
    return ((pl.col("timestamp_ns").diff() == 100_000_000)
            & ~segment_change.fill_null(False) & ~source_switch.fill_null(False)).fill_null(False)


# 【核心逻辑43】同时展开同组窗口，保留null并允许跨日连续滑动。
def rolling_factors(sources: dict[str, pl.Expr], windows: tuple, operation: str,
                    minimum_fraction: float) -> dict[str, pl.Expr]:
    """输入里层表达式、窗口及有效比例，返回完整因子名到表达式的映射。"""
    if not 0 < minimum_fraction <= 1:
        raise ValueError("min_fraction必须位于(0,1]")
    expressions = {}
    for name, expression in sources.items():
        values = expression.fill_nan(None)
        for window in windows:
            if operation == "mean_diff":
                short_seconds, long_seconds = window
                short, long = short_seconds * 10, long_seconds * 10
                expressions[f"{name}_mean_diff_{short_seconds}s_{long_seconds}s"] = (
                    values.rolling_mean(short, min_samples=math.ceil(short * minimum_fraction))
                    - values.rolling_mean(long, min_samples=math.ceil(long * minimum_fraction))
                )
            elif operation == "mean":
                expressions[f"{name}_mean_{window}s"] = values.rolling_mean(
                    window * 10, min_samples=math.ceil(window * 10 * minimum_fraction),
                )
            elif operation == "sum":
                expressions[f"{name}_sum_{window}s"] = values.rolling_sum(
                    window * 10, min_samples=math.ceil(window * 10 * minimum_fraction),
                )
            else:
                raise ValueError(f"未知窗口算子 {operation}")
    return {name: expression.cast(pl.Float64) for name, expression in expressions.items()}
