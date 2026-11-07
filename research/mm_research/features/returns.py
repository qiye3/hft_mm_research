"""完整时间序列的历史中间价收益，单位bps，缺失端点输出空值。"""

import polars as pl

GROUP_VERSION = "v2"
RETURN_FEATURE_COLUMNS = ["return_100ms_bps", "return_1s_bps", "return_5s_bps"]


# 【核心逻辑22】按精确历史时刻计算收益，跨午夜延续时间序列。
def return_expressions(interval_ns: int) -> dict[str, pl.Expr]:
    """输入网格间隔，返回100ms、1s和5s的历史收益表达式。

    两端Mid缺失则结果为空，时间差检查防止缺日期被压缩成较短窗口。
    """
    expressions = {}
    for suffix, samples in [("100ms", 1), ("1s", 10), ("5s", 50)]:
        previous = pl.col("mid_price").shift(samples)
        expected_time = pl.col("timestamp_ns") - samples * interval_ns
        expressions[f"return_{suffix}_bps"] = pl.when(
            pl.col("timestamp_ns").shift(samples) == expected_time
        ).then((pl.col("mid_price") / previous - 1) * 10000).otherwise(None)
    return expressions
