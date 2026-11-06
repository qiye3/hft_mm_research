"""现有研究版本的历史中间价收益特征，单位 bps。

add_return_features 输入连续网格和质量区段，输出三个过去周期的收益。
使用正向 shift 读取历史，区段起始预热保持为空。
"""

import polars as pl

GROUP_VERSION = "v1"

RETURN_FEATURE_COLUMNS = ["return_100ms_bps", "return_1s_bps", "return_5s_bps"]


# 【核心逻辑21】在连续盘口区段内计算100ms、1秒和5秒历史中间价收益。
def add_return_features(table: pl.DataFrame) -> pl.DataFrame:
    """输入含 mid_price、segment_id 的有序 100ms 表，返回新增历史收益的表。

    100ms、1s、5s 分别回看一点、十点、五十点，未预热值用于统一有效性判断。
    """
    expressions = []
    for suffix, samples in [("100ms", 1), ("1s", 10), ("5s", 50)]:
        previous = pl.col("mid_price").shift(samples).over("segment_id")
        expressions.append(((pl.col("mid_price") / previous - 1) * 10000).alias(f"return_{suffix}_bps"))
    return table.with_columns(expressions)
