"""现有研究版本的事件 OFI 窗口特征。

add_order_flow_features 从 C++ 基础变量 ofi_bbo 生成三个时间窗口，单位 BTC。
输入连续 100ms 网格和质量区段，输出新增 OFI 列的表，快照未知值继续为空。
"""

import polars as pl

from mm_research.features.expressions import rolling_sum_in_segment

ORDER_FLOW_FEATURE_COLUMNS = ["ofi_100ms", "ofi_1s", "ofi_5s"]


# 【核心逻辑66】将逐事件OFI累计为100ms、1秒和5秒订单流特征。
def add_order_flow_features(table: pl.DataFrame) -> pl.DataFrame:
    """输入有 ofi_bbo 与 segment_id 的连续网格，返回附加三个 OFI 窗口的表。

    1s、5s 使用完整十点、五十点窗口，快照源空值保持传播。
    """
    expressions = [pl.col("ofi_bbo").alias("ofi_100ms")]
    for suffix, samples in [("1s", 10), ("5s", 50)]:
        expressions.append(rolling_sum_in_segment(pl.col("ofi_bbo"), samples).alias(f"ofi_{suffix}"))
    return table.with_columns(expressions)
