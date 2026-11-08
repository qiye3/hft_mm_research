"""现有研究版本的事件 OFI 窗口特征。

add_order_flow_features 从 C++ 基础变量 ofi_bbo 生成三个时间窗口，单位 BTC。
输入连续100ms网格与缺失值，输出新增 OFI 列的表，快照未知值继续为空。
"""

import polars as pl


ORDER_FLOW_FEATURE_COLUMNS = ["ofi_100ms", "ofi_1s", "ofi_5s"]


# 【核心逻辑186】将逐事件OFI累计为100ms、1秒和5秒订单流特征。
def add_order_flow_features(table: pl.DataFrame, min_samples: dict[str, int]) -> pl.DataFrame:
    """输入含ofi_bbo的连续网格和窗口最少样本数，返回附加三个 OFI 窗口的表。

    1s、5s分别使用十点和五十点窗口，按min_samples处理缺失。
    """
    expressions = [pl.col("ofi_bbo").alias("ofi_100ms")]
    for suffix, samples in [("1s", 10), ("5s", 50)]:
        expressions.append(pl.col("ofi_bbo").fill_nan(None).rolling_sum(samples, min_samples=min_samples[suffix]).alias(f"ofi_{suffix}"))
    return table.with_columns(expressions)
