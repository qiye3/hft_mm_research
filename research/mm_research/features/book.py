"""现有研究版本的盘口特征，输入对齐后的十档基础表。

book_expressions 输出价差 Tick、分层数量失衡、深度与微价格偏移。
时钟对齐与盘口有效性由基础层负责，本模块只读取当前状态。
"""

import polars as pl

GROUP_VERSION = "v2"

BOOK_FEATURE_COLUMNS = [
    "spread_ticks", "imbalance_1", "imbalance_5", "imbalance_10",
    "bid_depth_10", "ask_depth_10", "microprice_minus_mid",
]


# 【核心逻辑18】构造盘口特征的原生表达式，每列只依赖所需的价格和数量。
def book_expressions(tick_size: float) -> dict[str, pl.Expr]:
    """输入Tick Size，返回字段名到表达式的映射，源空值继续传播。

    深度单位BTC，价差为Tick，微价格偏移为USDT/BTC。调用方选择列并组织执行。
    """
    expressions = {"spread_ticks": pl.col("spread") / tick_size}
    for levels in [1, 5, 10]:
        bid_columns = [pl.col(f"bid_quantity_{level}") for level in range(1, levels + 1)]
        ask_columns = [pl.col(f"ask_quantity_{level}") for level in range(1, levels + 1)]
        bid_depth = pl.sum_horizontal(bid_columns, ignore_nulls=False)
        ask_depth = pl.sum_horizontal(ask_columns, ignore_nulls=False)
        expressions[f"imbalance_{levels}"] = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        if levels == 10:
            expressions["bid_depth_10"] = bid_depth
            expressions["ask_depth_10"] = ask_depth
    microprice = (
        pl.col("ask_price_1") * pl.col("bid_quantity_1")
        + pl.col("bid_price_1") * pl.col("ask_quantity_1")
    ) / (pl.col("bid_quantity_1") + pl.col("ask_quantity_1"))
    expressions["microprice_minus_mid"] = microprice - pl.col("mid_price")
    return expressions
