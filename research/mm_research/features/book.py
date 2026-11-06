"""现有研究版本的盘口特征，输入对齐后的十档基础表。

add_book_features 输出价差 Tick、分层数量失衡、深度与微价格偏移。
时钟对齐与盘口有效性由基础层负责，本模块只读取当前状态。
"""

import polars as pl

BOOK_FEATURE_COLUMNS = [
    "spread_ticks", "imbalance_1", "imbalance_5", "imbalance_10",
    "bid_depth_10", "ask_depth_10", "microprice_minus_mid",
]


# 从当前盘口计算价差、分层深度与微价格偏移。
def add_book_features(table: pl.DataFrame, tick_size: float) -> pl.DataFrame:
    """输入前十档价格数量、中间价和价差及 Tick Size，返回新增盘口特征的表。

    深度单位 BTC，微价格偏移单位 USDT/BTC，保留既有公式与中间深度字段。
    非法盘口由统一 book_valid 剔除，支持单独检查这类公式。
    """
    expressions = [(pl.col("spread") / tick_size).alias("spread_ticks")]
    for levels in [1, 5, 10]:
        bid_depth = pl.sum_horizontal([pl.col(f"bid_quantity_{level}") for level in range(1, levels + 1)])
        ask_depth = pl.sum_horizontal([pl.col(f"ask_quantity_{level}") for level in range(1, levels + 1)])
        expressions.extend([
            bid_depth.alias(f"bid_depth_{levels}"),
            ask_depth.alias(f"ask_depth_{levels}"),
            ((bid_depth - ask_depth) / (bid_depth + ask_depth)).alias(f"imbalance_{levels}"),
        ])
    microprice = (
        pl.col("ask_price_1") * pl.col("bid_quantity_1")
        + pl.col("bid_price_1") * pl.col("ask_quantity_1")
    ) / (pl.col("bid_quantity_1") + pl.col("ask_quantity_1"))
    expressions.append((microprice - pl.col("mid_price")).alias("microprice_minus_mid"))
    return table.with_columns(expressions)
