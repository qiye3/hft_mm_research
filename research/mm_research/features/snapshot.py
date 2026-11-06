"""独立的 Polars 快照特征示例，使用当前前十档名义金额描述盘口不对称。

add_snapshot_notional_features 接收已向后对齐且带 book_valid 的快照网格。
返回双侧深度金额和金额失衡，可作为新增快照特征模块的参考。
"""

import polars as pl

from mm_research.features.expressions import normalized_imbalance

SNAPSHOT_EXAMPLE_COLUMNS = ["snapshot_bid_notional_10", "snapshot_ask_notional_10", "snapshot_notional_imbalance_10"]


# 用价格乘数量衡量双方十档挂单金额，再计算相对失衡。
def add_snapshot_notional_features(table: pl.DataFrame) -> pl.DataFrame:
    """输入包含十档价格数量及 book_valid 的网格，返回新增示例特征的表。

    双侧金额单位 USDT，失衡无量纲，正值表示买侧金额较大。
    无效盘口的示例特征置空，且所有公式只依赖当前快照。
    """
    table = table.with_columns([
        pl.sum_horizontal([
            pl.col(f"{side}_price_{level}") * pl.col(f"{side}_quantity_{level}")
            for level in range(1, 11)
        ]).alias(f"snapshot_{side}_notional_10")
        for side in ["bid", "ask"]
    ])
    table = table.with_columns(
        normalized_imbalance(
            pl.col("snapshot_bid_notional_10"), pl.col("snapshot_ask_notional_10"), empty_value=None,
        ).alias("snapshot_notional_imbalance_10"),
    )
    return table.with_columns([
        pl.when(pl.col("book_valid")).then(pl.col(column)).otherwise(None).alias(column)
        for column in SNAPSHOT_EXAMPLE_COLUMNS
    ])
