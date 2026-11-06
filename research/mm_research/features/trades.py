"""现有研究版本的主动成交特征。

add_trade_features 接收连续 100ms 买卖量与质量区段，增加历史成交量及成交失衡。
数量按 Lot 整数累计，原始成交读取和分桶由 data/base.py 提供。
"""

import polars as pl

from mm_research.features.expressions import normalized_imbalance, rolling_sum_in_segment

TRADE_FEATURE_COLUMNS = [
    "buy_volume_1s", "sell_volume_1s", "buy_volume_5s", "sell_volume_5s",
    "trade_imbalance_100ms", "trade_imbalance_1s",
]


# 先累计窗口主动买卖量，再计算相同窗口的成交失衡。
def add_trade_features(table: pl.DataFrame, lot_size: float) -> pl.DataFrame:
    """输入连续网格与数量步长，返回源字段和成交特征。

    完整无成交窗口失衡为零，未预热或未知窗口为空，数量单位 BTC。
    用于现有数据生产及独立类别研究。
    """
    expressions = []
    for suffix, samples in [("1s", 10), ("5s", 50)]:
        for source in ["buy_volume", "sell_volume"]:
            expressions.append(
                rolling_sum_in_segment(pl.col(source), samples, quantity_step=lot_size).alias(f"{source}_{suffix}")
            )
    table = table.with_columns(expressions)
    return table.with_columns([
        normalized_imbalance(pl.col(buy), pl.col(sell), empty_value=0.0).alias(f"trade_imbalance_{suffix}")
        for suffix, buy, sell in [
            ("100ms", "buy_volume", "sell_volume"),
            ("1s", "buy_volume_1s", "sell_volume_1s"),
        ]
    ])
