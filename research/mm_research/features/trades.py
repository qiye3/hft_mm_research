"""现有研究版本的主动成交特征。

add_trade_features 接收连续 100ms 买卖量与质量区段，增加历史成交量及成交失衡。
数量按 Lot 整数累计，build_trade_bins 从标准逐笔Parquet计算基础成交分桶。
"""

import datetime as dt
from pathlib import Path

import polars as pl

from mm_research.data.sources import DAY_NS, scan_standard_history

from mm_research.features.expressions import normalized_imbalance, rolling_sum_in_segment

GROUP_VERSION = "v1"

TRADE_FEATURE_COLUMNS = [
    "buy_volume_1s", "sell_volume_1s", "buy_volume_5s", "sell_volume_5s",
    "trade_imbalance_100ms", "trade_imbalance_1s",
]


# ==================== 标准逐笔的基础特征计算 ====================
# 【核心逻辑15】按模拟可见时刻汇总100ms成交笔数、主动买卖量和金额。
def build_trade_bins(folder: Path, date: str, interval_ns: int,
                     lot_size: float, market_data_latency_ns: int) -> pl.DataFrame:
    """输入标准版本目录与日期与延迟参数，返回数量、金额、笔数和区间质量。

    编号只作为成交身份，不参与质量切段。异常成交保留计数，其影响窗口标为无效。
    """
    table = scan_standard_history(folder, "trades", date).collect(engine="streaming")
    if table["exchange_timestamp_ns"].null_count():
        raise ValueError("成交时间缺失，无法确定受影响采样窗口")
    available = pl.col("exchange_timestamp_ns") + market_data_latency_ns
    # 价量和方向已由Data逐记录标注，Features只决定哪些标签会使当前成交窗口失效。
    valid = ~pl.col("quality_flags").list.eval(
        pl.element().is_in([
            "invalid_price", "invalid_quantity", "invalid_side", "content_conflict",
            "off_tick", "off_lot", "missing_trade_id",
        ])
    ).list.any()
    table = table.with_columns(
        available.alias("available_timestamp_ns"),
        (((available + interval_ns - 1) // interval_ns) * interval_ns).alias("timestamp_ns"),
        pl.when(valid).then((pl.col("quantity") / lot_size).round())
        .otherwise(None).cast(pl.Int64).alias("quantity_lots"),
        valid.alias("trade_valid"),
    )
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    table = table.filter(pl.col("timestamp_ns").is_between(start, start + DAY_NS, closed="left"))
    return table.group_by("timestamp_ns", maintain_order=True).agg(
        pl.len().cast(pl.Int64).alias("trade_count"),
        (pl.col("quantity_lots").filter((pl.col("side") == "buy") & pl.col("trade_valid")).sum() * lot_size).alias("buy_volume"),
        (pl.col("quantity_lots").filter((pl.col("side") == "sell") & pl.col("trade_valid")).sum() * lot_size).alias("sell_volume"),
        (pl.col("price") * pl.col("quantity")).filter((pl.col("side") == "buy") & pl.col("trade_valid")).sum().alias("buy_notional"),
        (pl.col("price") * pl.col("quantity")).filter((pl.col("side") == "sell") & pl.col("trade_valid")).sum().alias("sell_notional"),
        pl.col("trade_valid").all().alias("trade_valid"),
        pl.col("available_timestamp_ns").last().alias("last_trade_timestamp_ns"),
        pl.col("exchange_timestamp_ns").last().alias("last_trade_exchange_ns"),
    )


# ==================== 衍生成交特征 ====================
# 【核心逻辑18】累计1秒与5秒主动买卖量，计算对应窗口的成交失衡。
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
