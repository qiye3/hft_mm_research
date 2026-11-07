"""现有研究版本的主动成交特征。

trade_expressions在完整100ms序列上构造累计成交量及失衡，缺失值由min_samples处理。
数量按 Lot 整数累计，build_trade_bins 从标准逐笔Parquet计算基础成交分桶。
"""

import datetime as dt
from pathlib import Path

import polars as pl

from mm_research.data.sources import DAY_NS, scan_standard_history

from mm_research.features.expressions import normalized_imbalance

GROUP_VERSION = "v2"

TRADE_FEATURE_COLUMNS = [
    "buy_volume_1s", "sell_volume_1s", "buy_volume_5s", "sell_volume_5s",
    "trade_imbalance_100ms", "trade_imbalance_1s",
]


# ==================== 标准逐笔的基础特征计算 ====================
# 【核心逻辑16】按模拟可见时刻汇总100ms成交笔数、主动买卖量和金额。
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
# 【核心逻辑20】构造跨午夜且忽略缺失观测的成交窗口表达式。
def trade_expressions(lot_size: float, min_samples: dict[str, int]) -> dict[str, pl.Expr]:
    """输入Lot步长与各窗口最少样本数，返回成交量和失衡表达式。

    以整数Lot累计有效成交量，零成交窗口为零，观测不足时为空，不按完整窗口外推数量。
    """
    expressions = {}
    windows = {}
    for suffix, samples in [("1s", 10), ("5s", 50)]:
        windows[suffix] = {}
        for source in ["buy_volume", "sell_volume"]:
            lots = (pl.col(source) / lot_size).round().cast(pl.Int64)
            volume = lots.rolling_sum(samples, min_samples=min_samples[suffix]) * lot_size
            expressions[f"{source}_{suffix}"] = volume
            windows[suffix][source] = volume
    expressions["trade_imbalance_100ms"] = normalized_imbalance(
        pl.col("buy_volume"), pl.col("sell_volume"), empty_value=0.0,
    )
    expressions["trade_imbalance_1s"] = normalized_imbalance(
        windows["1s"]["buy_volume"], windows["1s"]["sell_volume"], empty_value=0.0,
    )
    return expressions
