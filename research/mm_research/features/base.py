"""base特征组，从标准盘口和逐笔计算100ms基础列。

结果与衍生组共同保存到日期命名的特征Parquet，供增量特征和独立标签复用。
"""

import datetime as dt
from pathlib import Path

import polars as pl

from mm_research.features.snapshot import build_snapshot_grid
from mm_research.features.trades import build_trade_bins

GROUP_VERSION = "v3"
KEY_COLUMNS = ["symbol", "date", "timestamp_ns"]
BOOK_COLUMNS = [f"{side}_{field}_{level}" for level in range(1, 11)
                for side in ["bid", "ask"] for field in ["price", "quantity"]]
MODEL_COLUMNS = [*BOOK_COLUMNS, "mid_price", "spread", "trade_count", "buy_volume",
                 "sell_volume", "buy_notional", "sell_notional"]
BASE_COLUMNS = [
    *BOOK_COLUMNS, "last_book_timestamp_ns", "book_exchange_ns", "snapshot_age_ns",
    "mid_price", "spread", "book_valid", "book_segment_id", "trade_count",
    "buy_volume", "sell_volume", "buy_notional", "sell_notional", "trade_valid",
    "last_trade_timestamp_ns", "last_trade_exchange_ns", "flow_valid",
    "last_available_timestamp_ns",
]


# 【核心逻辑12】关联100ms盘口和成交汇总，生成完整网格并将异常研究值置空。
def build_base_features(folder: Path, date: str, interval_ns: int, lot_size: float,
                        max_book_age_ns: int, market_data_latency_ns: int) -> pl.DataFrame:
    """输入标准行情与采样配置，返回可独立用于特征和标签的100ms表。

    只有前日来源存在时，首个跨午夜成交窗口才可确认为完整；其余已知无成交窗口填零。
    盘口与成交分别置空。book_segment_id仅供标签连续性判断，滚动特征直接跨缺失值计算。
    """
    base = build_snapshot_grid(folder, date, interval_ns, max_book_age_ns, market_data_latency_ns)
    bins = build_trade_bins(folder, date, interval_ns, lot_size, market_data_latency_ns)
    base = base.join(bins, on="timestamp_ns", how="left").with_columns(
        pl.col("trade_count").fill_null(0), pl.col("trade_valid").fill_null(True),
        pl.col("buy_volume", "sell_volume", "buy_notional", "sell_notional").fill_null(0.0),
    )
    calendar = dt.date.fromisoformat(date)
    previous = folder / "trades" / f"{calendar - dt.timedelta(days=1)}.parquet"
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    coverage = pl.lit(True) if previous.exists() else pl.col("timestamp_ns") >= start + market_data_latency_ns + interval_ns
    base = base.with_columns((pl.col("trade_valid") & coverage).alias("flow_valid"))
    numeric_books = [*BOOK_COLUMNS, "mid_price", "spread"]
    numeric_trades = ["buy_volume", "sell_volume", "buy_notional", "sell_notional", "trade_count"]
    base = base.with_columns([
        pl.when(pl.col("book_valid")).then(pl.col(column)).otherwise(None).alias(column)
        for column in numeric_books
    ])
    base = base.with_columns([
        pl.when(pl.col("flow_valid")).then(pl.col(column)).otherwise(None).alias(column)
        for column in numeric_trades
    ])
    return base.with_columns(
        pl.max_horizontal("last_book_timestamp_ns", "last_trade_timestamp_ns").alias("last_available_timestamp_ns"),
    ).drop("source_gap", "source_segment_id")
