"""读取官方成交 ZIP 与网盘二十五档快照，供标准行情生产使用。

load_official_trades 统一官方逐笔字段，scan_snapshot_source 流式映射完整盘口。
标准行情使用交易所时间，快照原始接收时间仅供来源质量检查，写入前移除。
"""

import datetime as dt
import gzip
from pathlib import Path
import zipfile

import polars as pl
import zstandard

from mm_research.data.quality import build_record_flags

DAY_NS = 86_400_000_000_000

TRADE_KEYS = ["exchange", "symbol", "trade_id"]
TRADE_VALUES = ["exchange_timestamp_ns", "price", "quantity", "side", "record_kind"]
TRADE_COLUMNS = [
    *TRADE_KEYS, "exchange_timestamp_ns", "price", "quantity", "side", "record_kind",
    "source", "source_row", "quality_flags",
]


# 将快照全部二十五档映射成统一字段，保留源行序供检查和追溯。
def scan_snapshot_source(filename: Path, source: str) -> pl.LazyFrame:
    """输入快照 CSV 和来源标识，返回流式二十五档表。

    时间从微秒转为纳秒，原始接收时间在标注来源缺口后移除。
    source_row 从第一条数据记录的1开始计数，同时间戳的连续状态保持源顺序。
    """
    schema = {"timestamp": pl.Int64, "local_timestamp": pl.Int64}
    levels = []
    for level in range(25):
        for raw_side, side in [("bids", "bid"), ("asks", "ask")]:
            for raw_field, field in [("price", "price"), ("amount", "quantity")]:
                raw = f"{raw_side}[{level}].{raw_field}"
                schema[raw] = pl.Float64
                levels.append(pl.col(raw).alias(f"{side}_{field}_{level + 1}"))
    return pl.scan_csv(filename, schema_overrides=schema).with_row_index("source_row", offset=1).select(
        "exchange", "symbol",
        (pl.col("timestamp") * 1000).alias("exchange_timestamp_ns"),
        (pl.col("local_timestamp") * 1000).alias("received_timestamp_ns"),
        *levels, pl.lit(source).cast(pl.Categorical).alias("source"), "source_row",
    )


# 直接读取官方日度成交档案，统一时间、方向和数量并保留字段异常。
def load_official_trades(filename: Path, source: str, tick_size: float, lot_size: float) -> pl.DataFrame:
    """输入 Binance USD-M 官方成交 ZIP、来源和价格数量步长，返回标准逐笔。

    包内必须包含一份 CSV，毫秒成交时间转为 UTC Unix 纳秒。
    行情延迟由回放统一配置，此表保留交易所时间和源行号，异常记录加标签。
    """
    with zipfile.ZipFile(filename) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"官方成交 ZIP 必须包含一份 CSV {filename.name}")
        with archive.open(members[0]) as stream:
            table = pl.read_csv(stream, schema_overrides={
                "id": pl.Int64, "time": pl.Int64, "price": pl.Float64,
                "qty": pl.Float64, "is_buyer_maker": pl.Boolean,
            })
    table = table.with_row_index("source_row", offset=1).select(
        pl.lit("binance-futures").alias("exchange"), pl.lit("BTCUSDT").alias("symbol"),
        pl.col("id").alias("trade_id"),
        (pl.col("time") * 1_000_000).alias("exchange_timestamp_ns"),
        pl.col("price").cast(pl.Float64), pl.col("qty").cast(pl.Float64).alias("quantity"),
        pl.when(pl.col("is_buyer_maker")).then(pl.lit("sell"))
        .when(~pl.col("is_buyer_maker")).then(pl.lit("buy")).otherwise(None).alias("side"),
        pl.lit("MARKET").alias("record_kind"),
        pl.lit(source).cast(pl.Categorical).alias("source"), "source_row",
    )
    rules = {
        "missing_trade_id": pl.col("trade_id").is_null(),
        "missing_exchange_time": pl.col("exchange_timestamp_ns").is_null(),
        "invalid_price": pl.col("price").is_null() | ~pl.col("price").is_finite() | (pl.col("price") <= 0),
        "invalid_quantity": pl.col("quantity").is_null() | ~pl.col("quantity").is_finite() | (pl.col("quantity") <= 0),
        "invalid_side": pl.col("side").is_null() | ~pl.col("side").is_in(["buy", "sell"]),
        "off_tick": ((pl.col("price") / tick_size).round() - pl.col("price") / tick_size).abs() > 1e-6,
        "off_lot": ((pl.col("quantity") / lot_size).round() - pl.col("quantity") / lot_size).abs() > 1e-6,
        "exchange_time_backwards": pl.col("exchange_timestamp_ns").diff() < 0,
    }
    return table.with_columns(build_record_flags(rules).alias("quality_flags")).select(TRADE_COLUMNS)


# 按源文件压缩格式建立文本流，供盘口与成交流共用。
def open_market_csv(filename: Path):
    """打开 gzip 或 Zstandard 压缩的行情 CSV。

    filename 为既有源文件，.zst/.zstd 使用 Zstandard，其余延续原有 gzip 读取方式。
    返回可用于 with 的文本流，换行交给 csv.reader 处理，关闭时同时释放底层文件。
    两种压缩格式共用后续字段、时间和盘口检查。
    """
    if filename.suffix in {".zst", ".zstd"}:
        return zstandard.open(filename, "rt", encoding="utf-8", newline="")
    return gzip.open(filename, "rt", newline="")


# 读取当前日和已有前日标准表，接回固定延迟后跨午夜到达的记录。
def scan_standard_history(folder: Path, kind: str) -> pl.LazyFrame:
    """输入标准日期目录及表类型，返回按交易所时间和源行号排序的行情。

    首个已下载日期的前日可能不可用，基础表会明确屏蔽无法确认的起始成交区间。
    """
    date = dt.date.fromisoformat(folder.name.removeprefix("date="))
    previous = folder.parent / f"date={date - dt.timedelta(days=1)}" / f"{kind}.parquet"
    filenames = [folder / f"{kind}.parquet"]
    if previous.exists():
        filenames.insert(0, previous)
    return pl.scan_parquet(filenames).with_row_index("_history_row").sort(
        ["exchange_timestamp_ns", "_history_row"], maintain_order=True,
    ).drop("_history_row")
