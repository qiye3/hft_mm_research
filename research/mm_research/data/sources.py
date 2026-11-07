"""读取官方成交ZIP、Tardis格式25档CSV及Crypto Lake的20档Parquet。

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

# 一天的纳秒数 = 24 * 60 * 60 * 1000 * 1000 * 1000 = 86_400_000_000_000
DAY_NS = 86_400_000_000_000

TRADE_KEYS = ["exchange", "symbol", "trade_id"] # 逐笔行情的key列
TRADE_VALUES = ["exchange_timestamp_ns", "price", "quantity", "side", "record_kind"] # 逐笔行情的value列
TRADE_COLUMNS = [
    *TRADE_KEYS, *TRADE_VALUES,
    "source", "source_row", "quality_flags", # 逐笔行情的全部列
]


# 【核心逻辑5】按明确来源格式读取全部快照档位，映射到统一价量与纳秒时间字段。
def scan_snapshot_source(filename: Path, source: str, source_format: str, depth: int) -> pl.LazyFrame:
    """输入快照文件、来源、声明格式与档数，返回标准字段的流式表。

    Tardis CSV微秒转纳秒，Crypto Lake Parquet原生纳秒保持原值。
    source_row 从第一条数据记录的1开始计数，同时间戳的连续状态保持源顺序。
    保留原始档数与行序，原始接收时间在标注来源缺口后移除。
    """
    if source_format == "cryptolake_parquet" and depth == 20:
        metadata = pl.read_parquet_metadata(filename)
        if metadata["exchange"] != "BINANCE_FUTURES" or metadata["symbol"] != "BTC-USDT-PERP":
            raise ValueError("Crypto Lake快照必须属于Binance BTCUSDT永续")
        levels = [
            pl.col(f"{side}_{level}_{raw_field}").alias(f"{side}_{field}_{level + 1}")
            for level in range(depth)
            for side in ["bid", "ask"]
            for raw_field, field in [("price", "price"), ("size", "quantity")]
        ]
        return pl.scan_parquet(filename).with_row_index("source_row", offset=1).select(
            pl.lit("binance-futures").alias("exchange"), pl.lit("BTCUSDT").alias("symbol"),
            pl.col("timestamp").alias("exchange_timestamp_ns"),
            pl.col("receipt_timestamp").alias("received_timestamp_ns"),
            *levels, pl.lit(source).cast(pl.Categorical).alias("source"), "source_row",
        )
    if source_format != "tardis_csv" or depth != 25:
        raise ValueError("盘口格式与档数必须为tardis_csv/25或cryptolake_parquet/20")
    schema = {"timestamp": pl.Int64, "local_timestamp": pl.Int64}

    levels = []
    for level in range(depth):
        for raw_side, side in [("bids", "bid"), ("asks", "ask")]:
            for raw_field, field in [("price", "price"), ("amount", "quantity")]:
                raw = f"{raw_side}[{level}].{raw_field}"
                schema[raw] = pl.Float64
                levels.append(pl.col(raw).alias(f"{side}_{field}_{level + 1}"))

    return pl.scan_csv(filename, schema_overrides=schema).with_row_index("source_row", offset=1).select( # 读取CSV文件，并添加源行号
        "exchange", "symbol", # 交易所和符号
        (pl.col("timestamp") * 1000).alias("exchange_timestamp_ns"), # 交易所时间，纳秒
        (pl.col("local_timestamp") * 1000).alias("received_timestamp_ns"), # 原始接收时间，纳秒
        *levels, pl.lit(source).cast(pl.Categorical).alias("source"), "source_row", # 盘口价格和数量，来源，源行号
    ) # 返回流式二十五档表

# 【核心逻辑7】读取官方成交ZIP，统一时间、价量和主动方向字段。
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

    # 转换为标准逐笔行情
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
        "missing_trade_id": pl.col("trade_id").is_null(), # 交易ID缺失
        "missing_exchange_time": pl.col("exchange_timestamp_ns").is_null(), # 交易所时间缺失
        "invalid_price": pl.col("price").is_null() | ~pl.col("price").is_finite() | (pl.col("price") <= 0), # 价格无效
        "invalid_quantity": pl.col("quantity").is_null() | ~pl.col("quantity").is_finite() | (pl.col("quantity") <= 0), # 数量无效
        "invalid_side": pl.col("side").is_null() | ~pl.col("side").is_in(["buy", "sell"]), # 方向无效
        "off_tick": ((pl.col("price") / tick_size).round() - pl.col("price") / tick_size).abs() > 1e-6, # 价格步长无效
        "off_lot": ((pl.col("quantity") / lot_size).round() - pl.col("quantity") / lot_size).abs() > 1e-6, # 数量步长无效
        "exchange_time_backwards": pl.col("exchange_timestamp_ns").diff() < 0, # 交易所时间倒退
    }

    return table.with_columns(build_record_flags(rules).alias("quality_flags")).select(TRADE_COLUMNS)


# 【非核心】按声明的压缩格式打开CSV文本流，拒绝未知后缀。
def open_market_csv(filename: Path):
    """打开 gzip 或 Zstandard 压缩的行情 CSV。

    filename只接受明确声明的.gz、.zst或.zstd后缀。
    返回可用于 with 的文本流，换行交给 csv.reader 处理，关闭时同时释放底层文件。
    两种压缩格式共用后续字段、时间和盘口检查。
    """
    if filename.suffix in {".zst", ".zstd"}:
        return zstandard.open(filename, "rt", encoding="utf-8", newline="")
    if filename.suffix == ".gz":
        return gzip.open(filename, "rt", newline="")
    raise ValueError(f"不支持的压缩行情格式 {filename.name}")


# 【核心逻辑14】合并本日与已有前日标准行情，按事件时间稳定排序。
def scan_standard_history(folder: Path, kind: str, date: str) -> pl.LazyFrame:
    """输入标准版本目录、表类型与日期，返回按交易所时间和源行号排序的行情。

    首个已下载日期的前日可能不可用，基础表会明确屏蔽无法确认的起始成交区间。
    """
    calendar = dt.date.fromisoformat(date)
    previous = folder / kind / f"{calendar - dt.timedelta(days=1)}.parquet"
    filenames = [folder / kind / f"{date}.parquet"]

    if previous.exists():
        filenames.insert(0, previous) # 将前一日标准行情插入到列表开头

    return pl.scan_parquet(filenames).with_row_index("_history_row").sort(
        ["exchange_timestamp_ns", "_history_row"], maintain_order=True, # 按交易所时间和源行号排序
    ).drop("_history_row")
