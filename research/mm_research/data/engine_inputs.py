"""以固定研究延迟连接原始L2和标准官方成交，再复用既有原生事件转换器。"""

import datetime as dt
import json
from pathlib import Path
import tempfile

import polars as pl
import zstandard

from mm_research.data.engine import convert
from mm_research.data.sources import DAY_NS, scan_market_csv


# 【非核心】逐块写出转换器需要的Tardis列，临时压缩CSV只用于同一研究时钟适配。
def write_engine_csv(table: pl.LazyFrame, filename: Path) -> None:
    """通过有界批次输出Zstandard CSV，避免额外保存整份文本字符串。"""
    with zstandard.open(filename, "wb") as stream:
        header = True
        for frame in table.collect_batches(chunk_size=100_000, maintain_order=True, engine="streaming"):
            frame.write_csv(stream, include_header=header)
            header = False


# 【核心逻辑142】按交易所时间加固定延迟准备L2与官方成交，生成双时间戳原生事件。
def prepare_engine_day(book: Path, trades: Path, date: str, destination: Path, latency_ns: int) -> dict:
    """保留全部深度事件，临时适配文件在转换结束后删除，初始快照允许早于UTC日界。"""
    manifest_file = destination / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest["fixed_market_latency_ns"] != latency_ns or manifest["date"] != date:
            raise ValueError("引擎数据时钟或日期变化，需要新的输出版本")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".engine-input-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        orderbook = scan_market_csv(book, {
            "timestamp": pl.Int64, "local_timestamp": pl.Int64, "price": pl.Float64, "amount": pl.Float64,
        })
        times = orderbook.select(pl.col("timestamp").min().alias("first"), pl.col("timestamp").max().alias("last")).collect().row(0)
        start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        if times[0] * 1000 > start + 60_000_000_000 or times[1] * 1000 < start + DAY_NS - 60_000_000_000:
            raise ValueError("L2源未覆盖指定UTC日两端")
        orderbook = orderbook.with_columns((pl.col("timestamp") + latency_ns // 1000).alias("local_timestamp"))
        orderbook = orderbook.sort("timestamp", maintain_order=True)
        write_engine_csv(orderbook, staging / "book.csv.zst")
        del orderbook
        fills = pl.scan_parquet(trades)
        if fills.filter(pl.col("quality_flags").list.len() > 0).select(pl.len()).collect().item():
            raise ValueError("实时引擎输入存在已标记异常成交，请明确核实后处理")
        fills = fills.select(
            "exchange", "symbol", (pl.col("exchange_timestamp_ns") // 1000).alias("timestamp"),
            ((pl.col("exchange_timestamp_ns") + latency_ns) // 1000).alias("local_timestamp"),
            pl.col("trade_id").alias("id"), "side", "price", pl.col("quantity").alias("amount"),
        ).sort("timestamp", maintain_order=True)
        write_engine_csv(fills, staging / "trades.csv.zst")
        # 完整日按交易所时间检查，固定延迟后的末尾回报可自然进入下一UTC日。
        events_staging = staging / "events"
        manifest = convert(staging / "book.csv.zst", staging / "trades.csv.zst", date, events_staging,
                           shard_rows=1_000_000, full_day=False)
        manifest.update(
            full_day_validated=True, timestamp_policy="exchange_timestamp_plus_fixed_latency",
            fixed_market_latency_ns=latency_ns, original_book=str(book), original_trades=str(trades),
            full_day_check="exchange_time_first_and_last_minute",
        )
        for kind, source in [("book", book), ("trades", trades)]:
            manifest[kind]["adapted_csv_bytes"] = manifest[kind]["bytes"]
            manifest[kind]["file"] = str(source)
            manifest[kind]["bytes"] = source.stat().st_size
        (events_staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        events_staging.replace(destination)
    return manifest
