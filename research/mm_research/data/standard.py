"""按日生产仅含盘口与逐笔的标准行情 Parquet，供研究模块直接读取。

run_standard_production 使用配置顺序处理日期，build_standard_day 在临时目录完成两份产物后发布。
来源、单位、核验范围与输入身份写入 Parquet 元数据，新生产返回统计，已有产物返回复用状态。
"""

from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile

import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.orderbook import annotate_snapshot_records
from mm_research.data.sources import load_official_trades, scan_snapshot_source
from mm_research.data.trades import prepare_official_trades

METADATA_KEY = "mm_standard"
SCHEMA_VERSION = 2


# ==================== 单日与多日生产 ====================
# 【核心逻辑3】按配置逐日生产并发布标准盘口和普通逐笔Parquet。
def run_standard_production(config_path: Path) -> list[dict]:
    """输入 YAML 配置，返回各日期的产物统计并打印进度。

    配置路径中的数据位置相对项目根目录，已完成的同定义日期直接复用。
    数据版本不可覆盖，不一致的输入或配置要求使用新的版本。
    """
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "data_standardize")
    summaries = []
    for date in config["dates"]:
        summary = build_standard_day(root, config, str(date))
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summaries


# 【核心逻辑4】连接快照整理与官方成交整理，完整生成两份单日标准行情。
def build_standard_day(root: Path, config: dict, date: str) -> dict:
    """输入项目位置、上游配置与 UTC 日期，新生产返回行数和标签统计。

    盘口保持源文件全部行与二十五档，成交直接使用官方日档。
    两份表只保存交易所时间，固定行情延迟由后续回放配置。
    临时产物使用目标目录所在文件系统，发布后分别位于snapshots/日期.parquet和trades/日期.parquet。
    已有同定义产物只读取元数据，返回日期与reused=True。
    """
    if config["tick_size"] <= 0 or config["lot_size"] <= 0 or config["snapshot_silence_ns"] <= 0:
        raise ValueError("价格、数量步长和盘口静默门限必须为正")
    sources = resolve_standard_sources(root, config, date)

    # 定义元数据
    definition = {
        "schema_version": SCHEMA_VERSION,
        "version": config["version"],
        "date": date,
        "timestamp_unit": "UTC Unix ns",
        "price_unit": "USDT/BTC",
        "quantity_unit": "BTC",
        "snapshot_depth": 25,
        "snapshot_silence_ns": config["snapshot_silence_ns"],
        "confirmed_snapshot_gaps": config["confirmed_snapshot_gaps"].get(date, []),
        "tick_size": config["tick_size"],
        "lot_size": config["lot_size"],
        "sources": sources,
        "source_row": "one-based data row, excluding header",
        "market_clock": "exchange_timestamp_ns",
        "receipt_time_model": "exchange timestamp plus configured fixed market data latency at replay",
        "snapshot_gap_clock": "original source received timestamp, used only for quality diagnostics",
        "conflict_policy": "keep selected source row and tag content_conflict",
    }
    destination = root / config["output_root"] / config["version"]
    with partition_lock(destination):
        files = {kind: destination / kind / f"{date}.parquet" for kind in ["snapshots", "trades"]}
        if any(filename.exists() for filename in files.values()):
            for filename in files.values():
                if not filename.exists() or read_standard_metadata(filename)["definition"] != definition:
                    raise ValueError(f"已有标准数据不完整或定义不同，请检查 {destination} 中的 {date}")
            return {"date": date, "reused": True}

        # 临时目录生成标准盘口和逐笔行情
        with tempfile.TemporaryDirectory(prefix=f".{date}.", dir=destination) as temporary:
            staging = Path(temporary)

            # 生成标准盘口行情
            snapshot = sources["snapshots"]
            snapshots = annotate_snapshot_records(
                scan_snapshot_source(root / snapshot["path"], snapshot["id"]),
                config["snapshot_silence_ns"], definition["confirmed_snapshot_gaps"],
            )
            snapshots.drop("received_timestamp_ns").sink_parquet(
                staging / "snapshots.parquet", compression="zstd", compression_level=3,
                row_group_size=100_000, maintain_order=True, engine="streaming",
                metadata={METADATA_KEY: json.dumps({
                    "kind": "snapshots", "partition_clock": "source snapshot file day",
                    "replay_order": "exchange timestamp, source row",
                    "sort_order": "original source row", "definition": definition,
                }, ensure_ascii=False, sort_keys=True)},
            )

            # 生成普通逐笔行情
            trade_source = sources["trades"]
            trades = prepare_official_trades(
                load_official_trades(
                    root / trade_source["path"], trade_source["id"],
                    config["tick_size"], config["lot_size"],
                ), date,
            )
            trades.write_parquet(
                staging / "trades.parquet", compression="zstd", compression_level=3, row_group_size=100_000,
                metadata={METADATA_KEY: json.dumps({
                    "kind": "trades", "partition_clock": "exchange trade UTC day",
                    "sort_order": "exchange timestamp, trade ID, source row",
                    "definition": definition,
                }, ensure_ascii=False, sort_keys=True)},
            )

            # 检查生成结果
            summary = summarize_standard_day(staging / "snapshots.parquet", staging / "trades.parquet")
            if summary["snapshot_rows"] == 0 or summary["trade_rows"] == 0:
                raise ValueError(f"标准行情缺少盘口或成交记录 {date}")
            # 两份文件在版本锁内发布，任一次移动失败时撤回本次已发布的文件。
            with ExitStack() as publication:
                for kind, filename in files.items():
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    (staging / f"{kind}.parquet").replace(filename)
                    publication.callback(filename.unlink)
                publication.pop_all()
        summary["date"] = date
        return summary


# ==================== 单一来源定位 ====================
# 【非核心】定位每日两份原始输入，记录文件位置与状态以核对缓存。
def resolve_standard_sources(root: Path, config: dict, date: str) -> dict:
    """输入配置与日期，返回两份必要源文件的位置、大小和修改时刻。

    source 与 source_row 共同定位原始记录，文件变化会阻止复用已有版本。
    """
    sources = {}
    for kind in ["snapshots", "trades"]:
        filename = root / config[kind].format(date=date)
        status = filename.stat()
        sources[kind] = {
            "id": f"{kind}:{date}",
            "path": os.path.relpath(filename, root),
            "bytes": status.st_size,
            "mtime_ns": status.st_mtime_ns,
        }
    return sources


# ==================== 元数据和逐日检查 ====================
# 【非核心】读取标准Parquet的来源与格式定义，供产物复用时核对。
def read_standard_metadata(filename: Path) -> dict:
    """输入标准 Parquet 文件，返回该文件内嵌的 kind、时钟和来源定义。"""
    return json.loads(pl.read_parquet_metadata(filename)[METADATA_KEY])


# 【非核心】检查25档字段并汇总记录数量和质量标签。
def summarize_standard_day(snapshots: Path, trades: Path) -> dict:
    """输入当日盘口和成交Parquet文件，返回行数和质量标签统计。

    全部二十五档通过列结构核对，统计扫描仅投影时间与标签，控制内存。
    """
    names = pl.read_parquet_schema(snapshots).names()
    for level in range(1, 26):
        for side in ["bid", "ask"]:
            for field in ["price", "quantity"]:
                if f"{side}_{field}_{level}" not in names:
                    raise ValueError("标准盘口缺少二十五档字段")
    snapshot_stats = pl.scan_parquet(snapshots).select(
        pl.len().alias("snapshot_rows"),
        pl.col("quality_flags").list.contains("gap_before_confirmed").sum().alias("confirmed_snapshot_gaps"),
        pl.col("quality_flags").list.contains("gap_before_suspected").sum().alias("suspected_snapshot_gaps"),
    ).collect().to_dicts()[0]
    trade_stats = pl.scan_parquet(trades).select(
        pl.len().alias("trade_rows"),
        pl.col("quality_flags").list.contains("content_conflict").sum().alias("trade_content_conflicts"),
        pl.col("quality_flags").list.contains("duplicate_collapsed").sum().alias("trade_duplicate_groups"),
        (pl.col("quality_flags").list.len() > 0).sum().alias("flagged_trades"),
    ).collect().to_dicts()[0]
    date = read_standard_metadata(trades)["definition"]["date"]
    return {"date": date, **snapshot_stats, **trade_stats}


# 【非核心】锁定版本目录，防止并发发布或更新互相覆盖文件。
@contextmanager
def partition_lock(folder: Path):
    """输入数据版本目录，在上下文内独占读写，退出时释放目录描述符。"""
    folder.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)
