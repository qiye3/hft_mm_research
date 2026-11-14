"""准备 M0/M1 共用的 Tardis 单日标准事件分片。

convert 编排 CSV 校验、双时钟路由和 NPZ 输出，append_source 读取单个压缩源文件，
write_routed_shards 负责分侧排序与分片。输入为 BTCUSDT 全深度 L2 和逐笔成交 CSV.gz 或 CSV.zst/.zstd，
输出为与 C++ hbt.Event 二进制布局兼容的 NPZ，以及含原始计数和最终盘口的 manifest。
最终盘口字典用于离线对照，回测中的盘口更新由 C++ 引擎执行。
"""
import datetime as dt
import json
import math
import shutil
import time
import zipfile
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as arrow_csv
from mm_research.data.sources import open_market_csv

BUY = 1 << 29
SELL = 1 << 28
LOCAL = 1 << 30
EXCHANGE = 1 << 31
DEPTH = 1
TRADE = 2
CLEAR = 3
SNAPSHOT = 4
# 字段顺序和 64 字节记录长度必须与 hbt::Event 一致，填充字段随记录显式置零。
EVENT_DTYPE = np.dtype([
    ("ev", "<u8"),
    ("exch_ts", "<i8"),
    ("local_ts", "<i8"),
    ("px", "<f8"),
    ("qty", "<f8"),
    ("order_id", "<u8"),
    ("ival", "<i8"),
    ("fval", "<f8"),
], align=True)


# ==================== 原始源流与时间边界校验 ====================
# 【非核心】检查单行价格与数量满足研究配置的有限值和整数网格约束。
def validate_grid(price: float, quantity: float, tick_size: float, lot_size: float) -> None:
    """验证原始行情的一组价格与数量。

    输入
        price 为正价格，quantity 为非负数量，零数量表示撤去价位。
        tick_size 和 lot_size 是调用方已验证为正的价格与数量步长。
    输出
        合法时返回 None，非法值或偏离整数网格时抛出 ValueError。
    用途
        在生成二进制事件前逐行检查，确保原生订单簿按配置取整时有一致的单位。
        网格检查验证的是数据与研究参数的相容性。
    """
    if not math.isfinite(price) or price <= 0 or not math.isfinite(quantity) or quantity < 0:
        raise ValueError(f"无效价格/数量：{price}, {quantity}")
    if abs(price / tick_size - round(price / tick_size)) > 1e-6:
        raise ValueError(f"价格不符合 tick_size：{price}")
    if abs(quantity / lot_size - round(quantity / lot_size)) > 1e-6:
        raise ValueError(f"数量不符合 lot_size：{quantity}")


# 【核心逻辑161】将一个压缩源文件校验并追加为标准记录，同时保存原始数据的对照统计。
def append_source(filename: Path, kind: str, output, tick_size: float, lot_size: float,
                  chunk_rows: int) -> dict:
    """读取完整 Tardis 源流，为后续双侧路由准备无路由标志的事件。

    输入
        filename 为带表头的 CSV.gz 或 CSV.zst/.zstd，kind 取 book 或 trades。
        output 是已打开的二进制输出流，记录追加到其当前位置。
        tick_size 和 lot_size 指定研究网格，chunk_rows 指定内存写出缓冲区容量。
    输出
        返回源文件大小、原始行分类、首尾本地纳秒时间、最小行情延迟和最终 Top-10。
        同时将 EVENT_DTYPE 记录写入 output，盘口快照前额外加入清空事件。
    用途
        验证品种、方向、源流时序和初始快照，并提供独立于 C++ 回放的最终盘口基准。
        列缺失、非法数值、空流或最终盘口交叉时抛出异常。
    """
    counts = Counter()
    first_local = None
    last_local = None
    minimum_latency = None
    previous_snapshot = False
    bids = {}
    asks = {}
    names = ["exchange", "symbol", "timestamp", "local_timestamp", "side", "price", "amount"]
    types = {"exchange": pa.string(), "symbol": pa.string(), "timestamp": pa.int64(),
             "local_timestamp": pa.int64(), "side": pa.string(), "price": pa.float64(), "amount": pa.float64()}
    if kind == "book":
        names.append("is_snapshot")
        types["is_snapshot"] = pa.string()
    with open_market_csv(filename) as source:
        reader = arrow_csv.open_csv(source.buffer,
            read_options=arrow_csv.ReadOptions(block_size=16 * 1024 * 1024),
            convert_options=arrow_csv.ConvertOptions(column_types=types, include_columns=names))
        for batch in reader:
            frame = pl.from_arrow(batch)
            if frame.null_count().sum_horizontal().item() != 0:
                raise ValueError("原始事件存在缺失字段")
            if frame.filter((pl.col("exchange") != "binance-futures") | (pl.col("symbol") != "BTCUSDT")).height:
                raise ValueError("仅接收 binance-futures BTCUSDT 行情")
            exchange_times = frame["timestamp"].to_numpy() * 1000
            local_times = frame["local_timestamp"].to_numpy() * 1000
            if np.any(local_times[1:] < local_times[:-1]) or (last_local is not None and local_times[0] < last_local):
                raise ValueError(f"源文件本地时间戳倒退：{filename}")
            if first_local is None:
                first_local = int(local_times[0])
            latency = int(np.min(local_times - exchange_times))
            minimum_latency = latency if minimum_latency is None else min(minimum_latency, latency)
            prices = frame["price"].to_numpy()
            quantities = frame["amount"].to_numpy()
            if not np.all(np.isfinite(prices) & (prices > 0) & np.isfinite(quantities) & (quantities >= 0)):
                raise ValueError("无效价格/数量")
            if np.any(np.abs(prices / tick_size - np.rint(prices / tick_size)) > 1e-6):
                raise ValueError("价格不符合 tick_size")
            if np.any(np.abs(quantities / lot_size - np.rint(quantities / lot_size)) > 1e-6):
                raise ValueError("数量不符合 lot_size")
            sides = frame["side"].to_numpy()
            buy_side = "bid" if kind == "book" else "buy"
            sell_side = "ask" if kind == "book" else "sell"
            if np.any((sides != buy_side) & (sides != sell_side)):
                raise ValueError("未知方向")
            buys = sides == buy_side
            snapshots = np.zeros(len(frame), dtype=bool)
            clears = np.zeros(len(frame), dtype=bool)
            if kind == "book":
                snapshot_values = frame["is_snapshot"].to_numpy()
                if np.any((snapshot_values != "true") & (snapshot_values != "false")):
                    raise ValueError("无效 is_snapshot")
                snapshots = snapshot_values == "true"
                if counts["rows"] == 0 and not snapshots[0]:
                    raise ValueError("盘口文件必须从初始快照开始")
                previous_flags = np.r_[previous_snapshot, snapshots[:-1]]
                previous_times = np.r_[local_times[0] if last_local is None else last_local, local_times[:-1]]
                clears = snapshots & (~previous_flags | (local_times != previous_times))
                counts["clear_events"] += int(clears.sum())
                counts[str(SNAPSHOT)] += int(snapshots.sum())
                counts[str(DEPTH)] += len(frame) - int(snapshots.sum())
                # 仅压缩独立最终盘口对照的重复价位，写入引擎的事件保持逐行完整。
                final_start = int(np.flatnonzero(clears)[-1]) if clears.any() else 0
                if clears.any():
                    bids.clear()
                    asks.clear()
                final_updates = frame.slice(final_start).group_by(["side", "price"], maintain_order=True).agg(pl.col("amount").last())
                for side, price, quantity in final_updates.iter_rows():
                    book = bids if side == "bid" else asks
                    if quantity == 0:
                        book.pop(price, None)
                    else:
                        book[price] = quantity
            else:
                counts[str(TRADE)] += len(frame)
            event_positions = np.arange(len(frame)) + np.cumsum(clears)
            events = np.zeros(len(frame) + int(clears.sum()), dtype=EVENT_DTYPE)
            event_kinds = np.where(snapshots, SNAPSHOT, DEPTH) if kind == "book" else TRADE
            events["ev"][event_positions] = event_kinds | np.where(buys, BUY, SELL)
            events["exch_ts"][event_positions] = exchange_times
            events["local_ts"][event_positions] = local_times
            events["px"][event_positions] = prices
            events["qty"][event_positions] = quantities
            clear_positions = event_positions[clears] - 1
            events["ev"][clear_positions] = CLEAR
            events["exch_ts"][clear_positions] = exchange_times[clears]
            events["local_ts"][clear_positions] = local_times[clears]
            events.tofile(output)
            counts["rows"] += len(frame)
            last_local = int(local_times[-1])
            previous_snapshot = bool(snapshots[-1])
    if not counts["rows"]:
        raise ValueError(f"源文件为空：{filename}")
    final_bids = sorted(bids.items(), reverse=True)[:10]
    final_asks = sorted(asks.items())[:10]
    if kind == "book" and (not final_bids or not final_asks or final_bids[0][0] >= final_asks[0][0]):
        raise ValueError("最终原始盘口为空或交叉")
    return {
        "file": str(filename.resolve()), "bytes": filename.stat().st_size, "counts": dict(counts),
        "first_local_timestamp_ns": first_local, "last_local_timestamp_ns": last_local,
        "minimum_feed_latency_ns": minimum_latency, "final_bids": final_bids, "final_asks": final_asks,
    }



# 【非核心】用两个源文件的首尾接收时刻检查它们是否覆盖指定 UTC 日的两端。
def validate_day(sources: list[dict], date: str) -> None:
    """验证单日样本的时间边界。

    输入
        sources 是 append_source 返回的统计字典列表，时间戳单位为纳秒。
        date 为 YYYY-MM-DD 格式的 UTC 自然日。
    输出
        每个源流均进入首尾一分钟窗口时返回 None，否则抛出 ValueError。
    用途
        防止截短文件被当成完整日样本。此处检查时间边界，流的数值与时序由源读取阶段检查。
    """
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    end = start + 86_400_000_000_000
    for source in sources:
        first = source["first_local_timestamp_ns"]
        last = source["last_local_timestamp_ns"]
        if not start <= first < start + 60_000_000_000 or not end - 60_000_000_000 <= last < end:
            raise ValueError(f"数据未覆盖给定 UTC 日的首尾一分钟：{source['file']}")


# ==================== 双时钟路由与分片落盘 ====================
# 【非核心】独立压缩一个事件分片，返回按调用顺序写入清单的文件信息。
def write_compressed_shard(destination: Path, filename: str, batch: np.ndarray) -> dict:
    """以一级DEFLATE写入标准NPZ，减少压缩耗时，事件内容与顺序保持原样。"""
    path = destination / filename
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        with archive.open("data.npy", "w", force_zip64=True) as member:
            np.lib.format.write_array(member, batch, allow_pickle=False)
    return {"file": filename, "rows": len(batch), "bytes": path.stat().st_size}


# 【核心逻辑162】为交易所与本地分别生成可见事件，并按可见时间稳定排序写出 NPZ。
def write_routed_shards(events: np.ndarray, destination: Path, shard_rows: int,
                        compression_workers: int = 1) -> list[dict]:
    """生成两个处理器各自有序的标准事件分片。

    输入
        events 为 EVENT_DTYPE 原始标准记录，每条含 exchange/local 两个时间戳。
        destination 为可写目录，shard_rows 为每个输出分片的最大物理记录数。
    输出
        返回包含 file、rows 和 bytes 的有序分片列表，文件成员名统一为 data.npy。
        每条输入分别输出一条 EXCH-only 和一条 LOCAL-only 记录，物理行数为输入的两倍。
    用途
        允许同一行情在两地以不同顺序出现，并检查各侧跨分片的时间非递减约束。
    """
    event_count = len(events)
    # 前半索引对应交易所副本，后半索引对应本地副本，取模可回到同一原始事件。
    visible_timestamps = np.empty(event_count * 2, dtype=np.int64)
    visible_timestamps[:event_count] = events["exch_ts"]
    visible_timestamps[event_count:] = events["local_ts"]
    order = np.argsort(visible_timestamps, kind="stable")
    # 相同可见时刻保留输入顺序，因此 convert 先追加的成交仍先于深度处理。
    del visible_timestamps
    previous_exchange = None
    previous_local = None
    shards = []
    pending = deque()
    with ThreadPoolExecutor(max_workers=compression_workers) as executor:
        for start in range(0, len(order), shard_rows):
            indices = order[start:start + shard_rows]
            exchange_mask = indices < event_count
            batch = events[indices % event_count].copy()
            batch["ev"] |= np.where(exchange_mask, np.uint64(EXCHANGE), np.uint64(LOCAL))
            for timestamp_field, mask, previous in [
                ("exch_ts", exchange_mask, previous_exchange),
                ("local_ts", ~exchange_mask, previous_local),
            ]:
                timestamps = batch[timestamp_field][mask]
                if len(timestamps) and (np.any(timestamps[1:] < timestamps[:-1]) or
                                        (previous is not None and timestamps[0] < previous)):
                    raise ValueError(f"路由事件顺序错误：{timestamp_field}")
                if len(timestamps):
                    if timestamp_field == "exch_ts":
                        previous_exchange = int(timestamps[-1])
                    else:
                        previous_local = int(timestamps[-1])
            filename = f"events-{start // shard_rows:05d}.npz"
            pending.append(executor.submit(write_compressed_shard, destination, filename, batch))
            # 限制在途批次，按提交顺序收集结果，保持清单与事件顺序不变。
            if len(pending) >= compression_workers:
                shards.append(pending.popleft().result())
                print(f"已写分片 {shards[-1]['file']}", flush=True)
        while pending:
            shards.append(pending.popleft().result())
    return shards


# ==================== 单日转换编排与命令行入口 ====================
# 【核心逻辑160】编排单日全深度行情转换，生成可被原生 C++ reader 直接消费的清单。
def convert(book: Path, trades: Path, date: str, destination: Path, tick_size: float = 0.1,
            lot_size: float = 0.001, shard_rows: int = 250_000, full_day: bool = True,
            compression_workers: int = 1) -> dict:
    """完成从两个原始压缩文件到标准事件和转换清单的离线准备。

    输入
        book 和 trades 分别为完整深度流及成交流，date 为样本 UTC 日期。
        destination 必须为空，tick_size、lot_size 指定正步长，shard_rows 控制单片大小。
        full_day 默认为真，测试夹具可设为假以复用转换逻辑处理短序列。
        compression_workers 控制独立分片压缩线程数，同时限制在途批次数量。
    输出
        返回并写出 manifest.json，其中包括源行计数、路由记录数、时间偏移与分片位置。
        NPZ 保存全部原始深度，策略观察档数在后续 C++ 读取阶段配置。
    用途
        为 M0 验证和 M1 机制实验建立共同数据口径。成功后清理 .work 暂存事件，
        中途异常会保留已经写出的文件供排查，重试需要选择空输出目录。
    """
    if not math.isfinite(tick_size) or tick_size <= 0 or not math.isfinite(lot_size) or lot_size <= 0:
        raise ValueError("tick_size 和 lot_size 必须为正有限数")
    if not 1 <= shard_rows <= 5_000_000:
        raise ValueError("shard_rows 必须在 1 到 5,000,000 之间，避免 ZIP64")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("输出目录必须为空，避免混合不同数据版本")
    destination.mkdir(parents=True, exist_ok=True)
    workspace = destination / ".work"
    workspace.mkdir()
    started = time.monotonic()
    raw_file = workspace / "events.bin"
    # 成交源先写入，稳定排序后同时间戳的成交优先于深度更新，避免重复扣减队列。
    with raw_file.open("wb") as output:
        print("转换 trades", flush=True)
        trades_info = append_source(trades, "trades", output, tick_size, lot_size, shard_rows)
        print("转换 incremental_book_L2", flush=True)
        book_info = append_source(book, "book", output, tick_size, lot_size, shard_rows)
    if full_day:
        validate_day([book_info, trades_info], date)
    minimum_latency = min(book_info["minimum_feed_latency_ns"], trades_info["minimum_feed_latency_ns"])
    if minimum_latency <= 0:
        raise ValueError("源行情存在非正接收延迟，请先核实时间戳，转换器不会自动平移时间")
    events = np.memmap(raw_file, dtype=EVENT_DTYPE, mode="r")
    shards = write_routed_shards(events, destination, shard_rows, compression_workers)
    counts = Counter(book_info["counts"])
    manifest = {
        "schema": "hbt.Event/64-byte-little-endian",
        "symbol": "BTCUSDT",
        "exchange": "binance-futures",
        "date": date,
        "full_day_validated": full_day,
        "timestamp_unit": "ns",
        "tick_size": tick_size,
        "lot_size": lot_size,
        "timestamp_policy": "source_local_time",
        "logical_events": len(events),
        "normalized_rows": len(events) * 2,
        "depth_updates": counts[str(DEPTH)],
        "snapshot_levels": counts[str(SNAPSHOT)],
        "clear_events": counts["clear_events"],
        "trade_events": trades_info["counts"]["rows"],
        "book": book_info,
        "trades": trades_info,
        "shards": shards,
        "normalized_bytes": sum(shard["bytes"] for shard in shards),
        "npz_compression_level": 1,
        "conversion_seconds": time.monotonic() - started,
        "routing": "Separate EXCH and LOCAL rows, stable by visible timestamp; trades before depth on ties.",
        "depth_policy": "All source levels retained; 5/10-level selection is observer-only.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    del events
    shutil.rmtree(workspace)
    return manifest
