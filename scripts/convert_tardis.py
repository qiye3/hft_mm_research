"""准备 M0/M1 共用的 Tardis 单日标准事件分片。

convert 编排 CSV 校验、双时钟路由和 NPZ 输出，append_source 读取单个压缩源文件，
write_routed_shards 负责分侧排序与分片。输入为 BTCUSDT 全深度 L2 和逐笔成交 CSV.gz 或 CSV.zst/.zstd，
输出为与 C++ hbt.Event 二进制布局兼容的 NPZ，以及含原始计数和最终盘口的 manifest。
最终盘口字典用于离线对照，回测中的盘口更新由 C++ 引擎执行。
"""
import argparse
import csv
import datetime as dt
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
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


# 【核心逻辑36】将一个压缩源文件校验并追加为标准记录，同时保存原始数据的对照统计。
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
    buffer = np.zeros(chunk_rows, dtype=EVENT_DTYPE)
    used = 0
    counts = Counter()
    first_local = None
    last_local = None
    minimum_latency = None
    previous_snapshot = False
    bids = {}
    asks = {}
    with open_market_csv(filename) as source:
        rows = csv.reader(source)
        columns = {name: index for index, name in enumerate(next(rows))}
        required = {"exchange", "symbol", "timestamp", "local_timestamp", "side", "price", "amount"}
        if kind == "book":
            required.add("is_snapshot")
        if not required.issubset(columns):
            raise ValueError(f"缺少 CSV 列：{required - columns.keys()}")
        for row in rows:
            if row[columns["exchange"]] != "binance-futures" or row[columns["symbol"]] != "BTCUSDT":
                raise ValueError("仅接收 binance-futures BTCUSDT 行情")
            # Tardis CSV 时间戳以微秒表示，标准事件在此统一放大为纳秒。
            exchange_ts = int(row[columns["timestamp"]]) * 1000
            local_ts = int(row[columns["local_timestamp"]]) * 1000
            if last_local is not None and local_ts < last_local:
                raise ValueError(f"源文件本地时间戳倒退：{filename}")
            if first_local is None:
                first_local = local_ts
            latency = local_ts - exchange_ts
            minimum_latency = latency if minimum_latency is None else min(minimum_latency, latency)
            price = float(row[columns["price"]])
            quantity = float(row[columns["amount"]])
            validate_grid(price, quantity, tick_size, lot_size)
            side = row[columns["side"]]
            allowed_sides = {"bid", "ask"} if kind == "book" else {"buy", "sell"}
            if side not in allowed_sides:
                raise ValueError(f"未知方向：{side}")
            flag = BUY if side in {"buy", "bid"} else SELL
            event_kind = TRADE
            is_snapshot = False
            if kind == "book":
                snapshot_value = row[columns["is_snapshot"]]
                if snapshot_value not in {"true", "false"}:
                    raise ValueError(f"无效 is_snapshot：{snapshot_value}")
                is_snapshot = snapshot_value == "true"
                if counts["rows"] == 0 and not is_snapshot:
                    raise ValueError("盘口文件必须从初始快照开始")
                new_snapshot = is_snapshot and (not previous_snapshot or local_ts != last_local)
                if new_snapshot:
                    # 同一快照的各价位共享本地接收时刻，只在该批快照起点清空一次。
                    # 日初和重连都按完整替换处理，旧价位不能继续残留在离线对照中。
                    buffer[used] = (CLEAR, exchange_ts, local_ts, 0, 0, 0, 0, 0)
                    used += 1
                    counts["clear_events"] += 1
                    bids.clear()
                    asks.clear()
                    if used == chunk_rows:
                        buffer.tofile(output)
                        used = 0
                event_kind = SNAPSHOT if is_snapshot else DEPTH
                # 全量保留深层更新，使它们在前方价位删除后仍可进入观察范围。
                side_book = bids if flag == BUY else asks
                if quantity == 0:
                    side_book.pop(price, None)
                else:
                    side_book[price] = quantity
            buffer[used] = (event_kind | flag, exchange_ts, local_ts, price, quantity, 0, 0, 0)
            used += 1
            counts["rows"] += 1
            counts[str(event_kind)] += 1
            last_local = local_ts
            previous_snapshot = is_snapshot
            if used == chunk_rows:
                buffer.tofile(output)
                used = 0
        if used:
            buffer[:used].tofile(output)
    if not counts["rows"]:
        raise ValueError(f"源文件为空：{filename}")
    final_bids = sorted(bids.items(), reverse=True)[:10]
    final_asks = sorted(asks.items())[:10]
    if kind == "book" and (not final_bids or not final_asks or final_bids[0][0] >= final_asks[0][0]):
        raise ValueError("最终原始盘口为空或交叉")
    return {
        "file": str(filename.resolve()),
        "bytes": filename.stat().st_size,
        "counts": dict(counts),
        "first_local_timestamp_ns": first_local,
        "last_local_timestamp_ns": last_local,
        "minimum_feed_latency_ns": minimum_latency,
        "final_bids": final_bids,
        "final_asks": final_asks,
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
# 【核心逻辑37】为交易所与本地分别生成可见事件，并按可见时间稳定排序写出 NPZ。
def write_routed_shards(events: np.ndarray, destination: Path, shard_rows: int) -> list[dict]:
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
        filename = f"events-{len(shards):05d}.npz"
        np.savez_compressed(destination / filename, data=batch)
        shards.append({"file": filename, "rows": len(batch), "bytes": (destination / filename).stat().st_size})
        print(f"已写分片 {filename}，累计 {start + len(batch):,}/{len(order):,} 行", flush=True)
    return shards


# ==================== 单日转换编排与命令行入口 ====================
# 【核心逻辑35】编排单日全深度行情转换，生成可被原生 C++ reader 直接消费的清单。
def convert(book: Path, trades: Path, date: str, destination: Path, tick_size: float = 0.1,
            lot_size: float = 0.001, shard_rows: int = 250_000, full_day: bool = True) -> dict:
    """完成从两个原始压缩文件到标准事件和转换清单的离线准备。

    输入
        book 和 trades 分别为完整深度流及成交流，date 为样本 UTC 日期。
        destination 必须为空，tick_size、lot_size 指定正步长，shard_rows 控制单片大小。
        full_day 默认为真，测试夹具可设为假以复用转换逻辑处理短序列。
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
    shards = write_routed_shards(events, destination, shard_rows)
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
        "conversion_seconds": time.monotonic() - started,
        "routing": "Separate EXCH and LOCAL rows, stable by visible timestamp; trades before depth on ties.",
        "depth_policy": "All source levels retained; 5/10-level selection is observer-only.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    del events
    shutil.rmtree(workspace)
    return manifest


# 【非核心】将命令行文件和样本日参数交给单日转换流程，并输出简要完成统计。
def main() -> None:
    """执行离线数据准备 CLI。

    输入来自 --book、--trades、--date、--output-dir 和可选步长、分片容量。
    成功时输出 manifest 及标准化文件，并在终端打印事件数和转换耗时。
    参数或源数据错误向调用进程传播，由批处理命令据退出状态判断是否继续回测。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tick-size", type=float, default=0.1)
    parser.add_argument("--lot-size", type=float, default=0.001)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    arguments = parser.parse_args()
    manifest = convert(arguments.book, arguments.trades, arguments.date, arguments.output_dir,
                       arguments.tick_size, arguments.lot_size, arguments.shard_rows)
    print(json.dumps({key: manifest[key] for key in ["logical_events", "normalized_rows", "conversion_seconds"]}))


if __name__ == "__main__":
    main()
