"""将 Tardis 单日全深度 L2/成交转换为框架支持的 ZIP32 NPZ 分片。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

BUY = 1 << 29
SELL = 1 << 28
LOCAL = 1 << 30
EXCHANGE = 1 << 31
DEPTH = 1
TRADE = 2
CLEAR = 3
SNAPSHOT = 4
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


def validate_grid(price: float, quantity: float, tick_size: float, lot_size: float) -> None:
    if not math.isfinite(price) or price <= 0 or not math.isfinite(quantity) or quantity < 0:
        raise ValueError(f"无效价格/数量：{price}, {quantity}")
    if abs(price / tick_size - round(price / tick_size)) > 1e-6:
        raise ValueError(f"价格不符合 tick_size：{price}")
    if abs(quantity / lot_size - round(quantity / lot_size)) > 1e-6:
        raise ValueError(f"数量不符合 lot_size：{quantity}")


def append_source(filename: Path, kind: str, output, tick_size: float, lot_size: float,
                  chunk_rows: int) -> dict:
    """原始数据只做格式校验及最终盘口核对，不在研究程序中另建订单簿。"""
    buffer = np.zeros(chunk_rows, dtype=EVENT_DTYPE)
    used = 0
    counts = Counter()
    first_local = None
    last_local = None
    minimum_latency = None
    previous_snapshot = False
    bids = {}
    asks = {}
    with gzip.open(filename, "rt", newline="") as source:
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
            if side not in {"buy", "sell", "bid", "ask"}:
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
                    buffer[used] = (CLEAR, exchange_ts, local_ts, 0, 0, 0, 0, 0)
                    used += 1
                    counts["clear_events"] += 1
                    bids.clear()
                    asks.clear()
                    if used == chunk_rows:
                        buffer.tofile(output)
                        used = 0
                event_kind = SNAPSHOT if is_snapshot else DEPTH
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


def validate_day(sources: list[dict], date: str) -> None:
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    end = start + 86_400_000_000_000
    for source in sources:
        first = source["first_local_timestamp_ns"]
        last = source["last_local_timestamp_ns"]
        if not start <= first < start + 60_000_000_000 or not end - 60_000_000_000 <= last < end:
            raise ValueError(f"数据未覆盖给定 UTC 日的首尾一分钟：{source['file']}")


def write_routed_shards(events: np.ndarray, destination: Path, shard_rows: int) -> list[dict]:
    """分别路由并按可见时间归并，保证两侧各自的时间顺序，保留同侧稳定顺序。"""
    event_count = len(events)
    # 不要求 exch_ts 与 local_ts 的排序一致；同一源事件各产生一条路由记录。
    visible_timestamps = np.empty(event_count * 2, dtype=np.int64)
    visible_timestamps[:event_count] = events["exch_ts"]
    visible_timestamps[event_count:] = events["local_ts"]
    order = np.argsort(visible_timestamps, kind="stable")
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


def convert(book: Path, trades: Path, date: str, destination: Path, tick_size: float = 0.1,
            lot_size: float = 0.001, shard_rows: int = 250_000, full_day: bool = True) -> dict:
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
    events = np.memmap(raw_file, dtype=EVENT_DTYPE, mode="r+")
    minimum_latency = min(book_info["minimum_feed_latency_ns"], trades_info["minimum_feed_latency_ns"])
    latency_offset = max(0, 1 - minimum_latency)
    if latency_offset:
        events["local_ts"] += latency_offset
        events.flush()
    shards = write_routed_shards(events, destination, shard_rows)
    counts = book_info["counts"]
    manifest = {
        "schema": "hbt.Event/64-byte-little-endian",
        "symbol": "BTCUSDT",
        "exchange": "binance-futures",
        "date": date,
        "full_day_validated": full_day,
        "timestamp_unit": "ns",
        "tick_size": tick_size,
        "lot_size": lot_size,
        "local_timestamp_offset_ns": latency_offset,
        "logical_events": len(events),
        "normalized_rows": len(events) * 2,
        "depth_updates": counts.get(str(DEPTH), 0),
        "snapshot_levels": counts.get(str(SNAPSHOT), 0),
        "clear_events": counts.get("clear_events", 0),
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


def main() -> None:
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
