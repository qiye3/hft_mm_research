"""用训练期逐笔数量的精确频数分布冻结大小单阈值，再生成六列基础成交桶。"""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.sources import DAY_NS, scan_standard_history
from mm_research.data.storage import artifact_ready, file_locks, read_metadata, scan_series, write_series_frames, write_json_atomic
from mm_research.execution import ordered_process_map

SIZE_COLUMNS = [f"trade_{size}_{side}_qty_100ms" for size in ["small", "mid", "large"] for side in ["buy", "sell"]]
INVALID_FLAGS = ["invalid_price", "invalid_quantity", "invalid_side", "content_conflict", "off_tick", "off_lot", "missing_trade_id"]


# 【核心逻辑27】从训练期数量整数Lot频数计算精确线性插值分位数，冻结Q50和Q90。
def fit_size_thresholds(standard: Path, train_dates: list[str], lot_size: float) -> dict:
    """聚合离散数量频数，无需将数十亿笔成交同时装入内存或全量排序。"""
    files = [standard / "trades" / f"{date}.parquet" for date in train_dates]
    with file_locks(files, shared=True):
        histogram = pl.scan_parquet(files).filter(
            ~pl.col("quality_flags").list.eval(pl.element().is_in(INVALID_FLAGS)).list.any(),
        ).select((pl.col("quantity") / lot_size).round().cast(pl.Int64).alias("lots")).group_by("lots").len().sort("lots").collect(engine="streaming")
    cumulative = np.cumsum(histogram["len"].cast(pl.UInt64).to_numpy(), dtype=np.uint64)
    lots = histogram["lots"].to_numpy()
    thresholds = {}
    for quantile, name in [(0.5, "q50"), (0.9, "q90")]:
        rank = (int(cumulative[-1]) - 1) * quantile
        lower = lots[np.searchsorted(cumulative, int(np.floor(rank)) + 1)]
        upper = lots[np.searchsorted(cumulative, int(np.ceil(rank)) + 1)]
        thresholds[name] = float((lower + (upper - lower) * (rank - np.floor(rank))) * lot_size)
    return {**thresholds, "train_start": train_dates[0], "train_end": train_dates[-1],
            "trades": int(cumulative[-1]), "method": "exact_lot_frequency_linear_quantile"}


# 【核心逻辑28】按固定延迟分桶逐笔大小单，未知成交窗口置空且空桶保留零。
def build_size_day(task: tuple) -> pl.DataFrame:
    """只读取当前及前日成交，Q50以下为小单，Q90及以上为大单。"""
    standard, features, date, dataset, thresholds = task
    trades = scan_standard_history(standard, "trades", date)
    latency = dataset["market_data_latency_ns"]
    interval = dataset["interval_ns"]
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    bucket = ((pl.col("exchange_timestamp_ns") + latency + interval - 1) // interval) * interval
    trades = trades.with_columns(bucket.alias("timestamp_ns")).filter(
        pl.col("timestamp_ns").is_between(start, start + DAY_NS, closed="left")
        & ~pl.col("quality_flags").list.eval(pl.element().is_in(INVALID_FLAGS)).list.any(),
    )
    masks = {
        "small": pl.col("quantity") < thresholds["q50"],
        "mid": pl.col("quantity").is_between(thresholds["q50"], thresholds["q90"], closed="left"),
        "large": pl.col("quantity") >= thresholds["q90"],
    }
    bins = trades.group_by("timestamp_ns").agg([
        ((pl.col("quantity") / dataset["lot_size"]).round().cast(pl.Int64)
         .filter(mask & (pl.col("side") == side)).sum() * dataset["lot_size"]).alias(f"trade_{size}_{side}_qty_100ms")
        for size, mask in masks.items() for side in ["buy", "sell"]
    ])
    quality, _ = scan_series(features, ["flow_valid"], [date])
    table = quality.join(bins, on="timestamp_ns", how="left", maintain_order="left").with_columns([
        pl.when(pl.col("flow_valid")).then(pl.col(column).fill_null(0.0)).otherwise(None).alias(column)
        for column in SIZE_COLUMNS
    ])
    return table.collect(engine="streaming")


# 【核心逻辑26】把冻结阈值与六列全日期成交量发布为新增Features组。
def build_size_features(root: Path, feature_config: dict, train_dates: list[str], workers: int) -> dict:
    """输入标准/Features配置和训练日期，保存阈值来源并复用同定义已有列。"""
    features = root / feature_config["feature_root"] / feature_config["feature_version"]
    standard = root / feature_config["standard_root"] / feature_config["standard_version"]
    dataset = read_metadata(features / "flow_valid.parquet")["dataset"]
    definition_file = features / "trade_size_thresholds.json"
    with file_locks([definition_file]):
        if definition_file.exists():
            thresholds = json.loads(definition_file.read_text())
            if thresholds["train_start"] != train_dates[0] or thresholds["train_end"] != train_dates[-1]:
                raise ValueError("大小单训练区间变化，需要新的Feature版本")
        else:
            thresholds = fit_size_thresholds(standard, train_dates, dataset["lot_size"])
            write_json_atomic(definition_file, thresholds)
    metadata = {
        column: {"kind": "feature", "name": column, "dataset": dataset,
                 "definition": {"group": "trade_size", "version": "v2", "thresholds": thresholds}}
        for column in SIZE_COLUMNS
    }
    sources = [standard / "trades" / f"{date}.parquet" for date in dataset["dates"]]
    sources.append(features / "flow_valid.parquet")
    with file_locks(sources, shared=True), file_locks([features / f"{column}.parquet" for column in SIZE_COLUMNS]):
        pending = [column for column in SIZE_COLUMNS if not artifact_ready(features / f"{column}.parquet", metadata[column], False)]
        if pending:
            tasks = [(standard, features, date, dataset, thresholds) for date in dataset["dates"]]
            write_series_frames(ordered_process_map(build_size_day, tasks, workers), features, pending, metadata)
    return thresholds
