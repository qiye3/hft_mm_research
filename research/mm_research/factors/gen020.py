"""Gen020同UTC半小时时段的20日事前基准，参考只取前20个完整日期。"""
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import file_locks, scan_series, write_parquet, read_metadata, artifact_ready
from mm_research.execution import ordered_process_map
from mm_research.factors.operators import rolling_factors
from mm_research.factors.sources import depth_quantity

VERSION = "seconds_v1"
SLOT_NS = 1800_000_000_000
SOURCE_BOUNDARY = "2025-10-01"


# 【核心逻辑79】逐时段汇总完整历史样本，任一历史缺失仅使对应变量基准失效。
def slot_statistics(past: np.ndarray, date: str, names: list[str]) -> list[dict]:
    """输入历史日×时段×100ms位置×变量，输出该日事前参考的小型记录表。"""
    valid = np.isfinite(past).all(axis=(0, 2))
    records = []
    for slot in range(past.shape[1]):
        record = {"date": date, "_slot": slot}
        for index, name in enumerate(names):
            values = past[:, slot, :, index].reshape(-1)
            acceptable = bool(valid[slot, index])
            record[f"_slot_{name}_mean"] = float(values.mean()) if acceptable else None
            record[f"_slot_{name}_std"] = float(values.std(ddof=1)) if acceptable else None
            record[f"_slot_{name}_q90"] = float(np.quantile(values, 0.9, method="linear")) if acceptable else None
        records.append(record)
    return records


# 【核心逻辑78】定义同时间段比较的价差、成交笔数与总深度。
def slot_sources(tick_size: float) -> dict[str, pl.Expr]:
    """三个基础量保持100ms粒度，盘口与成交缺失分别保留。"""
    return {
        "spread_level_ticks": ((pl.col("ask_price_1") - pl.col("bid_price_1")) / tick_size).round(0),
        "trade_count_100ms": pl.col("trade_count").cast(pl.Float64),
        "book_depth_total_l10": depth_quantity("bid", 10) + depth_quantity("ask", 10),
    }


# 【非核心】独立进程读取一天槽基准输入，避免主进程累积Polars分配器内存。
def read_slot_day(task: tuple) -> np.ndarray:
    """返回一天48个半小时槽的三列100ms样本，进程退出后释放读取缓存。"""
    features, columns, date, tick_size = task
    source, _ = scan_series(features, columns, [date])
    formulas = slot_sources(tick_size)
    values = source.select(*[expression.alias(name) for name, expression in formulas.items()]).collect(engine="streaming").to_numpy()
    return values.reshape(48, 18000, len(formulas))


# 【核心逻辑77】使用前20日同槽原始值拟合均值、样本标准差与精确线性Q90。
def build_slot_reference(features: Path, output: Path, config: dict, dataset: dict) -> Path:
    """每个历史日同槽18,000位置均有效才合格，来源切换重新累计20日。"""
    filename = output.parent / "metadata/gen020_slot_reference_20d_v1.parquet"
    formulas = slot_sources(config["tick_size"])
    columns = sorted({column for expression in formulas.values() for column in expression.meta.root_names()})
    metadata = {"kind": "slot_reference", "dataset": dataset, "definition": {
        "version": VERSION, "days": 20, "slot_seconds": 1800, "source_boundary": SOURCE_BOUNDARY,
        "inputs": {name: read_metadata(features / f"{name}.parquet")["definition"] for name in columns},
    }}
    with file_locks([features / f"{name}.parquet" for name in columns], shared=True), file_locks([filename]):
        if artifact_ready(filename, metadata, False):
            return filename
        history = deque(maxlen=20)
        records = []
        tasks = [(features, columns, date, config["tick_size"]) for date in dataset["dates"]]
        daily_values = ordered_process_map(read_slot_day, tasks, 1, recycle_workers=True)
        for date, values in zip(dataset["dates"], daily_values):
            if date == SOURCE_BOUNDARY:
                history.clear()
            # 当前日先生成参考，再进入历史队列，避免当前日期参与其自身基准。
            if len(history) == 20:
                records.extend(slot_statistics(np.stack(history), date, list(formulas)))
            history.append(values)
            print(f"UTC时段基准完成 {date}", flush=True)
        schema = {"date": pl.String, "_slot": pl.Int64}
        schema.update({f"_slot_{name}_{statistic}": pl.Float64 for name in formulas for statistic in ["mean", "std", "q90"]})
        write_parquet(pl.DataFrame(records, schema=schema).lazy(), filename, metadata)
    return filename


# 【核心逻辑81】按当前日期和UTC时段关联事前基准，保持100ms输入顺序。
def attach_slot_reference(source: pl.LazyFrame, filename: Path) -> pl.LazyFrame:
    """尚未取得20日完整历史时参考列为空，当前数据不参与参考拟合。"""
    source = source.with_columns(((pl.col("timestamp_ns") // SLOT_NS) % 48).alias("_slot"))
    return source.join(pl.scan_parquet(filename), on=["date", "_slot"], how="left", maintain_order="left")


# 【核心逻辑80】将当前量与历史槽均值、标准差和Q90比较后滚动。
def factor_expressions(config: dict) -> dict[str, pl.Expr]:
    """比例要求均值为正，zscore要求标准差为正，输出九种异常状态的四档均值。"""
    sources = {}
    for name, values in slot_sources(config["tick_size"]).items():
        mean = pl.col(f"_slot_{name}_mean")
        standard = pl.col(f"_slot_{name}_std")
        quantile = pl.col(f"_slot_{name}_q90")
        sources[f"utcslot_{name}_ratio_20d"] = values / pl.when(mean > 0).then(mean)
        sources[f"utcslot_{name}_zscore_20d"] = (values - mean) / pl.when(standard > 0).then(standard)
        sources[f"utcslot_{name}_q90flag_20d"] = (values > quantile).cast(pl.Float64)
    return rolling_factors(sources, config["aggregation_windows_s"], "mean", config["min_fraction"])
