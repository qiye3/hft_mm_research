"""Gen041训练期阈值、全历史因子生产及模型增量输入缓存。"""

import datetime as dt
import json
from pathlib import Path
import tempfile

import numpy as np
import polars as pl

from mm_research.data.storage import file_locks, scan_series, write_json_atomic, write_series_frames
from mm_research.execution import ordered_process_map
from mm_research.factors.gen041 import BOOK_COLUMNS, TRADE_COLUMNS, SECOND, attach_book_statistics, factor_expressions
from mm_research.features.trade_size import fit_size_thresholds
from mm_research.models.cross_market import publish_cache
from mm_research.multihorizon.common import context, model_inputs


# 【核心逻辑397】按本轮拟合前缀重算小单阈值，禁止使用未来训练截止日的旧阈值估计。
def prepare_small_threshold(filename: Path) -> None:
    """输出训练期Q50及复用决定；只有数值完全一致才可读取现有小单量。"""
    _, config, output = context(filename)
    marker = output / "small_threshold.json"
    if marker.exists():
        return
    source = Path(config["source"])
    manifest = json.loads((source / "e0/manifest.json").read_text())
    tune_rows = np.load(source / "e1/folds/final/tune_rows.npy", mmap_mode="r")
    # 用内部校准开始前的完整日拟合，连校准日也不参与阈值估计。
    dates = manifest["dates"][:int(tune_rows[0]) // 86400]
    threshold = fit_size_thresholds(Path(config["standard_folder"]), dates, config["lot_size"])
    existing = json.loads((Path(config["feature_folder"]) / "trade_size_thresholds.json").read_text())
    write_json_atomic(marker, {
        "fit": threshold, "existing": existing,
        "reuse_existing_small_columns": threshold["q50"] == existing["q50"],
        "rule": "quantity_strictly_less_than_fit_q50",
    })
    print("GEN041_THRESHOLD_READY", threshold["q50"], threshold["train_end"], flush=True)


# 【核心逻辑398】盘口先取整秒、成交单独汇总十格，按日回读3601秒历史供差分和长窗口使用。
def read_second_history(features: Path, date: str) -> pl.DataFrame:
    """只物化1s盘口矩阵；成交和OBI保留100ms聚合，来源质量由已有Features表达。"""
    calendar = dt.date.fromisoformat(date)
    start = int(dt.datetime.combine(calendar, dt.time(), dt.timezone.utc).timestamp()) * SECOND
    history = [str(calendar - dt.timedelta(days=1)), date]
    book_columns = [*BOOK_COLUMNS, "mid_price", "book_segment_id"]
    flow_columns = [*TRADE_COLUMNS, "imbalance_1"]
    paths = [features / f"{column}.parquet" for column in [*book_columns, *flow_columns]]
    with file_locks(paths, shared=True):
        book, _ = scan_series(features, book_columns, history, start_timestamp_ns=start - 3601 * SECOND)
        book = book.filter(pl.col("timestamp_ns") % SECOND == 0).collect(engine="streaming")
        flow, _ = scan_series(features, flow_columns, history, start_timestamp_ns=start - 3601 * SECOND)
        aggregates = [
            pl.when(pl.col(column).count() == 10).then(pl.col(column).sum()).alias(column)
            for column in TRADE_COLUMNS
        ]
        aggregates.append(pl.when(pl.col("imbalance_1").count() >= 8)
                          .then(pl.col("imbalance_1").mean()).alias("obi"))
        flow = flow.group_by(((pl.col("timestamp_ns") + SECOND - 1) // SECOND * SECOND).alias("timestamp_ns"))
        flow = flow.agg(*aggregates).collect(engine="streaming")
    return attach_book_statistics(book.join(flow, on="timestamp_ns", how="left").sort("timestamp_ns"))


# 【核心逻辑399】每日期间带历史独立生产36列，完成后发布检查点，失败重试不重算已完成日。
def produce_factor_day(task: tuple) -> str:
    """保存临时按日块及零活动统计；合并后正式产物仍为一个因子一个全日期文件。"""
    features, output, date, min_fraction = task
    folder = output / ".chunks"
    folder.mkdir(exist_ok=True)
    marker = folder / f"{date}.json"
    if marker.exists():
        return date
    source = read_second_history(features, date)
    expressions, quality = factor_expressions(min_fraction)
    factors = source.select(
        pl.lit("BTCUSDT").alias("symbol"), "date", "timestamp_ns",
        *[value.alias(name) for name, value in expressions.items()],
    ).filter(pl.col("date") == date)
    if factors.height != 86400:
        raise ValueError(f"Gen041日期未覆盖完整秒网格: {date}, {factors.height}")
    diagnostics = source.select("date", *[value.alias(name) for name, value in quality.items()])
    counts = diagnostics.filter(pl.col("date") == date).select(pl.exclude("date").sum()).to_dicts()[0]
    pending = folder / f"{date}.pending.parquet"
    factors.write_parquet(pending, compression="zstd")
    pending.replace(folder / f"{date}.parquet")
    write_json_atomic(marker, {"date": date, "rows": factors.height, "zero_activity_counts": counts})
    print("GEN041_DAY_READY", date, flush=True)
    return date


# 【核心逻辑400】有界并行生产全历史日期，顺序合并发布每个独立因子，保留可恢复检查点。
def produce_factors(filename: Path, workers: int, date: str | None = None) -> None:
    """单日模式用于真实数据验收；全量模式发布36个Parquet及统一完成清单。"""
    _, config, output = context(filename)
    threshold = json.loads((output / "small_threshold.json").read_text())
    if not threshold["reuse_existing_small_columns"]:
        raise ValueError("新Q50不同，须先按本轮阈值重建小单列，禁止复用旧列")
    manifest = json.loads((Path(config["source"]) / "e0/manifest.json").read_text())
    dates = [date] if date is not None else manifest["dates"]
    if date is None and (output / "factors_ready.json").exists():
        return
    tasks = [(Path(config["feature_folder"]), output, current, config["min_fraction"]) for current in dates]
    for _ in ordered_process_map(produce_factor_day, tasks, workers, recycle_workers=True):
        pass
    if date is not None:
        return
    expressions, _ = factor_expressions(config["min_fraction"])
    factor_folder = Path(config["factor_folder"])
    metadata = {name: {"kind": "factor", "name": name, "definition": config,
                       "small_threshold": threshold["fit"], "output_interval_ns": SECOND}
                for name in expressions}
    frames = (pl.read_parquet(output / ".chunks" / f"{current}.parquet") for current in dates)
    with file_locks([factor_folder / f"{name}.parquet" for name in expressions]):
        write_series_frames(frames, factor_folder, list(expressions), metadata)
    write_json_atomic(output / "factors_ready.json", {"columns": list(expressions), "dates": dates,
                                                     "rows": len(dates) * 86400, "status": "success"})


# 【核心逻辑401】按同一冻结行集读取上一完整秒Gen041值，内部拟合标准化后追加到原慢状态矩阵。
def cache_model_inputs(filename: Path) -> None:
    """输出E1长期、E2背离、E3全部三组缓存；原样本不因新增缺失而删除。"""
    _, config, output = context(filename)
    if (output / "completed.json").exists():
        return
    source = Path(config["source"])
    ready = json.loads((Path(config["factor_run"]) / "factors_ready.json").read_text())
    columns = ready["columns"]
    groups = {
        "e1": [index for index, name in enumerate(columns) if "_diff_" not in name and "_ratio_60s_" not in name],
        "e2": [index for index, name in enumerate(columns) if "_diff_" in name or "_ratio_60s_" in name],
        "e3": list(range(len(columns))),
    }
    row_sets = {split: np.load(source / f"e1/folds/final/{split}_rows.npy") for split in ["fit", "tune"]}
    segments = json.loads((source / "e0/splits.json").read_text())["segments"]
    for split in ["val_dev", "val_select"]:
        row_sets[split] = np.arange(segments[split]["start_row"], segments[split]["stop_row"])
    with tempfile.TemporaryDirectory(prefix="hft_gen041_inputs_", dir="/tmp") as temporary:
        raw = {split: np.lib.format.open_memmap(Path(temporary) / f"{split}.npy", mode="w+", dtype="float32",
                                              shape=(len(rows), len(columns))) for split, rows in row_sets.items()}
        for index, column in enumerate(columns):
            path = Path(config["factor_folder"]) / f"{column}.parquet"
            with file_locks([path], shared=True):
                values = pl.read_parquet(path, columns=[column])[column].to_numpy()
            for split, rows in row_sets.items():
                selected = values[np.maximum(rows - 1, 0)].astype("float32")
                selected[rows == 0] = np.nan
                raw[split][:, index] = selected
            del values, selected
        fit = np.asarray(raw["fit"], dtype="float64")
        mean = np.nanmean(fit, axis=0).astype("float32")
        scale = np.nanstd(fit, axis=0).astype("float32")
        if not np.isfinite(mean).all():
            raise ValueError("Gen041训练期存在全空列，不能进入模型")
        scale[scale < 1e-8] = 1
        del fit
        for group, indices in groups.items():
            destination = output / group / "all/inputs/180s"
            destination.mkdir(parents=True, exist_ok=True)
            for split, rows in row_sets.items():
                previous = np.load(Path(config["previous_macro_source"]) / f"all/inputs/180s/{split}_x.npy", mmap_mode="r")
                staging = Path(temporary) / "combined.npy"
                matrix = np.lib.format.open_memmap(staging, mode="w+", dtype="float32",
                                                  shape=(len(rows), previous.shape[1] + 2 * len(indices)))
                for begin in range(0, len(rows), 65536):
                    stop = min(begin + 65536, len(rows))
                    selected = np.asarray(raw[split][begin:stop, indices], dtype="float32")
                    matrix[begin:stop, :previous.shape[1]] = previous[begin:stop]
                    matrix[begin:stop, previous.shape[1]:] = model_inputs(selected, np.arange(stop - begin), mean[indices], scale[indices])
                matrix.flush()
                del matrix
                publish_cache(staging, destination / f"{split}_x.npy")
                print("GEN041_INPUT_READY", group, split, flush=True)
            np.savez(destination / "preprocessing.npz", mean=mean[indices], scale=scale[indices])
            write_json_atomic(destination / "completed.json", {
                "columns": [columns[index] for index in indices], "previous_macro_source": config["previous_macro_source"],
                "lag_seconds": 1, "order": "old_macro_matrix_then_new_values_then_new_missing_flags", "status": "success",
            })
        del raw
    write_json_atomic(output / "completed.json", {"status": "success", "groups": list(groups)})
