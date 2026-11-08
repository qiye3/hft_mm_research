"""独立Targets生产入口，逐日共享跨午夜上下文与五周期质量计算。"""

import datetime as dt
from pathlib import Path

import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.sources import DAY_NS
from mm_research.data.storage import artifact_ready, file_locks, read_metadata, scan_series, write_parquet
from mm_research.execution import ordered_process_map
from mm_research.targets.midprice import build_midprice_labels, horizon_name, prepare_label_context

LABEL_COLUMNS = ["mid_price", "book_valid", "book_segment_id"]
REASON_COLUMNS = ["invalid_future", "invalid_book", "invalid_gap", "invalid_source"]


# 【核心逻辑35】用标准快照的日界时间和已有缺口标签补齐跨日连续性。
def boundary_markers(standard: Path, boundary_date: str, dataset: dict, source_change: bool) -> tuple[list[int], list[int]]:
    """标记跨文件更新静默及源质量缺口，恢复事件归入其后第一个100ms网格。"""
    calendar = dt.date.fromisoformat(boundary_date)
    previous_date = str(calendar - dt.timedelta(days=1))
    available = {item["date"] for item in dataset["sources"] if item["kind"] == "snapshots"}
    if previous_date not in available or boundary_date not in available:
        return [], []
    previous = standard / "snapshots" / f"{previous_date}.parquet"
    current = standard / "snapshots" / f"{boundary_date}.parquet"
    latency = dataset["market_data_latency_ns"]
    interval = dataset["interval_ns"]
    boundary = int(dt.datetime.fromisoformat(boundary_date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    with file_locks([previous, current], shared=True):
        last = pl.scan_parquet(previous).select(pl.col("exchange_timestamp_ns").max()).collect().item()
        first = pl.scan_parquet(current).filter(pl.col("exchange_timestamp_ns") > last).select(
            pl.col("exchange_timestamp_ns").min(),
        ).collect().item()
        markers = []
        if first - last > dataset["max_book_age_ns"]:
            markers.append(((first + latency + interval - 1) // interval) * interval)
        queries = []
        for filename in [previous, current]:
            queries.append(pl.scan_parquet(filename).select("exchange_timestamp_ns", "quality_flags").filter(
                pl.col("exchange_timestamp_ns").is_between(boundary - 2_000_000_000, boundary + 2_000_000_000),
            ))
        flags = pl.concat(queries).filter(
            pl.col("quality_flags").list.contains("gap_before_confirmed")
            | pl.col("quality_flags").list.contains("gap_before_suspected"),
        ).select(((pl.col("exchange_timestamp_ns") + latency + interval - 1) // interval * interval).alias("marker")).collect()
        markers.extend(flags["marker"].to_list())
    switches = []
    if source_change:
        switches.append(((first + latency + interval - 1) // interval) * interval)
    return markers, switches


# 【核心逻辑34】按日期读取当日及次日最多300秒，建立标签专用连续上下文。
def load_label_context(root: Path, config: dict, date: str) -> tuple[pl.DataFrame, dict]:
    """读取现有Features，跨日区段重置在内存中校正，上游Parquet保持其原始定义。"""
    folder = root / config["feature_root"] / config["feature_version"]
    dataset = read_metadata(folder / "mid_price.parquet")["dataset"]
    next_date = str(dt.date.fromisoformat(date) + dt.timedelta(days=1))
    dates = [date]
    if next_date in dataset["dates"]:
        dates.append(next_date)
    start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    with file_locks([folder / f"{column}.parquet" for column in LABEL_COLUMNS], shared=True):
        scan, _ = scan_series(folder, LABEL_COLUMNS, dates)
        base = scan.filter(pl.col("timestamp_ns") < start + DAY_NS + max(config["horizons_ns"])).collect(engine="streaming")
    standard = root / config["standard_root"] / config["standard_version"]
    gaps = []
    source_markers = [int(dt.datetime.fromisoformat(value).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
                      for value in config["source_boundaries"]]
    for boundary_date in dates:
        boundary_gaps, switches = boundary_markers(standard, boundary_date, dataset, boundary_date in config["source_boundaries"])
        gaps.extend(boundary_gaps)
        source_markers.extend(switches)
    return prepare_label_context(base, dataset["interval_ns"], gaps, source_markers), dataset


# 【核心逻辑33】共享同日上下文生成五周期整秒标签，逐文件加锁并原子发布。
def build_labels_day(root: Path, config: dict, date: str, rebuild: bool = False) -> list[dict]:
    """返回各周期行数、有效量和原因计数，原因允许重叠，标签文件只保存研究所需列。"""
    folder = root / config["label_root"] / config["label_version"]
    context, dataset = load_label_context(root, config, date)
    summaries = []
    for horizon in config["horizons_ns"]:
        filename = folder / f"horizon={horizon_name(horizon)}" / f"{date}.parquet"
        metadata = {
            "kind": "labels", "dataset": dataset, "date": date, "horizon_ns": horizon,
            "definition": {"version": "cross_day_1s_v1", "output_interval_ns": config["output_interval_ns"],
                           "source_interval_ns": dataset["interval_ns"], "source_boundaries": config["source_boundaries"]},
        }
        labels = build_midprice_labels(context, dataset["tick_size"], dataset["interval_ns"], horizon).filter(
            (pl.col("date") == date) & (pl.col("timestamp_ns") % config["output_interval_ns"] == 0),
        )
        summary = labels.select(
            pl.len().alias("rows"), pl.col("label_valid").sum().alias("valid_rows"),
            *[pl.col(name).sum() for name in REASON_COLUMNS],
        ).row(0, named=True)
        summaries.append({"date": date, "horizon_ns": horizon, **summary})
        with file_locks([filename]):
            if not artifact_ready(filename, metadata, rebuild):
                write_parquet(labels.drop(REASON_COLUMNS).lazy(), filename, metadata)
    print(f"Targets完成 {date}", flush=True)
    return summaries


# 【非核心】把一个日期任务交给标签生产函数，供有界进程池运行。
def produce_labels_day(task: tuple) -> list[dict]:
    """返回一个日期的五周期统计，线程和进程预算由生产入口控制。"""
    root, config, date, rebuild = task
    return build_labels_day(root, config, date, rebuild)


# 【核心逻辑32】以独立Targets配置组织全日期生产，保存小型质量汇总。
def run_target_production(config_path: Path, rebuild: bool = False) -> None:
    """配置只声明输入Features、标准来源、输出周期和标签版本。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "targets")
    if config["output_interval_ns"] != 1_000_000_000:
        raise ValueError("Targets使用UTC整秒输出")
    tasks = [(root, config, date, rebuild) for date in config["dates"]]
    summaries = []
    for daily in ordered_process_map(produce_labels_day, tasks, config["workers"]):
        summaries.extend(daily)
    folder = root / config["label_root"] / config["label_version"]
    pl.DataFrame(summaries).write_parquet(folder / "quality_summary.parquet", compression="zstd")
