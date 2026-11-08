"""单变量全日期Features生产与X/y读取。

基础采样逐源日期读取，滚动公式处理完整时间序列，每个变量独立加锁与发布。
"""

import datetime as dt
import json
from pathlib import Path

import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.sources import DAY_NS
from mm_research.data.standard import read_standard_metadata
from mm_research.data.storage import (
    KEY_COLUMNS, artifact_ready, file_locks,
    read_metadata, scan_series, write_parquet, write_series_frames,
)
from mm_research.features.base import BASE_COLUMNS, build_base_features
from mm_research.execution import ordered_process_map
from mm_research.features.transform import GROUPS, group_expressions, requested_groups
from mm_research.targets.midprice import horizon_name


# 【核心逻辑9】组织完整日期区间的单特征文件生产。
def run_dataset_production(config_path: Path, rebuild: bool = False, workers: int = 1) -> None:
    """输入Features配置，按范围生产全日期单变量特征。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "features")
    dates = [str(date) for date in config["dates"]]
    calendar = [dt.date.fromisoformat(date) for date in dates]
    if not dates or any(right - left != dt.timedelta(days=1) for left, right in zip(calendar, calendar[1:])):
        raise ValueError("计算日期必须递增连续，缺失日期须保留为空时间网格")
    if config["interval_ns"] != 100_000_000 or not 0 <= config["market_data_latency_ns"] < DAY_NS:
        raise ValueError("研究使用100ms网格及小于一天的非负固定延迟")
    for suffix, size in [("1s", 10), ("5s", 50)]:
        if not 1 <= config["rolling_min_samples"][suffix] <= size:
            raise ValueError(f"{suffix}的min_samples超出窗口范围")
    groups = requested_groups(config)
    build_features(root, config, groups, rebuild, workers)
    print(f"Features生产完成 {dates[0]}—{dates[-1]}", flush=True)


# 【非核心】收集标准输入状态并固定网格定义，供不同单变量文件核对来源。
def input_definition(root: Path, config: dict) -> tuple[dict, list[Path]]:
    """返回共同数据定义和标准输入文件，包含可用前日数据以接回延迟事件。"""
    standard = root / config["standard_root"] / config["standard_version"]
    previous = str(dt.date.fromisoformat(str(config["dates"][0])) - dt.timedelta(days=1))
    sources = []
    filenames = []
    for date in [previous, *[str(date) for date in config["dates"]]]:
        for kind in ["snapshots", "trades"]:
            filename = standard / kind / f"{date}.parquet"
            if date == previous and not filename.exists():
                continue
            definition = read_standard_metadata(filename)["definition"]
            if (definition["schema_version"] != 2 or definition["version"] != config["standard_version"]
                    or definition["tick_size"] != config["tick_size"] or definition["lot_size"] != config["lot_size"]):
                raise ValueError(f"标准输入与研究配置不一致 {filename}")
            status = filename.stat()
            sources.append({"date": date, "kind": kind, "bytes": status.st_size, "mtime_ns": status.st_mtime_ns})
            filenames.append(filename)
    dataset = {
        "symbol": "BTCUSDT", "dates": [str(date) for date in config["dates"]],
        "base_version": GROUPS["base"]["version"],
        "standard_version": config["standard_version"], "sources": sources,
        "interval_ns": config["interval_ns"], "market_data_latency_ns": config["market_data_latency_ns"],
        "max_book_age_ns": config["max_book_age_ns"], "tick_size": config["tick_size"],
        "lot_size": config["lot_size"], "missing_values": "null_observations_v1",
    }
    return dataset, filenames


# 【非核心】生成单个特征的身份，分别记录公式版本与共同数据定义。
def feature_definition(column: str, group: str, dataset: dict, config: dict) -> dict:
    """输入字段、组和配置，返回用于产物复用的完整元数据。"""
    definition = {"group": group, "version": GROUPS[group]["version"]}
    if group != "base":
        definition["base_version"] = GROUPS["base"]["version"]
    if group == "trades":
        definition["min_samples"] = config["rolling_min_samples"]
    return {"kind": "feature", "name": column, "dataset": dataset, "definition": definition}


# 【核心逻辑10】同组一次读取共享依赖，并行表达式计算后按单变量文件发布。
def build_features(root: Path, config: dict, groups: list[str], rebuild: bool = False, workers: int = 1) -> None:
    """输入配置与组列表，按文件更新所选组，已有未选择的特征继续保留。"""
    folder = root / config["feature_root"] / config["feature_version"]
    dataset, sources = input_definition(root, config)
    with file_locks(sources, shared=True):
        build_base_files(root, config, dataset, folder, rebuild and "base" in config["feature_groups"], workers)
    for group in [name for name in groups if name != "base"]:
        expressions = group_expressions(group, config)
        metadata = {column: feature_definition(column, group, dataset, config) for column in expressions}
        pending = [column for column in expressions
                   if not artifact_ready(folder / f"{column}.parquet", metadata[column], rebuild)]
        if not pending:
            continue
        dependencies = sorted({
            column for name in pending for column in expressions[name].meta.root_names()
            if column not in KEY_COLUMNS
        })
        inputs = [folder / f"{name}.parquet" for name in dependencies]
        outputs = [folder / f"{name}.parquet" for name in pending]
        with file_locks(inputs, shared=True), file_locks(outputs):
            table, _ = scan_series(folder, dependencies)
            values = table.select(*KEY_COLUMNS, *[expressions[name].alias(name) for name in pending])
            frames = values.collect_batches(chunk_size=100_000, maintain_order=True, engine="streaming")
            write_series_frames(frames, folder, pending, metadata)
        print(f"Feature组完成 {group}", flush=True)


# 【核心逻辑11】有界并行逐日采样，按日期顺序追加并发布基础单变量文件。
def build_base_files(root: Path, config: dict, dataset: dict, folder: Path, rebuild: bool, workers: int = 1) -> None:
    """只保留workers份采样日，计算与写出重叠，输出布局及数值口径保持一致。"""
    metadata = {column: feature_definition(column, "base", dataset, config) for column in BASE_COLUMNS}
    outputs = [folder / f"{column}.parquet" for column in BASE_COLUMNS]
    with file_locks(outputs):
        pending = [column for column in BASE_COLUMNS
                   if not artifact_ready(folder / f"{column}.parquet", metadata[column], rebuild)]
        if not pending:
            return
        standard = root / config["standard_root"] / config["standard_version"]
        tasks = [(standard, date, config) for date in dataset["dates"]]
        frames = ordered_process_map(sample_base_day, tasks, workers)
        write_series_frames(frames, folder, pending, metadata)


# 【核心逻辑12】将一个日期采样任务交给已验证的基础特征函数。
def sample_base_day(task: tuple) -> pl.DataFrame:
    """返回一个完整采样日，有界调度保证进程间传输不会堆积全部日期。"""
    standard, date, config = task
    table = build_base_features(
        standard, date, config["interval_ns"], config["lot_size"],
        config["max_book_age_ns"], config["market_data_latency_ns"],
    )
    print(f"基础采样完成 {date}", flush=True)
    return table


# 【核心逻辑39】按共同时间键关联所选单列特征与同口径标签。
def load_samples(feature_root: Path, label_root: Path, dates: list[str], columns: list[str],
                 unit: str = "bps") -> tuple[pl.LazyFrame, str]:
    """返回有效样本查询及标签列名，调用方须持有输入锁直至物化。"""
    table, dataset = scan_series(feature_root, columns, dates)
    files = [label_root / f"{date}.parquet" for date in dates]
    label_metadata = [read_metadata(filename) for filename in files]
    if any(item["dataset"] != dataset for item in label_metadata):
        raise ValueError("特征与标签输入不一致")
    horizons = {item["horizon_ns"] for item in label_metadata}
    if len(horizons) != 1 or unit not in ["bps", "ticks", "usdt"]:
        raise ValueError("标签周期或单位不一致")
    target = f"y_mid_{horizon_name(horizons.pop())}_{unit}"
    labels = pl.scan_parquet(files).select("timestamp_ns", "label_valid", target)
    table = table.filter(pl.col("timestamp_ns") % 1_000_000_000 == 0).set_sorted("timestamp_ns")
    labels = labels.set_sorted("timestamp_ns")
    samples = table.join(labels, on="timestamp_ns", how="inner", maintain_order="left")
    valid = pl.all_horizontal([pl.col(name).is_not_null() & pl.col(name).is_finite() for name in [*columns, target]])
    return samples.filter(valid & pl.col("label_valid")).select(*KEY_COLUMNS, *columns, target).sort("timestamp_ns"), target


# 【核心逻辑38】在共享锁内物化X、y，保留调用方选择的特征顺序。
def load_xy(feature_root: Path, label_root: Path, dates: list[str], columns: list[str], unit: str = "bps"):
    """输入文件目录和样本选择，返回X、y与时间索引。"""
    inputs = [feature_root / f"{name}.parquet" for name in columns]
    inputs.extend(label_root / f"{date}.parquet" for date in dates)
    with file_locks(inputs, shared=True):
        scan, target = load_samples(feature_root, label_root, dates, columns, unit)
        table = scan.collect(engine="streaming")
    return table.select(columns).to_numpy(), table[target].to_numpy(), table.select(KEY_COLUMNS)


# 【非核心】汇总模型列和标签的逐日可用量，保存研究检查报告。
def summarize_dataset(feature_root: Path, label_root: Path, output: Path) -> dict:
    """输入单特征目录和标签目录，返回逐日样本规模并写入JSON。"""
    columns = [column for name in ["book", "trades", "returns"] for column in GROUPS[name]["model_columns"]]
    dates = read_metadata(feature_root / f"{columns[0]}.parquet")["dataset"]["dates"]
    days = []
    for date in dates:
        inputs = [feature_root / f"{name}.parquet" for name in columns] + [label_root / f"{date}.parquet"]
        with file_locks(inputs, shared=True):
            scan, target = load_samples(feature_root, label_root, [date], columns)
            statistics = scan.select(pl.len().alias("usable_rows"), pl.col(target).mean().alias("target_mean_bps")).collect().row(0, named=True)
        days.append({"date": date, "rows": DAY_NS // 100_000_000, **statistics})
    report = {"feature_columns": columns, "days": days, "rows": sum(day["rows"] for day in days),
              "usable_rows": sum(day["usable_rows"] for day in days)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
