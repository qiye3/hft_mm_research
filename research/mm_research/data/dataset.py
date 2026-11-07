"""单变量全日期Features生产、独立标签及X/y读取。

基础采样逐源日期读取，滚动公式处理完整时间序列，每个变量独立加锁与发布。
"""

from contextlib import ExitStack
import datetime as dt
import json
from pathlib import Path
import tempfile

import polars as pl
import pyarrow.parquet as pq

from mm_research.config import load_pipeline_config
from mm_research.data.sources import DAY_NS
from mm_research.data.standard import read_standard_metadata
from mm_research.data.storage import (
    KEY_COLUMNS, METADATA_KEY, SCHEMA_VERSION, artifact_ready, file_locks,
    read_metadata, scan_series, write_parquet,
)
from mm_research.features.base import BASE_COLUMNS, build_base_features
from mm_research.features.transform import GROUPS, group_expressions, requested_groups
from mm_research.targets.midprice import build_midprice_labels, horizon_name


# 【核心逻辑9】组织完整日期区间的单特征文件和独立标签生产。
def run_dataset_production(config_path: Path, rebuild: bool = False, only: str = "all") -> None:
    """输入Features/Targets配置，按范围生产全日期特征或逐日标签。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "features_targets")
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
    if only in ["all", "features"]:
        build_features(root, config, groups, rebuild)
    if only in ["all", "labels"]:
        for horizon in config["horizons_ns"]:
            for date in dates:
                build_labels_day(root, config, date, horizon, rebuild)
    print(f"研究生产完成 {dates[0]}—{dates[-1]} {only}", flush=True)


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


# 【核心逻辑10】复用基础单列，按各公式依赖读取全日期序列并逐变量发布。
def build_features(root: Path, config: dict, groups: list[str], rebuild: bool = False) -> None:
    """输入配置与组列表，按文件更新所选组，已有未选择的特征继续保留。"""
    folder = root / config["feature_root"] / config["feature_version"]
    dataset, sources = input_definition(root, config)
    with file_locks(sources, shared=True):
        build_base_files(root, config, dataset, folder, rebuild and "base" in config["feature_groups"])
    for group in [name for name in groups if name != "base"]:
        for column, expression in group_expressions(group, config).items():
            filename = folder / f"{column}.parquet"
            metadata = feature_definition(column, group, dataset, config)
            dependencies = sorted(set(expression.meta.root_names()) - set(KEY_COLUMNS))
            inputs = [folder / f"{name}.parquet" for name in dependencies]
            with file_locks(inputs, shared=True), file_locks([filename]):
                if artifact_ready(filename, metadata, rebuild):
                    continue
                table, _ = scan_series(folder, dependencies)
                values = table.select(*KEY_COLUMNS, expression.alias(column))
                write_parquet(values, filename, metadata)
            print(f"Feature完成 {column}", flush=True)


# 【核心逻辑11】逐日采样并追加单列临时行组，完整日期区间写完后发布。
def build_base_files(root: Path, config: dict, dataset: dict, folder: Path, rebuild: bool) -> None:
    """流式生成基础列，PyArrow负责追加Parquet行组，Polars负责全部计算。

    内存只保留一个采样日，临时文件写在目标盘，正式目录每个基础变量只有一个文件。
    """
    metadata = {column: feature_definition(column, "base", dataset, config) for column in BASE_COLUMNS}
    outputs = [folder / f"{column}.parquet" for column in BASE_COLUMNS]
    with file_locks(outputs):
        pending = [column for column in BASE_COLUMNS
                   if not artifact_ready(folder / f"{column}.parquet", metadata[column], rebuild)]
        if not pending:
            return
        standard = root / config["standard_root"] / config["standard_version"]
        with tempfile.TemporaryDirectory(prefix=".base-", dir=folder) as temporary:
            staging = Path(temporary)
            writers = {}
            with ExitStack() as streams:
                for date in dataset["dates"]:
                    base = build_base_features(
                        standard, date, config["interval_ns"], config["lot_size"],
                        config["max_book_age_ns"], config["market_data_latency_ns"],
                    )
                    for column in pending:
                        table = base.select(*KEY_COLUMNS, column).to_arrow()
                        if column not in writers:
                            definition = {**metadata[column], "schema_version": SCHEMA_VERSION}
                            schema = table.schema.with_metadata({METADATA_KEY: json.dumps(definition, sort_keys=True).encode()})
                            writer = pq.ParquetWriter(
                                staging / f"{column}.parquet", schema, compression="zstd", compression_level=3,
                                use_dictionary=["symbol", "date"], column_encoding={"timestamp_ns": "DELTA_BINARY_PACKED"},
                            )
                            writers[column] = streams.enter_context(writer)
                        writers[column].write_table(table, row_group_size=100_000)
                    print(f"基础采样完成 {date}", flush=True)
            for column in pending:
                (staging / f"{column}.parquet").replace(folder / f"{column}.parquet")


# 【核心逻辑23】从单列Mid与盘口质量生成指定日期和周期的独立标签。
def build_labels_day(root: Path, config: dict, date: str, horizon_ns: int, rebuild: bool = False) -> Path:
    """返回标签文件，其定义独立于衍生特征和因子。"""
    folder = root / config["feature_root"] / config["feature_version"]
    columns = ["mid_price", "book_valid", "book_segment_id"]
    inputs = [folder / f"{name}.parquet" for name in columns]
    filename = root / config["label_root"] / config["label_version"] / f"horizon={horizon_name(horizon_ns)}" / f"{date}.parquet"
    with file_locks(inputs, shared=True), file_locks([filename]):
        base, dataset = scan_series(folder, columns, [date])
        metadata = {"kind": "labels", "dataset": dataset, "date": date, "horizon_ns": horizon_ns,
                    "definition": "same_utc_day_continuous_book_v1"}
        if artifact_ready(filename, metadata, rebuild):
            return filename
        labels = build_midprice_labels(base.collect(engine="streaming"), config["tick_size"], config["interval_ns"], horizon_ns)
        write_parquet(labels.lazy(), filename, metadata)
    return filename


# 【核心逻辑26】按共同时间键关联所选单列特征与同口径标签。
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
    samples = table.join(labels, on="timestamp_ns", how="inner", validate="1:1")
    valid = pl.all_horizontal([pl.col(name).is_not_null() & pl.col(name).is_finite() for name in [*columns, target]])
    return samples.filter(valid & pl.col("label_valid")).select(*KEY_COLUMNS, *columns, target).sort("timestamp_ns"), target


# 【核心逻辑25】在共享锁内物化X、y，保留调用方选择的特征顺序。
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
