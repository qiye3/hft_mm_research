"""按日编排Features分组生产、独立标签与X/y加载。

base组与衍生组共同保存在每日日期命名的Parquet，组版本和输入身份内嵌文件元数据。
版本目录使用文件锁和单文件原子替换，增量更新失败时保留上一次完整文件。
"""

from contextlib import contextmanager
import datetime as dt
import json
from pathlib import Path
import tempfile

import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.sources import DAY_NS
from mm_research.data.standard import partition_lock, read_standard_metadata
from mm_research.features.base import BASE_COLUMNS, KEY_COLUMNS, build_base_features
from mm_research.features.transform import GROUPS, build_feature_group, group_definition, requested_groups
from mm_research.targets.midprice import build_midprice_labels, horizon_name

METADATA_KEY = "mm_research"
SCHEMA_VERSION = "feature_groups_v1"


# 【核心逻辑9】逐日组织特征组与预测标签的生产，复用特征文件中的基础列。
def run_dataset_production(config_path: Path, rebuild: bool = False, only: str = "all") -> None:
    """输入Features/Targets配置，逐日发布分组特征和独立标签。"""
    root = Path(__file__).resolve().parents[3]

    config = load_pipeline_config(config_path, "features_targets")
    if config["interval_ns"] != 100_000_000 or not 0 <= config["market_data_latency_ns"] < DAY_NS:
        raise ValueError("研究使用100ms网格及小于一天的非负固定延迟")
    if only not in ["all", "features", "labels"]:
        raise ValueError("生成范围须为all、features或labels")

    groups = requested_groups(config) if only != "labels" else ["base"]

    # 逐日生产特征组和预测标签
    for date in config["dates"]:
        date = str(date)
        filename = root / config["feature_root"] / config["feature_version"] / f"{date}.parquet"
        if only == "labels":
            if not filename.is_file():
                raise FileNotFoundError(f"请先生成包含base组的特征文件 {filename}")
            metadata = read_artifact_metadata(filename)
            if metadata["input_definition"] != input_definition(config):
                raise ValueError("标签配置与已有base组不同")
        else:
            filename, _ = build_features_day(root, config, date, groups, rebuild and only != "labels")
        if only in ["all", "labels"]:
            for horizon in config["horizons_ns"]:
                build_labels_day(root, config, date, filename, horizon, rebuild)
        print(f"分组研究数据已完成 {date} {only}", flush=True)


# 【非核心】整理来源版本和采样口径，供产物定义一致性检查使用。
def input_definition(config: dict) -> dict:
    """返回基础组公式版本、来源版本和采样参数，供缓存与标签核对。"""
    return {
        "standard_root": config["standard_root"],
        "standard_version": config["standard_version"],
        "base_group_version": GROUPS["base"]["version"],
        "sampling_clock": "exchange_timestamp_plus_fixed_latency_ns",
        "market_data_latency_ns": config["market_data_latency_ns"],
        "interval_ns": config["interval_ns"],
        "tick_size": config["tick_size"],
        "lot_size": config["lot_size"],
        "max_book_age_ns": config["max_book_age_ns"],
    }


# 【非核心】收集本日及前日输入的元数据，检查来源版本和交易步长。
def standard_sources(root: Path, config: dict, date: str) -> list[dict]:
    """返回本日和可用前日标准文件的定义、大小与修改时间。"""
    calendar = dt.date.fromisoformat(date)
    sources = []
    for source_date in [calendar - dt.timedelta(days=1), calendar]:
        folder = root / config["standard_root"] / config["standard_version"]
        for kind in ["snapshots", "trades"]:
            filename = folder / kind / f"{source_date}.parquet"
            if source_date != calendar and not filename.exists():
                continue
            metadata = read_standard_metadata(filename)
            definition = metadata["definition"]
            if definition["schema_version"] != 2 or definition["version"] != config["standard_version"]:
                raise ValueError("标准行情schema或版本与配置不同")
            for step in ["tick_size", "lot_size"]:
                if definition[step] != config[step]:
                    raise ValueError(f"标准行情与研究配置的{step}不同")
            status = filename.stat()
            sources.append({
                "date": str(source_date),
                "kind": kind,
                "definition": definition,
                "bytes": status.st_size,
                "mtime_ns": status.st_mtime_ns,
            })
    return sources


# 【非核心】读取Parquet内嵌定义，拒绝缺少定义或格式不符的产物。
def read_artifact_metadata(filename: Path) -> dict:
    """读取分组研究产物元数据，明确拒绝旧格式文件。"""
    metadata = pl.read_parquet_metadata(filename)
    if METADATA_KEY not in metadata:
        raise ValueError(f"研究文件缺少分组定义，请使用新输出版本 {filename}")
    definition = json.loads(metadata[METADATA_KEY])
    if definition["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"不支持的研究产物格式 {filename}")
    return definition


# 【非核心】写出临时Parquet并原子替换目标文件，保持数据与元数据同步。
def write_partition(table: pl.DataFrame, filename: Path, metadata: dict) -> dict:
    """输入单日表和定义，原子发布，失败时自动清理临时文件。"""
    if table.select(KEY_COLUMNS).unique().height != table.height:
        raise ValueError("研究表存在重复样本键")
    metadata = {**metadata, "schema_version": SCHEMA_VERSION, "rows": table.height}
    with tempfile.TemporaryDirectory(prefix=".publishing-", dir=filename.parent) as temporary:
        staging = Path(temporary) / filename.name
        table.write_parquet(
            staging, compression="zstd", compression_level=3, row_group_size=100_000,
            metadata={METADATA_KEY: json.dumps(metadata, ensure_ascii=False, sort_keys=True)},
        )
        staging.replace(filename)
    return metadata


# 【核心逻辑10】复用已有基础列，计算新增或更新的特征组并合并单日产物。
def build_features_day(root: Path, config: dict, date: str, groups: list[str],
                       rebuild: bool = False) -> tuple[Path, dict]:
    """返回当日特征文件与元数据，未选择的已有衍生组继续保留。

    输入身份变化时重算base及所选依赖组，采样口径变化要求使用新输出版本。
    """
    filename = root / config["feature_root"] / config["feature_version"] / f"{date}.parquet"
    definition = input_definition(config)
    standard = root / config["standard_root"] / config["standard_version"]
    with partition_lock(standard), partition_lock(filename.parent):
        sources = standard_sources(root, config, date)
        existing = read_artifact_metadata(filename) if filename.exists() else None
        if existing is not None and existing["input_definition"] != definition:
            raise ValueError("基础定义发生变化，请使用新的feature_version")
        fresh_base = existing is None or existing["sources"] != sources
        fresh_base = fresh_base or (rebuild and "base" in config["feature_groups"])
        if fresh_base:
            base = build_base_features(
                standard, date, config["interval_ns"], config["lot_size"],
                config["max_book_age_ns"], config["market_data_latency_ns"],
            ).with_columns((pl.col("book_valid") & pl.col("flow_valid")).alias("base_feature_valid"))
            base = base.select(*KEY_COLUMNS, *BASE_COLUMNS, "base_feature_valid")
            if base.filter(pl.col("last_available_timestamp_ns") > pl.col("timestamp_ns")).height:
                raise ValueError("base组包含未来行情")
            metadata = {
                "kind": "features",
                "version": config["feature_version"],
                "date": date,
                "input_definition": definition,
                "sources": sources,
                "groups": {"base": group_definition("base")},
            }
        else:
            metadata = existing
        affected = groups
        if fresh_base and existing:
            affected = list(dict.fromkeys([*groups, *existing["groups"]]))
        pending = [name for name in affected if name != "base" and
                   (fresh_base or rebuild or name not in metadata["groups"] or metadata["groups"][name] != group_definition(name))]
        if not fresh_base and not pending:
            return filename, metadata
        if not fresh_base:
            base = pl.read_parquet(filename, columns=[*KEY_COLUMNS, *BASE_COLUMNS, "base_feature_valid"])
        updates = {}
        for name in pending:
            updates[name] = build_feature_group(name, base, config)
        # 公式计算只使用base列；发布时才读取其他组，以便完整保留已有列。
        table = base if fresh_base else pl.read_parquet(filename)
        for name, values in updates.items():
            replaced = []
            if name in metadata["groups"]:
                previous = metadata["groups"][name]
                replaced = [*previous["columns"], previous["validity"]]
            table = table.drop(replaced).hstack(values.drop(KEY_COLUMNS))
            metadata["groups"][name] = group_definition(name)
        metadata["valid_rows"] = {
            name: int(table[group["validity"]].sum()) for name, group in metadata["groups"].items()
        }
        metadata = write_partition(table, filename, metadata)
        return filename, metadata


# 【核心逻辑22】从base组读取价格与盘口质量，生产指定周期的独立标签。
def build_labels_day(root: Path, config: dict, date: str, features: Path,
                     horizon_ns: int, rebuild: bool = False) -> Path:
    """读取时间、Mid和盘口质量，生成同周期bps、Ticks、USDT标签。"""
    folder = root / config["label_root"] / config["label_version"] / f"horizon={horizon_name(horizon_ns)}"
    filename = folder / f"{date}.parquet"
    with partition_lock(features.parent), partition_lock(folder):
        feature_metadata = read_artifact_metadata(features)
        metadata = {
            "kind": "labels",
            "version": config["label_version"],
            "date": date,
            "input_definition": feature_metadata["input_definition"],
            "sources": feature_metadata["sources"],
            "base_group": feature_metadata["groups"]["base"],
            "horizon_ns": horizon_ns,
        }
        if filename.exists() and not rebuild:
            previous = read_artifact_metadata(filename)
            if any(previous[key] != value for key, value in metadata.items()):
                raise ValueError("标签定义变化，请使用新的label_version或显式重建标签")
            return filename
        base = pl.read_parquet(features, columns=[*KEY_COLUMNS, "mid_price", "book_valid", "book_segment_id"])
        labels = build_midprice_labels(base, config["tick_size"], config["interval_ns"], horizon_ns)
        metadata["valid_rows"] = int(labels["label_valid"].sum())
        write_partition(labels, filename, metadata)
    return filename



# 【核心逻辑25】按样本键关联所选特征组和标签，筛选可用于研究的样本。
def load_samples(feature_root: Path, label_root: Path, dates: list[str] | None = None,
                       unit: str = "bps", groups: list[str] | None = None) -> tuple[pl.LazyFrame, list[str], str]:
    """返回按键关联的有效样本，逐日期检查来源、基础定义和所选组版本一致。"""
    selected_dates = sorted(
        path.stem for path in feature_root.glob("*.parquet")
    ) if dates is None else sorted(set(dates))
    if not selected_dates:
        raise ValueError("所选日期缺少特征文件")
    if unit not in ["bps", "ticks", "usdt"]:
        raise ValueError("标签单位必须为bps、ticks或usdt")
    reference = None
    selected_groups = list(dict.fromkeys(groups)) if groups is not None else None
    feature_paths = []
    label_paths = []
    for date in selected_dates:
        features = feature_root / f"{date}.parquet"
        labels = label_root / f"{date}.parquet"
        feature_metadata = read_artifact_metadata(features)
        label_metadata = read_artifact_metadata(labels)
        if feature_metadata["kind"] != "features" or label_metadata["kind"] != "labels":
            raise ValueError("研究文件类型不同")
        if selected_groups is None:
            selected_groups = [name for name in GROUPS if name != "base" and name in feature_metadata["groups"]]
        if not selected_groups or not set(selected_groups) <= feature_metadata["groups"].keys():
            raise ValueError("所选特征组不存在，请先生产或显式选择base组")
        if any(feature_metadata["groups"][name] != group_definition(name) for name in selected_groups):
            raise ValueError("特征组定义已过时，请先更新所选组")
        identity = {
            "input_definition": feature_metadata["input_definition"],
            "groups": {name: feature_metadata["groups"][name] for name in selected_groups},
            "horizon_ns": label_metadata["horizon_ns"],
        }
        if reference is not None and reference != identity:
            raise ValueError("跨日期的基础定义、特征组版本或标签周期不同")
        reference = identity
        if (feature_metadata["input_definition"] != label_metadata["input_definition"]
                or feature_metadata["sources"] != label_metadata["sources"]
                or feature_metadata["groups"]["base"] != label_metadata["base_group"]):
            raise ValueError("特征与标签的基础组或来源不同")
        feature_paths.append(features)
        label_paths.append(labels)
    columns = [column for name in selected_groups for column in reference["groups"][name]["model_columns"]]
    validity = [reference["groups"][name]["validity"] for name in selected_groups]
    target = f"y_mid_{horizon_name(reference['horizon_ns'])}_{unit}"
    feature_scan = pl.scan_parquet(feature_paths).select(*KEY_COLUMNS, *validity, *columns)
    label_scan = pl.scan_parquet(label_paths).select(*KEY_COLUMNS, "label_valid", target)
    samples = feature_scan.join(label_scan, on=KEY_COLUMNS, how="inner", validate="1:1")
    samples = samples.filter(pl.all_horizontal([pl.col(name) for name in [*validity, "label_valid"]]))
    return samples.select(*KEY_COLUMNS, *columns, target).sort(KEY_COLUMNS), columns, target


# 【非核心】取得特征版本目录的读取锁，避免物化矩阵期间文件被更新。
@contextmanager
def research_read_lock(feature_root: Path, dates: list[str] | None = None):
    """在上下文内读取一致版本，惰性扫描调用方也可显式使用此上下文。"""
    selected = sorted(
        path.stem for path in feature_root.glob("*.parquet")
    ) if dates is None else sorted(set(dates))
    if not selected or any(not (feature_root / f"{date}.parquet").is_file() for date in selected):
        raise ValueError("所选日期缺少特征文件")
    with partition_lock(feature_root):
        yield selected


# 【核心逻辑24】将关联后的样本物化为X、y和时间索引，固定特征列顺序。
def load_xy(feature_root: Path, label_root: Path, dates: list[str] | None = None,
                  unit: str = "bps", groups: list[str] | None = None):
    """返回X、y与样本索引，列顺序由所选组的声明确定。"""
    with research_read_lock(feature_root, dates) as selected:
        scan, columns, target = load_samples(feature_root, label_root, selected, unit, groups)
        table = scan.collect(engine="streaming")
    return table.select(columns).to_numpy(), table[target].to_numpy(), table.select(KEY_COLUMNS)


# 【非核心】逐日汇总样本可用量和数值分布，输出研究前检查报告。
def summarize_dataset(feature_root: Path, label_root: Path, output: Path) -> dict:
    """返回逐日可用行数与特征、bps标签统计，并保存报告。"""
    filenames = sorted(feature_root.glob("*.parquet"))
    if not filenames:
        raise ValueError("指定版本没有特征文件")
    days = []
    for filename in filenames:
        date = filename.stem
        with research_read_lock(feature_root, [date]):
            scan, columns, target = load_samples(feature_root, label_root, [date])
            expressions = [pl.len().alias("usable_rows"), pl.col(target).mean().alias("target_mean_bps"),
                           pl.col(target).std().alias("target_std_bps")]
            for column in columns:
                expressions.extend([pl.col(column).mean().alias(column + "_mean"),
                                    pl.col(column).min().alias(column + "_min"),
                                    pl.col(column).max().alias(column + "_max")])
            statistics = scan.select(expressions).collect(engine="streaming").row(0, named=True)
            metadata = read_artifact_metadata(filename)
        days.append({"date": date, "rows": metadata["rows"], "group_valid_rows": metadata["valid_rows"], **statistics})
    report = {
        "feature_columns": columns,
        "primary_unit": "bps",
        "days": days,
        "rows": sum(day["rows"] for day in days),
        "usable_rows": sum(day["usable_rows"] for day in days),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
