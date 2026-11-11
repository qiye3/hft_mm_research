"""按文档分组计算Factors，共享输入及表达式，临时分块后合并为单因子全日期文件。"""

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import datetime as dt
import json
import pickle
from pathlib import Path
import resource
import shutil
import time

import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.storage import KEY_COLUMNS, SCHEMA_VERSION, artifact_ready, file_locks, read_metadata, scan_series, write_json_atomic, write_parquet
from mm_research.execution import ordered_process_map

SUPPORTED_BATCHES = ("gen001", "gen002", "gen003", "gen004", "gen005", "gen006", "gen008", "gen009", "gen010")
EXTENDED_BATCHES = tuple(f"gen{index:03d}" for index in range(11, 38))
PRECOMPUTED_BATCHES = ("gen012", "gen014", "gen021", "gen022", "gen025", "gen026", "gen028", "gen031", "gen034", "gen037")


# 【非核心】仅加载当前已授权的具名公式组，未知组直接失败。
def factor_module(batch: str):
    """返回公式模块，Gen007及其他OFI分支不属于本轮范围。"""
    if batch not in (*SUPPORTED_BATCHES, *EXTENDED_BATCHES):
        raise ValueError(f"未支持的因子组 {batch}")
    return import_module(f"mm_research.factors.{batch}")


# 【非核心】把配置与输入定义记录为可复用的因子身份。
def factor_metadata(name: str, batch: str, dataset: dict, config: dict, inputs: dict) -> dict:
    """定义包含公式版本、窗口有效比例、来源及输出日期，分块大小不改变公式。"""
    metadata = {
        "kind": "factor", "name": name, "dataset": dataset,
        "definition": {
            "batch": batch, "version": factor_module(batch).VERSION,
            "dates": config["dates"], "min_fraction": config["min_fraction"],
            "tick_size": config["tick_size"], "inputs": inputs,
            "source_interval_ns": dataset["interval_ns"], "output_interval_ns": config["output_interval_ns"],
            "window_unit": "seconds", "storage_dtype": "float64",
        },
    }
    if batch == "gen037":
        metadata["definition"]["intermediate_interval_ns"] = 1_000_000_000
    return metadata


# 【核心逻辑44】一次读取同组依赖，带前日历史计算所有窗口并保存临时因子块。
def compute_factor_chunk(task: dict) -> dict:
    """按配置带足里层及外层历史，完整分块和状态原子发布后可用于恢复。"""
    checkpoint = Path(task['staging']) / f'{task["index"]:04d}' / 'completed.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as stream:
            return pickle.load(stream)
    started = time.perf_counter()
    cpu_started = time.process_time()
    config = task["config"]
    outer_seconds = max([*config['aggregation_windows_s'],
                         *[pair[1] for pair in config['mean_diff_windows_s']]])
    outer_ns = outer_seconds * 1_000_000_000
    features = Path(task["features"])
    previous = str(dt.date.fromisoformat(task["dates"][0]) - dt.timedelta(days=1))
    history = [previous, *task["dates"]]
    stateful = task["batch"] in ["gen014", "gen021"]
    if stateful:
        history = task["dates"]
    expressions = factor_module(task["batch"]).factor_expressions(config)
    expressions = {name: expressions[name] for name in task["names"]}
    dependencies = factor_module(task["batch"]).source_expressions(config) if task["batch"] in PRECOMPUTED_BATCHES else expressions
    columns = sorted({column for expression in dependencies.values() for column in expression.meta.root_names()
                      if column not in KEY_COLUMNS and not column.startswith("_slot_")})
    inputs = [features / f"{column}.parquet" for column in columns]
    with file_locks(inputs, shared=True):
        start_ns = int(dt.datetime.fromisoformat(task["dates"][0]).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        history_start = None if stateful else start_ns - outer_ns - 101_000_000_000
        source, _ = scan_series(features, columns, history, start_timestamp_ns=history_start)
        if task["batch"] in ["gen025", "gen034"]:
            # 有界里层最长100秒，加100秒外层；300秒覆盖预热，避免重算整个前日。
            source = source.filter(pl.col("timestamp_ns") >= start_ns - outer_ns - 101_000_000_000)
        next_state = None
        if stateful:
            # 无界事件状态携带真实锚点，滚动尾部长度随本轮最大窗口变化。
            previous_state = task["state"]
            raw = source.collect(engine="streaming")
            if previous_state is not None:
                raw = previous_state["raw"].vstack(raw)
            module = factor_module(task["batch"])
            if task["batch"] == "gen014":
                anchors = previous_state["anchors"] if previous_state is not None else None
                dependencies = module.source_expressions(config, anchors)
            else:
                anchor = previous_state["anchor"] if previous_state is not None else None
                dependencies = module.source_expressions(config, anchor)
            inner = raw.select(*KEY_COLUMNS, *[expression.alias(name) for name, expression in dependencies.items()])
            if previous_state is not None:
                inner = inner.slice(1)
            timestamp = inner["timestamp_ns"][-1]
            next_state = {"raw": raw.tail(1)}
            if task["batch"] == "gen014":
                anchors = {}
                for name in dependencies:
                    age = inner[name][-1]
                    anchors[name] = None if age is None else timestamp - round(age * 10) * 100_000_000
                next_state["anchors"] = anchors
            else:
                run = inner["trade_dir_signed_run_100ms"][-1]
                next_state["anchor"] = timestamp - (int(abs(run)) - 1) * 100_000_000 if run else timestamp
            if previous_state is not None:
                inner = previous_state["tail"].vstack(inner)
            next_state["tail"] = inner.tail(outer_seconds * 10 + 10)
            source = inner.lazy()
        elif task["batch"] in PRECOMPUTED_BATCHES:
            # 先形成里层状态，再按本轮最长外层窗口截取输出之前的历史。
            source = source.select(*KEY_COLUMNS, *[expression.alias(name) for name, expression in dependencies.items()])
            source = source.filter(pl.col("timestamp_ns") >= start_ns - outer_ns - 1_000_000_000)
            if task["batch"] in ["gen034", "gen037"]:
                source = source.filter(pl.col("timestamp_ns") % 1_000_000_000 == 0)
            if task["batch"] in ["gen022", "gen034"]:
                # 盘口墙状态或熵计数先物化，阻止多个窗口的查询图重复展开。
                source = source.collect(engine="streaming").lazy()
        if task["batch"] == "gen020":
            source = factor_module("gen020").attach_slot_reference(source, Path(task["slot_reference"]))
        # 滚动及lag在日期过滤之前执行，跨分块边界保留已收到的历史。
        values = source.select(*KEY_COLUMNS, *[expression.alias(name) for name, expression in expressions.items()])
        if stateful:
            values = values.filter(pl.col("date").is_in(config["dates"]))
        values = values.filter(
            pl.col("date").is_in(task["dates"]) & (pl.col("timestamp_ns") % config["output_interval_ns"] == 0)
        ).collect(engine="streaming")
        # 压缩写出释放GIL，用少量线程共享已物化表，避免36列压缩串行占用单核。
        with ThreadPoolExecutor(max_workers=config["write_threads"]) as writers:
            pending = []
            for name in task["names"]:
                filename = Path(task["staging"]) / f'{task["index"]:04d}' / f"{name}.parquet"
                pending.append(writers.submit(
                    write_parquet, values.lazy().select(*KEY_COLUMNS, name), filename, task["metadata"][name],
                ))
            for completed in pending:
                completed.result()
    result = {
        "batch": task["batch"], "first_date": task["dates"][0], "last_date": task["dates"][-1],
        "rows": values.height, "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "state": next_state,
    }
    pending = checkpoint.with_suffix('.part')
    with pending.open('wb') as stream:
        pickle.dump(result, stream)
    pending.replace(checkpoint)
    return result


# 【核心逻辑45】按日期顺序连接临时块，原子发布一个因子的全日期Parquet。
def merge_factor_chunks(task: dict) -> str:
    """合并只做纵向连接，保持全区间排序和每个因子一个文件的最终布局。"""
    filename = Path(task["output"]) / f'{task["name"]}.parquet'
    sources = [Path(task["staging"]) / f'{index:04d}' / filename.name for index in range(task["chunks"])]
    with file_locks([filename]):
        write_parquet(pl.scan_parquet(sources), filename, task["metadata"])
    return task["name"]


# 【核心逻辑40】按组分块并行计算，完整发布后释放临时数据，保留单因子缓存。
def run_factor_production(config_path: Path, rebuild: bool = False) -> None:
    """输入Factors配置，支持同组共享计算、跨日历史以及已有完整因子的复用。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "factors")
    if config["output_interval_ns"] != 1_000_000_000:
        raise ValueError("当前Factors使用UTC整秒输出")
    features = root / config["feature_root"] / config["feature_version"]
    output = root / config["factor_root"] / config["factor_version"]
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for batch in config["batches"]:
        expressions = factor_module(batch).factor_expressions(config)
        dependencies = factor_module(batch).source_expressions(config) if batch in PRECOMPUTED_BATCHES else expressions
        columns = sorted({column for expression in dependencies.values() for column in expression.meta.root_names()
                          if column not in KEY_COLUMNS and not column.startswith("_slot_")})
        with file_locks([features / f"{column}.parquet" for column in columns], shared=True):
            _, dataset = scan_series(features, columns)
            if not set(config["dates"]) <= set(dataset["dates"]):
                raise ValueError("Factor日期超出Features范围")
            definitions = {column: read_metadata(features / f"{column}.parquet")["definition"] for column in columns}
            slot_reference = None
            if batch == "gen020":
                slot_reference = factor_module(batch).build_slot_reference(features, output, config, dataset)
                definitions["slot_reference"] = read_metadata(slot_reference)["definition"]
            metadata = {name: factor_metadata(name, batch, dataset, config, definitions) for name in expressions}
            # 只复用元数据逐项相同的完整产物，符号链接保留原有来源且避免重复占盘。
            for name in expressions:
                filename = output / f'{name}.parquet'
                if filename.exists():
                    continue
                expected = {**metadata[name], 'schema_version': SCHEMA_VERSION}
                for version in config['reuse_factor_versions']:
                    existing = root / config['factor_root'] / version / filename.name
                    if existing.exists() and read_metadata(existing) == expected:
                        filename.symlink_to(existing.resolve())
                        break
            names = [name for name in expressions if not artifact_ready(output / f"{name}.parquet", metadata[name], rebuild)]
            if not names:
                print(f"Factor组复用 {batch}", flush=True)
                continue
            stateful = batch in ["gen014", "gen021"]
            chunk_days = min(7, config["chunk_days"]) if stateful else config["chunk_days"]
            state_dates = [date for date in dataset["dates"] if date <= config["dates"][-1]] if stateful else config["dates"]
            chunks = [state_dates[start:start + chunk_days]
                      for start in range(0, len(state_dates), chunk_days)]
            # 批次锁避免两个生产命令同时计算并发布同一组。输入始终由共享锁保护。
            with file_locks([output / batch]):
                temporary = output / '.chunks' / batch
                temporary.mkdir(parents=True, exist_ok=True)
                identity_file = temporary / 'definition.json'
                identity = {'config': config, 'metadata': metadata}
                if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
                    raise ValueError('分块恢复的配置或输入定义发生变化')
                identity_file.write_text(json.dumps(identity, ensure_ascii=False, sort_keys=True))
                tasks = [
                    {"config": config, "features": str(features), "batch": batch, "dates": dates,
                     "names": names, "staging": temporary, "index": index, "metadata": metadata,
                     "slot_reference": str(slot_reference) if slot_reference is not None else None, "state": None}
                    for index, dates in enumerate(chunks)
                ]
                workers = 1 if stateful else config["workers"]
                for index, usage in enumerate(ordered_process_map(
                    compute_factor_chunk, tasks, workers, recycle_workers=True,
                )):
                    carry = usage.pop("state")
                    if stateful and index + 1 < len(tasks):
                        tasks[index + 1]["state"] = carry
                        tasks[index]["state"] = None
                    report.append(usage)
                    print(json.dumps(usage, ensure_ascii=False), flush=True)
                merges = [
                    {"output": str(output), "staging": temporary, "chunks": len(chunks),
                     "name": name, "metadata": metadata[name]}
                    for name in names
                ]
                for name in ordered_process_map(merge_factor_chunks, merges, config["merge_workers"], recycle_workers=True):
                    print(f"Factor完成 {name}", flush=True)
                shutil.rmtree(temporary)
    write_json_atomic(output / 'runtime' / f'{"_".join(config["batches"])}.json', report)
