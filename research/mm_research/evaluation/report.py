"""按因子并行评估，训练期固定分组边界，保存分期指标、非重叠对照与诊断图。"""

from html import escape
import datetime as dt
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.storage import file_locks, read_metadata, write_json_atomic
from mm_research.evaluation.dataset import load_evaluation_targets, partition_sample_mask
from mm_research.evaluation.metrics import correlation_statistics, distribution_statistics, grouped_returns
from mm_research.execution import ordered_process_map


# 【非核心】按预先固定的时间边界标记样本归属，测试期保持独立。
def split_name(date: str, config: dict) -> str:
    """返回train、validation或test，日期使用UTC交易日。"""
    if date <= config["train_end"]:
        return "train"
    if date <= config["validation_end"]:
        return "validation"
    return "test"


# 【核心逻辑113】仅使用训练期因子值拟定分组及直方图边界。
def fit_group_boundaries(filename: Path, config: dict) -> dict:
    """精确训练分位数固定用于全部日期，重复分位边界改用负零正分组。"""
    next_date = dt.date.fromisoformat(config["train_end"]) + dt.timedelta(days=1)
    boundary = int(dt.datetime.combine(next_date, dt.time(), tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    values = pl.scan_parquet(filename).filter(
        pl.col("date").is_in(config["dates"]) & (pl.col("timestamp_ns") < boundary - config["purge_ns"])
    ).select(pl.col(filename.stem).alias("value")).filter(pl.col("value").is_finite())
    series = values.collect(engine="streaming")["value"]
    if series.is_empty():
        raise ValueError(f"训练期无有效因子 {filename.stem}")
    # 一次分区计算全部分位数，避免四个独立quantile各复制完整训练列。
    numbers = series.to_numpy(writable=True)
    del series
    lower = float(numbers.min())
    upper = float(numbers.max())
    cutoffs = np.quantile(numbers, [index / config["groups"] for index in range(1, config["groups"])],
                          method="linear", overwrite_input=True).tolist()
    mode = "quantile" if len(set(cutoffs)) == config["groups"] - 1 else "sign"
    return {"mode": mode, "cutoffs": cutoffs, "min": lower, "max": upper,
            "train_end": config["train_end"]}



# 【核心逻辑112】同批因子共享每日五周期标签读取，各自保存分布和预测指标。
def evaluate_factor_batch(task: tuple) -> list[dict]:
    """批次内共享标签和质量筛选，源因子只读，完成状态按因子独立发布。"""
    root, config, names = task
    factors = root / config["factor_root"] / config["factor_version"]
    labels = root / config["label_root"] / config["label_version"]
    states = {}
    results = []
    with file_locks([factors / f"{name}.parquet" for name in names], shared=True):
        for name in names:
            output = root / config["output"] / name
            output.mkdir(parents=True, exist_ok=True)
            identity = {
                "definition_version": "one_second_five_horizons_v3",
                "config": {key: value for key, value in config.items() if key not in ["workers", "factor_batch_size"]},
                "source": read_metadata(factors / f"{name}.parquet"),
            }
            completed = output / "completed.json"
            if completed.exists() and json.loads(completed.read_text()) == identity:
                results.append({"factor": name, "cached": True})
                continue
            boundaries = fit_group_boundaries(factors / f"{name}.parquet", config)
            lower, upper = boundaries["min"], boundaries["max"]
            if lower == upper:
                lower, upper = lower - 0.5, upper + 0.5
            states[name] = {
                "output": output, "identity": identity, "boundaries": boundaries,
                "edges": np.linspace(lower, upper, 81), "histogram": np.zeros(80, dtype=np.int64),
                "distributions": [], "correlations": [], "groups": [],
            }
        for date in config["dates"]:
            daily_values = {}
            for name, state in states.items():
                values = pl.scan_parquet(factors / f"{name}.parquet").filter(pl.col("date") == date).select(
                    "timestamp_ns", pl.col(name).alias("value"),
                ).collect(engine="streaming")
                daily_values[name] = values
                state["distributions"].append({"factor": name, "date": date, "split": split_name(date, config),
                                                **distribution_statistics(values), "unique_values": values["value"].drop_nulls().n_unique()})
                numbers = values["value"].to_numpy()
                state["histogram"] += np.histogram(
                    np.clip(numbers[np.isfinite(numbers)], state["edges"][0], state["edges"][-1]), bins=state["edges"],
                )[0]
            if not states:
                break
            dataset = next(iter(states.values()))["identity"]["source"]["dataset"]
            for horizon in config["horizons_ns"]:
                target = load_evaluation_targets(labels, date, horizon, dataset).with_columns(
                    (pl.col("label_valid") & partition_sample_mask(date, config)).fill_null(False).alias("label_valid"),
                )
                for name, values in daily_values.items():
                    state = states[name]
                    table = values.join(target, on="timestamp_ns", how="left")
                    keys = {"factor": name, "date": date, "split": split_name(date, config), "horizon_ns": horizon}
                    full_metrics = correlation_statistics(table)
                    full_groups = grouped_returns(table, config["groups"], state["boundaries"])
                    state["correlations"].append({**keys, "sampling": "all", **full_metrics})
                    if horizon == config["output_interval_ns"]:
                        nonoverlap_metrics = full_metrics
                        nonoverlap_groups = full_groups
                    else:
                        nonoverlap = table.filter(pl.col("timestamp_ns") % horizon == 0)
                        nonoverlap_metrics = correlation_statistics(nonoverlap)
                        nonoverlap_groups = grouped_returns(nonoverlap, config["groups"], state["boundaries"])
                    state["correlations"].append({**keys, "sampling": "nonoverlap", **nonoverlap_metrics})
                    for sampling, returns in [("all", full_groups), ("nonoverlap", nonoverlap_groups)]:
                        state["groups"].append(returns.with_columns(
                            *[pl.lit(value).alias(key) for key, value in keys.items()],
                            pl.lit(sampling).alias("sampling"),
                        ))
        for name, state in states.items():
            publish_factor_evaluation(name, config, state)
            results.append({"factor": name, "cached": False})
            print(f"评估完成 {name}", flush=True)
    return results


# 【核心逻辑119】汇总单因子的分期指标和固定分组价差，并发布可独立复用的结果。
def publish_factor_evaluation(name: str, config: dict, state: dict) -> None:
    """输入已完成的日级统计，保存结果表、图及与源定义绑定的完成状态。"""
    daily_ic = pl.DataFrame(state["correlations"], infer_schema_length=None, schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
    distributions = pl.DataFrame(state["distributions"], infer_schema_length=None)
    returns = pl.concat(state["groups"], how="vertical_relaxed")
    summary = daily_ic.group_by("factor", "split", "horizon_ns", "sampling").agg(
        pl.col("ic").count().alias("valid_ic_days"), pl.col("ic").mean().alias("ic_mean"),
        pl.col("ic").median().alias("ic_median"), pl.col("ic").std().alias("ic_std"),
        pl.col("rank_ic").mean().alias("rank_ic_mean"),
        (pl.col("ic").drop_nulls() > 0).mean().alias("positive_ic_day_ratio"),
        pl.col("paired_rows").sum().alias("paired_rows"),
    ).sort("split", "horizon_ns", "sampling")
    spreads = returns.filter(pl.col("actual_groups") == pl.col("requested_groups")).group_by("factor", "split", "date", "horizon_ns", "sampling").agg(
        (pl.col("mean_return_bps").sort_by("group").last()
         - pl.col("mean_return_bps").sort_by("group").first()).alias("top_bottom_bps"),
    ).group_by("factor", "split", "horizon_ns", "sampling").agg(
        pl.col("top_bottom_bps").mean(), pl.len().alias("complete_group_days"),
    )
    summary = summary.join(spreads, on=["factor", "split", "horizon_ns", "sampling"], how="left")
    output = state["output"]
    with file_locks([output / "evaluation"]):
        for filename, table in [("summary", summary), ("daily_distribution", distributions),
                                ("daily_ic", daily_ic), ("group_returns", returns)]:
            table.write_parquet(output / f"{filename}.parquet", compression="zstd")
        write_json_atomic(output / "group_boundaries.json", state["boundaries"])
        render_factor_report(output, name, summary, daily_ic, distributions, returns, state["histogram"], state["edges"])
        write_json_atomic(output / "completed.json", state["identity"])


# 【核心逻辑111】并行调度独立因子评估，并汇总分期结果及原始基线对照。
def run_evaluation(config_path: Path) -> dict:
    """配置明确训练和验证边界，按因子独立保存结果，失败后可复用已完成因子。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "evaluation")
    started = time.perf_counter()
    names = config["factors"]
    batches = [names[start:start + config["factor_batch_size"]] for start in range(0, len(names), config["factor_batch_size"])]
    for results in ordered_process_map(evaluate_factor_batch, [(root, config, batch) for batch in batches], config["workers"]):
        for result in results:
            if result["cached"]:
                print(f'评估复用 {result["factor"]}', flush=True)
    output = root / config["output"]
    summary = pl.concat([pl.read_parquet(output / name / "summary.parquet") for name in config["factors"]])
    summary.write_parquet(output / "summary.parquet", compression="zstd")
    links = "".join(f'<li><a href="{escape(name)}/report.html">{escape(name)}</a></li>' for name in config["factors"])
    (output / "report.html").write_text('<!doctype html><meta charset="utf-8"><h1>分期因子评估</h1><ul>' + links + '</ul>')
    return {"elapsed_seconds": time.perf_counter() - started, "summary": summary}


# 【非核心】绘制可直接检查的分布、逐日IC、多尺度IC与固定分组收益。
def render_factor_report(output: Path, name: str, summary: pl.DataFrame, daily_ic: pl.DataFrame,
                         distributions: pl.DataFrame, returns: pl.DataFrame,
                         counts: np.ndarray, edges: np.ndarray) -> None:
    """仅用已保存的评估表绘图，日分布额外展示有效率与分位变化。"""
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    figure.suptitle(name, fontsize=10)
    axes[0, 0].bar(edges[:-1], counts, width=np.diff(edges), align="edge")
    axes[0, 0].set(title="Distribution, training-range edge bins", ylabel="Samples")
    day_values = daily_ic.filter((pl.col("horizon_ns") == 1_000_000_000) & (pl.col("sampling") == "all")).sort("date")
    axes[0, 1].plot(day_values["ic"], label="Pearson")
    axes[0, 1].plot(day_values["rank_ic"], label="Spearman", alpha=0.7)
    axes[0, 1].axhline(0, color="gray", linewidth=0.5)
    axes[0, 1].set(title="Daily IC, 1s", xlabel="Trading-day index")
    axes[0, 1].legend()
    for split in ["train", "validation", "test"]:
        values = summary.filter((pl.col("split") == split) & (pl.col("sampling") == "all")).sort("horizon_ns")
        axes[1, 0].plot(values["horizon_ns"] / 1e9, values["ic_mean"], marker="o", label=split)
    axes[1, 0].set(title="Mean daily IC by horizon", xlabel="Seconds")
    axes[1, 0].legend()
    group_values = returns.filter((pl.col("split") == "test") & (pl.col("horizon_ns") == 1_000_000_000) & (pl.col("sampling") == "all")).group_by("group").agg(pl.col("mean_return_bps").mean()).sort("group")
    axes[1, 1].bar(group_values["group"], group_values["mean_return_bps"])
    axes[1, 1].set(title="Frozen-training groups, test 1s", ylabel="Future mid return, bps")
    axes[2, 0].plot(distributions["valid_rows"] / distributions["rows"])
    axes[2, 0].set(title="Daily factor validity", ylim=(0, 1.02))
    for column in ["q05", "median", "q95"]:
        axes[2, 1].plot(distributions[column], label=column)
    axes[2, 1].set(title="Daily quantiles")
    axes[2, 1].legend()
    figure.savefig(output / "diagnostics.png", dpi=120)
    plt.close(figure)
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>' + escape(name) + '</title><h1>' + escape(name)
        + '</h1><p>训练阈值固定，IC含全网格与非重叠对照。条件未来中间价收益以bps计。'
        + '直方图边界来自训练期，超范围值归入两端。逐日指标见同目录Parquet。</p>'
        + '<img style="max-width:100%" src="diagnostics.png" alt="因子评估">',
    )
