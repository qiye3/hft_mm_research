"""按因子并行评估，训练期固定分组边界，保存分期指标、非重叠对照与诊断图。"""

from html import escape
import datetime as dt
import json
import pickle
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.storage import file_locks, read_metadata, write_json_atomic
from mm_research.evaluation.dataset import load_evaluation_targets, partition_sample_mask
from mm_research.evaluation.metrics import (
    bootstrap_interval, bootstrap_summary, correlation_statistics, distribution_statistics, extreme_tail_days, grouped_returns,
    group_quality_statistics, oriented_statistics, single_factor_ols, train_ic_direction,
)
from mm_research.execution import ordered_process_map


# 【非核心】按预先固定的时间边界标记样本归属，测试期保持独立。
def split_name(date: str, config: dict) -> str:
    """返回train、validation或test，日期使用UTC交易日。"""
    if date <= config["train_end"]:
        return "train"
    if date <= config['validation_dev_end']:
        return 'val_dev'
    if date <= config["validation_end"]:
        return "val_select"
    return "test"


# 【核心逻辑113】仅使用训练期因子值拟定分组及直方图边界。
def fit_group_boundaries(filename: Path, config: dict) -> dict:
    """精确训练分位数固定用于全部日期，重复分位保持共同归组，另存P5/P95极端阈值。"""
    next_date = dt.date.fromisoformat(config["train_end"]) + dt.timedelta(days=1)
    boundary = int(dt.datetime.combine(next_date, dt.time(), tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
    values = pl.scan_parquet(filename).filter(
        pl.col("date").is_in(config["dates"])
        & (pl.col("timestamp_ns") + config['factor_lag_ns'] < boundary - config["purge_ns"])
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
    p05, p95 = np.quantile(numbers, [.05, .95], method="linear").tolist()
    mode = 'quantile'
    return {"mode": mode, "cutoffs": cutoffs, "min": lower, "max": upper,
            "p05": float(p05), "p95": float(p95), "train_end": config["train_end"]}



# 【核心逻辑112】同批因子共享每日配置周期标签读取，各自保存分布和预测指标。
def evaluate_factor_batch(task: tuple) -> list[dict]:
    """批次内共享标签和质量筛选，源因子只读，完成状态按因子独立发布。"""
    root, config, names = task
    factors = root / config["factor_root"] / config["factor_version"]
    labels = root / config["label_root"] / config["label_version"]
    train_dates = [date for date in config['dates'] if date <= config['train_end']]
    fold_boundaries = {
        int(dt.datetime.fromisoformat(train_dates[int(len(train_dates) * fraction)]).replace(
            tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
        for fraction in [.4, .6, .8]
    }
    states = {}
    results = []
    checkpoint = root / config['output'] / '.pending' / names[0] / 'state.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as stream:
            saved = pickle.load(stream)
        if saved['names'] != names:
            raise ValueError('评估检查点的批次列与当前配置不同')
        restored_states = saved['states']
    else:
        restored_states = {}
    with file_locks([factors / f"{name}.parquet" for name in names], shared=True):
        for name in names:
            output = root / config["output"] / name
            output.mkdir(parents=True, exist_ok=True)
            identity = {
                "definition_version": "coarse_windows_log_bps_seven_horizons_v3",
                "config": {key: value for key, value in config.items() if key not in ["workers", "factor_batch_size"]},
                "source": read_metadata(factors / f"{name}.parquet"),
            }
            completed = output / "completed.json"
            if completed.exists() and json.loads(completed.read_text()) == identity:
                results.append({"factor": name, "cached": True})
                continue
            if name in restored_states:
                state = restored_states[name]
                if state['identity'] != identity:
                    raise ValueError('评估检查点的数据或统计定义已经变化')
                states[name] = state
                continue
            boundaries = fit_group_boundaries(factors / f"{name}.parquet", config)
            lower, upper = boundaries["min"], boundaries["max"]
            if lower == upper:
                lower, upper = lower - 0.5, upper + 0.5
            states[name] = {
                "output": output, "identity": identity, "boundaries": boundaries,
                "edges": np.linspace(lower, upper, 81), "histogram": np.zeros(80, dtype=np.int64),
                "distributions": [], "correlations": [], "groups": [], "extremes": [], "ols": {},
                'incremental': [],
            }
        dates_to_compute = [] if restored_states or not states else config['dates']
        for date in dates_to_compute:
            daily_values = {}
            day_start = int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()) * 1_000_000_000
            # 已完成窗口仍参与共同样本掩码，其统计无需重算。
            for name in names:
                values = pl.scan_parquet(factors / f"{name}.parquet").filter(
                    pl.col('timestamp_ns').is_between(day_start - config['factor_lag_ns'],
                        day_start + 86400_000_000_000 - config['factor_lag_ns'], closed='left')
                ).select(
                    (pl.col("timestamp_ns") + config['factor_lag_ns']).alias('timestamp_ns'), pl.col(name).alias("value"),
                ).collect(engine="streaming")
                daily_values[name] = values
                if name not in states:
                    continue
                state = states[name]
                state["distributions"].append({"factor": name, "date": date, "split": split_name(date, config),
                                                **distribution_statistics(values), "unique_values": values["value"].drop_nulls().n_unique()})
                numbers = values["value"].to_numpy()
                state["histogram"] += np.histogram(
                    np.clip(numbers[np.isfinite(numbers)], state["edges"][0], state["edges"][-1]), bins=state["edges"],
                )[0]
            if not states:
                break
            common_masks = {}
            for name, values in daily_values.items():
                formula = re.sub(r'_(?:\d+s_)?\d+s$', '', name)
                valid = values['value'].is_finite().fill_null(False)
                common_masks[formula] = valid if formula not in common_masks else common_masks[formula] & valid
            dataset = next(iter(states.values()))["identity"]["source"]["dataset"]
            targets_by_horizon = {}
            fold_purge = pl.lit(True)
            boundary = day_start + 86400_000_000_000
            if boundary in fold_boundaries:
                fold_purge = ((pl.col('timestamp_ns') < boundary - config['purge_ns'])
                              & (pl.col('label_end_timestamp_ns') < boundary))
            for horizon in config["horizons_ns"]:
                target = load_evaluation_targets(labels, date, horizon, dataset).with_columns(
                    (pl.col("label_valid") & partition_sample_mask(date, config) & fold_purge).fill_null(False).alias("label_valid"),
                )
                targets_by_horizon[horizon] = target
                for name, state in states.items():
                    values = daily_values[name]
                    table = values.join(target, on="timestamp_ns", how="left")
                    keys = {"factor": name, "date": date, "split": split_name(date, config), "horizon_ns": horizon}
                    full_metrics = correlation_statistics(table)
                    full_groups = grouped_returns(table, config["groups"], state["boundaries"])
                    if horizon == config["output_interval_ns"]:
                        nonoverlap_table = table
                        nonoverlap_metrics = full_metrics
                        nonoverlap_groups = full_groups
                    else:
                        nonoverlap_table = table.filter(pl.col("timestamp_ns") % horizon == 0)
                        nonoverlap_metrics = correlation_statistics(nonoverlap_table)
                        nonoverlap_groups = grouped_returns(nonoverlap_table, config["groups"], state["boundaries"])
                    state["correlations"].append({**keys, "sampling": "all", **full_metrics})
                    state["correlations"].append({**keys, "sampling": "nonoverlap", **nonoverlap_metrics})
                    formula = re.sub(r'_(?:\d+s_)?\d+s$', '', name)
                    common_mask = common_masks[formula]
                    if common_mask.all():
                        common_metrics, common_groups = full_metrics, full_groups
                    else:
                        common_table = table.filter(common_mask)
                        common_metrics = correlation_statistics(common_table)
                        common_groups = grouped_returns(common_table, config['groups'], state['boundaries'])
                    state['correlations'].append({**keys, 'sampling': 'common', **common_metrics})
                    for sampling, returns in [("all", full_groups), ("nonoverlap", nonoverlap_groups)]:
                        state["groups"].append(returns.with_columns(
                            *[pl.lit(value).alias(key) for key, value in keys.items()],
                            pl.lit(sampling).alias("sampling"),
                        ))
                    state['groups'].append(common_groups.with_columns(
                        *[pl.lit(value).alias(key) for key, value in keys.items()], pl.lit('common').alias('sampling')))
                    if horizon == 180_000_000_000:
                        for start_s in [10, 60, 120]:
                            short = targets_by_horizon[start_s * 1_000_000_000].select(
                                'timestamp_ns', pl.col('target').alias('short_target'))
                            increment = table.join(short, on='timestamp_ns').with_columns(
                                (pl.col('target') - pl.col('short_target')).alias('target'))
                            state['incremental'].append({**keys, 'start_s': start_s, 'end_s': 180,
                                                         **correlation_statistics(increment)})
                    # 【核心逻辑290】极端分位逐日差值沿用训练期P5/P95，验证测试不重拟分位。
                    for sampling, sampled_table in [("all", table), ("nonoverlap", nonoverlap_table)]:
                        state["extremes"].append(extreme_tail_days(
                            sampled_table, name, date, keys["split"], horizon, sampling,
                            state["boundaries"]["p05"], state["boundaries"]["p95"]))
                    # 【核心逻辑292】每个预测周期独立累积各分区充分统计量，参数只在训练期拟合。
                    if name in config['ols_factors']:
                        valid = table.filter(pl.col("value").is_finite() & pl.col("target").is_finite()
                                             & pl.col("label_valid").fill_null(False))
                        sums = valid.select(
                            pl.len().alias("count"), pl.col("value").sum().alias("sum_x"),
                            (pl.col("value") ** 2).sum().alias("sum_x2"),
                            pl.col("target").sum().alias("sum_y"),
                            (pl.col("target") ** 2).sum().alias("sum_y2"),
                            (pl.col("value") * pl.col("target")).sum().alias("sum_xy"),
                        ).row(0)
                        statistics = state['ols'].setdefault(horizon, {})
                        current = statistics.get(keys['split'], [0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        statistics[keys['split']] = [left + right for left, right in zip(current, sums)]
        if states and not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.partial')
            with temporary.open('wb') as stream:
                pickle.dump({'names': names, 'states': states}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(checkpoint)
        for name in list(states):
            state = states.pop(name)
            publish_factor_evaluation(name, config, state)
            results.append({"factor": name, "cached": False})
            print(f"评估完成 {name}", flush=True)
        if checkpoint.exists():
            checkpoint.unlink()
            checkpoint.parent.rmdir()
    return results


# 【核心逻辑119】汇总单因子的分期指标和固定分组价差，并发布可独立复用的结果。
def publish_factor_evaluation(name: str, config: dict, state: dict) -> None:
    """输入已完成的日级统计，保存结果表、图及与源定义绑定的完成状态。"""
    daily_ic = pl.DataFrame(state["correlations"], infer_schema_length=None, schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
    distributions = pl.DataFrame(state["distributions"], infer_schema_length=None)
    returns = pl.concat(state["groups"], how="vertical_relaxed").rechunk()
    state["groups"].clear()
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
    # 【核心逻辑120/121】方向指标沿用训练期方向；sign组价差只在退化三组时有定义。
    directions = train_ic_direction(daily_ic)
    oriented = oriented_statistics(daily_ic, returns, directions)
    summary = (summary.join(directions, on=["horizon_ns", "sampling"], how="left")
               .join(oriented.drop("train_ic_direction"), on=["split", "horizon_ns", "sampling"], how="left"))
    if state["boundaries"]["mode"] == "sign":
        summary = summary.with_columns(
            (pl.col("sign_group_spread_bps") * pl.col("train_ic_direction")).alias("oriented_group_spread_bps"))
    else:
        summary = summary.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("sign_group_spread_bps"),
            pl.lit(None, dtype=pl.UInt32).alias("sign_spread_days"),
            (pl.col("top_bottom_bps") * pl.col("train_ic_direction")).alias("oriented_group_spread_bps"))
    # 【核心逻辑122/123】IC与端点组差的95%区间按交易日重采样，两种分组模式用各自端点。
    spread_group = 3 if state["boundaries"]["mode"] == "sign" else config["groups"]
    summary = summary.join(bootstrap_summary(daily_ic, returns, spread_group),
                           on=["split", "horizon_ns", "sampling"], how="left")
    # 【核心逻辑291】分组单调性与端点差为正日比例沿用训练方向，两种模式各自端点。
    summary = summary.join(group_quality_statistics(returns, directions, spread_group),
                           on=["split", "horizon_ns", "sampling"], how="left")
    # 【核心逻辑290】极端分位逐日差值先落表，再按有效日等权进入summary。
    extremes = pl.DataFrame(state['extremes'], infer_schema_length=None, schema_overrides={
        'high_bps': pl.Float64, 'low_bps': pl.Float64, 'extreme_spread_bps': pl.Float64,
    })
    summary = summary.join(extremes.group_by("split", "horizon_ns", "sampling").agg(
        pl.col("extreme_spread_bps").drop_nulls().mean().alias("extreme_5pct_spread_bps"),
        pl.col("extreme_spread_bps").is_not_null().sum().alias("extreme_valid_days"),
    ), on=["split", "horizon_ns", "sampling"], how="left")
    output = state["output"]
    with file_locks([output / "evaluation"]):
        for filename, table in [("summary", summary), ("daily_distribution", distributions),
                                ("daily_ic", daily_ic), ("group_returns", returns),
                                ("extreme_returns", extremes)]:
            table.write_parquet(output / f"{filename}.parquet", compression="zstd")
        if state['incremental']:
            incremental = pl.DataFrame(state['incremental'], schema_overrides={'ic': pl.Float64, 'rank_ic': pl.Float64})
            incremental.write_parquet(output / 'incremental_daily_ic.parquet', compression='zstd')
            rows = []
            for part in incremental.partition_by(['split', 'start_s', 'end_s']):
                rows.append({'factor': name, 'split': part['split'][0], 'start_s': part['start_s'][0],
                             'end_s': 180, 'ic_mean': part['ic'].mean(), 'rank_ic_mean': part['rank_ic'].mean(),
                             **bootstrap_interval(part['ic'].to_list())})
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(output / 'incremental_summary.parquet')
        # 【核心逻辑292】单因子OLS结果独立成表，参数全部来自训练期。
        if state["ols"]:
            pl.DataFrame([{"factor": name, 'horizon_ns': horizon, **single_factor_ols(statistics)}
                          for horizon, statistics in state['ols'].items()], infer_schema_length=None).write_parquet(
                output / "single_factor_ols.parquet", compression="zstd")
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
    for results in ordered_process_map(evaluate_factor_batch, [(root, config, batch) for batch in batches], config["workers"], recycle_workers=True):
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
    day_values = daily_ic.filter((pl.col("horizon_ns") == 180_000_000_000) & (pl.col("sampling") == "all")).sort("date")
    axes[0, 1].plot(day_values["ic"], label="Pearson")
    axes[0, 1].plot(day_values["rank_ic"], label="Spearman", alpha=0.7)
    axes[0, 1].axhline(0, color="gray", linewidth=0.5)
    axes[0, 1].set(title="Daily IC, 180s", xlabel="Trading-day index")
    axes[0, 1].legend()
    for split in ['train', 'val_dev', 'val_select', 'test']:
        values = summary.filter((pl.col("split") == split) & (pl.col("sampling") == "all")).sort("horizon_ns")
        axes[1, 0].plot(values["horizon_ns"] / 1e9, values["ic_mean"], marker="o", label=split)
    axes[1, 0].set(title="Mean daily IC by horizon", xlabel="Seconds")
    axes[1, 0].legend()
    group_values = returns.filter((pl.col("split") == "val_select") & (pl.col("horizon_ns") == 180_000_000_000) & (pl.col("sampling") == "all")).group_by("group").agg(pl.col("mean_return_bps").mean()).sort("group")
    axes[1, 1].bar(group_values["group"], group_values["mean_return_bps"])
    axes[1, 1].set(title="Frozen-training groups, val_select 180s", ylabel="Future mid return, bps")
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
