"""组织逐日单因子评估，并生成可检查的Parquet表格和HTML图表报告。"""

from html import escape
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.storage import file_locks, read_metadata
from mm_research.evaluation.dataset import load_evaluation_day
from mm_research.evaluation.metrics import correlation_statistics, distribution_statistics, grouped_returns
from mm_research.targets.midprice import horizon_name


# 【核心逻辑29】逐因子逐日评估，按周期汇总等权逐日IC和分组收益。
def run_evaluation(config_path: Path) -> dict:
    """输入Evaluation配置，保存指标表格、图和HTML，返回汇总表及耗时。

    同时只物化一个因子日与周期，完整因子列仅用于自身分布与直方图。
    """
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "evaluation")
    if config["groups"] < 2:
        raise ValueError("分组数至少为2")
    factors = root / config["factor_root"] / config["factor_version"]
    labels = root / config["label_root"] / config["label_version"]
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    daily_distributions = []
    daily_correlations = []
    group_tables = []
    overall = []
    histograms = {}
    source_definitions = {}
    for name in config["factors"]:
        filename = factors / f"{name}.parquet"
        # 同一因子的共享锁覆盖全部评估，防止日期之间切换到新文件。
        with file_locks([filename], shared=True):
            metadata = read_metadata(filename)
            source_definitions[name] = metadata["definition"]
            if not set(config["dates"]) <= set(metadata["definition"]["dates"]):
                raise ValueError(f"评估日期超出Factor范围 {name}")
            for date in config["dates"]:
                for index, horizon in enumerate(config["horizons_ns"]):
                    table = load_evaluation_day(filename, labels, date, horizon)
                    identity = {"factor": name, "date": date, "horizon_ns": horizon}
                    if index == 0:
                        daily_distributions.append({"factor": name, "date": date, **distribution_statistics(table)})
                    daily_correlations.append({**identity, **correlation_statistics(table)})
                    returns = grouped_returns(table, config["groups"]).with_columns(
                        pl.lit(name).alias("factor"), pl.lit(date).alias("date"),
                        pl.lit(horizon).alias("horizon_ns"),
                    )
                    group_tables.append(returns)
            values = pl.read_parquet(filename, columns=["date", name]).filter(
                pl.col("date").is_in(config["dates"])
            ).select(pl.col(name).alias("value"))
            overall.append({"factor": name, **distribution_statistics(values)})
            numbers = values["value"].to_numpy()
            histograms[name] = np.histogram(numbers[np.isfinite(numbers)], bins=80)
        print(f"评估完成 {name}", flush=True)
    daily_ic = pl.DataFrame(daily_correlations, infer_schema_length=None,
                           schema_overrides={"ic": pl.Float64, "rank_ic": pl.Float64})
    distributions = pl.DataFrame(daily_distributions, infer_schema_length=None)
    returns = pl.concat(group_tables, how="vertical_relaxed")
    summary = daily_ic.group_by("factor", "horizon_ns").agg(
        pl.col("ic").count().alias("valid_ic_days"),
        pl.col("ic").mean().alias("ic_mean"), pl.col("ic").std().alias("ic_std"),
        pl.col("rank_ic").mean().alias("rank_ic_mean"),
        (pl.col("ic") > 0).mean().alias("positive_ic_day_ratio"),
        pl.col("paired_rows").sum().alias("paired_rows"),
    )
    spreads = returns.filter(pl.col("actual_groups") == config["groups"]).group_by("factor", "date", "horizon_ns").agg(
        (pl.col("mean_return_bps").filter(pl.col("group") == config["groups"]).first()
         - pl.col("mean_return_bps").filter(pl.col("group") == 1).first()).alias("spread_bps"),
    ).group_by("factor", "horizon_ns").agg(
        pl.col("spread_bps").mean().alias("top_bottom_bps"), pl.len().alias("complete_group_days"),
    )
    summary = summary.join(spreads, on=["factor", "horizon_ns"], how="left")
    summary = summary.join(pl.DataFrame(overall, infer_schema_length=None), on="factor").sort("factor", "horizon_ns")
    for filename, table in [("summary", summary), ("daily_distribution", distributions),
                            ("daily_ic", daily_ic), ("group_returns", returns)]:
        table.write_parquet(
            output / f"{filename}.parquet", compression="zstd",
            metadata={"mm_evaluation": json.dumps({"config": config, "factors": source_definitions}, sort_keys=True)},
        )
    render_report(output, config, summary, daily_ic, returns, histograms)
    return {"elapsed_seconds": time.perf_counter() - started, "summary": summary}


# 【非核心】把指标表和四类诊断图组织为一个可在浏览器查看的报告。
def render_report(output: Path, config: dict, summary: pl.DataFrame, daily_ic: pl.DataFrame,
                  returns: pl.DataFrame, histograms: dict) -> None:
    """输入已计算指标和直方图，输出PNG图与HTML，绘图使用已保存的指标。"""
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    sections = []
    for name in config["factors"]:
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        figure.suptitle(name, fontsize=11)
        counts, edges = histograms[name]
        axes[0, 0].bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#3b729f")
        axes[0, 0].set(title="Factor distribution", xlabel="Factor value", ylabel="Samples")
        day_values = daily_ic.filter((pl.col("factor") == name) & (pl.col("horizon_ns") == 1_000_000_000)).sort("date")
        axes[0, 1].plot(day_values["date"], day_values["ic"], label="Pearson IC")
        axes[0, 1].plot(day_values["date"], day_values["rank_ic"], label="Rank IC", alpha=0.7)
        axes[0, 1].axhline(0, color="gray", linewidth=0.6)
        axes[0, 1].set(title="Daily IC, 1s", ylabel="IC")
        axes[0, 1].tick_params(axis="x", rotation=65, labelsize=6)
        axes[0, 1].legend(fontsize=8)
        horizons = summary.filter(pl.col("factor") == name).sort("horizon_ns")
        axes[1, 0].bar([horizon_name(value) for value in horizons["horizon_ns"]],
                       np.asarray(horizons["ic_mean"].to_numpy(), dtype=float), color="#3b729f")
        axes[1, 0].set(title="Mean daily IC by horizon", ylabel="Pearson IC")
        groups = returns.filter(
            (pl.col("factor") == name) & (pl.col("horizon_ns") == 1_000_000_000)
            & (pl.col("actual_groups") == config["groups"])
        ).group_by("group").agg(pl.col("mean_return_bps").mean()).sort("group")
        axes[1, 1].bar(groups["group"], groups["mean_return_bps"], color="#599c88")
        axes[1, 1].set(title="Equal-day group returns, 1s", xlabel="Group", ylabel="Future return (bps)")
        figure.savefig(figures / f"{name}.png", dpi=140)
        plt.close(figure)
        sections.append(f'<section><h2>{escape(name)}</h2><img src="figures/{escape(name)}.png" alt="因子诊断图"></section>')
    columns = ["factor", "horizon_ns", "ic_mean", "rank_ic_mean", "top_bottom_bps", "valid_ic_days", "missing_rate", "zero_rate"]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rows = []
    for row in summary.select(columns).iter_rows():
        cells = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row]
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>")
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Gen001 因子评估</title>
<style>body{max-width:1400px;margin:30px auto;padding:0 20px;font:15px/1.7 system-ui;color:#243447}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #dde3ea;padding:6px}img{width:100%}
section{margin:35px 0}h2{font-size:18px} .table{overflow:auto}</style><h1>Gen001 买卖压力持续水平</h1>"""
    content += f'<p>日期范围 {escape(str(config["dates"][0]))} 至 {escape(str(config["dates"][-1]))}。'
    content += "IC采用逐日等权汇总，零值率以有效因子样本为分母。收益单位为bps，分组差只汇总实际组数完整的日期。</p>"
    content += "<p>原始因子保留缺失值，滚动均值由配置的最少有效样本数决定。表中的分组收益反映未来Mid变化，交易执行成本另由策略回测计算。</p>"
    content += '<div class="table"><table><thead><tr>' + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    (output / "report.html").write_text(content + "".join(sections) + "</html>")
