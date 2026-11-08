"""读取共同因子与基线特征，按UTC时间划分且只在训练期选择候选。"""

from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import KEY_COLUMNS, file_locks, scan_series
from mm_research.evaluation.dataset import load_evaluation_targets, partition_sample_mask
from mm_research.models.linear import linear_statistics
from mm_research.factors.dataset import factor_module

RAW_COLUMNS = ["imbalance_1", "imbalance_5", "imbalance_10", "trade_imbalance_100ms", "trade_imbalance_1s", "return_1s_bps"]


# 【核心逻辑121】从训练期IC选择每组候选，并建立可实时移植的压力模型对照。
def select_model_columns(root: Path, config: dict) -> tuple[dict, dict]:
    """训练IC只决定输入列，验证期负责选模型参数，最终测试期保持封存。"""
    summary = pl.read_parquet(root / config["evaluation_root"] / "summary.parquet").filter(
        (pl.col("split") == "train") & (pl.col("sampling") == "nonoverlap")
        & (pl.col("horizon_ns") == 1_000_000_000) & pl.col("ic_mean").is_finite(),
    ).with_columns(pl.col("ic_mean").abs().alias("abs_ic"))
    groups = {}
    selected = []
    pressure_candidates = []
    for batch in config["batches"]:
        names = list(factor_module(batch).factor_expressions({"min_fraction": config["min_fraction"], "tick_size": 0.1}))
        ranked = summary.filter(pl.col("factor").is_in(names)).sort("abs_ic", descending=True)
        candidates = ranked.head(config["factors_per_batch"])["factor"].to_list()
        groups[batch] = candidates
        selected.extend(candidates)
        if batch in ["gen001", "gen002"]:
            pressure_candidates.extend(ranked.head(config["realtime_per_batch"])["factor"].to_list())
    return {
        "history": ["return_1s_bps"], "raw": RAW_COLUMNS,
        "factors": [*RAW_COLUMNS, *selected], "realtime": [*RAW_COLUMNS, *pressure_candidates],
    }, {"criterion": "absolute_train_nonoverlap_mean_daily_1s_ic", "selected_by_batch": groups,
        "train_end": config["train_end"], "realtime_factors": pressure_candidates}


# 【核心逻辑123】按时间键关联所选Features、Factors和标签，统一共同有效样本。
def load_model_day(root: Path, config: dict, date: str, columns: list[str], stride_ns: int) -> pl.DataFrame:
    """按固定绝对时点抽样，输出有序X/y，缺失值通过明确共同样本筛选处理。"""
    features = root / config["feature_root"] / config["feature_version"]
    factors = root / config["factor_root"] / config["factor_version"]
    raw_columns = [column for column in columns if column in RAW_COLUMNS]
    factor_columns = [column for column in columns if column not in RAW_COLUMNS]
    files = [features / f"{column}.parquet" for column in raw_columns]
    files += [factors / f"{column}.parquet" for column in factor_columns]
    labels = root / config["label_root"] / config["label_version"]
    with file_locks(files, shared=True):
        table, dataset = scan_series(features, raw_columns, [date])
        table = table.filter(pl.col("timestamp_ns") % stride_ns == 0)
        if factor_columns:
            values, factor_dataset = scan_series(factors, factor_columns, [date])
            if dataset != factor_dataset:
                raise ValueError("模型Features与Factors数据定义不同")
            table = table.join(values.select("timestamp_ns", *factor_columns), on="timestamp_ns", how="left", maintain_order="left")
        target = load_evaluation_targets(labels, date, 1_000_000_000, dataset).rename({"target": "y_mid_1s_bps"}).lazy()
        table = table.join(target, on="timestamp_ns", how="left", maintain_order="left")
        valid = pl.col("label_valid") & pl.all_horizontal([pl.col(column).is_finite() for column in columns])
        table = table.filter(valid & partition_sample_mask(date, config))
        return table.select(*KEY_COLUMNS, *columns, "y_mid_1s_bps").collect(engine="streaming")


# 【核心逻辑122】将一日共同样本归纳为可合并统计，进程间仅传递小型矩阵。
def model_day_statistics(task: tuple) -> tuple:
    """返回日期与充分统计量，训练阶段固定1秒间隔，避免重叠目标重复加权。"""
    root, config, date, columns = task
    table = load_model_day(root, config, date, columns, config["sample_stride_ns"])
    matrix = table.select(columns).to_numpy()
    target = table["y_mid_1s_bps"].to_numpy()
    return date, linear_statistics(matrix, target)


# 【核心逻辑128】按冻结模型生成测试期整秒预测，保留各基线共同样本。
def predict_model_day(task: tuple) -> dict:
    """保存逐日Parquet，预测只使用当前和历史输入，输出单位统一bps。"""
    root, config, date, columns, models = task
    table = load_model_day(root, config, date, columns, config["sample_stride_ns"])
    result = table.select(*KEY_COLUMNS, "y_mid_1s_bps")
    for name, model in models.items():
        matrix = table.select(model["columns"]).to_numpy()
        predictions = matrix @ np.asarray(model["coefficients"]) + model["intercept"]
        result = result.with_columns(pl.Series(f"prediction_{name}_bps", predictions))
    folder = root / config["output"] / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    result.write_parquet(folder / f"{date}.parquet", compression="zstd")
    return {"date": date, "rows": result.height}
