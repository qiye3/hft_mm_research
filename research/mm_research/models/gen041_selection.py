"""仅用内部拟合期选择Gen041输入，复用E3预处理与缓存，不访问开发标签。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.factors.gen041 import FAMILIES
from mm_research.models.cross_market import publish_cache
from mm_research.multihorizon.common import context


# 【核心逻辑405】用内部fit的逐日分类标签秩相关筛选输入，再从E3缓存复制同一预处理列。
def prepare_selected_inputs(filename: Path) -> None:
    """先冻结选择清单，随后才读取各验证输入矩阵；没有候选时明确停止，不补入弱列。"""
    from mm_research.models.cost_classifier import net_labels

    _, config, output = context(filename)
    destination = output / "all/inputs/180s"
    if (destination / "completed.json").exists():
        return
    existing = Path(config["e3_inputs"]) / "all/inputs/180s"
    metadata = json.loads((existing / "completed.json").read_text())
    columns = metadata["columns"]
    source = Path(config["source"])
    rows = np.load(source / "e1/folds/final/fit_rows.npy")
    boundary = int(np.load(source / "e1/folds/final/tune_rows.npy", mmap_mode="r")[0])
    market = np.load(source / "e0/market.npy", mmap_mode="r")
    payoff, valid = net_labels(market, rows, config["horizon_s"], 1)
    valid &= rows + config["horizon_s"] < boundary
    matrix = np.load(existing / "fit_x.npy", mmap_mode="r")
    prefix = matrix.shape[1] - 2 * len(columns)
    values = {"day": rows[valid] // 86400, "target": (payoff[valid] > 0).astype(float)}
    for index, column in enumerate(columns):
        observed = matrix[valid, prefix + len(columns) + index] == 0
        values[column] = np.where(observed, matrix[valid, prefix + index], np.nan)
    frame = pl.DataFrame(values).with_columns(pl.col(columns).fill_nan(None))
    daily = frame.group_by("day").agg(
        pl.len().alias("samples"),
        *[pl.corr(column, "target", method="spearman").alias(column) for column in columns],
        *[pl.col(column).count().alias(f"{column}_count") for column in columns],
    ).sort("day")
    del frame, values, matrix
    ranking, selected = select_from_daily(daily, columns, config)
    daily.write_parquet(output / "fit_daily.parquet", compression="zstd")
    ranking.write_csv(output / "fit_ranking.csv")
    write_json_atomic(output / "selection.json", {
        "selected": selected,
        "fit_last_row": int(rows[valid][-1]),
        "label_end_before_row": boundary,
        "target": "positive_long_1800s_bid_ask_return_after_fees",
        "status": "selected" if selected else "no_candidates",
    })
    if not selected:
        raise ValueError("Gen041没有满足预定训练稳定性规则的列，E4停止，不自动放宽门槛")
    indices = [columns.index(column) for column in selected]
    chosen = [
        *range(prefix),
        *[prefix + index for index in indices],
        *[prefix + len(columns) + index for index in indices],
    ]
    destination.mkdir(parents=True, exist_ok=True)
    for split in ["fit", "tune", "val_dev", "val_select"]:
        original = np.load(existing / f"{split}_x.npy", mmap_mode="r")
        temporary = Path("/tmp") / f"hft_gen041_selected_{split}.npy"
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(original), len(chosen)))
        for begin in range(0, len(original), 65536):
            stop = min(begin + 65536, len(original))
            matrix[begin:stop] = original[begin:stop, chosen]
        matrix.flush()
        del matrix
        publish_cache(temporary, destination / f"{split}_x.npy")
    with np.load(existing / "preprocessing.npz") as preprocessing:
        np.savez(destination / "preprocessing.npz", mean=preprocessing["mean"][indices], scale=preprocessing["scale"][indices])
    write_json_atomic(destination / "completed.json", {
        "columns": selected,
        "original_input": str(existing),
        "status": "success",
        "lag_seconds": 1,
        "order": "old_macro_matrix_then_selected_values_then_selected_missing_flags",
    })


# 【核心逻辑406】按预设覆盖、同号比例与家族上限排序，不能根据开发或复核成绩更换筛选口径。
def select_from_daily(daily: pl.DataFrame, columns: list[str], config: dict) -> tuple[pl.DataFrame, list[str]]:
    """稳定性分数为平均秩相关绝对值乘同号日比例，最多12列且每类最多2列。"""
    records = []
    for column in columns:
        eligible = daily.filter(
            (pl.col("samples") >= config["minimum_daily_samples"])
            & (pl.col(f"{column}_count") >= config["minimum_coverage"] * pl.col("samples"))
            & pl.col(column).is_finite(),
        )[column].to_numpy()
        mean = float(eligible.mean()) if len(eligible) else None
        agreement = float(np.mean(eligible * mean > 0)) if len(eligible) else None
        family = next(name for name in FAMILIES if column.startswith(name + "_"))
        accepted = (
            len(eligible) >= config["minimum_day_fraction"] * daily.height
            and agreement >= config["minimum_sign_agreement"]
            and abs(mean) > 0
        )
        records.append({
            "factor": column,
            "family": family,
            "valid_days": len(eligible),
            "mean_rank_correlation": mean,
            "sign_agreement": agreement,
            "score": abs(mean) * agreement if len(eligible) else None,
            "eligible": accepted,
        })
    ranking = pl.DataFrame(records).sort(["score", "factor"], descending=[True, False], nulls_last=True)
    selected = []
    families = {}
    for record in ranking.filter(pl.col("eligible")).iter_rows(named=True):
        family = record["family"]
        count = families.get(family, 0)
        if count < config["maximum_per_family"]:
            selected.append(record["factor"])
            families[family] = count + 1
        if len(selected) == config["maximum_factors"]:
            break
    return ranking, selected
