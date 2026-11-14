"""费用分类模型的同样本配对评价，概率从冻结净价值及训练盈亏幅度还原。"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context


# 【核心逻辑402】按相同未来标签掩码比较冻结模型，仅评价配置明确指定的开发分区。
def evaluate_fee_models(filename: Path) -> None:
    """输出总体、逐日、概率桶及按周成块抽样的误差增量，交易损益由独立账本计算。"""
    from mm_research.models.cost_classifier import net_labels

    _, config, output = context(filename)
    source = Path(config["source"])
    manifest = json.loads((source / "e0/manifest.json").read_text())
    segments = json.loads((source / "e0/splits.json").read_text())["segments"]
    bounds = segments[config["split"]]
    rows = np.arange(bounds["start_row"], bounds["stop_row"])
    market = np.load(source / "e0/market.npy", mmap_mode="r")
    day_numbers = rows // 86400
    daily = []
    overall = []
    calibration = []
    for side in [1, -1]:
        returns, valid = net_labels(market, rows, config["horizon_s"], side)
        valid &= rows + config["horizon_s"] < bounds["stop_row"]
        actual = (returns[valid] > 0).astype("float64")
        days = day_numbers[valid]
        for group, folder in config["models"].items():
            model = Path(folder) / "models" / f'h{config["horizon_s"]}_side{side}'
            metadata = json.loads((model / "completed.json").read_text())
            edge = np.load(model / f'{config["split"]}_net_edge_bps.npy', mmap_mode="r")
            payoff_range = metadata["train_win_mean_bps"] - metadata["train_loss_mean_bps"]
            probability = (edge[valid] - metadata["train_loss_mean_bps"]) / payoff_range
            if not np.isfinite(probability).all():
                raise ValueError(f"模型概率存在非有限值: {group}, {side}")
            probability = np.clip(probability, 0., 1.)
            overall.append({"group": group, "side": side, **binary_scores(actual, probability)})
            for day in np.unique(days):
                mask = days == day
                daily.append({"group": group, "side": side, "date": manifest["dates"][int(day)],
                              **binary_scores(actual[mask], probability[mask])})
            bins = np.minimum((probability * 10).astype("int64"), 9)
            for bucket in range(10):
                mask = bins == bucket
                count = int(mask.sum())
                calibration.append({"group": group, "side": side, "bin_lower": bucket / 10,
                                    "count": count, "mean_probability": float(probability[mask].mean()) if count else None,
                                    "observed_rate": float(actual[mask].mean()) if count else None})
    table = pl.DataFrame(daily)
    baseline = table.filter(pl.col("group") == config["baseline"]).select(
        "side", "date", *[pl.col(name).alias(f"baseline_{name}") for name in ["logloss", "brier", "auc"]],
    )
    table = table.join(baseline, on=["side", "date"], how="left").with_columns([
        (pl.col(name) - pl.col(f"baseline_{name}")).alias(f"delta_{name}") for name in ["logloss", "brier", "auc"]
    ])
    table.sort(["group", "side", "date"]).write_csv(output / "daily.csv")
    pl.DataFrame(overall).write_csv(output / "overall.csv")
    pl.DataFrame(calibration).write_csv(output / "calibration.csv")
    intervals = weekly_error_intervals(table, config["baseline"], config["bootstrap_repeats"], config["seed"])
    write_json_atomic(output / "completed.json", {
        "status": "success", "split": config["split"], "horizon_s": config["horizon_s"],
        "probability": "recovered_from_calibrated_edge_and_frozen_train_payoffs",
        "sample_mask": "shared_price_quality_and_label_end_inside_split_no_feature_deletion",
        "label": "positive_bid_ask_return_after_5bps_each_side_before_funding",
        "paired_week_bootstrap": intervals,
    })


# 【核心逻辑403】在同一二分类样本上计算概率误差及含并列值的精确AUC。
def binary_scores(actual: np.ndarray, probability: np.ndarray) -> dict:
    """类别单一时AUC为空，LogLoss和Brier仍按全部有效样本计算。"""
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    positive = actual > .5
    wins = int(positive.sum())
    losses = len(actual) - wins
    auc = None
    if wins and losses:
        ranks = pl.Series(probability).rank(method="average").to_numpy()
        auc = float((ranks[positive].sum() - wins * (wins + 1) / 2) / (wins * losses))
    return {
        "count": len(actual), "positive_rate": float(actual.mean()),
        "logloss": float(-(actual * np.log(clipped) + (1 - actual) * np.log1p(-clipped)).mean()),
        "brier": float(np.mean((probability - actual) ** 2)), "auc": auc,
    }


# 【核心逻辑404】对同日配对误差按UTC周成块重采样，保留重叠1800秒标签的组内相关性。
def weekly_error_intervals(daily: pl.DataFrame, baseline: str, repeats: int, seed: int) -> list[dict]:
    """对LogLoss/Brier报告加权均值和95%区间，不把秒级样本当作独立交易日。"""
    table = daily.filter(pl.col("group") != baseline).with_columns(
        pl.col("date").str.to_date().dt.truncate("1w").alias("week"),
    ).group_by("group", "side", "week").agg(
        pl.col("count").sum(),
        (pl.col("delta_logloss") * pl.col("count")).sum().alias("logloss_sum"),
        (pl.col("delta_brier") * pl.col("count")).sum().alias("brier_sum"),
    )
    records = []
    generator = np.random.default_rng(seed)
    for group, side in sorted(table.select("group", "side").unique().iter_rows()):
        sample = table.filter((pl.col("group") == group) & (pl.col("side") == side)).sort("week")
        selected = generator.integers(0, sample.height, size=(repeats, sample.height))
        weights = sample["count"].to_numpy()
        for metric in ["logloss", "brier"]:
            sums = sample[f"{metric}_sum"].to_numpy()
            estimates = sums[selected].sum(axis=1) / weights[selected].sum(axis=1)
            lower, upper = np.quantile(estimates, [.025, .975])
            records.append({"group": group, "side": side, "metric": metric, "weeks": sample.height,
                            "delta": float(sums.sum() / weights.sum()), "lower_95": float(lower), "upper_95": float(upper)})
    return records
