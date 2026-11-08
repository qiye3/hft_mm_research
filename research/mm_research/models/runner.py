"""固定时间切分、线性模型组训练、验证选择和一次性测试报告。"""

import json
from pathlib import Path
import time

import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.execution import ordered_process_map
from mm_research.models.dataset import model_day_statistics, predict_model_day, select_model_columns
from mm_research.models.linear import fit_linear_model, merge_statistics, score_linear_model, subset_statistics


# 【核心逻辑120】仅从训练与验证结果选模型，再对冻结模型报告最终测试指标。
def run_linear_baseline(config_path: Path) -> dict:
    """输入模型配置，保存训练列、时间区间、标准化、系数、对照指标及测试预测。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "models")
    started = time.perf_counter()
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    groups, selection = select_model_columns(root, config)
    columns = sorted({column for names in groups.values() for column in names})
    tasks = [(root, config, date, columns) for date in config["dates"]]
    combined = {}
    daily = []
    for date, statistics in ordered_process_map(model_day_statistics, tasks, config["workers"]):
        if date <= config["train_end"]:
            split = "train"
        elif date <= config["validation_end"]:
            split = "validation"
        else:
            split = "test"
        if split in combined:
            combined[split] = merge_statistics(combined[split], statistics)
        else:
            combined[split] = statistics
        daily.append((date, split, statistics))
        print(f"模型统计完成 {date}", flush=True)
    candidates = []
    selected_models = {}
    for group, names in groups.items():
        indices = [columns.index(name) for name in names]
        train = subset_statistics(combined["train"], indices)
        validation = subset_statistics(combined["validation"], indices)
        models = []
        for family, alphas in [("ols", [0.0]), ("ridge", config["ridge_alphas"]), ("lasso", config["lasso_alphas"])]:
            for alpha in alphas:
                model = fit_linear_model(train, family, alpha)
                model.update(columns=names, group=group, train_end=config["train_end"],
                             validation_end=config["validation_end"], selection=selection,
                             sample_stride_ns=config["sample_stride_ns"], min_fraction=config["min_fraction"],
                             state_interval_ns=100_000_000, inference_interval_ns=1_000_000_000)
                validation_metrics = score_linear_model(validation, model)
                models.append((validation_metrics["mse_bps2"], model))
                candidates.append({"group": group, "family": family, "alpha": alpha,
                                   "train": score_linear_model(train, model), "validation": validation_metrics})
        _, selected = min(models, key=lambda item: item[0])
        selected_models[group] = selected
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
    (output / "models.json").write_text(json.dumps(selected_models, ensure_ascii=False, indent=2) + "\n")
    metrics = {}
    daily_metrics = []
    for group, model in selected_models.items():
        indices = [columns.index(name) for name in model["columns"]]
        metrics[group] = {split: score_linear_model(subset_statistics(statistics, indices), model)
                          for split, statistics in combined.items()}
        for date, split, statistics in daily:
            daily_metrics.append({"group": group, "date": date, "split": split,
                                  **score_linear_model(subset_statistics(statistics, indices), model)})
    pl.DataFrame(daily_metrics, infer_schema_length=None).write_parquet(output / "daily_metrics.parquet", compression="zstd")
    test_dates = [date for date in config["dates"] if date > config["validation_end"]]
    for result in ordered_process_map(predict_model_day, [(root, config, date, columns, selected_models) for date in test_dates], config["workers"]):
        print(f"测试预测完成 {result['date']}", flush=True)
    realtime = selected_models["realtime"]
    # C++使用原单位系数，训练期中心化与标准化已代数合并到系数和截距中。
    coefficients = [f"intercept {realtime['intercept']:.17g}",
                    f"min_fraction {config['min_fraction']:.17g}",
                    "state_interval_ns 100000000", "inference_interval_ns 1000000000"]
    coefficients += [f"{name} {coefficient:.17g}" for name, coefficient in zip(realtime["columns"], realtime["coefficients"])]
    (output / "realtime_model.txt").write_text("\n".join(coefficients) + "\n")
    report = {
        "config": config, "feature_selection": selection, "candidates": candidates,
        "selected_metrics_nonoverlap": metrics, "elapsed_seconds": time.perf_counter() - started,
        "evaluation_sampling_ns": config["sample_stride_ns"], "test_prediction_sampling_ns": config["sample_stride_ns"],
        "training_standardization": "training_only_centered_population_scale",
        "split_boundary": "cross_day_labels_300s_purge_and_label_end_before_boundary",
        "common_valid_samples": True,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
