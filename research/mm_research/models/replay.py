"""真实L2策略回测与逐行数值核对，研究预测与交易账户结果分开保存。"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
import yaml

from mm_research.config import load_pipeline_config
from mm_research.data.engine_inputs import prepare_engine_day
from mm_research.data.storage import file_locks, scan_series, write_json_atomic
from mm_research.evaluation.trading import analyze_run
from mm_research.factors import gen001, gen002
from mm_research.features.expressions import normalized_imbalance
from mm_research.features.trades import trade_expressions
from mm_research.models.dataset import RAW_COLUMNS


# 【核心逻辑183】从实时记录重新计算原始压力及因子，独立核对C++系数推理。
def audit_realtime_signal(samples: Path, model: dict, lot_size: float) -> dict:
    """同一原生行情输入用Polars重新计算，核对窗口、预热、null和最终bps预测。"""
    floats = ["mid_price", "buy_volume", "sell_volume", "prediction_bps", *RAW_COLUMNS]
    floats += [f"{side}_depth_{depth}" for side in ["bid", "ask"] for depth in [1, 5, 10]]
    floats += [f"input_{name}" for name in model["columns"]]
    schema = {column: pl.Float64 for column in floats}
    schema.update(timestamp_ns=pl.Int64, book_valid=pl.Int8, flow_valid=pl.Int8, inference_update=pl.Int8)
    actual = pl.read_csv(samples, schema_overrides=schema).fill_nan(None)
    # 与正式研究列一样按整数Lot累计，已知空成交窗严格为零。
    trade_1s = trade_expressions(lot_size, {"1s": 8, "5s": 40})["trade_imbalance_1s"]
    reference = actual.select(
        "timestamp_ns",
        *[normalized_imbalance(pl.col(f"bid_depth_{depth}"), pl.col(f"ask_depth_{depth}"), empty_value=None).alias(f"imbalance_{depth}")
          for depth in [1, 5, 10]],
        normalized_imbalance(pl.col("buy_volume"), pl.col("sell_volume"), empty_value=0).alias("trade_imbalance_100ms"),
        trade_1s.alias("trade_imbalance_1s"),
        ((pl.col("mid_price") / pl.col("mid_price").shift(10) - 1) * 10_000).alias("return_1s_bps"),
    )
    expressions = {**gen001.factor_expressions({"min_fraction": model["min_fraction"]}), **gen002.factor_expressions({"min_fraction": model["min_fraction"]})}
    reference = reference.with_columns([
        expressions[name].alias(name) for name in model["columns"] if name not in RAW_COLUMNS
    ])
    errors = {}
    for name in model["columns"]:
        expected = reference[name].to_numpy()
        update = actual["inference_update"].to_numpy().astype(bool)
        observed = actual[f"input_{name}"].to_numpy()[update]
        expected = expected[update]
        np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-10, equal_nan=True)
        valid = np.isfinite(expected)
        errors[name] = {"valid_rows": int(valid.sum()), "max_absolute_error": float(np.max(np.abs(observed[valid] - expected[valid])))}
    expected_prediction = reference.select(model["columns"]).to_numpy() @ np.asarray(model["coefficients"]) + model["intercept"]
    update = actual["inference_update"].to_numpy().astype(bool)
    indices = np.maximum.accumulate(np.where(update, np.arange(len(update)), -1))
    held_prediction = expected_prediction[np.maximum(indices, 0)]
    valid_state = actual["book_valid"].to_numpy().astype(bool) & actual["flow_valid"].to_numpy().astype(bool)
    expected_prediction = np.where((indices >= 0) & valid_state, held_prediction, np.nan)
    actual_prediction = actual["prediction_bps"].to_numpy()
    np.testing.assert_allclose(actual_prediction, expected_prediction, rtol=1e-8, atol=1e-8, equal_nan=True)
    return {"rows": actual.height, "feature_errors": errors,
            "valid_predictions": int(np.isfinite(actual_prediction).sum()),
            "max_prediction_error_bps": float(np.nanmax(np.abs(actual_prediction - expected_prediction)))}


# 【非核心】量化L2实时状态与研究快照输入的来源差异，保存实际一致率。
def compare_research_inputs(root: Path, samples: Path, model_config: dict, date: str) -> dict:
    """计算定义对照之外，单独核对两份来源在同一20ms时钟下的观测差异。"""
    columns = [*RAW_COLUMNS, "mid_price", "buy_volume", "sell_volume"]
    folder = root / model_config["feature_root"] / model_config["feature_version"]
    with file_locks([folder / f"{column}.parquet" for column in columns], shared=True):
        research, _ = scan_series(folder, columns, [date])
        research = research.select("timestamp_ns", *columns).collect(engine="streaming")
    current = pl.read_csv(samples, schema_overrides={column: pl.Float64 for column in columns}).select("timestamp_ns", *columns).fill_nan(None)
    paired = research.join(current, on="timestamp_ns", how="inner", suffix="_replay")
    results = {}
    for column in columns:
        expected = paired[column].to_numpy()
        observed = paired[f"{column}_replay"].to_numpy()
        valid = np.isfinite(expected) & np.isfinite(observed)
        differences = np.abs(expected[valid] - observed[valid])
        results[column] = {"paired_rows": int(valid.sum()),
                           "equal_ratio": float(np.isclose(expected[valid], observed[valid], rtol=1e-9, atol=1e-10).mean()),
                           "max_absolute_difference": float(differences.max())}
    return results


# 【核心逻辑141】在最终测试日对照原生固定报价与实时线性模型报价，并独立核对账本。
def run_linear_replay(config_path: Path) -> dict:
    """使用同一完整L2日、官方成交、费率、延迟和仓位预算，模型系数保持冻结。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "linear_replay")
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "completed.json"
    model_root = root / config["model_output"]
    model = json.loads((model_root / "models.json").read_text())["realtime"]
    identity = {"config": config, "model": model,
                "strategies": {"baseline": yaml.safe_load((root / config["baseline_config"]).read_text()),
                               "alpha": yaml.safe_load((root / config["alpha_config"]).read_text())}}
    run_identity = output / "run_identity.json"
    if run_identity.exists() and json.loads(run_identity.read_text()) != identity:
        raise ValueError("回测模型或配置变化，需要新的输出版本")
    write_json_atomic(run_identity, identity)
    if completed.exists():
        return json.loads((output / "report.json").read_text())
    manifest = prepare_engine_day(root / config["book"], root / config["trades"], config["date"],
                                  root / config["engine_output"], config["market_data_latency_ns"])
    subprocess.run(["cmake", "--build", "workspace/build/release", "--target", "baseline_mm", "linear_mm", "replay_probe", "-j", "2"], cwd=root, check=True)
    probe_config = output / "replay_probe.yaml"
    probe_config.write_text(yaml.safe_dump({
        "source": str(root / config["engine_output"] / "manifest.json"), "output": str(output / "replay_probe.json"),
        "date": config["date"], "symbol": "BTCUSDT", "tick_size": manifest["tick_size"], "lot_size": manifest["lot_size"],
        "depth_levels": 10, "snapshot_interval_ns": 100_000_000, "latency_sample_every": 0,
    }))
    if not (output / "replay_probe.json").exists():
        subprocess.run([str(root / "workspace/build/release/replay_probe"), str(probe_config)], cwd=root, check=True)
    subprocess.run([sys.executable, "scripts/validate_replay.py", str(root / config["engine_output"] / "manifest.json"),
                    str(output / "replay_probe.json")], cwd=root, check=True)
    summaries = {}
    for case, binary, configuration in [("baseline", "baseline_mm", config["baseline_config"]), ("alpha", "linear_mm", config["alpha_config"])]:
        strategy = identity["strategies"][case]
        directory = root / strategy["output"]
        command = [str(root / "workspace/build/release" / binary), configuration]
        if case == "alpha":
            command += [str(model_root / "realtime_model.txt"), str(output / "realtime_samples.csv")]
        if not (directory / "summary.json").exists():
            subprocess.run(command, cwd=root, check=True)
        summaries[case] = analyze_run(directory, manifest)
    parity = audit_realtime_signal(output / "realtime_samples.csv", model, manifest["lot_size"])
    model_config = json.loads((model_root / "report.json").read_text())["config"]
    source_comparison = compare_research_inputs(root, output / "realtime_samples.csv", model_config, config["date"])
    report = {"date": config["date"], "fixed_market_latency_ns": config["market_data_latency_ns"],
              "model_group": "realtime", "max_quote_shift_bps": 1.0, "strategy_results": summaries,
              "realtime_polars_parity": parity, "l2_snapshot_source_comparison": source_comparison,
              "pnl_difference_usdt": summaries["alpha"]["net_pnl"] - summaries["baseline"]["net_pnl"]}
    write_json_atomic(output / "report.json", report)
    write_json_atomic(completed, identity)
    return report
