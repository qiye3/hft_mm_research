"""服务器研究流水线，分阶段设置线程预算，记录进程退出状态与运行日志。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mm_research.config import load_pipeline_config
from mm_research.data.storage import write_json_atomic
from mm_research.features.trade_size import build_size_features


# 【非核心】用阶段独立线程预算执行命令，失败时保留日志并终止后续阶段。
def run_stage(root: Path, output: Path, name: str, command: list[str], threads: int) -> None:
    """GNU time记录阶段耗时和峰值内存，所有网络访问使用服务器自身出口。"""
    environment = os.environ.copy()
    for variable in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
        environment.pop(variable, None)
    environment.update(POLARS_MAX_THREADS=str(threads), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", NO_PROXY="*", no_proxy="*")
    print(f"阶段开始 {name}", flush=True)
    with (output / f"{name}.log").open("w") as log:
        subprocess.run(["/usr/bin/time", "-v", *command], cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f"阶段完成 {name}", flush=True)


# 【核心逻辑25】依次完成因子、评估、线性模型和真实事件回测，保留可恢复的阶段产物。
def run_research(config_path: Path) -> None:
    """读取各阶段所属配置，避免同一时间启动超出CPU和内存预算的计算。"""
    root = Path(__file__).resolve().parents[2]
    config = load_pipeline_config(config_path, "research")
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    feature_config = load_pipeline_config(root / config["feature_config"], "features")
    model_config = load_pipeline_config(root / config["model_config"], "models")
    train_dates = [date for date in feature_config["dates"] if date <= model_config["train_end"]]
    build_size_features(root, feature_config, train_dates, workers=2)
    python = sys.executable
    run_stage(root, output, "targets", [python, "scripts/build_labels.py", config["target_config"]], 2)
    run_stage(root, output, "factors", [python, "scripts/build_factors.py", config["factor_config"]], 5)
    run_stage(root, output, "evaluation", [python, "scripts/evaluate_factors.py", config["evaluation_config"]], 2)
    run_stage(root, output, "models", [python, "scripts/train_linear_baseline.py", config["model_config"]], 2)
    run_stage(root, output, "replay", [python, "scripts/run_linear_replay.py", config["replay_config"]], 5)
    write_json_atomic(output / "completed.json", {"completed_at_unix": time.time(), "config": config})
    print("RESEARCH_PIPELINE_COMPLETE", flush=True)
