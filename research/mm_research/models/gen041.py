"""Gen041生产前的训练期阈值冻结，明确是否可复用现有小单基础列。"""

import json
from pathlib import Path

from mm_research.data.storage import write_json_atomic
from mm_research.features.trade_size import fit_size_thresholds
from mm_research.multihorizon.common import context


# 【核心逻辑397】按本轮拟合前缀重算小单阈值，禁止使用未来训练截止日的旧阈值估计。
def prepare_small_threshold(filename: Path) -> None:
    """输出训练期Q50及复用决定；只有数值完全一致才可读取现有小单量。"""
    import numpy as np

    _, config, output = context(filename)
    marker = output / "small_threshold.json"
    if marker.exists():
        return
    source = Path(config["source"])
    manifest = json.loads((source / "e0/manifest.json").read_text())
    tune_rows = np.load(source / "e1/folds/final/tune_rows.npy", mmap_mode="r")
    # 用内部校准开始前的完整日拟合，连校准日也不参与阈值估计。
    dates = manifest["dates"][:int(tune_rows[0]) // 86400]
    threshold = fit_size_thresholds(Path(config["standard_folder"]), dates, config["lot_size"])
    existing = json.loads((Path(config["feature_folder"]) / "trade_size_thresholds.json").read_text())
    write_json_atomic(marker, {
        "fit": threshold, "existing": existing,
        "reuse_existing_small_columns": threshold["q50"] == existing["q50"],
        "rule": "quantity_strictly_less_than_fit_q50",
    })
    print("GEN041_THRESHOLD_READY", threshold["q50"], threshold["train_end"], flush=True)
