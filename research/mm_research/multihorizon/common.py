"""共享期限、冻结配置与有界矩阵读取。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from mm_research.data.storage import write_json_atomic

HORIZONS = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300])
SECOND = 1_000_000_000
DAY = 86400


# 【非核心】只对计划要求冻结的小型清单计算摘要，不重复扫描原始行情文件。
def definition_digest(value: dict) -> str:
    """输入JSON定义，返回稳定SHA256作为配置与分区身份。"""
    content = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(content).hexdigest()


# 【非核心】从明确配置载入独立研究版本，所有数据路径相对当前项目根目录。
def context(filename: Path) -> tuple[Path, dict, Path]:
    """返回根目录、冻结参数和产物目录，配置变化必须使用新的版本。"""
    root = Path.cwd()
    config = yaml.safe_load(filename.read_text())
    output = root / config['output']
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'config.json'
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        raise ValueError('研究配置变化，需要新的输出版本')
    if not frozen.exists():
        write_json_atomic(frozen, config)
    return root, config, output


# 【非核心】只读映射完整时序，按所需位置拷贝小块，避免整表扩展到Float64。
def open_dataset(output: Path) -> tuple:
    """返回清单、分区、原始输入、Float64标签及市场状态矩阵。"""
    folder = output / 'e0'
    manifest = json.loads((folder / 'manifest.json').read_text())
    splits = json.loads((folder / 'splits.json').read_text())
    matrix = np.load(folder / 'x.npy', mmap_mode='r')
    targets = np.load(folder / 'y.npy', mmap_mode='r')
    market = np.load(folder / 'market.npy', mmap_mode='r')
    return manifest, splits, matrix, targets, market


# 【核心逻辑256】将输入缺失填入本折训练均值并附加缺失标记，统计量不接触预测折。
def model_inputs(matrix, rows: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """返回有界Float32批次，标准化后截断±20并附加同宽缺失通道。"""
    values = np.array(matrix[rows], dtype=np.float32)
    missing = ~np.isfinite(values)
    values = np.where(missing, mean, values)
    values = np.clip((values - mean) / scale, -20, 20)
    return np.concatenate([values, missing.astype(np.float32)], axis=1)
