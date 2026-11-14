"""使用冻结的家族均衡180秒树权重生成历史复核方向预测，不读取任何标签。"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from mm_research.models.input_comparison import load_input_group
from mm_research.multihorizon.common import HORIZONS, SECOND


# 【核心逻辑417】严格复用冻结输入和树权重，按完整复核网格预测，不拟合、不读取未来标签。
def main() -> None:
    """固定本轮已选择模型；输出两列timestamp_ns与prediction_log_bps供入场确认读取。"""
    import cupy as cp
    import xgboost as xgb

    filename = Path('config/models/feature_model_family_balanced_20261114.yaml')
    config = yaml.safe_load(filename.read_text())
    source = Path(config['source'])
    output = Path(config['output'])
    folder = output / 'models/180s_combined_xgb_seed17'
    destination = folder / 'val_select.parquet'
    if destination.exists():
        raise FileExistsError('冻结复核预测已存在，不覆盖历史产物')
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments']['val_select']
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    preprocessing = np.load(source / 'e1/folds/final/preprocessing.npz')
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    mean = float(preprocessing['target_mean'][target_index])
    scale = float(preprocessing['target_scale'][target_index])
    inputs = [load_input_group(config, output, 180, part, 'val_select') for part in ['old', 'new']]
    model = xgb.Booster(model_file=str(folder / 'model.ubj'))
    model.set_param({'device': 'cuda', 'nthread': 2})
    count = bounds['stop_row'] - bounds['start_row']
    prediction = np.empty(count, dtype='float64')
    for begin in range(0, count, 65536):
        stop = min(count, begin + 65536)
        batch = cp.asarray(np.concatenate([matrix[begin:stop] for matrix in inputs], axis=1))
        prediction[begin:stop] = cp.asnumpy(model.inplace_predict(batch)) * scale + mean
    rows = np.arange(bounds['start_row'], bounds['stop_row'], dtype='int64')
    frame = pl.DataFrame({
        'timestamp_ns': manifest['start_timestamp_ns'] + rows * SECOND,
        'prediction_log_bps': prediction,
    })
    temporary = destination.with_suffix('.pending.parquet')
    frame.write_parquet(temporary)
    temporary.replace(destination)
    print(f'冻结复核预测已生成：{count}行', flush=True)


if __name__ == '__main__':
    main()
