"""直接预测费用后交易价值，比较稳健损失与训练期截尾目标。"""

import fcntl
from pathlib import Path
import json

import numpy as np

from mm_research.data.storage import write_json_atomic
from mm_research.models.cost_classifier import net_labels
from mm_research.models.intraday import input_parts
from mm_research.multihorizon.common import context


# 【核心逻辑383】用费用后连续收益训练价值模型，稳健处理只影响训练目标，评估保留全部行情。
def train_net_value(filename: Path, horizon: int, side: int, family: str) -> None:
    """沿用冻结输入与时间分区，输出各分区的方向性稳健净价值bps。"""
    import cupy as cp
    import xgboost as xgb
    from mm_research.models.tree180 import GpuBatchIterator

    _, config, output = context(filename)
    folder = output / 'models' / f'{family}_h{horizon}_side{side}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completed.json').exists():
        return
    source = Path(config['source'])
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    row_sets = {split: np.load(source / f'e1/folds/final/{split}_rows.npy') for split in ['fit', 'tune']}
    targets = {}
    positions = {}
    for split in ['fit', 'tune']:
        values, valid = net_labels(market, row_sets[split], horizon, side)
        boundary = int(row_sets['tune'][0]) if split == 'fit' else segments['train']['stop_row']
        chosen = valid & (row_sets[split] + horizon < boundary)
        positions[split] = np.flatnonzero(chosen)
        targets[split] = values[chosen]
    center = float(targets['fit'].mean())
    scale = float(targets['fit'].std())
    with Path('/tmp/hft_mm_research_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        matrices = {}
        temporary_files = []
        for split in ['fit', 'tune']:
            parts = input_parts(config, Path(config['macro_source']), split)
            selected = positions[split]
            temporary = Path('/tmp') / f'hft_net_value_{split}.npy'
            values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32',
                shape=(len(selected), sum(part.shape[1] for part in parts)))
            for begin in range(0, len(selected), 65536):
                rows = selected[begin:begin + 65536]
                values[begin:begin + len(rows)] = np.concatenate([part[rows] for part in parts], axis=1)
            values.flush()
            labels = (targets[split] - center) / scale
            if family == 'clipped' and split == 'fit':
                labels = np.clip(labels, -3., 3.)
            reference = matrices['fit'] if split == 'tune' else None
            matrices[split] = xgb.QuantileDMatrix(
                GpuBatchIterator(values, labels.astype('float32')), max_bin=64, ref=reference, nthread=2,
            )
            temporary_files.append(temporary)
            del values
            cp.get_default_memory_pool().free_all_blocks()
        objective = {'huber': 'reg:pseudohubererror', 'clipped': 'reg:squarederror'}[family]
        model = xgb.train({
            'objective': objective, 'eval_metric': 'mae', 'tree_method': 'hist', 'device': 'cuda',
            'max_depth': 4, 'max_bin': 64, 'min_child_weight': 500, 'eta': .05,
            'subsample': .8, 'colsample_bytree': .8, 'lambda': 10, 'seed': 17, 'nthread': 2,
        }, matrices['fit'], num_boost_round=800, evals=[(matrices['tune'], 'tune')],
            early_stopping_rounds=50, verbose_eval=50)
        model = model[:model.best_iteration + 1]
        model.save_model(str(folder / 'model.ubj'))
        rounds = model.num_boosted_rounds()
        del matrices
        cp.get_default_memory_pool().free_all_blocks()
        for temporary in temporary_files:
            temporary.unlink()
        for split in ['tune', 'val_dev', 'val_select']:
            parts = input_parts(config, Path(config['macro_source']), split)
            prediction = np.empty(len(parts[0]), dtype='float64')
            for begin in range(0, len(prediction), 65536):
                stop = min(begin + 65536, len(prediction))
                values = cp.asarray(np.concatenate([part[begin:stop] for part in parts], axis=1))
                prediction[begin:stop] = cp.asnumpy(model.inplace_predict(values)) * scale + center
            np.save(folder / f'{split}_net_edge_bps.npy', prediction)
    write_json_atomic(folder / 'completed.json', {
        'status': 'success', 'family': family, 'horizon_s': horizon, 'side': side,
        'seed': 17, 'rounds': rounds, 'target_center_bps': center, 'target_scale_bps': scale,
        'fit_rows': len(positions['fit']), 'tune_rows': len(positions['tune']),
        'evaluation_labels': 'all_valid_original_returns_without_tail_deletion',
    })
