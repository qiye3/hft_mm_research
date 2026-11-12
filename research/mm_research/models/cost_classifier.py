"""预测扣除双边费用后单边交易获利概率，以Train内部数据校准交易价值。"""

import fcntl
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit

from mm_research.data.storage import write_json_atomic
from mm_research.models.intraday import input_parts
from mm_research.multihorizon.common import context, SECOND


# 【非核心】根据当前与未来一档报价构造单边交易净收益，双边各收5bps。
def net_labels(market: np.ndarray, rows: np.ndarray, horizon: int, side: int) -> tuple:
    """标签使用研究可见的一档价格，最终策略另用到达时刻十档执行核算。"""
    future_rows = rows + horizon
    if side == 1:
        entry = np.asarray(market[rows, 2])
        exit_price = np.asarray(market[future_rows, 1])
    else:
        entry = np.asarray(market[rows, 1])
        exit_price = np.asarray(market[future_rows, 2])
    valid = (np.isfinite(entry) & np.isfinite(exit_price) & (entry > 0) & (exit_price > 0)
             & (market[rows, 8] > .5) & (market[future_rows, 8] > .5)
             & (market[rows, 9] == market[future_rows, 9]))
    values = np.full(len(rows), np.nan)
    ratio = exit_price[valid] / entry[valid]
    values[valid] = side * (ratio - 1) * 10000 - 5 * (1 + ratio)
    return values, valid


# 【非核心】在Train内部尾部校准概率，后续区间使用固定斜率和截距。
def probability_calibration(probability: np.ndarray, actual: np.ndarray) -> tuple:
    """返回Platt校准参数，采用稳定对数损失，不读取开发与复核标签。"""
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))

    # 【非核心】仅两个概率校准参数，轻微正则防止内部尾部少数样本造成无限斜率。
    def loss(parameters: np.ndarray) -> float:
        """返回二元对数损失及校准斜率正则项。"""
        scores = parameters[0] * logits + parameters[1]
        return float(np.mean(np.logaddexp(0, scores) - actual * scores) + 1e-4 * parameters[0] ** 2)

    fitted = minimize(loss, np.array([1., 0.]), method='L-BFGS-B')
    if not fitted.success:
        raise RuntimeError(f'概率校准未收敛 {fitted.message}')
    return float(fitted.x[0]), float(fitted.x[1])


# 【核心逻辑367】直接训练真实费用门槛下的交易获利分类器，训练与早停按目标周期隔离。
def train_cost_classifier(filename: Path, horizon: int, side: int, seed: int,
                          max_depth: int = 4, min_child_weight: float = 50.,
                          fit_lookback_days: int | None = None) -> None:
    """两个方向单独拟合，输出校准概率和由训练盈亏幅度换算的预期净收益。"""
    import cupy as cp
    import xgboost as xgb
    from mm_research.models.tree180 import GpuBatchIterator

    _, config, output = context(filename)
    folder = output / 'models' / f'h{horizon}_side{side}'
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
        selected = valid & (row_sets[split] + horizon < boundary)
        if split == 'fit' and fit_lookback_days is not None:
            selected &= row_sets[split] >= boundary - fit_lookback_days * 86400
        positions[split] = np.flatnonzero(selected)
        targets[split] = values[selected]
    with Path('/tmp/hft_mm_research_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        matrices = {}
        temporary_files = []
        for split in ['fit', 'tune']:
            parts = input_parts(config, Path(config['macro_source']), split)
            selected = positions[split]
            temporary = Path('/tmp') / f'hft_cost_classifier_{split}.npy'
            values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32',
                shape=(len(selected), sum(part.shape[1] for part in parts)))
            for begin in range(0, len(selected), 65536):
                rows = selected[begin:begin + 65536]
                values[begin:begin + len(rows)] = np.concatenate([part[rows] for part in parts], axis=1)
            values.flush()
            reference = matrices['fit'] if split == 'tune' else None
            labels = (targets[split] > 0).astype('float32')
            matrices[split] = xgb.QuantileDMatrix(GpuBatchIterator(values, labels), max_bin=64, ref=reference, nthread=2)
            temporary_files.append(temporary)
            del values
            cp.get_default_memory_pool().free_all_blocks()
        model = xgb.train({
            'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
            'device': 'cuda', 'max_depth': max_depth, 'max_bin': 64, 'min_child_weight': min_child_weight,
            'eta': .05, 'subsample': .8, 'colsample_bytree': .8, 'lambda': 10,
            'seed': seed, 'nthread': 2,
        }, matrices['fit'], num_boost_round=800, evals=[(matrices['tune'], 'tune')],
            early_stopping_rounds=50, verbose_eval=50)
        model = model[:model.best_iteration + 1]
        model.save_model(str(folder / 'model.ubj'))
        del matrices
        cp.get_default_memory_pool().free_all_blocks()
        for file in temporary_files:
            file.unlink()
        probabilities = {}
        for split in ['tune', 'val_dev', 'val_select']:
            parts = input_parts(config, Path(config['macro_source']), split)
            probability = np.empty(len(parts[0]), dtype='float64')
            for begin in range(0, len(probability), 65536):
                stop = min(begin + 65536, len(probability))
                values = cp.asarray(np.concatenate([part[begin:stop] for part in parts], axis=1))
                probability[begin:stop] = cp.asnumpy(model.inplace_predict(values))
            probabilities[split] = probability
        slope, intercept = probability_calibration(
            probabilities['tune'][positions['tune']], (targets['tune'] > 0).astype('float64'),
        )
        fit = targets['fit']
        win_mean = float(fit[fit > 0].mean())
        loss_mean = float(fit[fit <= 0].mean())
        for split, probability in probabilities.items():
            probability = np.clip(probability, 1e-6, 1 - 1e-6)
            calibrated = expit(slope * np.log(probability / (1 - probability)) + intercept)
            edge = calibrated * win_mean + (1 - calibrated) * loss_mean
            np.save(folder / f'{split}_net_edge_bps.npy', edge)
    write_json_atomic(folder / 'completed.json', {
        'status': 'success', 'horizon_s': horizon, 'side': side, 'seed': seed,
        'max_depth': max_depth, 'min_child_weight': min_child_weight,
        'fit_lookback_days': fit_lookback_days, 'fit_rows': len(positions['fit']),
        'calibration_slope': slope, 'calibration_intercept': intercept,
        'train_win_mean_bps': win_mean, 'train_loss_mean_bps': loss_mean,
        'train_positive_share': float((fit > 0).mean()),
    })


# 【核心逻辑368】仅在预期净收益大于零时生成方向信号，并冻结训练分位门槛。
def export_cost_signals(filename: Path) -> None:
    """交易方向由当前概率决定，完整开发与复核时钟始终输出，不使用未来标签筛行。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    signals = output / 'signals'
    signals.mkdir(exist_ok=True)
    thresholds = {}
    for horizon in config['horizons_s']:
        group = f'cost_h{horizon}'
        for split in ['tune', 'val_dev', 'val_select']:
            buy = np.load(output / f'models/h{horizon}_side1/{split}_net_edge_bps.npy')
            sell = np.load(output / f'models/h{horizon}_side-1/{split}_net_edge_bps.npy')
            positive = np.maximum(buy, sell) > 0
            score = np.where(positive, np.where(buy >= sell, buy, -sell), 0.)
            if split == 'tune':
                thresholds[group] = {str(value): float(np.quantile(np.abs(score), value)) for value in config['quantiles']}
                continue
            bounds = segments[split]
            rows = np.arange(bounds['start_row'], bounds['stop_row'])
            pl.DataFrame({
                'timestamp_ns': manifest['start_timestamp_ns'] + rows * SECOND,
                'expected_net_edge_bps': score,
            }).write_parquet(signals / f'{group}_{split}.parquet', compression='zstd')
    profit = Path(config['profit_output'])
    profit.mkdir(parents=True, exist_ok=True)
    write_json_atomic(profit / 'thresholds.json', {'fit': 'train_internal_net_edge_distribution', 'groups': thresholds})
    write_json_atomic(output / 'completed.json', {'status': 'success', 'signal_meaning': 'signed_expected_net_edge_bps'})
