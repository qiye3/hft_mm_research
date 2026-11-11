"""训练期前向OOF、多期限模型与完整网格预测。"""

import json
import os
from pathlib import Path
import shutil
import time

import lightgbm as lgb
import numpy as np
import polars as pl
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn

from mm_research.data.storage import write_json_atomic
from mm_research.models.neural import FactorNetwork
from mm_research.multihorizon.common import HORIZONS, DAY, SECOND, context, open_dataset, model_inputs


# 【核心逻辑259】每折独立估计输入和标签尺度，早停区间位于该折训练末尾且隔离300秒。
def prepare_fold(output: Path, config: dict, fold_id: str) -> dict:
    """原始特征均值、缺失通道与训练缓存仅使用本折训练前缀。"""
    manifest, splits, matrix, targets, market = open_dataset(output)
    folder = output / 'e1' / 'folds' / fold_id
    folder.mkdir(parents=True, exist_ok=True)
    state_file = folder / 'prepared.json'
    if state_file.exists():
        return json.loads(state_file.read_text())
    if fold_id == 'final':
        cutoff = splits['segments']['train']['stop_row'] - 300
    else:
        cutoff = next(fold['train_stop'] for fold in splits['oof_folds'] if fold['id'] == fold_id)
    tune_start = cutoff - 14 * DAY
    rows = np.load(output / 'e0/rows/train.npy', mmap_mode='r')
    sampled = rows[(rows < cutoff) & (rows % config['train_stride_s'] == 0)]
    fit = sampled[sampled < tune_start - 300]
    tune = sampled[sampled >= tune_start]
    counts = np.zeros(matrix.shape[1])
    sums = np.zeros_like(counts)
    squares = np.zeros_like(counts)
    target_sum = np.zeros(16)
    target_square = np.zeros(16)
    increment_sum = np.zeros(16)
    increment_square = np.zeros(16)
    for begin in range(0, len(fit), 32768):
        positions = fit[begin:begin + 32768]
        values = np.asarray(matrix[positions], dtype=np.float64)
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.)
        counts += finite.sum(axis=0)
        sums += clean.sum(axis=0)
        squares += (clean * clean).sum(axis=0)
        labels = np.asarray(targets[positions])
        increments = np.diff(np.column_stack([np.zeros(len(positions)), labels]), axis=1)
        target_sum += labels.sum(axis=0)
        target_square += (labels * labels).sum(axis=0)
        increment_sum += increments.sum(axis=0)
        increment_square += (increments * increments).sum(axis=0)
    if np.any(counts == 0):
        absent = [manifest['feature_names'][index] for index in np.flatnonzero(counts == 0)]
        raise ValueError(f'本折训练前缀缺少候选有效观测 {absent}')
    mean = sums / counts
    scale = np.sqrt(np.maximum(squares / counts - mean * mean, 1e-12))
    target_mean = target_sum / len(fit)
    target_scale = np.sqrt(np.maximum(target_square / len(fit) - target_mean ** 2, 1e-12))
    increment_mean = increment_sum / len(fit)
    increment_scale = np.sqrt(np.maximum(increment_square / len(fit) - increment_mean ** 2, 1e-12))
    np.savez(folder / 'preprocessing.npz', mean=mean.astype('float32'), scale=scale.astype('float32'),
             target_mean=target_mean, target_scale=target_scale,
             increment_mean=increment_mean, increment_scale=increment_scale)
    for name, positions in [('fit', fit), ('tune', tune)]:
        np.save(folder / f'{name}_rows.npy', positions)
        normalized = np.lib.format.open_memmap(folder / f'{name}_x.npy', mode='w+', dtype='float32',
                                              shape=(len(positions), matrix.shape[1] * 2))
        for begin in range(0, len(positions), 32768):
            normalized[begin:begin + 32768] = model_inputs(matrix, positions[begin:begin + 32768], mean, scale)
        normalized.flush()
        del normalized
    state = {'fold_id': fold_id, 'train_stop': cutoff, 'fit_rows': len(fit), 'tune_rows': len(tune),
             'fit_last_row': int(fit[-1]), 'tune_first_row': int(tune[0]),
             'preprocessing_fitted_before': tune_start - 300,
             'prediction_uses_future_labels': False}
    write_json_atomic(state_file, state)
    return state


# 【核心逻辑260】独立树按相同输入、行集和折内早停训练，逐期限发布检查点。
def train_trees(output: Path, config: dict, model_id: str, fold_id: str) -> dict:
    """M1训练16期限；M0重训原LightGBM的1/10秒部分，60秒交由原LSTM。"""
    manifest, splits, matrix, targets, market = open_dataset(output)
    fold = output / 'e1/folds' / fold_id
    destination = output / 'e1/models' / model_id / fold_id
    destination.mkdir(parents=True, exist_ok=True)
    prep = np.load(fold / 'preprocessing.npz')
    fit_rows = np.load(fold / 'fit_rows.npy', mmap_mode='r')
    tune_rows = np.load(fold / 'tune_rows.npy', mmap_mode='r')
    fit = np.load(fold / 'fit_x.npy', mmap_mode='r')
    tune = np.load(fold / 'tune_x.npy', mmap_mode='r')
    indices = range(16) if model_id == 'M1' else [0, 5]
    rounds = {}
    for index in indices:
        horizon = int(HORIZONS[index])
        weights = destination / f'tree_{horizon}s.txt'
        if weights.exists():
            rounds[str(horizon)] = lgb.Booster(model_file=str(weights)).current_iteration()
            continue
        train_labels = ((np.asarray(targets[fit_rows, index]) - prep['target_mean'][index]) / prep['target_scale'][index])
        tune_labels = ((np.asarray(targets[tune_rows, index]) - prep['target_mean'][index]) / prep['target_scale'][index])
        parameters = {'objective': 'regression', 'metric': 'l2', 'verbosity': -1,
                      'num_threads': config['training_threads'], 'seed': config['seed'],
                      'num_leaves': config['lightgbm']['num_leaves'] if model_id == 'M1' else 63,
                      'learning_rate': config['lightgbm']['learning_rate'],
                      'min_data_in_leaf': config['lightgbm']['min_data_in_leaf'],
                      'deterministic': True, 'force_col_wise': True}
        trained = lgb.train(parameters, lgb.Dataset(fit, label=train_labels),
                            num_boost_round=config['lightgbm']['n_estimators'],
                            valid_sets=[lgb.Dataset(tune, label=tune_labels)],
                            callbacks=[lgb.early_stopping(config['lightgbm']['early_stopping_rounds'], verbose=False)])
        trained.save_model(str(weights))
        rounds[str(horizon)] = trained.best_iteration
        print('TREE_READY', model_id, fold_id, horizon, trained.best_iteration, flush=True)
    return {'family': 'independent_lightgbm' if model_id == 'M1' else 'legacy_lightgbm_lstm', 'rounds': rounds}


class CurveNetwork(nn.Module):
    """共享128维主干，输出累计、分段收益或两项指数衰减幅度。"""

    # 【核心逻辑261】按模型定义构造共享MLP和显式曲线输出头。
    def __init__(self, features: int, mode: str, taus: tuple = (5, 60)):
        """M2/M3输出16个标准化目标，M4输出两个幅度并重建16期限曲线。"""
        super().__init__()
        self.mode = mode
        self.encoder = nn.Sequential(nn.Linear(features, 128), nn.GELU(), nn.Dropout(.1),
                                     nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1))
        self.head = nn.Linear(128, 2 if mode == 'M4' else 16)
        seconds = np.arange(300)
        basis = np.stack([np.cumsum(np.exp(-seconds / tau))[HORIZONS - 1] for tau in taus])
        self.register_buffer('basis', torch.tensor(basis, dtype=torch.float32))

    # 【核心逻辑262】当前输入经过共享主干生成收益曲线，不添加跨期限单调约束。
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """返回M4的bps曲线，或M2/M3的标准化累计／增量。"""
        amplitudes = self.head(self.encoder(values))
        return amplitudes @ self.basis if self.mode == 'M4' else amplitudes


# 【核心逻辑263】神经模型只用折内训练与早停数据，按批读取并保存最优权重。
def train_network(output: Path, config: dict, model_id: str, fold_id: str, taus: tuple = (5, 60)) -> dict:
    """原M0的60秒LSTM使用120秒历史，其他模型使用同一当前因子向量。"""
    manifest, splits, matrix, targets, market = open_dataset(output)
    fold = output / 'e1/folds' / fold_id
    destination = output / 'e1/models' / model_id / fold_id
    if model_id == 'M4':
        destination = output / 'e1/models' / f'M4_tau{taus[0]}_{taus[1]}' / fold_id
    destination.mkdir(parents=True, exist_ok=True)
    prep = np.load(fold / 'preprocessing.npz')
    fit_rows = np.load(fold / 'fit_rows.npy', mmap_mode='r')
    tune_rows = np.load(fold / 'tune_rows.npy', mmap_mode='r')
    fit = np.load(fold / 'fit_x.npy', mmap_mode='r')
    tune = np.load(fold / 'tune_x.npy', mmap_mode='r')
    torch.manual_seed(config['seed'])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    legacy = model_id == 'M0'
    model = FactorNetwork('lstm', matrix.shape[1] * 2, 120, 128) if legacy else CurveNetwork(matrix.shape[1] * 2, model_id, taus)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003 if legacy else config['mlp']['learning_rate'])
    weights = destination / 'network.pt'
    progress_file = destination / 'network_progress.json'
    target_kind = 'increment' if model_id == 'M3' else 'target'
    target_mean = prep[f'{target_kind}_mean']
    target_scale = prep[f'{target_kind}_scale']
    batch_size = 256 if legacy else config['mlp']['batch_size']
    max_epochs = 1 if legacy else config['mlp']['max_epochs']
    generator = np.random.default_rng(config['seed'])
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(max_epochs):
        order = generator.permutation(len(fit_rows))
        model.train()
        total_loss = 0.
        for begin in range(0, len(order), batch_size):
            selected = order[begin:begin + batch_size]
            positions = fit_rows[selected]
            if legacy:
                history_rows = positions[:, None] - np.arange(119, -1, -1)
                values = model_inputs(matrix, history_rows.reshape(-1), prep['mean'], prep['scale'])
                values = values.reshape(len(positions), 120, -1)
            else:
                values = np.asarray(fit[selected])
            actual = np.asarray(targets[positions])
            if model_id == 'M3':
                actual = np.diff(np.column_stack([np.zeros(len(actual)), actual]), axis=1)
            normalized = (actual - target_mean) / target_scale
            expected = normalized[:, 10] if legacy else normalized
            optimizer.zero_grad(set_to_none=True)
            predicted = model(torch.as_tensor(values, device=device))
            if model_id == 'M4':
                predicted = (predicted - torch.as_tensor(target_mean, device=device)) / torch.as_tensor(target_scale, device=device)
            loss = ((predicted - torch.as_tensor(expected, dtype=torch.float32, device=device)) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(selected)
        model.eval()
        validation_loss = 0.
        with torch.no_grad():
            for begin in range(0, len(tune_rows), batch_size):
                positions = tune_rows[begin:begin + batch_size]
                if legacy:
                    history_rows = positions[:, None] - np.arange(119, -1, -1)
                    values = model_inputs(matrix, history_rows.reshape(-1), prep['mean'], prep['scale']).reshape(len(positions), 120, -1)
                else:
                    values = np.asarray(tune[begin:begin + batch_size])
                actual = np.asarray(targets[positions])
                if model_id == 'M3':
                    actual = np.diff(np.column_stack([np.zeros(len(actual)), actual]), axis=1)
                expected = (actual - target_mean) / target_scale
                predicted = model(torch.as_tensor(values, device=device))
                if model_id == 'M4':
                    predicted = (predicted - torch.as_tensor(target_mean, device=device)) / torch.as_tensor(target_scale, device=device)
                if legacy:
                    expected = expected[:, 10]
                loss = ((predicted - torch.as_tensor(expected, dtype=torch.float32, device=device)) ** 2).mean()
                validation_loss += float(loss) * len(positions)
        validation_loss /= len(tune_rows)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), weights)
        record = {'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_loss': best_loss,
                  'training_loss': total_loss / len(fit_rows), 'tune_loss': validation_loss,
                  'taus': list(taus), 'legacy_sequence_s': 120 if legacy else 1}
        write_json_atomic(progress_file, record)
        print('NETWORK_EPOCH', model_id, fold_id, record, flush=True)
        if epoch + 1 - best_epoch >= config['mlp']['patience']:
            break
    return record


# 【核心逻辑264】逐块输出完整预测网格，推理不以未来标签有效性选择可交易时刻。
def predict_fold(output: Path, config: dict, model_id: str, fold_id: str) -> None:
    """Train OOF只预测前向区间，最终模型只生成两段验证；Test留到最终冻结之后。"""
    manifest, splits, matrix, targets, market = open_dataset(output)
    fold = output / 'e1/folds' / fold_id
    prep = np.load(fold / 'preprocessing.npz')
    destination = output / 'e1/models' / model_id / fold_id
    prediction_folder = output / 'e1/predictions' / model_id
    prediction_folder.mkdir(parents=True, exist_ok=True)
    frozen_model = FrozenCurveModel(output, model_id, fold_id, config['training_threads'])
    if fold_id == 'final':
        intervals = [(name, bounds['start_row'], bounds['stop_row']) for name, bounds in splits['segments'].items()
                     if name in ['val_dev', 'val_select']]
    else:
        specification = next(fold for fold in splits['oof_folds'] if fold['id'] == fold_id)
        intervals = [(fold_id, specification['predict_start'], specification['predict_stop'])]
    import pyarrow.parquet as pq
    for name, start, stop in intervals:
        file = prediction_folder / f'{name}.parquet'
        writer = None
        try:
            for begin in range(start, stop, config['inference_chunk_rows']):
                rows = np.arange(begin, min(stop, begin + config['inference_chunk_rows']))
                predicted = frozen_model.predict_rows(matrix, rows)
                timestamp = manifest['start_timestamp_ns'] + rows * SECOND
                table = pl.DataFrame({'decision_ts_ns': timestamp,
                                      'ready_ts_ns': timestamp + config['prediction_latency_ns'],
                                      'row': rows, 'model_id': [model_id] * len(rows),
                                      'model_version': [f'{model_id}/{fold_id}'] * len(rows),
                                      'fold_id': [name] * len(rows), 'mid': market[rows, 0],
                                      'pred_curve_bps': predicted,
                                      'pred_increment_bps': np.diff(np.column_stack([np.zeros(len(rows)), predicted]), axis=1)})
                arrow = table.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(file, arrow.schema, compression='zstd')
                writer.write_table(arrow)
        finally:
            if writer is not None:
                writer.close()
        print('PREDICTIONS_READY', model_id, name, flush=True)


# 【核心逻辑265】按模型和折发布独立训练状态，M3/M4需已有M1/M2可复现结果。
def train_model(filename: Path, model_id: str, fold_id: str) -> None:
    """训练与预测在独立进程内完成，完成标记只在权重和预测都已发布后写入。"""
    root, config, output = context(filename)
    if model_id == 'preprocess':
        prepare_fold(output, config, fold_id)
        return
    folder = output / 'e1/models' / model_id / fold_id
    complete = folder / 'completed.json'
    if complete.exists():
        return
    started = time.monotonic()
    records = {}
    if model_id in ['M0', 'M1']:
        records['trees'] = train_trees(output, config, model_id, fold_id)
        write_json_atomic(folder / 'weights_complete.json', {'model_id': model_id, 'fold_id': fold_id,
                          'tree_horizons': list(records['trees']['rounds'])})
    if model_id != 'M1':
        if model_id == 'M4':
            candidates = []
            for taus in config['tau_candidates_s']:
                candidate = train_network(output, config, model_id, fold_id, tuple(taus))
                candidates.append(candidate)
            selected = min(candidates, key=lambda candidate: candidate['best_loss'])
            taus = selected['taus']
            source = output / 'e1/models' / f'M4_tau{taus[0]}_{taus[1]}' / fold_id
            folder.mkdir(parents=True, exist_ok=True)
            for name in ['network.pt', 'network_progress.json']:
                shutil.copyfile(source / name, folder / name)
            records['network'] = selected
            records['tau_candidates'] = candidates
        else:
            records['network'] = train_network(output, config, model_id, fold_id)
    predict_fold(output, config, model_id, fold_id)
    write_json_atomic(complete, {'model_id': model_id, 'fold_id': fold_id,
                               'elapsed_seconds': time.monotonic() - started, 'records': records})


class FrozenCurveModel:
    """恢复单一权重版本，完整日期回放与研究预测共用相同预处理。"""

    # 【核心逻辑282】明确指定模型和训练折，恢复其输入尺度、风险和真实权重。
    def __init__(self, output: Path, model_id: str, fold_id: str, threads: int):
        """该对象仅读取权重，不读取标签有效性，也不会重新拟合参数。"""
        manifest, splits, matrix, targets, market = open_dataset(output)
        self.model_id = model_id
        self.threads = threads
        self.prep = np.load(output / 'e1/folds' / fold_id / 'preprocessing.npz')
        folder = output / 'e1/models' / model_id / fold_id
        self.trees = {}
        if model_id in ['M0', 'M1']:
            indices = range(16) if model_id == 'M1' else [0, 5]
            self.trees = {index: lgb.Booster(model_file=str(folder / f'tree_{int(HORIZONS[index])}s.txt')) for index in indices}
        self.network = None
        if model_id != 'M1':
            if model_id == 'M0':
                self.network = FactorNetwork('lstm', matrix.shape[1] * 2, 120, 128)
            else:
                progress = json.loads((folder / 'network_progress.json').read_text())
                self.network = CurveNetwork(matrix.shape[1] * 2, model_id, tuple(progress['taus']))
            self.network.load_state_dict(torch.load(folder / 'network.pt', map_location='cpu', weights_only=True))
            self.network = self.network.cuda().eval()

    # 【核心逻辑283】按显式当前时间行推理，目标标签及其是否可评估不参与信号是否可用。
    def predict_rows(self, matrix, rows: np.ndarray) -> np.ndarray:
        """返回16期限原始bps曲线，M0缺少的期限保持NaN。"""
        prep = self.prep
        values = model_inputs(matrix, rows, prep['mean'], prep['scale'])
        predicted = np.full((len(rows), 16), np.nan)
        for index, tree in self.trees.items():
            predicted[:, index] = tree.predict(values, num_threads=self.threads) * prep['target_scale'][index] + prep['target_mean'][index]
        if self.network is None:
            return predicted
        if self.model_id == 'M0':
            for begin in range(0, len(rows), 256):
                positions = rows[begin:begin + 256]
                history_rows = positions[:, None] - np.arange(119, -1, -1)
                history = model_inputs(matrix, history_rows.reshape(-1), prep['mean'], prep['scale']).reshape(len(positions), 120, -1)
                with torch.no_grad():
                    result = self.network(torch.as_tensor(history, device='cuda')).cpu().numpy()
                predicted[begin:begin + len(positions), 10] = result * prep['target_scale'][10] + prep['target_mean'][10]
            return predicted
        with torch.no_grad():
            result = self.network(torch.as_tensor(values, device='cuda')).cpu().numpy().astype('float64')
        if self.model_id == 'M2':
            result = result * prep['target_scale'] + prep['target_mean']
        elif self.model_id == 'M3':
            result = np.cumsum(result * prep['increment_scale'] + prep['increment_mean'], axis=1)
        return result
