"""180秒专用GPU基线，复用冻结输入与训练期预处理，独立保存模型和验证预测。"""

import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch import nn
import yaml

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import HORIZONS, SECOND


class Target180MLP(nn.Module):
    """按实验建议使用256/128/64三层表示，输出标准化180秒收益。"""

    # 【核心逻辑296】保持单目标输出，缺失通道与数值通道共同进入全连接网络。
    def __init__(self, channels: int):
        """输入已冻结的值与缺失通道总数，建立回归网络。"""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    # 【核心逻辑298】只使用当前已完成整秒输入，不展开未来位置。
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """返回每个输入位置的一个标准化回归值。"""
        return self.layers(values).squeeze(-1)


# 【非核心】权重和优化器一起原子保存，重启从完整epoch继续。
def save_checkpoint(filename: Path, state: dict) -> None:
    """临时文件成功写完才替换检查点。"""
    temporary = filename.with_suffix('.partial')
    torch.save(state, temporary)
    temporary.replace(filename)


# 【核心逻辑295】复用训练前缀缓存，将训练与内部早停数据分别放入显存。
def train_cached(config: dict, folder: Path, source: Path, seed: int, learning_rate: float):
    """训练期最后14天只用于早停，外部两段Validation均不参与梯度或早停。"""
    fold = source / 'e1/folds/final'
    preprocessing = np.load(fold / 'preprocessing.npz')
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    target_mean = float(preprocessing['target_mean'][target_index])
    target_scale = float(preprocessing['target_scale'][target_index])
    targets = np.load(source / 'e0/y.npy', mmap_mode='r')
    cached = {}
    for split in ['fit', 'tune']:
        rows = np.load(fold / f'{split}_rows.npy', mmap_mode='r')
        values = np.load(fold / f'{split}_x.npy', mmap_mode='r')
        cached[split] = (
            torch.tensor(values, device='cuda'),
            torch.tensor((targets[rows, target_index] - target_mean) / target_scale,
                         dtype=torch.float32, device='cuda'),
        )
    torch.manual_seed(seed)
    model = Target180MLP(cached['fit'][0].shape[1]).cuda()
    model = fit_cached_tensors(model, cached, folder, config, learning_rate)
    return model, preprocessing, target_mean, target_scale


# 【核心逻辑297】共享GPU批训练及完整epoch恢复，MLP与序列模型采用相同早停规则。
def fit_cached_tensors(model, cached: dict, folder: Path, config: dict, learning_rate: float):
    """输入两段显存张量、网络及超参数，返回内部验证最优网络。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_loss = float('inf')
    best_epoch = 0
    first_epoch = 0
    checkpoint = folder / 'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cuda', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        first_epoch = state['epoch']
        best_loss = state['best_loss']
        best_epoch = state['best_epoch']
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
    fit_values, fit_labels = cached['fit']
    tune_values, tune_labels = cached['tune']
    batch_size = config['batch_size']
    for epoch in range(first_epoch, config['max_epochs']):
        if best_epoch > 0 and epoch - best_epoch >= config['patience']:
            break
        started = time.monotonic()
        order = torch.randperm(len(fit_values), device='cuda')
        model.train()
        training_loss = torch.zeros((), device='cuda')
        for begin in range(0, len(order), batch_size):
            positions = order[begin:begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(fit_values[positions]) - fit_labels[positions]).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_loss += loss.detach() * len(positions)
        model.eval()
        tune_loss = torch.zeros((), device='cuda')
        with torch.no_grad():
            for begin in range(0, len(tune_values), batch_size):
                predicted = model(tune_values[begin:begin + batch_size])
                expected = tune_labels[begin:begin + batch_size]
                tune_loss += (predicted - expected).square().sum()
        tune_loss_value = float(tune_loss / len(tune_values))
        if tune_loss_value < best_loss:
            best_loss = tune_loss_value
            best_epoch = epoch + 1
            save_checkpoint(folder / 'model.pt', model.state_dict())
        record = {
            'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_loss': best_loss,
            'training_loss': float(training_loss / len(fit_values)),
            'tune_loss': tune_loss_value, 'elapsed_seconds': time.monotonic() - started,
        }
        save_checkpoint(checkpoint, {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1, 'best_loss': best_loss, 'best_epoch': best_epoch,
            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
        })
        write_json_atomic(folder / 'progress.json', record)
        with (folder / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print('EPOCH', record, flush=True)
        if epoch + 1 - best_epoch >= config['patience']:
            break
    model.load_state_dict(torch.load(folder / 'model.pt', weights_only=True))
    model.eval()
    return model


# 【核心逻辑299】按完整验证网格分批推理，标签仅参与评价有效性，不决定预测是否输出。
def predict_validation(model, preprocessing, source: Path, folder: Path,
                       target_mean: float, target_scale: float, config: dict,
                       *, sequence_input: bool = False, prediction_batch_size: int = 65536) -> None:
    """发布两段验证的整秒预测、逐日IC与三个预先固定的非重叠相位。"""
    from mm_research.multihorizon.common import model_inputs

    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    matrix = np.load(source / 'e0/x.npy', mmap_mode='r')
    targets = np.load(source / 'e0/y.npy', mmap_mode='r')
    metrics = {}
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    for split in ['val_dev', 'val_select']:
        bounds = segments[split]
        filename = folder / f'{split}.parquet'
        temporary = filename.with_suffix('.partial')
        writer = None
        try:
            batch_size = prediction_batch_size
            for begin in range(bounds['start_row'], bounds['stop_row'], batch_size):
                rows = np.arange(begin, min(begin + batch_size, bounds['stop_row']))
                if sequence_input:
                    values = rows
                else:
                    values = model_inputs(matrix, rows, preprocessing['mean'], preprocessing['scale'])
                with torch.no_grad():
                    prediction = model(torch.tensor(values, device='cuda')).cpu().numpy()
                prediction = prediction.astype('float64') * target_scale + target_mean
                actual = np.asarray(targets[rows, target_index])
                valid = np.isfinite(actual) & (rows < bounds['stop_row'] - 300)
                table = pl.DataFrame({
                    'timestamp_ns': manifest['start_timestamp_ns'] + rows * SECOND,
                    'prediction_180s_log_bps': prediction,
                    'target_180s_log_bps': actual,
                    'evaluation_valid': valid,
                }).to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(filename)
        frame = pl.read_parquet(filename).filter(pl.col('evaluation_valid'))
        prediction = pl.col('prediction_180s_log_bps')
        actual = pl.col('target_180s_log_bps')
        statistics = [
            pl.len().alias('rows'),
            pl.corr(prediction, actual).alias('ic'),
            pl.corr(prediction, actual, method='spearman').alias('rank_ic'),
            (1 - (prediction - actual).pow(2).sum() /
             (actual - target_mean).pow(2).sum()).alias('r2_vs_train_mean'),
        ]
        metrics[split] = {'all': frame.select(statistics).row(0, named=True), 'phases': {}}
        for phase in config['nonoverlap_phases_s']:
            sample = frame.filter((pl.col('timestamp_ns') // SECOND) % 180 == phase)
            metrics[split]['phases'][str(phase)] = sample.select(statistics).row(0, named=True)
        daily = frame.with_columns((pl.col('timestamp_ns') // (86400 * SECOND)).alias('utc_day'))
        daily.group_by('utc_day').agg(statistics).sort('utc_day').write_parquet(folder / f'{split}_daily.parquet')
        print('VALIDATION', split, metrics[split], flush=True)
    write_json_atomic(folder / 'metrics.json', metrics)


# 【核心逻辑294】每个种子与学习率独立运行并记录数据契约，失败由研究队列隔离。
def run_target180(filename: Path, seed: int, learning_rate: float) -> None:
    """读取GPU实验配置，复用旧146列基线，输出权重、预测及资源摘要。"""
    config = yaml.safe_load(filename.read_text())
    source = Path(config['source'])
    folder = Path(config['output']) / f'mlp_lr{learning_rate}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    contract = {
        'config': config, 'seed': seed, 'learning_rate': learning_rate,
        'input_version': 'frozen_146_previous_windows',
        'target': '180s_log_bps', 'selection': 'inner_train_tail_14days',
        'split_source': str(source / 'e0/splits.json'),
    }
    contract_file = folder / 'contract.json'
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError('实验契约变化，请使用新的输出版本')
    write_json_atomic(contract_file, contract)
    if (folder / 'completed.json').exists():
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model, preprocessing, target_mean, target_scale = train_cached(config, folder, source, seed, learning_rate)
    predict_validation(model, preprocessing, source, folder, target_mean, target_scale, config)
    resource_summary = {
        'elapsed_seconds': time.monotonic() - started,
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
    }
    write_json_atomic(folder / 'resource_summary.json', resource_summary)
    write_json_atomic(folder / 'completed.json', {'status': 'success', **resource_summary})
