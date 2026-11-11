"""180秒序列GPU基线，训练期选列后缓存全日期输入，按需索引完整过去120秒。"""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional
import yaml

from mm_research.data.storage import write_json_atomic
from mm_research.models.target180 import Target180MLP, fit_cached_tensors, predict_validation
from mm_research.multihorizon.common import HORIZONS, model_inputs


# 【核心逻辑301】训练前缀选列与缓存生产共用同一计算，允许GPU先训练已完成的前缀。
def select_training_features(config: dict, source: Path):
    """返回冻结列序及完整选列依据，所有统计仅来自拟合前缀。"""
    fold = source / 'e1/folds/final'
    rows = np.load(fold / 'fit_rows.npy', mmap_mode='r')[::20]
    targets = np.load(source / 'e0/y.npy', mmap_mode='r')
    matrix = np.load(source / 'e0/x.npy', mmap_mode='r')
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    values = np.asarray(matrix[rows], dtype='float64')
    labels = np.asarray(targets[rows, target_index])
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    clean = np.where(valid, values, 0.0)
    mean = clean.sum(axis=0) / count
    label_mean = (np.where(valid, labels[:, None], 0.0)).sum(axis=0) / count
    centered_values = np.where(valid, values - mean, 0.0)
    centered_labels = np.where(valid, labels[:, None] - label_mean, 0.0)
    correlation = (centered_values * centered_labels).sum(axis=0) / np.sqrt(
        (centered_values ** 2).sum(axis=0) * (centered_labels ** 2).sum(axis=0))
    eligible = np.flatnonzero(np.isfinite(correlation) & (count >= len(rows) * 0.8))
    selected = eligible[np.argsort(-np.abs(correlation[eligible]), kind='stable')[:config['selected_features']]]
    if len(selected) != config['selected_features']:
        raise ValueError('训练前缀可用列不足以建立指定维度序列')
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    selection = {
        'names': [manifest['feature_names'][int(index)] for index in selected],
        'indices': selected.tolist(), 'fit_stride': 20,
        'criterion': 'abs_180s_pearson_fit_prefix',
        'fit_last_row': int(rows[-1]), 'correlations': correlation[selected].tolist(),
        'channels': len(selected) * 2, 'dtype': 'float32',
    }
    return selected, selection


# 【核心逻辑300】仅按拟合前缀的180秒相关选48列，冻结全日期输入的列清单。
def prepare_sequence(config: dict) -> None:
    """仅保存列名与筛选依据，规范化矩阵由训练进程直接加载显存。"""
    source = Path(config['source'])
    destination = Path(config['output']) / 'inputs'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'completed.json').exists():
        return
    _, selection = select_training_features(config, source)
    write_json_atomic(destination / 'selection.json', selection)
    write_json_atomic(destination / 'completed.json', {
        'status': 'success', 'storage': 'gpu_only_normalization_from_frozen_source',
    })


# 【核心逻辑303】顺序读取原矩阵并规范化入选列，共享给显存加载和临时内存缓存。
def normalized_blocks(source: Path, selected: np.ndarray, preprocessing):
    """逐批返回位置范围和Float32值/缺失通道，不重复物化全日期CPU矩阵。"""
    with (source / 'e0/x.npy').open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError('冻结模型输入必须使用当前NPY v1.0')
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        if fortran_order or dtype != np.dtype('float32'):
            raise ValueError('冻结模型输入必须为Float32行主序')
        for begin in range(0, shape[0], 131072):
            stop = min(begin + 131072, shape[0])
            values = np.fromfile(stream, dtype=dtype, count=(stop - begin) * shape[1])
            values = values.reshape(stop - begin, shape[1])[:, selected]
            normalized = model_inputs(values, np.arange(len(values)),
                                      preprocessing['mean'][selected], preprocessing['scale'][selected])
            yield begin, stop, normalized


# 【核心逻辑305】将同一定义的全日期输入放入显存，显式支持临时内存文件供后续模型复用。
def load_gpu_inputs(source: Path, selected: np.ndarray, preprocessing,
                    normalized_input: Path | None = None) -> torch.Tensor:
    """默认直接规范化冻结原矩阵，指定内存文件时只读取该文件，读取失败直接报错。"""
    if normalized_input is not None:
        values = np.load(normalized_input, mmap_mode='r')
        gpu_inputs = torch.empty(values.shape, dtype=torch.float32, device='cuda')
        for begin in range(0, len(values), 262144):
            stop = min(begin + 262144, len(values))
            gpu_inputs[begin:stop] = torch.tensor(values[begin:stop], device='cuda')
        return gpu_inputs
    matrix = np.load(source / 'e0/x.npy', mmap_mode='r')
    gpu_inputs = torch.empty((len(matrix), len(selected) * 2), dtype=torch.float32, device='cuda')
    for begin, stop, normalized in normalized_blocks(source, selected, preprocessing):
        gpu_inputs[begin:stop] = torch.from_numpy(normalized).to('cuda')
        if begin % (131072 * 16) == 0:
            print('GPU_CACHE_LOADED_ROWS', stop, len(matrix), flush=True)
    return gpu_inputs


# 【核心逻辑302】一次构造临时RAM输入，后续种子和模型直接复用，最终汇总后清理。
def prepare_memory_cache(filename: Path, destination: Path) -> None:
    """显式输出到/dev/shm，原标准数据和磁盘结果保持独立。"""
    if destination.exists():
        print('RAM_CACHE_ALREADY_READY', str(destination), flush=True)
        return
    config = yaml.safe_load(filename.read_text())
    source = Path(config['source'])
    selection = json.loads((Path(config['output']) / 'inputs/selection.json').read_text())
    selected = np.asarray(selection['indices'])
    preprocessing = np.load(source / 'e1/folds/final/preprocessing.npz')
    matrix = np.load(source / 'e0/x.npy', mmap_mode='r')
    temporary = destination.with_suffix('.partial.npy')
    with temporary.open('wb') as stream:
        np.lib.format.write_array_header_1_0(stream, {
            'descr': '<f4', 'fortran_order': False, 'shape': (len(matrix), len(selected) * 2),
        })
        for _, stop, normalized in normalized_blocks(source, selected, preprocessing):
            stream.write(normalized.tobytes(order='C'))
            print('RAM_CACHE_ROWS', stop, len(matrix), flush=True)
    temporary.replace(destination)
    print('RAM_CACHE_READY', str(destination), flush=True)


# 【核心逻辑308】以三项确定性矩阵乘法实现同一三点因果膨胀卷积，避开慢速CUDA卷积反传。
def causal_dilated_convolution(values: torch.Tensor, convolution: nn.Conv1d) -> torch.Tensor:
    """权重和偏置保持Conv1d定义，三个输入分别来自t-2d、t-d和t。"""
    dilation = convolution.dilation[0]
    steps = values.shape[-1]
    zeros = values.new_zeros(values.shape[0], values.shape[1], dilation * 2)
    padded = torch.cat([zeros, values], dim=-1)
    oldest = functional.linear(padded[:, :, :steps].transpose(1, 2), convolution.weight[:, :, 0])
    middle = functional.linear(padded[:, :, dilation:dilation + steps].transpose(1, 2),
                               convolution.weight[:, :, 1])
    latest = functional.linear(padded[:, :, dilation * 2:dilation * 2 + steps].transpose(1, 2),
                               convolution.weight[:, :, 2])
    return (oldest + middle + latest + convolution.bias).transpose(1, 2)


class RowSequence180(nn.Module):
    """缓存保持单份显存张量，batch按行号收集历史，权重文件仅保存网络参数。"""

    # 【核心逻辑306】建立120秒GRU或因果TCN，感受野覆盖整个输入历史。
    def __init__(self, history_values: torch.Tensor, family: str, history_s: int):
        """输入全日期缓存和网络类型，历史张量作为只读数据而非模型权重。"""
        super().__init__()
        self.history_values = history_values
        self.family = family
        self.register_buffer('offsets', torch.arange(history_s - 1, -1, -1, device='cuda'))
        if family == 'mlp':
            self.encoder = Target180MLP(history_values.shape[1])
            return
        self.projection = nn.Sequential(nn.Linear(history_values.shape[1], 64), nn.LayerNorm(64))
        if family == 'gru':
            self.encoder = nn.GRU(64, 64, num_layers=1, batch_first=True)
        elif family == 'tcn':
            self.encoder = nn.ModuleList([
                nn.Conv1d(64, 64, kernel_size=3, dilation=dilation)
                for dilation in [1, 2, 4, 8, 16, 32, 64]
            ])
        else:
            raise ValueError(f'未知序列模型 {family}')
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    # 【核心逻辑307】每个预测位置只收集当前及过去119个已完成秒，缺失通道保持原样。
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """输入当前预测行号，返回标准化180秒收益，跨午夜使用连续时间行。"""
        if self.family == 'mlp':
            return self.encoder(self.history_values[positions])
        history_rows = positions[:, None] - self.offsets[None, :]
        encoded = self.projection(self.history_values[history_rows])
        if self.family == 'gru':
            encoded, _ = self.encoder(encoded)
            representation = encoded[:, -1]
        else:
            encoded = encoded.transpose(1, 2)
            for convolution in self.encoder:
                encoded = encoded + functional.gelu(causal_dilated_convolution(encoded, convolution))
            representation = encoded[:, :, -1]
        return self.head(representation).squeeze(-1)


# 【核心逻辑304】按训练与早停行号动态收集历史，复用MLP的检查点和共同验证统计。
def run_sequence(filename: Path, family: str, seed: int, *, normalized_input: Path | None = None,
                 batch_size: int | None = None) -> None:
    """完整120秒序列在显存按批读取，禁止预展开重叠样本。"""
    config = yaml.safe_load(filename.read_text())
    if batch_size is not None:
        config['batch_size'] = batch_size
    source = Path(config['source'])
    model_name = f'mlp_current_seed{seed}' if family == 'mlp' else f'{family}_120s_seed{seed}'
    folder = Path(config['output']) / model_name
    folder.mkdir(parents=True, exist_ok=True)
    contract = {'config': config, 'family': family, 'seed': seed,
                'target': '180s_log_bps', 'input': 'fit_selected_48_previous_windows',
                'sequence_s': 1 if family == 'mlp' else 120, 'selection': 'inner_train_tail_14days'}
    if family == 'tcn':
        contract['convolution_backend'] = 'three_linear_causal_v1'
    contract_file = folder / 'contract.json'
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError('序列实验契约变化，请使用新的输出版本')
    write_json_atomic(contract_file, contract)
    if (folder / 'completed.json').exists():
        return
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()
    input_folder = Path(config['output']) / 'inputs'
    selection = json.loads((input_folder / 'selection.json').read_text())
    selected = np.asarray(selection['indices'])
    fold = source / 'e1/folds/final'
    preprocessing = np.load(fold / 'preprocessing.npz')
    gpu_inputs = load_gpu_inputs(source, selected, preprocessing, normalized_input)
    print('GPU_INPUTS_READY', len(gpu_inputs), flush=True)
    targets = np.load(source / 'e0/y.npy', mmap_mode='r')
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    target_mean = float(preprocessing['target_mean'][target_index])
    target_scale = float(preprocessing['target_scale'][target_index])
    cached = {}
    for split in ['fit', 'tune']:
        rows = np.load(fold / f'{split}_rows.npy', mmap_mode='r')
        rows = rows[rows >= config['history_s'] - 1]
        cached[split] = (
            torch.tensor(rows, device='cuda'),
            torch.tensor((targets[rows, target_index] - target_mean) / target_scale,
                         dtype=torch.float32, device='cuda'),
        )
    model = RowSequence180(gpu_inputs, family, config['history_s']).cuda()
    model = fit_cached_tensors(model, cached, folder, config, config['learning_rate'])
    predict_validation(model, preprocessing, source, folder, target_mean, target_scale, config,
                       sequence_input=True, prediction_batch_size=config['batch_size'])
    resources = {
        'elapsed_seconds': time.monotonic() - started,
        'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'input_storage': 'ram' if normalized_input is not None else 'frozen_source',
    }
    write_json_atomic(folder / 'resource_summary.json', resources)
    write_json_atomic(folder / 'completed.json', {'status': 'success', **resources})
