"""180秒GPU树基线，缓存准备期间使用现成146列矩阵训练。"""

import json
from pathlib import Path
import time

import numpy as np
import cupy as cp
import torch
import xgboost as xgb
import yaml

from mm_research.data.storage import write_json_atomic
from mm_research.models.target180 import predict_validation
from mm_research.multihorizon.common import HORIZONS


class GpuBatchIterator(xgb.DataIter):
    """限制GPU分箱批次，避免整块输入与临时排序数组同时挤满显存。"""

    # 【非核心】只读训练缓存，GPU每次最多接收262144行。
    def __init__(self, values, labels):
        """保存已有矩阵映射和标签，不额外复制整表。"""
        super().__init__(release_data=True)
        self.values = values
        self.labels = labels
        self.begin = 0

    # 【非核心】XGBoost构造量化矩阵时会多遍读取同一固定行集。
    def reset(self):
        """每遍从第一行开始，不改变输入顺序。"""
        self.begin = 0

    # 【核心逻辑310】逐批把现成矩阵送入CUDA分箱，维持有限显存占用。
    def next(self, input_data):
        """将下一批数值与标签交给XGBoost，耗尽后明确结束。"""
        if self.begin == len(self.values):
            return False
        stop = min(self.begin + 262144, len(self.values))
        input_data(data=cp.asarray(self.values[self.begin:stop]),
                   label=cp.asarray(self.labels[self.begin:stop]))
        self.begin = stop
        return True


class TreePredictor:
    """复用统一验证输出接口，训练分箱、树学习和预测均在GPU执行。"""

    # 【非核心】预测时仅保留选出的最优树轮数，与早停成绩保持一致。
    def __init__(self, booster):
        """接收已经截取至最佳迭代的树模型。"""
        self.booster = booster

    # 【核心逻辑311】通过DLPack共享GPU批次，由冻结树输出标准化收益。
    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """返回与MLP相同形状的预测，标签及未来状态不参与推理。"""
        prediction = self.booster.inplace_predict(cp.from_dlpack(values))
        return torch.from_dlpack(prediction)


# 【核心逻辑309】用现成训练缓存拟合CUDA直方图树，训练尾部14天只作早停。
def run_tree(filename: Path) -> None:
    """复用同一行集和标准化口径，发布180秒XGBoost模型及完整验证结果。"""
    config = yaml.safe_load(filename.read_text())
    source = Path(config['source'])
    folder = Path(config['output']) / 'xgb_depth6_seed17'
    folder.mkdir(parents=True, exist_ok=True)
    contract = {
        'config': config, 'input': 'frozen_146_previous_windows',
        'target': '180s_log_bps', 'selection': 'inner_train_tail_14days',
        'device': 'cuda', 'seed': 17,
    }
    contract_file = folder / 'contract.json'
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError('树实验契约变化，请使用新的输出版本')
    write_json_atomic(contract_file, contract)
    if (folder / 'completed.json').exists():
        return
    started = time.monotonic()
    fold = source / 'e1/folds/final'
    preprocessing = np.load(fold / 'preprocessing.npz')
    targets = np.load(source / 'e0/y.npy', mmap_mode='r')
    target_index = int(np.flatnonzero(HORIZONS == 180)[0])
    target_mean = float(preprocessing['target_mean'][target_index])
    target_scale = float(preprocessing['target_scale'][target_index])
    weights = folder / 'model.ubj'
    if weights.exists():
        booster = xgb.Booster(model_file=str(weights))
    else:
        matrices = {}
        for split in ['fit', 'tune']:
            rows = np.load(fold / f'{split}_rows.npy', mmap_mode='r')
            values = np.load(fold / f'{split}_x.npy', mmap_mode='r')
            labels = ((targets[rows, target_index] - target_mean) / target_scale).astype('float32')
            reference = matrices['fit'] if split == 'tune' else None
            iterator = GpuBatchIterator(values, labels)
            matrices[split] = xgb.QuantileDMatrix(iterator, max_bin=64,
                                                 ref=reference, nthread=2)
            del iterator
            cp.get_default_memory_pool().free_all_blocks()
            print('MATRIX_READY', split, flush=True)
        parameters = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'tree_method': 'hist', 'device': 'cuda', 'max_depth': 6,
            'max_bin': 64, 'min_child_weight': 500, 'eta': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'lambda': 10, 'seed': 17, 'nthread': 2,
        }
        booster = xgb.train(parameters, matrices['fit'], num_boost_round=1000,
                            evals=[(matrices['tune'], 'tune')], early_stopping_rounds=50,
                            verbose_eval=25)
        best_iteration = booster.best_iteration + 1
        booster = booster[:best_iteration]
        temporary = weights.with_suffix('.partial.ubj')
        booster.save_model(str(temporary))
        write_json_atomic(folder / 'progress.json', {'best_epoch': best_iteration})
        temporary.replace(weights)
        del matrices
    booster.set_param({'device': 'cuda', 'nthread': 2})
    predict_validation(TreePredictor(booster), preprocessing, source, folder,
                       target_mean, target_scale, config)
    resources = {
        'elapsed_seconds': time.monotonic() - started, 'gpu': torch.cuda.get_device_name(),
        'xgboost': xgb.__version__, 'prediction_device': 'cuda',
        'peak_gpu_allocated_gib': None,
    }
    write_json_atomic(folder / 'resource_summary.json', resources)
    write_json_atomic(folder / 'completed.json', {'status': 'success', **resources})
