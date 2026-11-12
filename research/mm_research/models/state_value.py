"""将既有费用分类概率转换为事前波动环境下的交易价值。"""

import json
from pathlib import Path
import shutil

import numpy as np
from scipy.special import expit

from mm_research.data.storage import write_json_atomic
from mm_research.models.cost_classifier import net_labels, probability_calibration
from mm_research.multihorizon.common import context, DAY
from mm_research.strategies.stable_strategy import visible_management


# 【核心逻辑389】按连续秒级前缀计算事前波动，分块结果保持完整窗口定义。
def prepare_state_volatility(filename: Path) -> None:
    """原始市场矩阵输入；输出全日期600秒事前波动，最初601秒和缺失窗口保留NaN。"""
    _, config, output = context(filename)
    destination = output / 'past600_volatility.npy'
    if destination.exists():
        return
    market = np.load(Path(config['source']) / 'e0/market.npy', mmap_mode='r')
    temporary = Path('/tmp/hft_state_value_volatility.npy')
    volatility = np.lib.format.open_memmap(temporary, mode='w+', dtype='float64', shape=(len(market),))
    volatility[:601] = np.nan
    for begin in range(601, len(market), DAY):
        stop = min(begin + DAY, len(market))
        volatility[begin:stop] = visible_management(market, begin, stop)[:, 2]
    volatility.flush()
    del volatility
    pending = destination.with_suffix('.pending')
    shutil.copyfile(temporary, pending)
    pending.replace(destination)
    temporary.unlink()


# 【非核心】波动状态只由冻结分界和当前已知波动决定，未知窗口单独编码。
def volatility_states(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """返回低/中/高波动0—2，缺失窗口返回-1，不用未来值填补。"""
    states = np.searchsorted(thresholds, values, side='right').astype('int8')
    states[~np.isfinite(values)] = -1
    return states


# 【非核心】把旧分类器保存的价值还原为其校准概率，再应用固定的状态收益幅度。
def conditional_values(probability: np.ndarray, states: np.ndarray, definitions: list[dict], recalibrate: bool) -> np.ndarray:
    """输入当时模型概率与可见状态；未知状态保留NaN，不引入替代信号。"""
    values = np.full(len(probability), np.nan)
    for state, definition in enumerate(definitions):
        selected = states == state
        calibrated = probability[selected]
        if recalibrate:
            clipped = np.clip(calibrated, 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1 - clipped))
            calibrated = expit(definition['slope'] * logits + definition['intercept'])
        values[selected] = calibrated * definition['win_mean_bps'] + (1 - calibrated) * definition['loss_mean_bps']
    return values


# 【核心逻辑390】拟合前缀估计状态收益幅度，Train尾部校准概率，开发复核只读取冻结映射。
def build_state_value(filename: Path, seed: int) -> None:
    """复用既有GPU分类器，不重新挑种子；分别输出条件幅度与条件概率两组预测。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    model_source = Path(config['models'][str(seed)])
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    volatility = np.load(output / 'past600_volatility.npy', mmap_mode='r')
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    fit_rows = np.load(source / 'e1/folds/final/fit_rows.npy')
    tune_rows = np.load(source / 'e1/folds/final/tune_rows.npy')
    thresholds = np.nanquantile(volatility[fit_rows], [1 / 3, 2 / 3])
    fit_states = volatility_states(volatility[fit_rows], thresholds)
    tune_states = volatility_states(volatility[tune_rows], thresholds)
    horizon = config['horizon_s']
    for side in [1, -1]:
        original = model_source / 'models' / f'h{horizon}_side{side}'
        metadata = json.loads((original / 'completed.json').read_text())
        fit_labels, fit_valid = net_labels(market, fit_rows, horizon, side)
        fit_valid &= fit_rows + horizon < tune_rows[0]
        tune_labels, tune_valid = net_labels(market, tune_rows, horizon, side)
        tune_valid &= tune_rows + horizon < segments['train']['stop_row']
        tune_edges = np.load(original / 'tune_net_edge_bps.npy')
        amplitude = metadata['train_win_mean_bps'] - metadata['train_loss_mean_bps']
        tune_probability = (tune_edges - metadata['train_loss_mean_bps']) / amplitude
        definitions = []
        for state in range(3):
            fitting = fit_labels[fit_valid & (fit_states == state)]
            tuning = tune_valid & (tune_states == state)
            slope, intercept = probability_calibration(tune_probability[tuning], (tune_labels[tuning] > 0).astype(float))
            definition = {
                'state': state, 'fit_rows': len(fitting), 'tune_rows': int(tuning.sum()),
                'win_mean_bps': float(fitting[fitting > 0].mean()),
                'loss_mean_bps': float(fitting[fitting <= 0].mean()),
                'slope': slope, 'intercept': intercept,
            }
            if not np.isfinite([definition['win_mean_bps'], definition['loss_mean_bps'], slope, intercept]).all():
                raise ValueError('状态内收益幅度或概率校准不可用')
            definitions.append(definition)
        for split in ['val_dev', 'val_select']:
            bounds = segments[split]
            states = volatility_states(volatility[bounds['start_row']:bounds['stop_row']], thresholds)
            edges = np.load(original / f'{split}_net_edge_bps.npy')
            probability = (edges - metadata['train_loss_mean_bps']) / amplitude
            for method in ['payoff', 'calibrated']:
                folder = output / 'models' / f'{method}_seed{seed}_side{side}'
                folder.mkdir(parents=True, exist_ok=True)
                prediction = conditional_values(probability, states, definitions, method == 'calibrated')
                np.save(folder / f'{split}_net_edge_bps.npy', prediction)
                write_json_atomic(folder / 'definition.json', {
                    'horizon_s': horizon, 'side': side, 'seed': seed,
                    'thresholds_bps': thresholds.tolist(), 'states': definitions,
                    'method': method, 'unknown_state': 'no_entry',
                })
    write_json_atomic(output / f'seed{seed}_completed.json', {'status': 'success'})
