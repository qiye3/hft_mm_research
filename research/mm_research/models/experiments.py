"""有明确截止时间的模型实验队列，验证集选择模型，冻结后统一评价测试集。"""

import datetime as dt
import json
import random
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import polars as pl
import pyarrow.parquet as pq
import torch
from torch import nn
import yaml

from mm_research.data.storage import write_json_atomic
from mm_research.models.experiment_data import HORIZONS, prepare_experiment_data
from mm_research.models.neural import FactorNetwork


# 【核心逻辑134】按输入列训练IC选择每个目标的候选，排序只使用训练期统计。
def selected_columns(metadata: dict, horizon_index: int, count: int) -> np.ndarray:
    """固定保留六列原始基线，其余按训练绝对IC挑选，输出稳定列顺序。"""
    correlations = np.abs(np.asarray(metadata['train_ic'][horizon_index], dtype=np.float64))
    order = np.argsort(-correlations, kind='stable')
    selected = list(range(6))
    selected.extend(int(index) for index in order if index >= 6 and np.isfinite(correlations[index]))
    return np.asarray(selected[:count], dtype=np.int64)


# 【核心逻辑136】从完整时间网格读取过去序列，缺失用训练均值及显式标记处理。
def history_batch(matrix, indices: np.ndarray, columns: np.ndarray, sequence: int) -> np.ndarray:
    """标准化矩阵的零对应训练均值，附加等宽缺失通道，不跨越未来时点。"""
    positions = indices[:, None] + np.arange(1 - sequence, 1)
    values = np.asarray(matrix[positions[:, :, None], columns[None, None, :]])
    missing = ~np.isfinite(values)
    values = np.nan_to_num(values, nan=0, copy=False)
    return np.concatenate([values, missing.astype(np.float32)], axis=-1)


# 【非核心】以固定时间位置抽取验证样本，不根据实际收益选择样本。
def evaluation_indices(targets, metadata: dict, horizon_index: int, split: str, maximum: int | None) -> np.ndarray:
    """验证统一均匀筛查，最终测试使用全部有效整秒样本。"""
    start = metadata['validation_start'] if split == 'validation' else metadata['test_start']
    stop = metadata['validation_stop'] if split == 'validation' else metadata['rows']
    indices = np.arange(start, stop, dtype=np.int64)
    indices = indices[np.isfinite(targets[indices, horizon_index])]
    if maximum is not None and len(indices) > maximum:
        indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]
    return indices


# 【核心逻辑138】按批次推理并累积误差与相关，最终测试可保存完整整秒预测。
def evaluate_model(model, matrix, targets, indices: np.ndarray, columns: np.ndarray,
                   trial: dict, metadata: dict, destination: Path | None = None) -> dict:
    """输入冻结模型与明确样本位置，返回bps指标，神经网络仅调用eval模式。"""
    horizon_index = HORIZONS.index(trial['horizon_seconds'])
    scale = metadata['target_scale_bps'][horizon_index]
    totals = np.zeros(7, dtype=np.float64)
    writer = None
    if trial['family'] != 'lightgbm':
        model.eval()
    for offset in range(0, len(indices), 2048):
        batch = indices[offset:offset + 2048]
        inputs = history_batch(matrix, batch, columns, trial['sequence_seconds'])
        if trial['family'] == 'lightgbm':
            prediction = model.predict(inputs[:, -1], num_threads=2)
        else:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                prediction = model(torch.from_numpy(inputs).to('cuda')).float().cpu().numpy() * scale
        actual = np.asarray(targets[batch, horizon_index], dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        totals += [len(batch), actual.sum(), prediction.sum(), actual @ actual,
                   prediction @ prediction, actual @ prediction, (actual - prediction) @ (actual - prediction)]
        if destination is not None:
            table = pa.table({
                'timestamp_ns': (metadata['start_timestamp_seconds'] + batch) * 1_000_000_000,
                'target_bps': actual, 'prediction_bps': prediction,
            })
            if writer is None:
                writer = pq.ParquetWriter(destination.with_suffix(".partial.parquet"), table.schema, compression='zstd')
            writer.write_table(table)
    if writer is not None:
        writer.close()
        destination.with_suffix(".partial.parquet").replace(destination)
    count, sum_y, sum_prediction, sum_yy, sum_prediction2, sum_cross, error = totals
    variance_y = max(sum_yy - sum_y ** 2 / count, 0)
    variance_prediction = max(sum_prediction2 - sum_prediction ** 2 / count, 0)
    denominator = np.sqrt(variance_y * variance_prediction)
    return {'rows': int(count), 'rmse_bps': float(np.sqrt(error / count)), 'mse_bps2': float(error / count),
            'ic': float((sum_cross - sum_y * sum_prediction / count) / denominator) if denominator > 0 else None,
            'r2': float(1 - error / variance_y) if variance_y > 0 else None,
            'zero_prediction_mse_bps2': float(sum_yy / count)}


# 【核心逻辑133】在固定训练步数和时间预算内拟合神经网络，保留验证误差最低状态。
def train_neural(trial: dict, config: dict, metadata: dict, matrix, targets, output: Path, deadline: float) -> dict:
    """均匀抽取全部训练日期的合法秒点，随机种子、列顺序和标准化随检查点保存。"""
    seed = trial['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = np.random.default_rng(seed)
    horizon_index = HORIZONS.index(trial['horizon_seconds'])
    columns = selected_columns(metadata, horizon_index, config['selected_columns'])
    sequence = trial['sequence_seconds']
    eligible = np.flatnonzero(np.isfinite(targets[:metadata['train_stop'], horizon_index]))
    eligible = eligible[eligible >= sequence - 1]
    validation = evaluation_indices(targets, metadata, horizon_index, 'validation', config['validation_samples'])
    model = FactorNetwork(trial['family'], len(columns) * 2, sequence, trial['hidden']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial['learning_rate'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    scale = metadata['target_scale_bps'][horizon_index]
    best = float('inf')
    started = time.time()
    history = []
    steps = 0
    for step in range(trial['steps']):
        if time.time() >= min(deadline, started + config['maximum_trial_seconds']):
            break
        model.train()
        batch = generator.choice(eligible, size=trial['batch_size'], replace=True)
        inputs = history_batch(matrix, batch, columns, sequence)
        actual = np.asarray(targets[batch, horizon_index] / scale, dtype=np.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            prediction = model(torch.from_numpy(inputs).to('cuda'))
            loss = nn.functional.mse_loss(prediction, torch.from_numpy(actual).to('cuda'))
        if not torch.isfinite(loss):
            raise ValueError(f'训练损失非有限 {trial}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        steps = step + 1
        if steps % config['validate_every_steps'] == 0 or steps == trial['steps']:
            metrics = evaluate_model(model, matrix, targets, validation, columns, trial, metadata)
            history.append({'steps': steps, **metrics})
            if metrics['mse_bps2'] < best:
                best = metrics['mse_bps2']
                torch.save({'state_dict': model.state_dict(), 'trial': trial, 'columns': columns.tolist(),
                            'target_scale_bps': scale, 'validation': metrics}, output / 'best.pt')
            print(json.dumps({'trial': output.name, 'steps': steps, 'validation': metrics}), flush=True)
    if not history or history[-1]['steps'] != steps:
        metrics = evaluate_model(model, matrix, targets, validation, columns, trial, metadata)
        history.append({'steps': steps, **metrics})
        if metrics['mse_bps2'] < best:
            torch.save({'state_dict': model.state_dict(), 'trial': trial, 'columns': columns.tolist(),
                        'target_scale_bps': scale, 'validation': metrics}, output / 'best.pt')
    checkpoint = torch.load(output / 'best.pt', weights_only=True)
    return {'trial': trial, 'validation': checkpoint['validation'], 'steps_completed': steps,
            'seconds': time.time() - started, 'training_draws': steps * trial['batch_size'],
            'validation_history': history, 'columns': columns.tolist()}


# 【核心逻辑139】用相同训练候选与固定验证集拟合树模型，明确记录训练抽样规模。
def train_lightgbm(trial: dict, config: dict, metadata: dict, matrix, targets, output: Path, deadline: float) -> dict:
    """最多使用预设训练秒点数量，验证早停决定树数，目标仍以bps表达。"""
    import lightgbm as lgb
    started = time.time()
    horizon_index = HORIZONS.index(trial['horizon_seconds'])
    columns = selected_columns(metadata, horizon_index, config['selected_columns'])
    indices = np.flatnonzero(np.isfinite(targets[:metadata['train_stop'], horizon_index]))
    if len(indices) > config['tree_training_samples']:
        indices = indices[np.linspace(0, len(indices) - 1, config['tree_training_samples'], dtype=np.int64)]
    validation = evaluation_indices(targets, metadata, horizon_index, 'validation', config['validation_samples'])
    training_x = history_batch(matrix, indices, columns, 1)[:, -1]
    validation_x = history_batch(matrix, validation, columns, 1)[:, -1]
    model = lgb.LGBMRegressor(n_estimators=1000, num_leaves=trial['hidden'], learning_rate=.03,
                              max_depth=-1, min_child_samples=200, reg_lambda=1, n_jobs=3,
                              random_state=trial['seed'], verbosity=-1)
    best_score = float('inf')
    best_iteration = 0
    best_evaluation = []

    # 【非核心】树模型也遵守每次实验的时间预算，结束时保留已观察到的最佳验证轮次。
    def stop_at_budget(environment):
        """LightGBM迭代回调，仅使用验证误差记录最佳轮次。"""
        nonlocal best_score, best_iteration, best_evaluation
        score = environment.evaluation_result_list[0][2]
        if score < best_score:
            best_score = score
            best_iteration = environment.iteration
            best_evaluation = environment.evaluation_result_list
        if time.time() >= min(deadline, started + config['maximum_trial_seconds']):
            raise lgb.callback.EarlyStopException(best_iteration, best_evaluation)

    model.fit(training_x, targets[indices, horizon_index], eval_set=[(validation_x, targets[validation, horizon_index])],
              callbacks=[lgb.early_stopping(50, verbose=False), stop_at_budget])
    model.booster_.save_model(str(output / 'best.txt'))
    metrics = evaluate_model(model, matrix, targets, validation, columns, trial, metadata)
    return {'trial': trial, 'validation': metrics, 'steps_completed': model.best_iteration_,
            'training_rows': len(indices), 'seconds': time.time() - started, 'columns': columns.tolist()}


# 【核心逻辑129】运行预先列出的模型队列，到截止时间前冻结选择并测试最佳模型。
def run_model_experiments(config_path: Path) -> None:
    """队列按验证期MSE选每周期赢家，所有测试结果在选择结束后统一生成。"""
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(config_path.read_text())
    output = root / config['output']
    output.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat(config['training_deadline']).timestamp()
    metadata = prepare_experiment_data(root, config)
    matrix = np.load(root / config['dataset_output'] / 'x.npy', mmap_mode='r')
    targets = np.load(root / config['dataset_output'] / 'y.npy', mmap_mode='r')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    if not torch.cuda.is_available():
        raise RuntimeError('深度实验要求可用CUDA GPU')
    write_json_atomic(output / 'environment.json', {'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__,
                                                   'polars': pl.__version__, 'arrow': pa.__version__,
                                                   'python': sys.version,
                                                   'gpu': torch.cuda.get_device_name(), 'config': config})
    results = []
    failures = []
    for index, trial in enumerate(config['trials']):
        directory = output / f'{index:03d}_{trial["family"]}_{trial["horizon_seconds"]}s'
        report_file = directory / 'report.json'
        if report_file.exists():
            report = json.loads(report_file.read_text())
            if report['trial'] != trial:
                raise ValueError('实验配置变化，需要新输出版本')
            results.append(report)
            continue
        if time.time() + 60 >= deadline:
            break
        directory.mkdir(parents=True, exist_ok=True)
        print(f'模型实验开始 {directory.name}', flush=True)
        # 单次实验进程退出后释放CUDA上下文，失败不会阻断其他独立候选。
        with (directory / 'training.log').open('w') as log:
            process = subprocess.run([sys.executable, 'scripts/run_model_experiments.py', str(config_path),
                                      '--trial-index', str(index)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            failures.append({'trial': trial, 'returncode': process.returncode, 'directory': str(directory.relative_to(root))})
        else:
            results.append(json.loads(report_file.read_text()))
        write_json_atomic(output / 'progress.json', {'completed_trials': len(results), 'failed_trials': failures,
                          'total_trials': len(config['trials']), 'updated_at': time.time(), 'results': results})
    winners = {}
    for report in results:
        horizon = str(report['trial']['horizon_seconds'])
        if horizon not in winners or report['validation']['mse_bps2'] < winners[horizon]['validation']['mse_bps2']:
            winners[horizon] = report
    if not winners:
        raise RuntimeError('本轮没有成功完成验证的模型，详见逐实验日志')
    write_json_atomic(output / 'frozen_selection.json', winners)
    test_results = {}
    for horizon, report in winners.items():
        trial = report['trial']
        directory = root / report['directory']
        columns = np.asarray(report['columns'], dtype=np.int64)
        if trial['family'] == 'lightgbm':
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(directory / 'best.txt'))
        else:
            checkpoint = torch.load(directory / 'best.pt', weights_only=True)
            model = FactorNetwork(trial['family'], len(columns) * 2, trial['sequence_seconds'], trial['hidden']).cuda()
            model.load_state_dict(checkpoint['state_dict'])
        indices = evaluation_indices(targets, metadata, HORIZONS.index(int(horizon)), 'test', None)
        destination = output / f'test_predictions_{horizon}s.parquet'
        test_results[horizon] = evaluate_model(model, matrix, targets, indices, columns, trial, metadata, destination)
        write_json_atomic(output / 'test_results.json', test_results)
        print(json.dumps({'test_horizon': horizon, 'result': test_results[horizon]}), flush=True)
    write_json_atomic(output / 'completed.json', {'completed_at': time.time(), 'trials': len(results), 'failed_trials': failures,
                                                  'configured_trials': len(config['trials']), 'test_results': test_results})
    print('MODEL_EXPERIMENTS_COMPLETE', flush=True)


# 【核心逻辑132】在独立进程中执行一个已列明候选，结果原子发布后退出。
def run_single_trial(config_path: Path, index: int) -> None:
    """使用共享矩阵和固定期限，训练失败保持原异常供父进程记录。"""
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(config_path.read_text())
    metadata = json.loads((root / config['dataset_output'] / 'metadata.json').read_text())
    matrix = np.load(root / config['dataset_output'] / 'x.npy', mmap_mode='r')
    targets = np.load(root / config['dataset_output'] / 'y.npy', mmap_mode='r')
    trial = config['trials'][index]
    directory = root / config['output'] / f'{index:03d}_{trial["family"]}_{trial["horizon_seconds"]}s'
    directory.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat(config['training_deadline']).timestamp()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    if trial['family'] == 'lightgbm':
        report = train_lightgbm(trial, config, metadata, matrix, targets, directory, deadline)
    else:
        report = train_neural(trial, config, metadata, matrix, targets, directory, deadline)
    report['directory'] = str(directory.relative_to(root))
    write_json_atomic(directory / 'report.json', report)
