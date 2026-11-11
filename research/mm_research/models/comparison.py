"""训练期选列、共同样本和有限内存矩阵下的因子增量与模型对照。"""

import json
import os
import sys
import tarfile
from pathlib import Path
import tempfile

import numpy as np

from mm_research.config import load_pipeline_config
from mm_research.data.storage import write_json_atomic
from mm_research.models.comparison_contract import select_comparison_inputs, comparison_indices
from mm_research.models.comparison_metrics import predict_comparison, prediction_totals, prediction_metrics, evaluate_comparison
from mm_research.models.experiment_data import prepare_experiment_data, history_batch
from mm_research.models.linear import linear_statistics, merge_statistics, subset_statistics, fit_linear_model, score_linear_model


# 【非核心】读取当前正式配置，运行中及恢复时要求配置完全相同。
def comparison_context(config_path: Path) -> tuple:
    """矩阵始终只读映射，实际内存由计算块及受控子进程决定。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, 'model_comparison')
    output = root / config['output']
    contract = json.loads((output / 'contract.json').read_text())
    if contract['config'] != config:
        raise ValueError('正式实验配置变化，需要新的输出版本')
    metadata = json.loads((root / config['dataset_output'] / 'metadata.json').read_text())
    selection = json.loads((output / 'selection.json').read_text())
    matrix = np.load(root / config['dataset_output'] / 'x.npy', mmap_mode='r')
    targets = np.load(root / config['dataset_output'] / 'y.npy', mmap_mode='r')
    indices = {path.stem: np.load(path, mmap_mode='r') for path in (output / 'rows').glob('*.npy')}
    return root, config, output, metadata, selection, matrix, targets, indices


# 【核心逻辑133】冻结训练期候选与唯一研究矩阵，所有后续模型读取同一份数据契约。
def prepare_comparison(config_path: Path) -> None:
    """输入配置，输出选择清单、训练标准化、共同时间位置及原子发布的磁盘矩阵。"""
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, 'model_comparison')
    output = root / config['output']
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text())['config'] != config:
        raise ValueError('正式实验配置变化，需要新的输出版本')
    snapshot = output / 'source_snapshot.tar.gz'
    if not snapshot.exists():
        with tarfile.open(snapshot, 'w:gz') as archive:
            sources = sorted((root / 'research').rglob('*.py'))
            sources += [root / 'scripts/run_model_comparison.py', config_path.resolve(), root / 'pyproject.toml',
                        root / 'config/research/model_comparison.yaml', root / 'scripts/run_overnight.py',
                        root / 'scripts/run_overnight_server.sh']
            for source in sources:
                archive.add(source, arcname=str(source.relative_to(root)))
    selection = select_comparison_inputs(root, config)
    metadata = prepare_experiment_data(root, config, selection['inputs'], selection['derivations'])
    dates = metadata['identity']['dataset']['dates']
    if dates[0] != config['start_date'] or dates[-1] != config['end_date']:
        raise ValueError('研究区间与配置不一致')
    targets = np.load(root / config['dataset_output'] / 'y.npy', mmap_mode='r')
    indices = comparison_indices(metadata, targets, config)
    (output / 'rows').mkdir(exist_ok=True)
    for split, rows in indices.items():
        np.save(output / 'rows' / f'{split}.npy', rows)
    write_json_atomic(output / 'selection.json', selection)
    write_json_atomic(contract_path, {'config': config, 'row_counts': {name: len(rows) for name, rows in indices.items()},
                      'selection': selection['criterion'], 'missing': 'training_mean_plus_missing_indicator',
                      'normalization': 'training_label_valid_only_clip20', 'boundary_purge_seconds': 300,
                      'test_status': 'previously_inspected_historical_reassessment', 'matrix_columns': metadata['columns'],
                      'source_snapshot': snapshot.name, 'python': sys.version, 'numpy': np.__version__})
    (output / 'models').mkdir(exist_ok=True)
    print('COMPARISON_DATA_READY', flush=True)


# 【非核心】在共同调参位置计算MSE，验证集只负责选择模型参数。
def tuning_metrics(record: dict, model, matrix, targets, rows, metadata) -> dict:
    """按块预测，避免将验证期全量矩阵转换成Float64。"""
    totals = np.zeros(7)
    for offset in range(0, len(rows), 4096):
        batch = rows[offset:offset + 4096]
        predicted = predict_comparison(record, model, matrix, batch, metadata)
        actual = np.asarray(targets[batch, 0], dtype=np.float64)
        totals += prediction_totals(actual, predicted)
    return prediction_metrics(totals)


# 【核心逻辑140】共用一次训练与调参统计，逐组拟合线性基线并按相同验证MSE选参数。
def train_comparison_linear(config_path: Path) -> None:
    """充分统计只读相同六百万训练行，OLS/Ridge/Lasso复用原有拟合接口。"""
    from threadpoolctl import threadpool_limits
    root, config, output, metadata, selection, matrix, targets, indices = comparison_context(config_path)
    if (output / 'linear_records.json').exists():
        return
    count = len(metadata['columns'])
    columns = np.arange(count)
    statistics = {}
    with threadpool_limits(limits=config['threads'], user_api='blas'):
        for split in ['fit', 'tune']:
            accumulated = None
            rows = indices[split]
            for offset in range(0, len(rows), config['chunk_rows']):
                batch = rows[offset:offset + config['chunk_rows']]
                values = history_batch(matrix, batch, columns, 1)[:, 0].astype(np.float64)
                current = linear_statistics(values, np.asarray(targets[batch, 0], dtype=np.float64))
                accumulated = current if accumulated is None else merge_statistics(accumulated, current)
            statistics[split] = accumulated
    records = []
    failures = []
    for group, names in selection['groups'].items():
        selected = [metadata['columns'].index(name) for name in names]
        expanded = selected + [index + count for index in selected]
        train = subset_statistics(statistics['fit'], expanded)
        validation = subset_statistics(statistics['tune'], expanded)
        families = {'ridge': config['ridge_alphas']}
        if group in ['base90', 'full']:
            families.update(ols=[0.], lasso=config['lasso_alphas'])
        for family, alphas in families.items():
            candidates = []
            for alpha in alphas:
                try:
                    model = fit_linear_model(train, family, alpha)
                except (ValueError, np.linalg.LinAlgError) as error:
                    failures.append({'group': group, 'family': family, 'alpha': alpha, 'error': str(error)})
                    continue
                score = score_linear_model(validation, model)
                candidates.append((score['mse_bps2'], model, score))
            if not candidates:
                continue
            _, model, score = min(candidates, key=lambda item: item[0])
            name = f'{family}_{group}'
            write_json_atomic(output / 'models' / f'{name}.json', model)
            record = {'id': name, 'family': family, 'group': group, 'columns': selected, 'sequence_seconds': 1,
                      'model_file': f'models/{name}.json', 'validation_tune': score,
                      'training_rows': len(indices['fit']), 'alpha': model['alpha']}
            records.append(record)
            print('LINEAR_READY', name, score['mse_bps2'], flush=True)
    write_json_atomic(output / 'linear_records.json', records)
    write_json_atomic(output / 'linear_failures.json', failures)


# 【核心逻辑141】顺序写入有限内存训练文件，树模型共用训练行和训练均值缺失表达。
def train_comparison_tree(config_path: Path, group: str) -> None:
    """训练输入使用只读映射，LightGBM构建直方图时无需整表浮点复制。"""
    import lightgbm as lgb
    root, config, output, metadata, selection, matrix, targets, indices = comparison_context(config_path)
    name = f'lightgbm_{group}'
    if (output / f'{name}.json').exists():
        return
    columns = np.array([metadata['columns'].index(column) for column in selection['groups'][group]])
    record = {'id': name, 'family': 'lightgbm', 'group': group, 'columns': columns.tolist(), 'sequence_seconds': 1,
              'model_file': f'models/{name}.txt', 'training_rows': len(indices['fit'])}
    best = np.inf
    candidates = []
    with tempfile.TemporaryDirectory(prefix=f'.tree-{group}-', dir=output) as temporary:
        path = Path(temporary) / 'training.npy'
        with path.open('wb') as stream:
            np.lib.format.write_array_header_2_0(stream, {'descr': '<f4', 'fortran_order': False,
                                                         'shape': (len(indices['fit']), len(columns) * 2)})
            for offset in range(0, len(indices['fit']), config['chunk_rows']):
                batch = indices['fit'][offset:offset + config['chunk_rows']]
                stream.write(history_batch(matrix, batch, columns, 1)[:, 0].tobytes())
        training = np.load(path, mmap_mode='r')
        validation = history_batch(matrix, indices['tune'], columns, 1)[:, 0]
        for leaves in config['tree']['leaves']:
            model = lgb.LGBMRegressor(n_estimators=config['tree']['rounds'], num_leaves=leaves,
                learning_rate=.05, min_child_samples=1000, max_bin=63, reg_lambda=1.,
                subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=config['seed'],
                n_jobs=config['threads'], verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(training, targets[indices['fit'], 0], eval_set=[(validation, targets[indices['tune'], 0])],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
            score = tuning_metrics(record, model.booster_, matrix, targets, indices['tune'], metadata)
            candidates.append({'leaves': leaves, 'best_iteration': model.best_iteration_, **score})
            if score['mse_bps2'] < best:
                best = score['mse_bps2']
                model.booster_.save_model(str(output / record['model_file']))
                record.update(validation_tune=score, leaves=leaves, best_iteration=model.best_iteration_)
            del model
    record['candidates'] = candidates
    write_json_atomic(output / f'{name}.json', record)
    print('TREE_READY', name, best, flush=True)


# 【核心逻辑143】GRU在同一份训练行上完整训练，每轮使用同一行集的确定性排列。
def train_comparison_gru(config_path: Path, sequence: int) -> None:
    """GPU分配上限80%，与线性及树模型共用缺失通道、目标单位和调参位置。"""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from mm_research.models.neural import FactorNetwork
    root, config, output, metadata, selection, matrix, targets, indices = comparison_context(config_path)
    name = f'gru_base90_{sequence}s'
    if (output / f'{name}.json').exists():
        return
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    columns = np.array([metadata['columns'].index(column) for column in selection['groups']['base90']])
    model = FactorNetwork('gru', len(columns) * 2, sequence, config['gru']['hidden']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    scale = metadata['target_scale_bps'][0]
    record = {'id': name, 'family': 'gru', 'group': 'base90', 'columns': columns.tolist(), 'sequence_seconds': sequence,
              'model_file': f'models/{name}.pt', 'hidden': config['gru']['hidden'], 'training_rows': len(indices['fit'])}
    best = np.inf
    for epoch in range(config['gru']['epochs']):
        order = np.random.default_rng(config['seed'] + epoch).permutation(len(indices['fit']))
        model.train()
        for offset in range(0, len(order), config['gru']['batch_size']):
            batch = indices['fit'][order[offset:offset + config['gru']['batch_size']]]
            values = history_batch(matrix, batch, columns, sequence)
            response = np.asarray(targets[batch, 0], dtype=np.float32) / scale
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = torch.nn.functional.mse_loss(model(torch.from_numpy(values).cuda()), torch.from_numpy(response).cuda())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            if offset % (config['gru']['batch_size'] * 500) == 0:
                print('GRU_PROGRESS', sequence, epoch, offset, flush=True)
        model.eval()
        score = tuning_metrics(record, model, matrix, targets, indices['tune'], metadata)
        if score['mse_bps2'] < best:
            best = score['mse_bps2']
            torch.save(model.state_dict(), output / record['model_file'])
            record.update(validation_tune=score, selected_epoch=epoch + 1)
    record['training_draws'] = len(indices['fit']) * config['gru']['epochs']
    write_json_atomic(output / f'{name}.json', record)
    print('GRU_READY', name, best, flush=True)


# 【核心逻辑144】所有候选调参结束后冻结选择，测试标签只进入后续评价阶段。
def freeze_comparison(config_path: Path) -> None:
    root, config, output, metadata, selection, matrix, targets, indices = comparison_context(config_path)
    linear_file = output / 'linear_records.json'
    records = json.loads(linear_file.read_text()) if linear_file.exists() else []
    expected = [f'ridge_{group}' for group in selection['groups']]
    expected += [f'{family}_{group}' for family in ['ols', 'lasso'] for group in ['base90', 'full']]
    expected += [f'lightgbm_{group}' for group in selection['groups']]
    expected += [f'gru_base90_{sequence}s' for sequence in config['gru']['sequences']]
    for name in expected:
        if name.startswith(('lightgbm_', 'gru_')) and (output / f'{name}.json').exists():
            records.append(json.loads((output / f'{name}.json').read_text()))
    if not records:
        raise RuntimeError('没有成功完成调参的模型，详见独立任务日志')
    missing = sorted(set(expected) - {record['id'] for record in records})
    winner = min(records, key=lambda record: record['validation_tune']['mse_bps2'])['id']
    write_json_atomic(output / 'frozen.json', {'winner_by_validation': winner, 'records': records,
                      'selection_uses_test': False, 'missing_training_cases': missing, 'row_counts': {name: len(rows) for name, rows in indices.items()}})


# 【非核心】恢复冻结模型，权重、列顺序和序列长度均由同一清单指定。
def load_comparison_model(record: dict, output: Path):
    path = output / record['model_file']
    if record['family'] == 'lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    if record['family'] == 'gru':
        import torch
        from mm_research.models.neural import FactorNetwork
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.8)
        model = FactorNetwork('gru', len(record['columns']) * 2, record['sequence_seconds'], record['hidden']).cuda()
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.eval()
    return json.loads(path.read_text())


# 【核心逻辑145】冻结后统一评价全部验证和测试行，分组阈值只取共同训练样本。
def evaluate_comparison_family(config_path: Path, neural: bool, model_id: str | None = None) -> None:
    root, config, output, metadata, selection, matrix, targets, indices = comparison_context(config_path)
    frozen = json.loads((output / 'frozen.json').read_text())
    for record in frozen['records']:
        if (record['family'] == 'gru') != neural or (model_id is not None and record['id'] != model_id):
            continue
        directory = output / 'evaluation' / record['id']
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'report.json').exists():
            continue
        model = load_comparison_model(record, output)
        predictions = []
        for offset in range(0, len(indices['threshold']), 4096):
            predictions.append(predict_comparison(record, model, matrix, indices['threshold'][offset:offset + 4096], metadata))
        thresholds = np.quantile(np.concatenate(predictions), [.2, .4, .6, .8]).tolist()
        result = {'model': record, 'thresholds': thresholds}
        for split in ['validation', 'test']:
            result[split] = evaluate_comparison(record, model, matrix, targets, indices[split], metadata,
                                                thresholds, directory / f'{split}.parquet')
        write_json_atomic(directory / 'report.json', result)
        print('EVALUATED', record['id'], result['test']['horizons']['1'], flush=True)
        del model
