"""盘点已有模型、验证选择和预测产物，历史探索与正式对照分别记录。"""

import json
from pathlib import Path

import polars as pl


# 【核心逻辑227】按真实产物读取全部模型批次，配置未完成数和成功模型数分别汇总。
def collect_model_inventory(root: Path, output: Path) -> dict:
    """输入项目根目录，输出逐模型明细、批次状态和用于交易的冻结候选索引。"""
    model_root = root / 'workspace/results/models'
    records = []
    batches = []
    for batch in ['overnight_legacy_v1', 'overnight_expanded_v1']:
        folder = model_root / batch
        completed = json.loads((folder / 'completed.json').read_text())
        frozen = json.loads((folder / 'frozen_selection.json').read_text())
        winning_directories = {value['directory'] for value in frozen.values()}
        batch_records = []
        for path in sorted(folder.glob('*/report.json')):
            report = json.loads(path.read_text())
            trial = report['trial']
            validation = report['validation']
            record = {
                'batch': batch, 'model_id': path.parent.name, 'family': trial['family'],
                'horizon_s': trial['horizon_seconds'], 'sequence_s': trial['sequence_seconds'],
                'input_columns': len(report['columns']), 'training_samples_or_draws': report.get('training_draws', report.get('training_rows')),
                'validation_samples': validation['rows'], 'validation_ic': validation['ic'],
                'validation_r2': validation['r2'], 'validation_mse_bps2': validation['mse_bps2'],
                'selected_by_validation': report['directory'] in winning_directories,
                'weights': str(path.parent / ('best.txt' if trial['family'] == 'lightgbm' else 'best.pt')),
                'status': 'trained', 'evaluation_scope': 'historical_exploration',
            }
            batch_records.append(record)
        records.extend(batch_records)
        batches.append({'batch': batch, 'status': 'completed_at_budget', 'trained_models': len(batch_records),
                        'configured_trials': completed['configured_trials'], 'failed_trials': len(completed['failed_trials']),
                        'frozen_horizons': list(frozen), 'test_metrics': completed['test_results']})
    for batch in ['pilot_linear_1s_v1', 'linear_1s_v1']:
        folder = model_root / batch
        models = json.loads((folder / 'models.json').read_text())
        for name, model in models.items():
            records.append({'batch': batch, 'model_id': name, 'family': model['family'], 'horizon_s': 1,
                            'sequence_s': 1, 'input_columns': len(model['columns']),
                            'training_samples_or_draws': model['training_samples'], 'selected_by_validation': True,
                            'weights': str(folder / 'models.json'), 'status': 'trained', 'evaluation_scope': 'linear_baseline'})
        batches.append({'batch': batch, 'status': 'completed', 'trained_models': len(models)})
    for folder in sorted(model_root.glob('comparison_*')):
        if not (folder / 'contract.json').exists():
            batches.append({'batch': folder.name, 'status': 'preparation_incomplete', 'trained_models': 0})
            continue
        case_records = []
        if (folder / 'linear_records.json').exists():
            case_records.extend(json.loads((folder / 'linear_records.json').read_text()))
        for path in sorted(folder.glob('*.json')):
            if path.stem.startswith(('lightgbm_', 'gru_')):
                case_records.append(json.loads(path.read_text()))
        frozen_file = folder / 'frozen.json'
        winner = json.loads(frozen_file.read_text())['winner_by_validation'] if frozen_file.exists() else None
        for case in case_records:
            score = case['validation_tune']
            records.append({'batch': folder.name, 'model_id': case['id'], 'family': case['family'],
                            'horizon_s': 1, 'sequence_s': case['sequence_seconds'], 'input_columns': len(case['columns']),
                            'training_samples_or_draws': case['training_rows'], 'validation_samples': score['rows'],
                            'validation_ic': score['ic'], 'validation_r2': score['r2'], 'validation_mse_bps2': score['mse_bps2'],
                            'selected_by_validation': case['id'] == winner, 'weights': str(folder / case['model_file']),
                            'status': 'trained', 'evaluation_scope': 'train_only_selection_common_rows'})
        completed_file = folder / 'completed.json'
        state = json.loads(completed_file.read_text()) if completed_file.exists() else {'status': 'running'}
        batches.append({'batch': folder.name, 'trained_models': len(case_records), **state})
    validation_root = root / 'workspace/results/information_validation'
    for folder in sorted(validation_root.glob('*/*')):
        model_file = folder / 'models.json'
        if model_file.exists():
            models = json.loads(model_file.read_text())
            for name, model in models.items():
                records.append({'batch': f'information_validation/{folder.parent.name}/{folder.name}',
                                'model_id': name, 'family': model['family'], 'horizon_s': 10 if folder.name == 'diagnostic_10s' else 1,
                                'sequence_s': 1, 'input_columns': len(model['columns']),
                                'training_samples_or_draws': model['training_samples'],
                                'weights': str(model_file), 'status': 'trained', 'evaluation_scope': 'paired_ablation'})
        contract = folder / 'capacity/contract.json'
        if contract.exists():
            specification = json.loads(contract.read_text())
            for name, columns in specification['columns'].items():
                weights = folder / 'capacity' / f'{name}.{ "txt" if name.startswith("tree") else "json"}'
                records.append({'batch': f'information_validation/{folder.parent.name}/{folder.name}/capacity',
                                'model_id': name, 'family': 'lightgbm' if name.startswith('tree') else 'ridge',
                                'horizon_s': 1, 'sequence_s': 1, 'input_columns': len(columns),
                                'training_samples_or_draws': specification['training_samples'], 'weights': str(weights),
                                'status': 'trained' if weights.exists() else 'pending', 'evaluation_scope': 'common_training_capacity'})
    references_file = output / 'reference_models.json'
    if references_file.exists():
        references = json.loads(references_file.read_text())
        for name, model in references.items():
            records.append({
                'batch': 'strategy_reference_20261109',
                'model_id': name,
                'family': model['family'],
                'horizon_s': 1,
                'sequence_s': 1,
                'input_columns': len(model['columns']),
                'training_samples_or_draws': model['training_samples'],
                'validation_samples': model['validation']['rows'],
                'validation_ic': model['validation']['ic'],
                'validation_r2': model['validation']['r2'],
                'weights': str(references_file),
                'status': 'trained',
                'evaluation_scope': 'execution_attribution_reference',
            })
        batches.append({
            'batch': 'strategy_reference_20261109',
            'status': 'completed',
            'trained_models': len(references),
        })
    explorations = []
    for path in sorted((root / 'workspace/results/info_increment').glob('*results.json')):
        data = json.loads(path.read_text())
        models = archived_model_metrics(data)
        explorations.append({'source': str(path), 'models': models,
                             'status': 'archived_exploration_separate_selection_and_sample_rules'})
    result = {'batches': batches, 'models': records, 'archived_explorations': explorations}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model_inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    pl.DataFrame(records, infer_schema_length=None).write_csv(output / 'model_inventory.csv')
    return result


# 【非核心】按旧研究文件的嵌套实验名提取模型指标，条件收益表保持为独立研究产物。
def archived_model_metrics(node: dict, prefix: str = '') -> list[dict]:
    """只登记含预测IC的验证结果，保留文件内原有模型路径和评价口径。"""
    records = []
    for name, value in node.items():
        if not isinstance(value, dict):
            continue
        identifier = f'{prefix}/{name}' if prefix else name
        if isinstance(value.get('val'), dict) and 'ic' in value['val']:
            records.append({'id': identifier, 'validation': value['val'], 'test': value.get('test')})
        else:
            records.extend(archived_model_metrics(value, identifier))
    return records
