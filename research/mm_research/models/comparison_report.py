"""从冻结清单和共同样本评价生成统一研究结论，历史探索报告单独标记。"""

import datetime as dt
import json
from pathlib import Path

from mm_research.data.storage import write_json_atomic
from mm_research.models.comparison import comparison_context
from mm_research.models.comparison_metrics import paired_day_comparison


# 【核心逻辑147】汇总共同样本上的因子增量、模型差异及五周期衰减，生成统一报告。
def write_comparison_report(config_path: Path) -> None:
    """模型优胜者来自冻结验证选择，报告不按测试成绩重新选模型。"""
    root, config, output, metadata, selection, matrix, targets, indices = comparison_context(config_path)
    if (output / 'completed.json').exists() and json.loads((output / 'completed.json').read_text())['status'] == 'complete':
        return
    frozen = json.loads((output / 'frozen.json').read_text())
    reports = {record['id']: json.loads((output / 'evaluation' / record['id'] / 'report.json').read_text())
               for record in frozen['records'] if (output / 'evaluation' / record['id'] / 'report.json').exists()}
    missing_evaluations = [record['id'] for record in frozen['records'] if record['id'] not in reports]
    expected = {split: len(indices[split]) for split in ['validation', 'test']}
    for name, result in reports.items():
        for split in expected:
            if result[split]['horizons']['1']['rows'] != expected[split]:
                raise ValueError(f'模型评估行集不一致 {name}/{split}')
    comparisons = {}
    pairs = [
        (f'lightgbm_{group}', 'lightgbm_base90')
        for group in ['representatives16', 'depth_expanded', 'sequence_expanded', 'full']
    ]
    pairs += [
        ('lightgbm_without_depth', 'lightgbm_full'),
        ('lightgbm_without_obi', 'lightgbm_full'),
        ('lightgbm_core', 'lightgbm_history'),
        ('ridge_core', 'ridge_history'),
        ('lightgbm_base90', 'ridge_base90'),
        ('lightgbm_full', 'ridge_full'),
        ('gru_base90_1s', 'lightgbm_base90'),
        ('gru_base90_30s', 'gru_base90_1s'),
        ('gru_base90_120s', 'gru_base90_1s'),
    ]
    unavailable_comparisons = []
    for candidate, baseline in pairs:
        if candidate not in reports or baseline not in reports:
            unavailable_comparisons.append([candidate, baseline])
            continue
        comparisons[f'{candidate}__vs__{baseline}'] = paired_day_comparison(
            reports[candidate]['validation']['daily'], reports[baseline]['validation']['daily'], config['seed'])
    timestamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# 统一口径的高频因子与模型研究结论', '', f'生成时间为 {timestamp}。', '',
        '## 数据和实验口径', '',
        '研究区间为2025-05-09至2026-10-21，共531天。100ms Features和整秒Float64 Factors沿用已有正式产物。主模型预测未来1s Mid变化，单位bps。',
        '训练截至2026-05-31，验证为2026-06-01至08-31，回看测试为2026-09-01至10-21。训练与验证末端剔除300秒，并检查每条标签目标时间。',
        '候选排序、方向与覆盖门槛仅使用训练期。90列由训练排名80列和训练期选出的10个家族代表组成。16列代表、深度扩池及序列输入同样只用训练期确定。',
        '所有模型共用有效标签时间戳和同一份训练抽样。特征缺失使用训练均值填补，并附加缺失标记；标准化和clip20在训练、验证及测试一致执行。',
        f"训练候选 {len(indices['train']):,} 行，固定拟合 {len(indices['fit']):,} 行，调参 {len(indices['tune']):,} 行，完整验证 {len(indices['validation']):,} 行，回看测试 {len(indices['test']):,} 行。",
        '数据按日写入Float32磁盘矩阵，模型分块读取。线性模型使用Float64充分统计，LightGBM读取磁盘训练矩阵，GRU读取有界过去序列。',
        '当前Test已在前期研究中被接触，本轮属于统一口径的历史回看。泛化结论需要在之后封存的新数据上确认。', '',
        '## 模型结果', '',
        '模型表IC按整个区间合并计算。增量表先计算每天IC，再比较同一天两个模型的差。', '',
        '| 模型 | 输入列 | 历史秒数 | 完整验证IC | 回看测试IC | 回看测试R² |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for record in frozen['records']:
        if record['id'] not in reports:
            continue
        result = reports[record['id']]
        validation = result['validation']['horizons']['1']
        test = result['test']['horizons']['1']
        ic_validation = '空' if validation['ic'] is None else f"{validation['ic']:.5f}"
        ic_test = '空' if test['ic'] is None else f"{test['ic']:.5f}"
        lines.append(f"| {record['id']} | {len(record['columns'])} | {record['sequence_seconds']} | {ic_validation} | {ic_test} | {test['r2']:.3%} |")
    lines.extend(['', f"冻结的验证期优胜模型为 `{frozen['winner_by_validation']}`。", '',
                  '## 验证期配对增量', '',
                  'IC比较量为候选减基线，MSE减少量为基线减候选，正值表示改善。Bootstrap在同一批日期上重采样日级差值。去组实验与完整输入模型比较。', '',
                  '| 对照 | 日均IC差 | IC差95%区间 | IC改善日 | 日均MSE减少bps² | MSE减少95%区间 |', '| --- | ---: | --- | ---: | ---: | --- |'])
    for name, result in comparisons.items():
        lower, upper = result['ci95']
        mse_lower, mse_upper = result['mse_improvement_ci95']
        lines.append(
            f"| {name} | {result['mean_daily_ic_difference']:+.5f} | "
            f"[{lower:+.5f}, {upper:+.5f}] | {result['positive_day_fraction']:.1%} | "
            f"{result['mean_daily_mse_improvement_bps2']:+.6f} | "
            f"[{mse_lower:+.6f}, {mse_upper:+.6f}] |"
        )
    if frozen['winner_by_validation'] in reports:
        winner = reports[frozen['winner_by_validation']]
        lines.extend(['', '## 冻结模型的多周期衰减', '',
                      '后四个周期使用同一1s模型的预测，只评价方向性衰减。分组阈值来自固定训练样本，未来Mid变化的分组价差用于研究交易成本预算。', '',
                      '| 周期 | 全样本IC | 非重叠IC | 训练阈值高低组价差bps |', '| --- | ---: | ---: | ---: |'])
        for horizon in ['1', '10', '60', '180', '300']:
            result = winner['test']['horizons'][horizon]
            independent = winner['test']['nonoverlap_horizons'][horizon]
            means = winner['test']['train_fixed_group_mean_bps'][horizon]
            spread = '空' if means[0] is None or means[-1] is None else f'{means[-1] - means[0]:.5f}'
            lines.append(f"| {horizon}s | {result['ic']:.5f} | {independent['ic']:.5f} | {spread} |")
    lines.extend(['', '## 对现有研究的统一解释', '',
        '因子数、有效测试日和样本覆盖率分别记录。正式因子共1236项；1s至少50个有效测试日的为1198项，10s为1189项。低覆盖因子单独诊断。',
        '此前A1残差数字对应逐个因子控制OBI。OBI自身残差为零来自对自身回归，不能据此解释为其他六组完全重构OBI。净深度控制OBI后仍保留约0.1207训练期残差IC。',
        '日度IC序列聚类描述因子表现的共同变化。数值相关、残差关系和同样本模型消融分别回答不同问题。',
        '旧90因子实验使用Test选列；v4、v6发生过整表复制OOM。它们保留为历史探索，本轮冻结配置和共同样本结果作为当前比较口径。',
        '成交方向与盘口压力的条件关系存在观察依据，其模型增量以本轮共同样本验证结果衡量。',
        '交易经济性仍需通过实际订单、成交、手续费和库存账本核对。分组未来Mid价差与预测IC用于决定后续回放实验。', '',
        '## 可复现入口', '',
        '`config/models/comparison.yaml` 声明数据、选列、样本和模型参数。',
        '`config/research/model_comparison.yaml` 声明阶段依赖、内存和并发控制。',
        '`scripts/run_model_comparison.py` 为所有阶段的正式入口。',
        '选择清单、共同时间位置、训练标准化、模型权重、冻结选择、逐行预测和逐日统计保存在本报告同目录。'])
    if frozen['missing_training_cases'] or missing_evaluations:
        lines.extend(['', '## 失败隔离记录', '',
                      '以下模型缺少完成产物，其他模型的共同样本评价独立保存。',
                      str({'training': frozen['missing_training_cases'], 'evaluation': missing_evaluations})])
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json_atomic(output / 'comparisons.json', comparisons)
    write_json_atomic(output / 'completed.json', {'completed_at': timestamp, 'models': len(reports), 'status': 'partial' if frozen['missing_training_cases'] or missing_evaluations else 'complete',
                      'missing_training_cases': frozen['missing_training_cases'], 'missing_evaluations': missing_evaluations,
                      'unavailable_comparisons': unavailable_comparisons,
                      'winner_by_validation': frozen['winner_by_validation'], 'row_counts': expected,
                      'test_status': 'historical_reassessment'})
    print('COMPARISON_COMPLETE', len(reports), flush=True)
