"""汇总常驻队列的真实状态、成功评价与模型结果，失败任务保留独立说明。"""

import datetime as dt
import json
from pathlib import Path

import polars as pl
import yaml

from mm_research.data.storage import write_json_atomic


# 【核心逻辑157】汇总已成功的分组评估与独立模型结果，生成可直接阅读的夜间报告。
def collect_night_results(config_path: Path) -> None:
    """只纳入状态为success的任务，未完成及失败项在报告中明确列出。"""
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(config_path.read_text())
    output = root / config['output']
    state = json.loads((output / 'status.json').read_text())
    config = state['config']
    grouped = {}
    lines = ['# 夜间研究运行报告', '', f"更新时间为 {dt.datetime.now(dt.timezone.utc).isoformat()}。", '',
             '## 任务状态', '', '| 任务 | 状态 | 峰值匿名内存 GiB | 原因 |', '| --- | --- | --- | --- |']
    for task in config['tasks']:
        record = state['tasks'][task['id']]
        memory = record.get('peak_anonymous_bytes', 0) / 1024 ** 3
        lines.append(f"| {task['id']} | {record['status']} | {memory:.2f} | {record.get('reason', '')} |")
        if task['id'].startswith('evaluation_') and record['status'] == 'success':
            evaluation = yaml.safe_load((root / task['command'][-1]).read_text())
            folder = root / evaluation['output']
            grouped.setdefault(folder, []).extend(folder / name / 'summary.parquet' for name in evaluation['factors'])
    for folder in grouped:
        # 恢复队列只重跑部分组，汇总保留同版本此前已完成的全部因子。
        files = sorted(filename for filename in folder.glob("*/summary.parquet")
                       if (filename.parent / "completed.json").exists())
        tables = [pl.read_parquet(filename) for filename in files]
        summary = pl.concat(tables)
        summary.write_parquet(folder / 'summary.parquet', compression='zstd')
        factors = sorted(summary['factor'].unique().to_list())
        links = ''.join(f'<li><a href="{name}/report.html">{name}</a></li>' for name in factors)
        (folder / 'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>已完成分组评价</h1><ul>'+links+'</ul>')
        lines.extend(['', f"{folder.relative_to(root)} 已汇总 {len(factors)} 个成功因子。"])
    lines.extend(['', '## 模型结果', ''])
    for name in ['overnight_legacy_v1', 'overnight_expanded_v1']:
        folder = root / 'workspace/results/models' / name
        completed = folder / 'completed.json'
        if completed.exists():
            result = json.loads(completed.read_text())
            lines.extend([f'### {name}', '', f"成功完成 {result['trials']} 项实验。", '',
                          '| 目标秒数 | 测试样本 | RMSE bps | IC | R² |', '| --- | --- | --- | --- | --- |'])
            for horizon, metrics in result['test_results'].items():
                lines.append(f"| {horizon} | {metrics['rows']} | {metrics['rmse_bps']:.6g} | {metrics['ic']} | {metrics['r2']} |")
        else:
            lines.extend([f'{name} 尚未完成冻结测试。', ''])
    lines.extend(['', '## 口径', '',
                  'Factors保持Float64。神经模型共享缓存使用Float32，训练期标准化，缺失输入使用训练均值并增加缺失通道。',
                  '模型候选只用训练期选列，模型与参数只用验证集选择，测试预测为整秒。',
                  'Gen030缺少现货盘口，尚未进入生产。失败组在状态表保留，扩展模型明确记录未使用的组。'])
    (output / 'report.md').write_text('\n'.join(lines)+'\n')
    write_json_atomic(output / 'summary.json', {'updated_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                                                'task_status': {name: record['status'] for name, record in state['tasks'].items()}})
