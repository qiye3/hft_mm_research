"""按实际完成产物生成整夜研究状态，缺失阶段保留具体原因。"""

import datetime as dt
import json
from pathlib import Path

import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.multihorizon.common import context


# 【核心逻辑275】将数据、模型、仓位与回放的真实进度汇总为可阅读的晨报。
def write_report(filename: Path) -> None:
    """所有数字来自本轮正式产物，未完成和受数据限制的验收单独列出。"""
    root, config, output = context(filename)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
    lines = ['# 多期限预测与交易研究进度', '', f'更新时间为 {now}。', '',
             f'本轮截止时间为 {config["stop_at"]}。', '', '## 数据准备', '']
    manifest = output / 'e0/manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text())
        quality = json.loads((output / 'e0/data_quality.json').read_text())
        lines.append(f'已准备{len(data["dates"])}天、{data["rows"]:,}个整秒位置、{len(data["feature_names"])}个固定输入，16期限累计及分段对数bps标签。')
        lines.append('分区共同有效行数为 ' + str(quality['effective_rows']) + '。')
        lines.append('特征采用上一完整秒并计入计算延迟；候选按构造组均匀固定，旧测试IC未参与选列。')
    else:
        progress = output / 'e0/progress.json'
        lines.append('准备进行中 ' + (progress.read_text() if progress.exists() else '等待首批数据'))
    inventory = output / 'l2/archive_inventory.json'
    if inventory.exists():
        sources = json.loads(inventory.read_text())
        lines.append(f'网盘原始L2清单共{len(sources)}日，原档合计{sum(row["entry"]["size"] for row in sources) / 1024**3:.2f}GiB。')
    conversions = []
    for file in sorted((output / 'l2/validation').glob('*.json')):
        conversions.append(json.loads(file.read_text()))
    lines.append(f'本轮已完成转换记录{len(conversions)}日，逐日状态见l2/validation。')
    metrics = output / 'e1/metrics_by_horizon.csv'
    if metrics.exists():
        lines.extend(['', '## 模型对照', '', '| 模型 | 分区 | 期限s | IC | OOS-R² |', '| --- | --- | ---: | ---: | ---: |'])
        table = pl.read_csv(metrics)
        for row in table.filter(pl.col('horizon_s').is_in([1, 10, 60, 120, 300])).iter_rows(named=True):
            lines.append(f'| {row["model_id"]} | {row["split"]} | {row["horizon_s"]} | {row["ic"]} | {row["oos_r2"]} |')
    inventory_metrics = output / 'e3/policy_metrics.csv'
    if inventory_metrics.exists():
        lines.extend(['', '## 动态仓位验证', '', '| 模型 | 分区 | 策略 | 费用前USDT | 手续费USDT | 净收益USDT |',
                      '| --- | --- | --- | ---: | ---: | ---: |'])
        table = pl.read_csv(inventory_metrics).filter(pl.col('split') == 'val_select')
        for row in table.iter_rows(named=True):
            lines.append(f'| {row["model_id"]} | {row["split"]} | {row["id"]} | {row["gross_pnl_usdt"]:.2f} | '
                         f'{row["fees_usdt"]:.2f} | {row["net_pnl_usdt"]:.2f} |')
    status = output / 'night/status.json'
    if status.exists():
        state = json.loads(status.read_text())
        lines.extend(['', '## 任务状态', '', '| 任务 | 状态 |', '| --- | --- |'])
        for name, record in state['tasks'].items():
            lines.append(f'| {name} | {record["status"]} |')
    lines.extend(['', '## 验收范围和后续数据', '',
        'E0—E3沿用既有行情及因子生产，研究目录与原数据版本隔离。研究使用普通用户Maker 2/Taker 5 bps情景，账户实际历史费率尚未提供。',
        '现有历史Test已被前期研究接触，本轮属于重新冻结分区后的历史复核，新的未接触封存验收需要未来独立数据。',
        'P4持续性概率缩放属于条件扩展，本轮不默认启用。E4逐日L2准备和事件回放分别记录，转换成功与原生回放成功不混为一谈。',
        '模型与分区、时钟、标签、费用和具体命令均保存在本轮配置及产物。Git提交由用户决定。', ''])
    (output / 'report.md').write_text('\n'.join(lines))
    write_json_atomic(output / 'progress_summary.json', {'updated_at': now, 'stop_at': config['stop_at'],
                       'e0_ready': manifest.exists(), 'model_metrics_ready': metrics.exists(),
                       'inventory_metrics_ready': inventory_metrics.exists(), 'converted_l2_days': len(conversions)})
