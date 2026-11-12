"""将已有180秒Alpha用于宽报价风险保护，直接复用原生委托执行。"""

import json
from pathlib import Path
import subprocess

import polars as pl

from mm_research.data.storage import write_json_atomic
from mm_research.strategies.forecast_research import research_context
from mm_research.strategies.wide_quotes import execute_quote_day, report_wide, select_wide


# 【非核心】本轮单独构建原生策略和直接相关的库存退出测试。
def build_defensive(filename: Path) -> None:
    """构建结果归本轮目录，源订单簿、排队与撮合代码来自固定第三方依赖。"""
    config, source, output = research_context(filename)
    build = Path(config['binary_folder'])
    subprocess.run(['cmake', '-S', '.', '-B', str(build), '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON'], check=True)
    subprocess.run(['cmake', '--build', str(build), '-j2', '--target', 'strategy_replay', 'test_execution_policy'], check=True)
    subprocess.run(['ctest', '--test-dir', str(build), '-R', '^mm.execution_policy$', '--output-on-failure'], check=True)


# 【核心逻辑365】直接复用三种子信号和Train门槛，导出既定原生日期所需的完整预测网格。
def prepare_defensive(filename: Path) -> None:
    """四套规则只改变信号组和开仓侧保护门槛，强反向信号均触发原生减仓。"""
    config, source, output = research_context(filename)
    folder = output / 'signals'
    folder.mkdir(exist_ok=True)
    thresholds = json.loads(Path(config['thresholds']).read_text())['groups']
    calibrations = {}
    variants = []
    for group in ['flow', 'spot']:
        for quantile in [.9, .99]:
            name = f'{group}_q{int(quantile * 100)}'
            calibrations[name] = {
                'entry_bps': thresholds[group][str(quantile)],
                'exit_bps': thresholds[group]['0.9'], 'scale_bps': 1.,
            }
            variants.append({
                'signal_id': name, 'mode': 'defensive', 'policy': 'alpha_protection',
                'width_bps': 8., 'ttl_ms': 1000, 'hold_s': 1800, 'dual': False,
            })
        for split in ['val_dev', 'val_select']:
            signals = pl.read_parquet(Path(config['signals']) / f'{group}_{split}.parquet')
            for date in config['dates'][split]:
                daily = signals.filter(pl.col('timestamp_ns').cast(pl.Datetime('ns')).dt.date() == pl.lit(date).str.to_date())
                shared = folder / f'{group}_{date}.csv'
                daily.select('timestamp_ns', pl.col('prediction_log_bps').alias('prediction_bps')).write_csv(shared)
                for quantile in [.9, .99]:
                    name = f'{group}_q{int(quantile * 100)}'
                    destination = folder / f'{name}_{date}.csv'
                    if not destination.exists():
                        destination.symlink_to(shared.name)
    write_json_atomic(output / 'calibration.json', calibrations)
    write_json_atomic(output / 'variants.json', {'variants': variants})


# 【核心逻辑366】强反向Alpha在实际仓位上触发IOC减仓，开发后冻结一个规则复核。
def run_defensive_day(filename: Path, date: str, split: str) -> None:
    """盘口和在途订单沿用原生处理，输出可逐笔核对的Maker/Taker现金流。"""
    config, source, output = research_context(filename)
    selected = 'variants.json' if split == 'val_dev' else 'selected.json'
    key = 'variants' if split == 'val_dev' else 'candidates'
    variants = json.loads((output / selected).read_text())[key]
    execute_quote_day(config, source, output, date, split, variants)


# 【非核心】复用报价研究的冻结选择与汇总，保持日期和费用口径一致。
def finish_defensive(filename: Path, stage: str) -> None:
    """按开发净收益选择后，完整展示Alpha保护规则的实际原生复核结果。"""
    if stage == 'defensive_select':
        select_wide(filename)
    else:
        report_wide(filename, '固定报价半宽8bps与1800秒持仓上限，比较flow/spot信号及Train q90/q99门槛。开发12日冻结一个正收益规则后运行12个历史复核日。')
