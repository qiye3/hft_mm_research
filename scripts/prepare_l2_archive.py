"""盘点已授权网盘L2原档，分日期下载并复用正式转换和原生回放验证。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import importlib.util
import json
import os
import shutil
from pathlib import Path
import re
import subprocess
import sys
import time
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research'))
from mm_research.data.engine_inputs import prepare_engine_day
from mm_research.data.storage import write_json_atomic


# 【非核心】复用现有网盘接口，凭证仅从用户已批准的明确私有文件读取。
def downloader(credential_file: Path):
    """返回现有下载模块和凭证，日志不含授权链接或Token。"""
    specification = importlib.util.spec_from_file_location('hft_baidu_downloader', Path(__file__).with_name('download_baidu.py'))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    token = re.search(r'access_token=([^&#\s`<>]+)', credential_file.read_text()).group(1)
    return module, token


# 【核心逻辑257】从实际目录项建立日期、压缩后缀和大小清单，不假定所有文件都叫zst。
def inventory_archive(module, token: str, folder: Path) -> list[dict]:
    """遍历2025年10月起至2026年10月的日目录，保存可下载来源范围。"""
    output = folder / 'archive_inventory.json'
    if output.exists():
        return json.loads(output.read_text())
    base = '/STUDY/Binance-BTC-Futures/L2增量 2020-2026/BTCUSDT'
    directories = []
    for year in [2025, 2026]:
        months = module.request_metadata('file', {'method': 'list', 'dir': f'{base}/{year}', 'start': 0, 'limit': 1000}, token)['list']
        for month in months:
            number = int(month['server_filename'])
            if year == 2025 and number < 10:
                continue
            days = module.request_metadata('file', {'method': 'list', 'dir': month['path'], 'start': 0, 'limit': 1000}, token)['list']
            directories.extend(day for day in days if day['isdir'])
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(module.request_metadata, 'file', {
            'method': 'list', 'dir': day['path'], 'start': 0, 'limit': 1000,
        }, token): day for day in directories}
        for future in as_completed(futures):
            day = futures[future]
            listing = future.result()['list']
            files = [entry for entry in listing if not entry['isdir'] and entry['server_filename'].startswith('BTCUSDT.csv.')]
            if len(files) != 1:
                raise ValueError(f'日目录未包含唯一L2源文件 {day["path"]}')
            entry = files[0]
            date = '-'.join(day['path'].split('/')[-3:])
            records.append({'date': date, 'entry': entry})
    records.sort(key=lambda item: item['date'])
    write_json_atomic(output, records)
    print('L2_INVENTORY', len(records), sum(item['entry']['size'] for item in records), flush=True)
    return records


# 【核心逻辑258】按日期下载原始L2并转换为既有双时间戳事件缓存，失败保留现场。
def prepare_day(root: Path, folder: Path, record: dict, module, token: str) -> dict:
    """输入一天来源，输出转换指标；标准官方成交继续复用已生产的Parquet。"""
    date = record['date']
    entry = record['entry']
    raw_folder = folder / 'event_cache' / date / 'source'
    extension = entry['server_filename'].removeprefix('BTCUSDT.csv.')
    raw = raw_folder / f'book.csv.{extension}'
    conversion = folder / 'event_cache' / date / 'events'
    report = folder / 'validation' / f'{date}.json'
    if report.exists():
        return json.loads(report.read_text())
    started = time.monotonic()
    download = module.download_file(entry, raw, token)
    manifest = prepare_engine_day(raw, root / 'workspace/data/standard/server_market_v1/trades' / f'{date}.parquet',
                                  date, conversion, 20_000_000)
    result = {'date': date, 'status': 'converted', 'source': entry['path'],
              'raw_bytes': raw.stat().st_size, 'normalized_bytes': manifest['normalized_bytes'],
              'logical_events': manifest['logical_events'], 'conversion_seconds': manifest['conversion_seconds'],
              'download_seconds': download['download_seconds'], 'elapsed_seconds': time.monotonic() - started,
              'manifest': str(conversion / 'manifest.json'),
              'native_replay_validation': 'pending', 'full_depth_retained': True,
              'event_cache_retained': True, 'managed_cache': True}
    write_json_atomic(report, result)
    print('L2_CONVERTED', date, result['elapsed_seconds'], flush=True)
    return result


# 【核心逻辑276】原生观察器消费全部事件，再复用已有验收核对分类、时间和最终十档。
def verify_native_day(root: Path, folder: Path, result: dict) -> dict:
    """转换成功与原生回放成功分别记录，输出只归属本轮研究目录。"""
    if result['native_replay_validation'] == 'passed':
        return result
    replay_folder = folder / 'replay'
    replay_folder.mkdir(exist_ok=True)
    date = result['date']
    replay_file = replay_folder / f'{date}.json'
    config = {'source': result['manifest'], 'date': date, 'symbol': 'BTCUSDT',
              'tick_size': .1, 'lot_size': .001, 'depth_levels': 10,
              'snapshot_interval_ns': 100_000_000, 'latency_sample_every': 0, 'output': str(replay_file)}
    config_file = replay_folder / f'{date}.yaml'
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    with (replay_folder / f'{date}.log').open('w') as log:
        subprocess.run([str(root / 'workspace/build/release/replay_probe'), str(config_file)],
                       cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('validate_replay.py')),
                        result['manifest'], str(replay_file)], cwd=root,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result.update(status='validated', native_replay_validation='passed', replay_report=str(replay_file))
    write_json_atomic(folder / 'validation' / f'{date}.json', result)
    return result


# 【非核心】一天转换放入独立受限进程，内存不与模型共享，异常保留逐日日志。
def run_day_worker(root: Path, folder: Path, date: str, credential_file: Path) -> dict:
    """记录本轮日期任务的退出码，成功也需已有原生验收记录。"""
    logs = folder / 'logs'
    logs.mkdir(exist_ok=True)
    command = ['prlimit', '--data=34359738368', sys.executable, '-u', str(Path(__file__).resolve()),
               '--credential-file', str(credential_file), '--output', str(folder), '--date', date]
    started = time.monotonic()
    from mm_research.overnight import stop_task
    with (logs / f'{date}.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            exit_code = process.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            stop_task(process.pid)
            process.wait()
            return {'date': date, 'status': 'failed', 'stage': 'date_timeout',
                    'elapsed_seconds': time.monotonic() - started}
    return {'date': date, 'status': 'success' if exit_code == 0 else 'failed',
            'exit_code': exit_code, 'elapsed_seconds': time.monotonic() - started}


# 【核心逻辑277】全日期限时排队，优先覆盖各研究分区，仅释放本轮独占目录内已验收缓存。
def run_archive_queue(root: Path, folder: Path, records: list[dict], credential_file: Path,
                      stop_at: str, module, token: str) -> None:
    """网盘原档及既有normalized保持独立，非优先日只保留来源和完整验收报告。"""
    import numpy as np
    from mm_research.overnight import memory_state
    deadline = dt.datetime.fromisoformat(stop_at).timestamp()
    dates = [record['date'] for record in records]
    bounds = [('2025-10-01', '2026-03-22'), ('2026-03-23', '2026-05-14'),
              ('2026-05-15', '2026-07-06'), ('2026-07-07', '2026-10-21')]
    pools = []
    for first, last in bounds:
        pool = [date for date in dates if first <= date <= last]
        chosen = np.linspace(0, len(pool) - 1, min(12, len(pool)), dtype=int)
        pools.append([pool[index] for index in chosen])
    priority = []
    for index in range(12):
        for pool in pools:
            if index < len(pool):
                priority.append(pool[index])
    priority = list(dict.fromkeys(priority))
    ordered = priority + [date for date in dates if date not in priority]
    write_json_atomic(folder / 'batch_plan.json', {'all_dates': ordered, 'retain_event_cache_dates': priority,
                      'parallel_conversions': 2, 'parallel_downloads': 4, 'stop_at': stop_at,
                      'cache_root': str(folder / 'event_cache')})
    progress_file = folder / 'batch_progress.json'
    completed = json.loads(progress_file.read_text())['results'] if progress_file.exists() else []
    done = {record['date'] for record in completed}
    entries = {record['date']: record['entry'] for record in records}
    with ThreadPoolExecutor(max_workers=2) as workers, ThreadPoolExecutor(max_workers=4) as downloads:
        pending = {}
        prefetch = {}
        remaining = [date for date in ordered if date not in done]
        while remaining or pending:
            override = folder / 'queue_priority.json'
            if override.exists():
                schedule = json.loads(override.read_text())['dates']
                ranks = {date: index for index, date in enumerate(schedule)}
                remaining.sort(key=lambda date: ranks[date])
            if time.time() < deadline - 600 and shutil.disk_usage(root).free > 45 * 1024 ** 3:
                for date in remaining[:4]:
                    if date not in prefetch:
                        entry = entries[date]
                        extension = entry['server_filename'].removeprefix('BTCUSDT.csv.')
                        raw = folder / 'event_cache' / date / 'source' / f'book.csv.{extension}'
                        prefetch[date] = downloads.submit(module.download_file, entry, raw, token)
            while remaining and len(pending) < 2 and time.time() < deadline - 600:
                if shutil.disk_usage(root).free < 45 * 1024 ** 3:
                    break
                required_gib = 30 if not pending else 40
                if memory_state()['available_bytes'] < required_gib * 1024 ** 3:
                    break
                ready = [date for date in remaining if date in prefetch and prefetch[date].done()]
                if not ready:
                    break
                date = ready[0]
                try:
                    prefetch.pop(date).result()
                except (OSError, RuntimeError, ValueError) as error:
                    remaining.remove(date)
                    completed.append({'date': date, 'status': 'failed', 'stage': 'download',
                                      'error_type': type(error).__name__})
                    print('L2_DOWNLOAD_FAILED', date, type(error).__name__, flush=True)
                    continue
                remaining.remove(date)
                pending[workers.submit(run_day_worker, root, folder, date, credential_file)] = date
                print('L2_STARTED', date, flush=True)
            for future in [future for future in pending if future.done()]:
                date = pending.pop(future)
                result = future.result()
                if result['status'] == 'success' and date not in priority:
                    report_file = folder / 'validation' / f'{date}.json'
                    validation = json.loads(report_file.read_text())
                    generated = folder / 'event_cache' / date
                    if Path(validation['manifest']).resolve().parent != (generated / 'events').resolve():
                        raise ValueError('清理目标不属于本批生成的日期缓存')
                    if validation['native_replay_validation'] != 'passed':
                        raise ValueError('尚未验收的事件缓存不能按成功任务清理')
                    shutil.rmtree(generated)
                    validation['event_cache_retained'] = False
                    write_json_atomic(report_file, validation)
                completed.append(result)
                print('L2_FINISHED', date, result['status'], flush=True)
            for marker in (folder / 'consumed').glob('*.json'):
                date = marker.stem
                report_file = folder / 'validation' / f'{date}.json'
                validation = json.loads(report_file.read_text())
                if validation['managed_cache'] and validation['event_cache_retained']:
                    generated = folder / 'event_cache' / date
                    if Path(validation['manifest']).resolve().parent != (generated / 'events').resolve():
                        raise ValueError('已消费缓存不属于本轮独占目录')
                    if validation['native_replay_validation'] != 'passed':
                        raise ValueError('已消费缓存缺少原生验收')
                    shutil.rmtree(generated)
                    validation.update(event_cache_retained=False, release_reason='all_frozen_execution_cases_complete')
                    write_json_atomic(report_file, validation)
            write_json_atomic(progress_file, {'results': completed, 'remaining_dates': remaining,
                              'running_dates': list(pending.values()), 'stop_at': stop_at})
            if time.time() >= deadline - 600 and not pending:
                break
            time.sleep(10)
    write_json_atomic(folder / 'batch_completed.json', {'results': completed, 'unprocessed_dates': remaining,
                      'deadline': stop_at})


# 【非核心】明确选择首日试跑或完整日期队列，按时限和磁盘余量停止新增任务。
def main():
    """盘点与转换分开执行，跨训练和验证优先，所有未完成日期如实记录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--credential-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--date')
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--stop-at', default='2026-11-10T08:00:00+08:00')
    arguments = parser.parse_args()
    root = Path.cwd()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.batch:
        write_json_atomic(arguments.output / 'runtime.json', {
            'pid': os.getpid(), 'credential_file': str(arguments.credential_file.resolve()),
            'output': str(arguments.output.resolve()),
        })
    module, token = downloader(arguments.credential_file)
    if arguments.date:
        date = arguments.date
        directory = '/STUDY/Binance-BTC-Futures/L2增量 2020-2026/BTCUSDT/' + date.replace('-', '/')
        listing = module.request_metadata('file', {'method': 'list', 'dir': directory, 'start': 0, 'limit': 1000}, token)['list']
        files = [entry for entry in listing if not entry['isdir'] and entry['server_filename'].startswith('BTCUSDT.csv.')]
        if len(files) != 1:
            raise ValueError('首日日目录没有唯一L2文件')
        result = prepare_day(root, arguments.output, {'date': date, 'entry': files[0]}, module, token)
        verify_native_day(root, arguments.output, result)
    else:
        records = inventory_archive(module, token, arguments.output)
        if arguments.batch:
            run_archive_queue(root, arguments.output, records, arguments.credential_file,
                              arguments.stop_at, module, token)


if __name__ == '__main__':
    main()
