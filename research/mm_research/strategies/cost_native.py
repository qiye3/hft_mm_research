"""对冻结的费用后分类信号进行连续原生回放，空仓且全无信号的日期可安全跳过。"""

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import polars as pl
import yaml

from mm_research.data.engine_inputs import prepare_engine_day
from mm_research.data.storage import file_locks, write_json_atomic
from mm_research.multihorizon.common import context, DAY, SECOND
from mm_research.multihorizon.inventory import funding_inputs
from mm_research.strategies.experiments import write_case_configs


# 【核心逻辑372】用所有可发单时刻建立原生回放区间，额外保留最大持仓时长覆盖的后续日期。
def plan_native(filename: Path) -> None:
    """区间选择不读取收益，包含全部潜在入场日期，并保存空仓跳日的证明。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    blocks = []
    for split in config['splits']:
        table = pl.read_parquet(Path(config['signals']) / f'cost_h1800_{split}.parquet').with_row_index('row')
        eligible = table.filter(
            pl.col('expected_net_edge_bps').is_finite() & (pl.col('expected_net_edge_bps') != 0)
            & (pl.col('row') >= 120) & (pl.col('row') < table.height - config['hold_s'] - 20),
        )
        active = eligible.select(pl.col('timestamp_ns').cast(pl.Datetime('ns')).dt.date().alias('date')).unique()['date'].to_list()
        last_date = dt.date.fromisoformat(manifest['dates'][(segments[split]['stop_row'] - 1) // DAY])
        dates = set(active)
        for date in active:
            for extra in range(1, (config['hold_s'] + DAY - 1) // DAY + 1):
                following = date + dt.timedelta(days=extra)
                if following <= last_date:
                    dates.add(following)
        groups = []
        for date in sorted(dates):
            if not groups or date != groups[-1][-1] + dt.timedelta(days=1):
                groups.append([])
            groups[-1].append(date)
        previous = None
        for index, group in enumerate(groups):
            identifier = f'{split}_{index:02d}'
            blocks.append({
                'id': identifier, 'split': split, 'dates': list(map(str, group)),
                'previous_block': previous, 'first_date': str(group[0]), 'last_date': str(group[-1]),
            })
            previous = identifier
    write_json_atomic(output / 'blocks.json', {
        'blocks': blocks, 'selection': 'all_nonzero_signals_plus_maximum_hold_calendar',
        'skipped_days': 'no_possible_entry_and_flat_after_previous_maximum_hold',
    })
    tasks = []
    command = ['.venv-models/bin/python', '-u', 'scripts/run_cost_native.py', str(filename)]
    for block in blocks:
        data = 'data_' + block['id']
        data_dependencies = ['plan']
        if block['previous_block']:
            data_dependencies.append('data_' + block['previous_block'])
            previous = next(item for item in blocks if item['id'] == block['previous_block'])
            if previous['previous_block']:
                data_dependencies.append('replay_' + previous['previous_block'])
        tasks.append(native_task(data, command + ['data', '--block', block['id']], data_dependencies, 'data', 24, 34))
        dependencies = [data, 'build']
        if block['previous_block']:
            dependencies.append('replay_' + block['previous_block'])
        tasks.append(native_task('replay_' + block['id'], command + ['replay', '--block', block['id']], dependencies, 'cpu', 2, 20))
    cleanup = native_task('cleanup_credential', command + ['cleanup'], [], 'cpu', 1, 2)
    cleanup['after_finished'] = [task['id'] for task in tasks if task['id'].startswith('data_')]
    tasks.append(cleanup)
    report = native_task('report', command + ['report'], [], 'cpu', 3, 8)
    report['after_finished'] = [task['id'] for task in tasks if task['id'].startswith('replay_')]
    tasks.append(report)
    extension = Path(config['queue_directory']) / 'blocks.yaml'
    extension.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(extension, {'tasks': tasks})


# 【非核心】日期准备与区段回放复用监督器的资源准入、失败重试和独立日志。
def native_task(identifier: str, command: list[str], dependencies: list[str], lane: str,
                estimate: int, limit: int) -> dict:
    """返回公开队列结构，不另建调度或重试系统。"""
    return {
        'id': identifier, 'command': command, 'threads': 4 if lane == 'data' else 2, 'estimated_peak_gib': estimate,
        'memory_limit_gib': limit, 'allocation_limit_gib': 64,
        'requires_success': dependencies, 'after_finished': [], 'lane': lane,
        'max_retries': 2, 'retry_delay_seconds': 60, 'priority': 10 if lane == 'data' else 20,
    }


# 【非核心】从已存盘清单定位一个固定连续区间。
def block_definition(output: Path, identifier: str) -> dict:
    """返回已冻结区间定义，找不到指定任务时直接报错。"""
    return next(block for block in json.loads((output / 'blocks.json').read_text())['blocks'] if block['id'] == identifier)


# 【非核心】预取和正式转换共用稳定文件锁，防止并发追加同一个断点文件。
def download_native_raw(entry: dict, raw: Path, token: str) -> None:
    """调用已有断点下载器，完成文件可复用，凭证仅在内存中传递。"""
    specification = importlib.util.spec_from_file_location('native_baidu_download', Path('scripts/download_baidu.py'))
    downloader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(downloader)
    with file_locks([raw]):
        downloader.download_file(entry, raw, token)


# 【核心逻辑377】预取后续区间原档，使网络传输与原生回放及事件转换重叠。
def prefetch_native(filename: Path, excluded_blocks: list[str]) -> None:
    """按冻结区间清单下载未缓存原档，四路网络传输，不提前生成大体积事件文件。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    inventory = {record['date']: record['entry'] for record in json.loads((source / 'l2/archive_inventory.json').read_text())}
    blocks = json.loads((output / 'blocks.json').read_text())['blocks']
    pending = []
    for block in blocks:
        # 启动时显式排除已由旧下载进程占用的区间，后续转换均使用同一文件锁。
        if block['id'] in excluded_blocks:
            continue
        for date in block['dates']:
            if (source / f'l2/event_cache/{date}/events/manifest.json').exists():
                continue
            entry = inventory[date]
            extension = entry['server_filename'].removeprefix('BTCUSDT.csv.')
            raw = output / 'raw' / date / f'book.csv.{extension}'
            if not raw.exists():
                pending.append((entry, raw))
    if not pending:
        return
    token = re.search(r'access_token=([^&#\s`<>]+)', Path(config['credential_file']).read_text()).group(1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_native_raw, entry, raw, token) for entry, raw in pending]
        for future, (_, raw) in zip(futures, pending):
            future.result()
            print(f'原档预取完成 {raw.parent.name}', flush=True)


# 【核心逻辑373】复用既有日期缓存，只下载和转换原生验证缺少的L2原档。
def prepare_native_block(filename: Path, identifier: str, date: str | None = None) -> None:
    """准备冻结区间的全部日期或指定一天，供正式队列和并行准备任务共用。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    block = block_definition(output, identifier)
    inventory = {record['date']: record['entry'] for record in json.loads((source / 'l2/archive_inventory.json').read_text())}
    (output / 'days').mkdir(exist_ok=True)
    dates = block['dates']
    if date is not None:
        if date not in dates:
            raise ValueError(f'指定日期不属于冻结回放区间 {identifier}: {date}')
        dates = [date]
    # 每日独立子进程释放排序内存，并发准入由监督队列控制。
    for selected_date in dates:
        prepare_native_date(filename, config, output, inventory, selected_date)


# 【非核心】同一天取得独占锁后转换和发布标记，正式队列复用并行准备的日期。
def prepare_native_date(filename: Path, config: dict, output: Path, inventory: dict, date: str) -> None:
    """复用已有事件或下载原档，子进程退出后发布单日清单，避免跨日内存累积。"""
    source = Path(config['source'])
    marker = output / 'days' / f'{date}.json'
    with file_locks([marker]):
        if marker.exists():
            return
        existing = source / f'l2/event_cache/{date}/events/manifest.json'
        if existing.exists():
            write_json_atomic(marker, {'date': date, 'manifest': str(existing.resolve()), 'created_here': False})
            return
        entry = inventory[date]
        extension = entry['server_filename'].removeprefix('BTCUSDT.csv.')
        raw = output / 'raw' / date / f'book.csv.{extension}'
        if not raw.exists() or raw.stat().st_size != entry['size']:
            token = re.search(r'access_token=([^&#\s`<>]+)', Path(config['credential_file']).read_text()).group(1)
            download_native_raw(entry, raw, token)
        destination = output / f'event_cache/{date}/events'
        environment = os.environ.copy()
        environment['PYTHONPATH'] = 'research'
        subprocess.run([sys.executable, '-u', 'scripts/run_cost_native.py', str(filename), 'convert', '--date', date],
                       env=environment, check=True)
        write_json_atomic(marker, {'date': date, 'manifest': str((destination / 'manifest.json').resolve()), 'created_here': True})


# 【非核心】一天转换独立退出，释放排序与路由的临时分配。
def convert_native_day(filename: Path, date: str) -> None:
    """复用已有固定20ms研究时钟转换器，保留全部必要深度事件。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    inventory = {record['date']: record['entry'] for record in json.loads((source / 'l2/archive_inventory.json').read_text())}
    entry = inventory[date]
    extension = entry['server_filename'].removeprefix('BTCUSDT.csv.')
    prepare_engine_day(output / 'raw' / date / f'book.csv.{extension}',
        Path(config['standard_trades']) / f'{date}.parquet', date, output / f'event_cache/{date}/events', 20_000_000)


# 【核心逻辑374】连续日期交给原生引擎，跨午夜保留持仓并按前一区间现金续接账户。
def replay_native_block(filename: Path, identifier: str) -> None:
    """区间包括所有潜在入场和持仓日期，两个区间间跳过的日期可证明全程空仓。"""
    _, config, output = context(filename)
    block = block_definition(output, identifier)
    result = output / 'block_results' / f'{identifier}.json'
    if result.exists():
        release_native_cache(output, block)
        return
    shards = []
    events = 0
    for date in block['dates']:
        metadata = json.loads((output / 'days' / f'{date}.json').read_text())
        manifest_file = Path(metadata['manifest'])
        manifest = json.loads(manifest_file.read_text())
        events += manifest['logical_events']
        for shard in manifest['shards']:
            shards.append({'file': str((manifest_file.parent / shard['file']).resolve()), 'bytes': shard['bytes']})
    folder = output / 'blocks' / identifier
    folder.mkdir(parents=True, exist_ok=True)
    combined = folder / 'events.json'
    write_json_atomic(combined, {'logical_events': events, 'shards': shards})
    signals = pl.read_parquet(Path(config['signals']) / f'cost_h1800_{block["split"]}.parquet').with_row_index('row')
    signals = signals.with_columns(
        pl.when((pl.col('row') >= 120) & (pl.col('row') < signals.height - config['hold_s'] - 20))
        .then(pl.col('expected_net_edge_bps')).otherwise(0.).alias('prediction_bps'),
    )
    first = int(np.datetime64(block['first_date'], 'ns').astype('int64'))
    last = int(np.datetime64(block['last_date'], 'ns').astype('int64')) + DAY * SECOND
    signal_columns = ['timestamp_ns', 'prediction_bps']
    if config['mode'] in ['directional_risk', 'directional_maker_risk']:
        signal_columns.append('risk_bps')
    signals.filter(pl.col('timestamp_ns').is_between(first, last, closed='left')).select(
        signal_columns,
    ).write_csv(folder / 'signals.csv')
    capital = 10000.
    if block['previous_block']:
        previous = json.loads((output / 'block_results' / f'{block["previous_block"]}.json').read_text())
        capital = previous['ending_account_equity_usdt']
    trading = {
        'date': block['first_date'], 'source': str(combined.resolve()), 'output': str(output),
        'initial_capital': capital, 'order_quantity': config['order_quantity'], 'max_position': config['order_quantity'],
        'maker_fee': .0002, 'taker_fee': .0005, 'entry_latency_ns': 1_000_000,
        'response_latency_ns': 500_000, 'decision_interval_ns': 100_000_000,
        'prediction_latency_ns': 1_000_000, 'warmup_seconds': 0, 'liquidation_buffer_seconds': 20,
        'max_alpha_bps': 1000., 'taker_slippage_ticks': 1,
    }
    case = {
        'id': identifier, 'mode': config['mode'], 'queue_model': 'risk_adverse',
        'entry_bps': 0., 'exit_bps': 0., 'scale_bps': 1., 'inventory_skew_bps': 0.,
        'cost_filter': False, 'offset_ticks': 0, 'ttl_ms': config['ttl_ms'], 'hold_s': config['hold_s'],
        'quote_half_width_bps': 0., 'minimum_notional_usdt': 100.,
    }
    trading_file, policy_file = write_case_configs(Path.cwd(), trading, case, output)
    settings = yaml.safe_load(trading_file.read_text())
    settings['quote_stop_timestamp_ns'] = last - 20 * SECOND
    trading_file.write_text(yaml.safe_dump(settings, sort_keys=False))
    run = output / 'runs' / identifier
    if not (run / 'summary.json').exists():
        with (folder / 'replay.log').open('w') as log:
            subprocess.run([str(Path(config['binary_folder']) / 'strategy_replay'), str(trading_file),
                str(policy_file), str(folder / 'signals.csv')], stdout=log, stderr=subprocess.STDOUT, check=True)
    analyze_native_block(config, output, block, run, capital, events)


# 【非核心】大区间账本逐表流式压缩，只把实际成交读入内存。
def compress_native_frames(run: Path) -> None:
    """保留订单、成交、权益、清算权益和IOC用途，避免长区间CSV常驻内存。"""
    float_fields = {
        'orders': ['price', 'quantity', 'leaves_quantity', 'exec_price', 'exec_quantity'],
        'fills': ['price', 'quantity', 'fee', 'position', 'balance', 'average_cost', 'realized_gross', 'mid', 'net_pnl'],
        'equity': ['mid', 'position', 'balance', 'fees', 'inventory_value', 'realized_gross', 'unrealized', 'net_pnl',
                   'equity', 'reserved_buy', 'reserved_sell', 'running_max_drawdown'],
        'liquidation_equity': ['mid', 'bid', 'ask', 'position', 'net_pnl_liquidated'],
        'ioc_actions': [],
    }
    for name, fields in float_fields.items():
        destination = run / f'{name}.parquet'
        if destination.exists():
            continue
        schema = {field: pl.Float64 for field in fields}
        schema['local_timestamp_ns'] = pl.Int64
        if name in ['orders', 'fills', 'ioc_actions']:
            schema['order_id'] = pl.UInt64
        if name in ['orders', 'fills']:
            schema.update(exchange_timestamp_ns=pl.Int64, side=pl.Int8, maker=pl.Int8)
        if name == 'fills':
            schema['fill_id'] = pl.UInt64
        if name == 'ioc_actions':
            schema['purpose'] = pl.String
        # 中断只能留下临时文件，正式文件存在即表示该表已完整发布。
        pending = destination.with_suffix('.parquet.pending')
        pl.scan_csv(run / f'{name}.csv', schema_overrides=schema, null_values='nan').sink_parquet(pending, compression='zstd')
        pending.replace(destination)


# 【核心逻辑375】独立核对原生成交、手续费、结算时实际仓位和连续区间权益。
def analyze_native_block(config: dict, output: Path, block: dict, run: Path, capital: float, expected_events: int) -> None:
    """资金费按真实交易所成交顺序计算，所有区间末端必须空仓且无在途订单。"""
    compress_native_frames(run)
    fills = pl.read_parquet(run / 'fills.parquet').sort('exchange_timestamp_ns')
    summary = json.loads((run / 'summary.json').read_text())
    gross = float(fills.select((-pl.col('side') * pl.col('price') * pl.col('quantity')).sum()).item())
    fees = float(fills.select((pl.col('price') * pl.col('quantity') * pl.when(pl.col('maker') == 1).then(.0002).otherwise(.0005)).sum()).item())
    if abs(gross - summary['balance']) > 1e-7 or abs(fees - summary['fees']) > 1e-7:
        raise ValueError('原生连续区间现金或费用核对失败')
    if summary['position'] != 0 or summary['final_working_orders'] != 0:
        raise ValueError('原生连续区间末端仍有仓位或在途委托')
    if summary['local_events'] != expected_events or summary['exchange_events'] != expected_events:
        raise ValueError('连续原生回放未消费完整事件')
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    first_row = manifest['dates'].index(block['first_date']) * DAY
    last_row = (manifest['dates'].index(block['last_date']) + 1) * DAY
    times, rates, marks = funding_inputs(source, manifest, first_row, last_row)
    exchange_times = fills['exchange_timestamp_ns'].to_numpy()
    positions = fills.select((pl.col('side') * pl.col('quantity')).cum_sum().alias('position'))['position'].to_numpy()
    payments = []
    for timestamp, rate, mark in zip(times, rates, marks):
        index = int(np.searchsorted(exchange_times, int(timestamp * SECOND), side='left')) - 1
        position = float(positions[index]) if index >= 0 else 0.
        payments.append({'timestamp_ns': int(timestamp * SECOND), 'position_btc': position,
                         'funding_payment_usdt': float(position * rate * mark)})
    funding = sum(record['funding_payment_usdt'] for record in payments)
    net = gross - fees - funding
    result = {
        'id': block['id'], 'split': block['split'], 'first_date': block['first_date'], 'last_date': block['last_date'],
        'initial_account_equity_usdt': capital, 'ending_account_equity_usdt': capital + net,
        'gross_pnl_usdt': gross, 'fees_usdt': fees, 'funding_paid_usdt': funding,
        'net_pnl_usdt': net, 'fills': fills.height, 'final_position_btc': summary['position'],
        'logical_events': expected_events, 'ledger_reconciled': True,
    }
    write_json_atomic(run / 'funding_cashflows.json', {'records': payments})
    (output / 'block_results').mkdir(exist_ok=True)
    write_json_atomic(output / 'block_results' / f'{block["id"]}.json', result)
    for name in ['orders', 'fills', 'equity', 'liquidation_equity', 'ioc_actions']:
        (run / f'{name}.csv').unlink(missing_ok=True)
    release_native_cache(output, block)


# 【非核心】区间账本发布后释放本轮新建事件缓存，原始行情和复现清单继续保存。
def release_native_cache(output: Path, block: dict) -> None:
    """只删除当前实验创建的事件目录，原有跨实验缓存继续共享。"""
    for date in block['dates']:
        marker = output / 'days' / f'{date}.json'
        metadata = json.loads(marker.read_text())
        if not metadata['created_here'] or metadata.get('cache_released', False):
            continue
        folder = output / 'event_cache' / date
        manifest = folder / 'events/manifest.json'
        archived = output / 'days' / f'{date}.manifest.json'
        if manifest.exists():
            archived.write_text(manifest.read_text())
        if folder.exists():
            shutil.rmtree(folder)
        metadata['cache_released'] = True
        write_json_atomic(marker, metadata)


# 【非核心】全部下载任务进入终态后删除本轮临时私有凭证。
def cleanup_credential(filename: Path) -> None:
    """删除明确的本轮私有文件，来源原档与交易结果继续保留。"""
    _, config, output = context(filename)
    Path(config['credential_file']).unlink(missing_ok=True)
    write_json_atomic(output / 'credential_removed.json', {'removed': True})


# 【核心逻辑376】按原分区累计所有原生连续区间的实际净收益，列出任何缺失区间。
def report_native(filename: Path) -> None:
    """账户在空仓区间间延续现金，完整发单区间全部纳入结果。"""
    _, config, output = context(filename)
    blocks = json.loads((output / 'blocks.json').read_text())['blocks']
    records = []
    missing = []
    for block in blocks:
        result = output / 'block_results' / f'{block["id"]}.json'
        if result.exists():
            records.append(json.loads(result.read_text()))
        else:
            missing.append(block['id'])
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    if not records:
        raise ValueError('原生验证没有完成的区间')
    table = pl.DataFrame(records)
    table.write_csv(summary / 'blocks.csv')
    totals = table.group_by('split').agg(
        pl.col('net_pnl_usdt').sum(), pl.col('gross_pnl_usdt').sum(), pl.col('fees_usdt').sum(),
        pl.col('funding_paid_usdt').sum(), pl.col('fills').sum(), pl.col('logical_events').sum(),
    )
    totals.write_csv(summary / 'totals.csv')
    lines = ['# 费用后分类信号的原生连续验证｜2026-11-11', '',
        '冻结30分钟获利分类器与配置持有规则。Maker单侧2bps、Taker单侧5bps，实际限价单、IOC、部分成交、在途回报和库存使用原生引擎。',
        '全部可发单日期及最长持仓所需后续日期均重放。可证明全程无入场且空仓的日期跳过，区间间现金续接。', '',
        '| 分区 | 费用前USDT | 手续费USDT | 资金费USDT | 净收益USDT | 成交数 |',
        '| --- | --- | --- | --- | --- | --- |']
    for row in totals.iter_rows(named=True):
        lines.append(f"| {row['split']} | {row['gross_pnl_usdt']:.3f} | {row['fees_usdt']:.3f} | {row['funding_paid_usdt']:.3f} | {row['net_pnl_usdt']:.3f} | {row['fills']} |")
    lines.append(f'缺失区间数为{len(missing)}。全部原始来源、事件清单、订单、成交、资金费和账户权益保存在本轮目录。')
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    if not missing:
        report_native_account(config, output, blocks)
    write_json_atomic(output / 'completed.json', {'status': 'success' if not missing else 'incomplete', 'missing_blocks': missing})


# 【核心逻辑394】全部原生区间完成后按完整日历计算账户年化、真实费用占比和1倍敞口门禁。
def report_native_account(config: dict, output: Path, blocks: list[dict]) -> None:
    """复用逐笔和连续权益核算，零交易区间保留在完整日历中，不缩短年化时间。"""
    from mm_research.strategies.cost_attribution import native_round_trips, summarize_native_risk

    frames = []
    maker_notional = 0.
    taker_notional = 0.
    for block in blocks:
        run = output / 'runs' / block['id']
        frame = native_round_trips(run)
        if not frame.is_empty():
            frames.append(frame.with_columns(pl.lit(block['split']).alias('split')))
        fills = pl.read_parquet(run / 'fills.parquet', columns=['maker', 'price', 'quantity'])
        for maker in [1, 0]:
            notional = float(fills.filter(pl.col('maker') == maker)
                             .select((pl.col('price') * pl.col('quantity')).sum()).item())
            if maker == 1:
                maker_notional += notional
            else:
                taker_notional += notional
    if frames:
        trades = pl.concat(frames)
    else:
        trades = pl.DataFrame(schema={'split': pl.String, 'net_pnl_usdt': pl.Float64})
    summary = output / 'summary'
    trades.write_parquet(summary / 'native_trades.parquet', compression='zstd')
    summarize_native_risk(config, summary, output, blocks, trades, tuple(config['splits']))
    daily = pl.read_csv(summary / 'native_daily.csv', try_parse_dates=True)
    monthly = daily.group_by('split', pl.col('date').dt.strftime('%Y-%m').alias('month')).agg(pl.col('net_pnl_usdt').sum())
    weekly = daily.group_by('split', pl.col('date').dt.truncate('1w').alias('week')).agg(pl.col('net_pnl_usdt').sum())
    monthly.sort(['split', 'month']).write_csv(summary / 'monthly.csv')
    weekly.sort(['split', 'week']).write_csv(summary / 'weekly.csv')
    risk = json.loads((summary / 'native_risk.json').read_text())
    for record in risk['results']:
        profits = weekly.filter(pl.col('split') == record['split'])['net_pnl_usdt']
        record['excluding_best_week_usdt'] = record['net_pnl_usdt'] - float(profits.max())
        record['passes_annual_exposure_gate'] = (
            record['annualized_return'] > .30 and record['peak_gross_leverage'] <= 1.
            and record['excluding_best_week_usdt'] > 0.
        )
    write_json_atomic(summary / 'native_account.json', {
        'capital_usdt': 10000., 'target_annualized_return': .30, 'max_gross_leverage': 1.,
        'maker_notional_usdt': maker_notional, 'taker_notional_usdt': taker_notional,
        'results': risk['results'], 'monthly_concentration': 'reported_separately_not_implied_by_annual_gate',
    })


# 【核心逻辑407】冻结E4净价值与事前600秒风险，按完整开发时钟输出原生策略输入。
def prepare_risk_signals(filename: Path) -> None:
    """风险仅用决策前数据，预测和风险共用时间键；不导出成交时刻或未来退出请求。"""
    from mm_research.strategies.stable_strategy import visible_management

    _, config, output = context(filename)
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    segments = json.loads((source / 'e0/splits.json').read_text())['segments']
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    destination = Path(config['signals'])
    destination.mkdir(parents=True, exist_ok=True)
    for split in config['splits']:
        bounds = segments[split]
        begin, stop = bounds['start_row'], bounds['stop_row']
        model = Path(config['model_source']) / 'models'
        buy = np.load(model / f'h1800_side1/{split}_net_edge_bps.npy')
        sell = np.load(model / f'h1800_side-1/{split}_net_edge_bps.npy')
        volatility = visible_management(market, begin, stop)[:, 2]
        risk = np.maximum(20., 2. * volatility)
        active = (np.maximum(buy, sell) > 0) & np.isfinite(risk)
        score = np.where(active, np.where(buy >= sell, buy, -sell), 0.)
        frame = pl.DataFrame({
            'timestamp_ns': manifest['start_timestamp_ns'] + np.arange(begin, stop) * SECOND,
            'expected_net_edge_bps': score,
            'risk_bps': risk,
        })
        pending = destination / f'cost_h1800_{split}.parquet.pending'
        frame.write_parquet(pending, compression='zstd')
        pending.replace(destination / f'cost_h1800_{split}.parquet')
    write_json_atomic(output / 'risk_inputs.json', {
        'model_source': config['model_source'],
        'risk': 'max_20bps_or_2_times_strictly_past_600_second_volatility',
        'execution': 'native_actual_fills_no_frozen_exit_schedule',
        'splits': config['splits'],
    })
