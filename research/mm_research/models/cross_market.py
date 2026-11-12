"""直接在原146列训练缓存上做跨市场增量实验，不建立额外的标准行情生产阶段。"""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import polars as pl
import yaml

from mm_research.data.storage import write_json_atomic
from mm_research.factors.cross_market import cross_market_factors
from mm_research.multihorizon.common import context, model_inputs, SECOND

BAR_COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore',
]


# 【非核心】大矩阵先写实例本地盘，再顺序发布共享盘，避免网络文件系统逐页写放大。
def publish_cache(temporary: Path, destination: Path) -> None:
    """只保留一份当前写入矩阵，发布成功后删除本批临时文件。"""
    pending = destination.with_suffix('.partial.npy')
    with temporary.open('rb') as source:
        with pending.open('wb') as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
    pending.replace(destination)
    temporary.unlink()


# 【非核心】旧训练矩阵直接链接，只有两段完整验证输入需要按既有预处理分块写出。
def prepare_old_cross_inputs(filename: Path) -> None:
    """缓存放在共享盘，避免实例临时盘配额不足；不重新训练旧模型。"""
    _, config, output = context(filename)
    source = Path(config['source'])
    destination = Path(config['local_cache'])
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'completed.json').exists():
        return
    fold = source / 'e1/folds/final'
    for split in ['fit', 'tune']:
        target = destination / f'{split}_x.npy'
        if not target.exists():
            target.symlink_to((fold / f'{split}_x.npy').resolve())
    raw = np.load(source / 'e0/x.npy', mmap_mode='r')
    preprocessing = np.load(fold / 'preprocessing.npz')
    splits = json.loads((source / 'e0/splits.json').read_text())['segments']
    for split in ['val_dev', 'val_select']:
        bounds = splits[split]
        temporary = Path('/tmp') / f'hft_cross_old_{split}.npy'
        cached = np.lib.format.open_memmap(
            temporary, mode='w+', dtype='float32',
            shape=(bounds['stop_row'] - bounds['start_row'], raw.shape[1] * 2),
        )
        for begin in range(bounds['start_row'], bounds['stop_row'], 65536):
            stop = min(begin + 65536, bounds['stop_row'])
            cached[begin - bounds['start_row']:stop - bounds['start_row']] = model_inputs(
                raw, np.arange(begin, stop), preprocessing['mean'], preprocessing['scale'],
            )
        cached.flush()
        del cached
        publish_cache(temporary, destination / f'{split}_x.npy')
        print('OLD_INPUT_READY', split, flush=True)
    write_json_atomic(destination / 'completed.json', {'status': 'success', 'train_inputs': 'shared_original_files'})


# 【非核心】每个信息组使用既有训练函数所需的同一配置结构，输出独立保存。
def group_configuration(config: dict, group: str) -> Path:
    """生成该信息组的运行配置，原训练行、学习率、种子及早停规则保持不变。"""
    folder = Path(config['output']) / group
    folder.mkdir(parents=True, exist_ok=True)
    settings = {
        'source': config['source'], 'output': str(folder),
        'local_cache': config['local_cache'], 'batch_size': 8192,
        'max_epochs': 20, 'patience': 4, 'learning_rate': 0.001,
    }
    filename = folder / 'runtime.yaml'
    filename.write_text(yaml.safe_dump(settings, sort_keys=False))
    return filename


# 【核心逻辑345】实验直接下载所需月份的两个公开档案，完整文件原子发布且断点复用。
def download_month(task: tuple) -> None:
    """输入源类型、月份和原档目录；HTTP错误直接交给已有队列有限重试。"""
    source, month, folder = task
    if source == 'spot':
        relative = f'spot/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-{month}.zip'
    else:
        relative = f'futures/um/monthly/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-{month}.zip'
    destination = folder / source / relative.rsplit('/', 1)[-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    temporary = destination.with_suffix('.partial')
    with opener.open('https://data.binance.vision/data/' + relative, timeout=120) as response:
        with temporary.open('wb') as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)
    print('DOWNLOADED', source, month, destination.stat().st_size, flush=True)


# 【非核心】来源字段和单位已经明确，月档只读取实验所需列。
def read_month(folder: Path, source: str, month: str) -> pl.DataFrame:
    """现货使用微秒时间、分钟溢价使用毫秒，均延迟到区间结束后的下一个整秒。"""
    interval = '1s' if source == 'spot' else '1m'
    filename = folder / source / f'BTCUSDT-{interval}-{month}.zip'
    schema = {
        name: pl.Int64 if name in ['open_time', 'close_time', 'count'] else pl.Float64
        for name in BAR_COLUMNS
    }
    with zipfile.ZipFile(filename) as archive:
        with archive.open(archive.namelist()[0]) as stream:
            table = pl.read_csv(stream, has_header=source != 'spot', schema=schema)
    if source == 'spot':
        return table.select(
            (pl.col('open_time') * 1000 + 2 * SECOND).alias('timestamp_ns'),
            pl.col('close').alias('spot_close'), 'volume',
            pl.col('taker_buy_volume').alias('buy_volume'),
        )
    return table.select(
        (pl.col('open_time') * 1_000_000 + 61 * SECOND).alias('timestamp_ns'),
        (pl.col('close') * 10000).alias('premium_bps'),
    )


# 【核心逻辑346】月档直接进入因子计算，跨月携带历史尾部，生成与原训练矩阵同时间键的缓存。
def prepare_cross_inputs(filename: Path) -> None:
    """仅计算至历史复核末尾，逐月生成20列，不重放全部永续逐笔或生产新标准层。"""
    _, config, output = context(filename)
    if (output / 'features_ready.json').exists():
        return
    source = Path(config['source'])
    manifest = json.loads((source / 'e0/manifest.json').read_text())
    bounds = json.loads((source / 'e0/splits.json').read_text())['segments']
    stop = bounds['val_select']['stop_row']
    start = manifest['start_timestamp_ns']
    end = start + stop * SECOND
    months = sorted({date[:7] for date in manifest['dates'] if date <= '2026-07-06'})
    raw = Path(config['raw_folder'])
    tasks = [(kind, month, raw) for month in months for kind in ['spot', 'premium']]
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(download_month, tasks))
    market = np.load(source / 'e0/market.npy', mmap_mode='r')
    temporary = Path('/tmp/hft_cross_features_20261111.npy')
    cache = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(stop, 20))
    cache[:] = np.nan
    tail = None
    premium_tail = None
    column_names = None
    for month in months:
        spot = read_month(raw, 'spot', month).filter(pl.col('timestamp_ns') < end)
        premium = read_month(raw, 'premium', month)
        if premium_tail is not None:
            premium = pl.concat([premium_tail, premium])
        joined = spot.join_asof(premium, on='timestamp_ns', strategy='backward', tolerance=61 * SECOND)
        rows = (joined['timestamp_ns'].to_numpy() - start) // SECOND
        mid = np.full(len(rows), np.nan)
        valid = (rows >= 1) & (rows <= len(market))
        mid[valid] = market[rows[valid] - 1, 0]
        joined = joined.with_columns(pl.Series('perp_mid', mid))
        original_start = joined['timestamp_ns'][0]
        if tail is not None:
            joined = pl.concat([tail, joined])
        factors = cross_market_factors(joined, config['windows_s']).filter(
            (pl.col('timestamp_ns') >= max(start, original_start)) & (pl.col('timestamp_ns') < end),
        )
        column_names = factors.columns[1:]
        positions = (factors['timestamp_ns'].to_numpy() - start) // SECOND
        cache[positions] = factors.select(column_names).to_numpy().astype('float32')
        tail = joined.tail(1202)
        premium_tail = premium.tail(1)
        print('FACTORS_READY', month, factors.height, flush=True)
    cache.flush()
    del cache
    publish_cache(temporary, output / 'cross.npy')
    write_json_atomic(output / 'features_ready.json', {
        'columns': column_names, 'rows': stop, 'windows_s': config['windows_s'],
        'availability': 'closed_spot_second_plus_20ms_rounded_up_to_decision_second',
        'perp_mid': 'existing_market_at_previous_second',
    })


# 【核心逻辑348】新增列仅按原fit行标准化，各信息组与旧模型共用全部训练和评价行。
def cache_group(filename: Path, group: str) -> None:
    """写入已有GPU训练器读取的四份矩阵，追加缺失通道并保持列顺序。"""
    _, config, output = context(filename)
    runtime = group_configuration(config, group)
    destination = runtime.parent / 'inputs/180s'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'completed.json').exists():
        return
    source = Path(config['source'])
    metadata = json.loads((output / 'features_ready.json').read_text())
    columns = metadata['columns']
    chosen = feature_group_indices(columns, group)
    values = np.load(output / 'cross.npy', mmap_mode='r')
    row_sets = {
        split: np.load(source / f'e1/folds/final/{split}_rows.npy')
        for split in ['fit', 'tune']
    }
    fit_values = np.asarray(values[row_sets['fit']][:, chosen], dtype='float64')
    mean = np.nanmean(fit_values, axis=0).astype('float32')
    scale = np.nanstd(fit_values, axis=0).astype('float32')
    scale[scale < 1e-8] = 1
    if not np.isfinite(mean).all():
        raise ValueError('训练期新增列完全缺失，不能拟合本组模型')
    del fit_values
    splits = json.loads((source / 'e0/splits.json').read_text())['segments']
    for split in ['val_dev', 'val_select']:
        row_sets[split] = np.arange(splits[split]['start_row'], splits[split]['stop_row'])
    for split, rows in row_sets.items():
        temporary = Path('/tmp') / f'hft_cross_{group}_{split}.npy'
        matrix = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(len(rows), len(chosen) * 2))
        for begin in range(0, len(rows), 65536):
            selected = np.array(values[rows[begin:begin + 65536]][:, chosen], dtype='float32')
            matrix[begin:begin + len(selected)] = model_inputs(selected, np.arange(len(selected)), mean, scale)
        matrix.flush()
        del matrix
        publish_cache(temporary, destination / f'{split}_x.npy')
    np.savez(destination / 'preprocessing.npz', mean=mean, scale=scale)
    write_json_atomic(destination / 'completed.json', {'columns': [columns[index] for index in chosen], 'status': 'success'})


# 【非核心】消融按信息含义选择已有列，保持原来的变量顺序。
def feature_group_indices(columns: list[str], group: str) -> list[int]:
    """成交压力与价格组严格分开，列筛选不读取预测目标。"""
    groups = {
        'all': columns,
        'spot': [name for name in columns if name.startswith('spot_')],
        'premium': [name for name in columns if name.startswith('premium_')],
        'flow': [name for name in columns if name.startswith('spot_imbalance_')],
        'price': [name for name in columns if name.startswith('spot_') and not name.startswith('spot_imbalance_')],
    }
    return [index for index, name in enumerate(columns) if name in groups[group]]


# 【核心逻辑349】调用已有GPU训练器，仅补未运行的种子或消融组。
def train_cross_model(filename: Path, group: str, family: str, seed: int) -> None:
    """通过既有GPU锁串行使用单卡，参数与旧输入对照保持一致。"""
    from mm_research.models.input_comparison import train_input_group

    _, config, _ = context(filename)
    runtime = group_configuration(config, group)
    with Path('/tmp/hft_mm_research_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        input_group = 'old' if group == 'baseline' else 'combined'
        train_input_group(runtime, 180, input_group, family, seed)


# 【核心逻辑350】将新信息组与已经完成的旧模型按日期配对，输出真实增量和失败清单。
def report_cross_models(filename: Path) -> None:
    """汇总模型、逐日IC差和日期重采样区间，不把未运行模型记为零收益。"""
    _, config, output = context(filename)
    records = []
    missing = []
    random = np.random.default_rng(42)
    for group in ['spot', 'premium', 'all']:
        for family, seeds in [('xgb', [17]), ('mlp', [17, 42, 73])]:
            for seed in seeds:
                model = output / group / 'models' / f'180s_combined_{family}_seed{seed}'
                baseline = Path(config['baseline']) / f'180s_old_{family}_seed{seed}'
                if not (model / 'completed.json').exists():
                    missing.append(str(model))
                    continue
                metrics = json.loads((model / 'metrics.json').read_text())
                old_metrics = json.loads((baseline / 'metrics.json').read_text())
                for split in ['val_dev', 'val_select']:
                    paired = pl.read_parquet(model / f'{split}_daily.parquet').select('utc_day', 'ic').join(
                        pl.read_parquet(baseline / f'{split}_daily.parquet').select('utc_day', 'ic'),
                        on='utc_day', suffix='_old',
                    ).with_columns((pl.col('ic') - pl.col('ic_old')).alias('delta')).filter(pl.col('delta').is_finite())
                    differences = paired['delta'].to_numpy()
                    bootstrap = random.choice(differences, size=(1000, len(differences)), replace=True).mean(axis=1)
                    records.append({
                        'group': group, 'family': family, 'seed': seed, 'split': split,
                        'ic': metrics[split]['all']['ic'], 'old_ic': old_metrics[split]['all']['ic'],
                        'rank_ic': metrics[split]['all']['rank_ic'], 'r2': metrics[split]['all']['r2'],
                        'mean_daily_delta_ic': float(differences.mean()),
                        'positive_day_share': float((differences > 0).mean()),
                        'ci95_low': float(np.quantile(bootstrap, 0.025)),
                        'ci95_high': float(np.quantile(bootstrap, 0.975)),
                    })
    write_json_atomic(output / 'completed.json', {'status': 'success' if not missing else 'incomplete', 'missing': missing})
    if not records:
        raise ValueError('本批没有完成的模型，不能生成对照成绩')
    summary = output / 'summary'
    summary.mkdir(exist_ok=True)
    pl.DataFrame(records).write_csv(summary / 'models.csv')
    lines = [
        '# 跨市场180秒信息增量模型结果｜2026-11-11', '',
        '旧146列基线直接复用。新增输入分为现货16列、溢价4列、合并20列。',
        '主目标为180秒对数收益bps，训练截至2026-03-22；开发和历史复核分别截至05-14、07-06。',
        '新增输入只用已结束K线，永续Mid取上一整秒。所有模型共用原拟合行和内部早停行。', '',
        '| 新信息组 | 模型 | 种子 | 分区 | IC | 旧IC | 日均IC增量 | 日期95%区间 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for record in records:
        lines.append(
            f"| {record['group']} | {record['family']} | {record['seed']} | {record['split']} | "
            f"{record['ic']:.5f} | {record['old_ic']:.5f} | {record['mean_daily_delta_ic']:.5f} | "
            f"[{record['ci95_low']:.5f}, {record['ci95_high']:.5f}] |"
        )
    lines.extend(['', '上述区间为逐日配对重采样的描述性结果；本批多组比较未作多重比较调整，不能单凭最佳区间晋级。',
                  '此表评价预测增量，交易费用后收益需由既有回放链路另行计算。',
                  f'未完成模型数：{len(missing)}。详细指标见models.csv；原权重、预测和逐日指标保存在各信息组目录。'])
    (summary / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(records, ensure_ascii=False, indent=2), flush=True)
