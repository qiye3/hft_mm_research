"""服务器已发布评测的只读适配。仅加载统计表，不扫描原始行情和因子全量值。"""
from contextlib import contextmanager
import fcntl
from functools import partial
from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import time
from urllib.parse import parse_qs, unquote, urlsplit

import polars as pl

APP = Path(__file__).resolve().parents[2]


# 【非核心】共享已有评估锁，避免读到发布过程中的混合文件。
@contextmanager
def published_result(directory):
    """输入单个因子目录，在上游独占写入之间读取完整结果。"""
    with (directory / '.locks/evaluation.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        yield


# 【非核心】JSON边界将统计意义上的NaN输出为null。
def finite_json(value):
    """将嵌套结果中的非有限浮点转为JSON空值。"""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Results:
    """显式配置评测批次，按完成标记收录，列表默认展示测试期1s。"""

    # 【非核心】所有路径只来自部署配置，不接受请求参数拼接目录。
    def __init__(self, project, config):
        """记录项目位置和明确列出的评估批次。"""
        self.project = project
        self.roots = [project / root for root in config['evaluation_roots']]
        self.updated = 0
        self.directories = {}
        self.items = []
        self.manifest = {}

    # 【非核心】仅刷新轻量索引，用户打开资料库时不逐因子请求详情。
    def refresh(self):
        """每分钟重新发现已完成评测，保持列表和路径映射一起切换。"""
        if time.monotonic() - self.updated < 60:
            return
        directories = {}
        items = []
        dates = set()
        for root in self.roots:
            for marker in sorted(root.glob('*/completed.json')):
                directory = marker.parent
                name = directory.name
                if name in directories:
                    raise ValueError(f'Duplicate factor: {name}')
                with published_result(directory):
                    metadata = json.loads(marker.read_text())
                    summary = pl.read_parquet(directory / 'summary.parquet')
                    distribution = pl.read_parquet(directory / 'daily_distribution.parquet')
                definition = metadata['source']['definition']
                config = metadata['config']
                dates.update(config['dates'])
                selected = summary.filter((pl.col('split') == 'test') &
                    (pl.col('sampling') == 'all') & (pl.col('horizon_ns') == 1_000_000_000)).row(0, named=True)
                test_distribution = distribution.filter(pl.col('split') == 'test')
                test_rows = test_distribution['rows'].sum()
                zero_rate = (test_distribution['zero_rate'] * test_distribution['valid_rows']).sum() / test_distribution['valid_rows'].sum()
                deviation = selected['ic_std']
                items.append({
                    'id': name, 'display_name': '', 'gen': definition['batch'].replace('gen', 'Gen'),
                    'definition': {key: value for key, value in definition.items() if key != 'dates'},
                    'zero_rate': zero_rate,
                    'metrics': {
                        'ic': selected['ic_mean'], 'rank_ic': selected['rank_ic_mean'],
                        'spread': selected['top_bottom_bps'],
                        'icir': selected['ic_mean'] / deviation if deviation and selected['ic_mean'] is not None else None,
                        'positive': selected['positive_ic_day_ratio'],
                        'valid': selected['paired_rows'] / test_rows,
                        'days': selected['valid_ic_days'],
                    },
                })
                directories[name] = directory
        ordered_dates = sorted(dates)
        self.manifest = {
            'mode': 'production_results', 'run_id': 'server_completed_evaluations',
            'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'dates': ordered_dates, 'factor_ids': sorted(directories),
            'scopes': ['train', 'validation', 'test'],
            'horizons_ms': [1000, 10000, 60000, 180000, 300000],
            'list_scope': 'test', 'list_horizon_ms': 1000,
        }
        self.directories = directories
        self.items = items
        self.updated = time.monotonic()

    # 【非核心】保留上游日指标口径，只对选定范围做展示聚合。
    def evaluation(self, name, split, horizon, sampling):
        """读取已完成表，返回日IC、分组收益、分布和计算身份。"""
        directory = self.directories[name]
        condition = ((pl.col('split') == split) & (pl.col('horizon_ns') == horizon * 1_000_000) &
                     (pl.col('sampling') == sampling))
        with published_result(directory):
            metadata = json.loads((directory / 'completed.json').read_text())
            boundaries = json.loads((directory / 'group_boundaries.json').read_text())
            summary = pl.read_parquet(directory / 'summary.parquet')
            daily_all = pl.read_parquet(directory / 'daily_ic.parquet')
            daily = daily_all.filter(condition).sort('date')
            groups = pl.read_parquet(directory / 'group_returns.parquet').filter(condition)
            distribution = pl.read_parquet(directory / 'daily_distribution.parquet').filter(pl.col('split') == split).sort('date')
        selected = summary.filter(condition).row(0, named=True)
        group_summary = groups.group_by('group').agg(
            pl.col('mean_return_bps').mean().alias('bps'),
            pl.col('samples').sum().alias('samples'),
            pl.col('date').n_unique().alias('valid_days'),
        ).sort('group').to_dicts()
        config = {key: value for key, value in metadata['config'].items() if key not in ['dates', 'factors']}
        return {
            'id': name, 'summary': selected, 'daily_ic': daily.to_dicts(),
            'groups': group_summary, 'boundaries': boundaries,
            'daily_groups': groups.select('date', 'group', 'samples', 'mean_return_bps').sort('date', 'group').to_dicts(),
            'daily_horizons': daily_all.filter((pl.col('split') == split) & (pl.col('sampling') == sampling)).select('date', 'horizon_ns', 'ic', 'rank_ic').sort('date', 'horizon_ns').to_dicts(),
            'split_summary': summary.filter((pl.col('horizon_ns') == horizon * 1_000_000) & (pl.col('sampling') == sampling)).to_dicts(),
            'distribution': distribution.to_dicts(),
            'horizons': summary.filter((pl.col('split') == split) & (pl.col('sampling') == sampling)).sort('horizon_ns').to_dicts(),
            'config': config, 'definition': {key: value for key, value in metadata['source']['definition'].items() if key != 'dates'},
            'definition_version': metadata['definition_version'],
        }


class Handler(SimpleHTTPRequestHandler):
    """静态站点和只读评估接口共用单进程，避免并发放大内存。"""

    # 【非核心】生产入口显式加载生产视图，原本机预览入口保持独立。
    def do_GET(self):
        """在HTTP边界校验因子、切分、周期和采样选项。"""
        route = unquote(urlsplit(self.path).path)
        if route == '/js/data.js':
            self.path = '/js/production_data.js'
        if route == '/js/evaluation.js':
            self.path = '/js/production_evaluation.js'
        if not route.startswith('/api/'):
            super().do_GET()
            return
        self.server.results.refresh()
        results = self.server.results
        query = parse_qs(urlsplit(self.path).query)
        if route == '/api/v1/config':
            payload = results.manifest
        elif route == '/api/v1/factors':
            payload = {'items': results.items}
        elif route == '/api/v1/health':
            payload = {'ok': True, 'read_only': True, 'factors': len(results.items)}
        elif route.startswith('/api/v1/evaluations/'):
            name = route.removeprefix('/api/v1/evaluations/')
            split = query.get('scope', ['test'])[0]
            horizon = query.get('horizon', ['1000'])[0]
            sampling = query.get('sampling', ['all'])[0]
            if name not in results.directories:
                self.send_error(404, 'Unknown completed factor')
                return
            if split not in results.manifest['scopes'] or horizon not in [str(value) for value in results.manifest['horizons_ms']] or sampling not in ['all', 'nonoverlap']:
                self.send_error(422, 'Invalid split, horizon or sampling')
                return
            payload = results.evaluation(name, split, int(horizon), sampling)
        else:
            self.send_error(404, 'Unknown endpoint')
            return
        body = json.dumps(finite_json(payload), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


# 【非核心】仅监听服务器回环地址，通过SSH使用现有账号授权访问。
def serve(project, config_path, port):
    """加载显式评估清单，在指定回环端口提供看板。"""
    config = json.loads(config_path.read_text())
    server = HTTPServer(('127.0.0.1', port), partial(Handler, directory=str(APP / 'frontend')))
    server.results = Results(project, config)
    server.results.refresh()
    print(f'FactorBoard ready: {len(server.results.items)} completed factors', flush=True)
    server.serve_forever()
