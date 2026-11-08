"""本机只读路由；参数校验集中在HTTP边界。"""
from http.server import SimpleHTTPRequestHandler
import json
import polars as pl
from urllib.parse import parse_qs, unquote, urlsplit

from .dataset import RESULTS, SHORT_NAMES, time_series
from .statistics import SCOPES
from .views import correlations, current_run, evaluation, read_json


class PreviewHandler(SimpleHTTPRequestHandler):
    """前端静态目录与只读API，主研究目录不通过静态服务暴露。"""

    # 【非核心】单一边界校验输入，内部只接受已知因子、日期和研究选项。
    def do_GET(self) -> None:
        """构造API响应；无效选择明确报错，不切回默认数据。"""
        route = unquote(urlsplit(self.path).path)
        if not route.startswith('/api/'):
            super().do_GET()
            return
        query = parse_qs(urlsplit(self.path).query)
        root, manifest = current_run()
        scope = query.get('scope', ['full'])[0]
        mode = query.get('mode', ['daily'])[0]
        horizon_text = query.get('horizon', ['1000'])[0]
        if scope not in SCOPES or mode not in ['daily', 'fixed'] or horizon_text not in ['100', '500', '1000', '5000']:
            self.send_error(422, 'Invalid scope, grouping mode or horizon')
            return
        if route == '/api/v1/config':
            payload = manifest
        elif route == '/api/v1/health':
            payload = {'ok': True, 'run_id': manifest['run_id'], 'read_only': True}
        elif route == '/api/v1/factors':
            # 原列表的零值率直接读取已有核心汇总，不从直方图近似推断。
            summary = pl.read_parquet(RESULTS / 'summary.parquet').filter(pl.col('horizon_ns') == 1_000_000_000)
            zero_rates = dict(summary.select('factor', 'zero_rate').iter_rows())
            payload = {'items': [{'id': name, 'display_name': SHORT_NAMES[name], 'zero_rate': zero_rates[name]}
                                 for name in manifest['factor_ids']]}
        elif route == '/api/v1/compare/correlation':
            names = query.get('ids', [''])[0].split(',')
            method = query.get('method', ['pearson'])[0]
            if not (len(names) >= 2 and len(set(names)) == len(names)) or any(name not in manifest['factor_ids'] for name in names) or method not in ['pearson', 'spearman']:
                self.send_error(422, 'Invalid correlation selection')
                return
            payload = correlations(root, names, scope, method)
        elif route.startswith('/api/v1/evaluations/'):
            name = route.removeprefix('/api/v1/evaluations/')
            if name not in manifest['factor_ids']:
                self.send_error(404, 'Unknown factor')
                return
            payload = evaluation(root, name, scope, mode, int(horizon_text))
        elif route.startswith('/api/v1/series/'):
            name = route.removeprefix('/api/v1/series/')
            date = query.get('date', [''])[0]
            start = query.get('start', ['0'])[0]
            duration = query.get('duration', ['100'])[0]
            if name not in manifest['factor_ids'] or date not in manifest['dates'] or not start.isdigit() or not duration.isdigit():
                self.send_error(422, 'Invalid series selection')
                return
            if not 0 <= int(start) < 86400 or not 1 <= int(duration) <= 600 or int(start) + int(duration) > 86400:
                self.send_error(422, 'Time range outside UTC day')
                return
            expected_metadata = read_json(str(root / f'{name}.json'))['metadata']
            payload = time_series(name, date, int(start), int(duration), int(horizon_text), expected_metadata)
        else:
            self.send_error(404, 'Unknown read-only endpoint')
            return
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)
