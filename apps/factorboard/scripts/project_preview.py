"""看板独立入口：prepare更新诊断快照，serve只读浏览研究结果。"""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from project_board.prepare import prepare
from project_board.server import PreviewHandler


# 【非核心】本机启动看板，不安装依赖，不触发主项目研究任务。
def main() -> None:
    """接收prepare或serve，结果只保存在看板自己的workspace目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'serve'])
    parser.add_argument('--port', type=int, default=8765)
    arguments = parser.parse_args()
    if arguments.command == 'prepare':
        prepare()
        return
    frontend = Path(__file__).resolve().parents[1] / 'frontend'
    handler = partial(PreviewHandler, directory=str(frontend))
    server = ThreadingHTTPServer(('127.0.0.1', arguments.port), handler)
    print(f'http://127.0.0.1:{arguments.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
