"""生产看板入口，读取已有评估统计。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from project_board.production import serve


# 【非核心】部署参数只在入口解析，数据根目录和评估清单显式指定。
def main():
    """启动只读服务，不运行研究任务。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    arguments = parser.parse_args()
    serve(arguments.project, arguments.config, arguments.port)


if __name__ == '__main__':
    main()
