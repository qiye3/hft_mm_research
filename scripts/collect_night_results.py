"""汇总已完成的夜间因子评价与模型结果，读取任务状态并生成报告。"""
import argparse
from pathlib import Path
from mm_research.evaluation.night_summary import collect_night_results


# 【非核心】读取常驻队列配置，按已保存任务状态整理结果。
def main() -> None:
    """输出位于队列自身目录，失败项与成功项同时记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    arguments = parser.parse_args()
    collect_night_results(arguments.config)


if __name__ == '__main__':
    main()
