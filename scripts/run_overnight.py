"""Linux服务器常驻入口，任务失败互相隔离，内存与线程预算由YAML声明。"""
import argparse
from pathlib import Path
from mm_research.overnight import run_overnight


# 【非核心】读取夜间队列配置，常驻运行至队列任务进入最终状态。
def main() -> None:
    """输入config/research/下的队列配置，输出独立状态、资源记录和任务日志。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    arguments = parser.parse_args()
    run_overnight(arguments.config)


if __name__ == '__main__':
    main()
