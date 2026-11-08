"""运行有时间预算的树模型和GPU时序模型实验，配置声明输入、期限与候选。"""
import argparse
from pathlib import Path
from mm_research.models.experiments import run_model_experiments, run_single_trial


# 【非核心】读取本轮实验配置，调用研究包的模型队列。
def main() -> None:
    """输入YAML路径，结果保存在配置的独立输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--trial-index', type=int)
    arguments = parser.parse_args()
    if arguments.trial_index is None:
        run_model_experiments(arguments.config)
    else:
        run_single_trial(arguments.config, arguments.trial_index)


if __name__ == '__main__':
    main()
