"""180秒序列基线入口，缓存准备与每次训练都是独立队列任务。"""
import argparse
import fcntl
from pathlib import Path

from mm_research.models.sequence180 import prepare_sequence, prepare_memory_cache, run_sequence
import yaml


# 【非核心】显式指定缓存准备或序列模型训练，保持任务失败范围独立。
def main() -> None:
    """配置声明输入、输出、历史长度和训练预算。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('stage', choices=['prepare', 'cache', 'train'])
    parser.add_argument('--family', choices=['mlp', 'gru', 'tcn'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--normalized-input', type=Path)
    parser.add_argument('--batch-size', type=int)
    arguments = parser.parse_args()
    if arguments.stage == 'prepare':
        prepare_sequence(yaml.safe_load(arguments.config.read_text()))
    elif arguments.stage == 'cache':
        if arguments.normalized_input is None:
            parser.error('cache需要显式指定normalized-input输出路径')
        prepare_memory_cache(arguments.config, arguments.normalized_input)
    else:
        if arguments.family is None or arguments.seed is None:
            parser.error('train需要显式指定family和seed')
        lock = Path('/tmp/hft_mm_research_gpu.lock')
        with lock.open('a') as descriptor:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            run_sequence(arguments.config, arguments.family, arguments.seed,
                         normalized_input=arguments.normalized_input, batch_size=arguments.batch_size)


if __name__ == '__main__':
    main()
