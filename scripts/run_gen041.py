"""Gen041阈值准备、全历史因子生产和模型输入缓存的分阶段入口。"""

import argparse
from pathlib import Path

from mm_research.models.gen041 import prepare_small_threshold, produce_factors, cache_model_inputs


# 【非核心】配置明确标准逐笔、旧Features及本轮输出，不修改旧生产版本。
def main() -> None:
    """按阶段配置运行；因子支持单日验收与完整历史生产，缓存独立发布。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("stage", choices=["threshold", "factors", "cache"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--date")
    arguments = parser.parse_args()
    if arguments.stage == "threshold":
        prepare_small_threshold(arguments.config)
    elif arguments.stage == "factors":
        produce_factors(arguments.config, arguments.workers, arguments.date)
    else:
        cache_model_inputs(arguments.config)


if __name__ == "__main__":
    main()
