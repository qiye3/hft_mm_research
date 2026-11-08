"""将Tardis L2及逐笔压缩CSV转换成原生引擎NPZ，核心实现在data/engine.py。"""

import argparse
import json
from pathlib import Path

from mm_research.data.engine import convert


# 【非核心】将命令行文件和样本日参数交给单日转换流程，并输出简要完成统计。
def main() -> None:
    """执行离线数据准备 CLI。

    输入来自 --book、--trades、--date、--output-dir 和可选步长、分片容量。
    成功时输出 manifest 及标准化文件，并在终端打印事件数和转换耗时。
    参数或源数据错误向调用进程传播，由批处理命令据退出状态判断是否继续回测。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tick-size", type=float, default=0.1)
    parser.add_argument("--lot-size", type=float, default=0.001)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    arguments = parser.parse_args()
    manifest = convert(arguments.book, arguments.trades, arguments.date, arguments.output_dir,
                       arguments.tick_size, arguments.lot_size, arguments.shard_rows)
    print(json.dumps({key: manifest[key] for key in ["logical_events", "normalized_rows", "conversion_seconds"]}))


if __name__ == "__main__":
    main()
