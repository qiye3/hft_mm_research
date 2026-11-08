"""独立验收 M0 完整日回放的分类数量、时间边界和最终盘口。

validate 对照转换清单和 C++ JSON 报告，main 提供两个文件参数的命令行入口。
输入为既有 manifest 和回放汇总，输出为核对结果，差异以异常中断执行。
检查基于已保存的统计和原始最终盘口，无需再次载入整日事件分片。
"""
import argparse
import json
import math
from pathlib import Path


# 【非核心】将实际双侧消费结果与转换阶段保存的原始统计和最终盘口逐项核对。
def validate(manifest: dict, replay: dict) -> None:
    """检查一个完整日回放是否与离线数据准备阶段一致。

    输入
        manifest 为 convert_tardis 生成的清单，replay 为 C++ write_report 生成的汇总。
        两者应使用同一天、同一品种和相同的标准事件数据。
    输出
        全部一致时返回 None 并打印完成信息，计数、时间或价位差异触发 ValueError。
    用途
        同时检查两个处理器各自恰好消费一次逻辑事件，并确认分片缓存全部释放。
        首尾本地时间按原始两源的最早和最晚边界计算，价格数量采用绝对误差比较。
    """
    if manifest["timestamp_policy"] not in {"source_local_time", "exchange_timestamp_plus_fixed_latency"}:
        raise ValueError("回放校验需要明确的源接收时间或固定延迟时钟")
    for name in ["normalized_rows", "depth_updates", "snapshot_levels", "clear_events", "trade_events"]:
        if manifest[name] != replay[name]:
            raise ValueError(f"{name} 不一致：{manifest[name]} != {replay[name]}")
    if replay["date"] != manifest["date"] or replay["symbol"] != manifest["symbol"]:
        raise ValueError("品种或日期不一致")
    if replay["local_events"] != manifest["logical_events"] or replay["exchange_events"] != manifest["logical_events"]:
        raise ValueError("两侧没有分别恰好处理全部逻辑事件")
    if replay["shards_loaded"] != len(manifest["shards"]) or replay["shards_released"] != len(manifest["shards"]):
        raise ValueError("分片没有全部加载或释放")
    for boundary, operation in [("first", min), ("last", max)]:
        key = f"{boundary}_local_timestamp_ns"
        expected = operation(manifest["book"][key], manifest["trades"][key])
        if replay[key] != expected:
            raise ValueError(f"回放时间边界错误：{key}")
    for side in ["bids", "asks"]:
        expected_levels = manifest["book"][f"final_{side}"][:replay["depth_levels"]]
        actual_levels = replay[f"final_{side}"]
        if len(actual_levels) != len(expected_levels):
            raise ValueError(f"最终 {side} 档数不一致")
        for actual, expected in zip(actual_levels, expected_levels, strict=True):
            if not all(math.isclose(actual_value, expected_value, rel_tol=0, abs_tol=1e-8)
                       for actual_value, expected_value in zip(actual, expected, strict=True)):
                raise ValueError(f"最终 {side} 不一致：{actual} != {expected}")
    print("实际双侧回放与原始 CSV 的事件计数、时间边界、最终盘口一致。")


# 【非核心】从命令行加载清单与回放报告，执行完整日一致性检查。
def main() -> None:
    """读取位置参数 manifest 和 report 对应的两个 JSON 文件。

    输出为 validate 的成功提示，文件读取、JSON 格式和一致性错误直接导致 CLI 失败。
    本入口适合在 benchmark_replay 完成后作为独立验收步骤调用。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    validate(json.loads(arguments.manifest.read_text()), json.loads(arguments.report.read_text()))


if __name__ == "__main__":
    main()
