"""将实际 C++ 回放计数、时间覆盖和最终 Top-N 与原始 CSV 转换清单核对。"""
import argparse
import json
import math
from pathlib import Path


def validate(manifest: dict, replay: dict) -> None:
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
        expected = operation(manifest["book"][key], manifest["trades"][key]) + manifest["local_timestamp_offset_ns"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    validate(json.loads(arguments.manifest.read_text()), json.loads(arguments.report.read_text()))


if __name__ == "__main__":
    main()
