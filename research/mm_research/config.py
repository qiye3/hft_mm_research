"""当前两阶段YAML配置契约，拒绝重复键、缺项、未知字段及错误入口。"""

from pathlib import Path

import yaml

PIPELINE_FIELDS = {
    "data_standardize": {
        "stage", "version", "output_root", "dates", "snapshots", "trades",
        "tick_size", "lot_size", "snapshot_silence_ns", "confirmed_snapshot_gaps",
    },
    "features_targets": {
        "stage", "standard_root", "standard_version", "dates", "feature_groups",
        "feature_root", "feature_version", "label_root", "label_version",
        "market_data_latency_ns", "interval_ns", "tick_size", "lot_size",
        "max_book_age_ns", "horizons_ns",
    },
}


# 【非核心】拒绝配置中的重复键，避免参数被静默覆盖。
class StrictYamlLoader(yaml.SafeLoader):
    """保持SafeLoader的标量类型，只接受字符串键且不允许重复。"""

    # 【非核心】将YAML映射节点转为字典，并检查键类型与唯一性。
    def construct_mapping(self, node, deep=False):
        values = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML配置键必须为字符串，日期键请加引号")
            if key in values:
                raise ValueError(f"重复YAML配置键 {key}")
            values[key] = self.construct_object(value_node, deep=deep)
        return values


# 【非核心】按入口所属阶段读取配置，检查必填字段和未知字段。
def load_pipeline_config(filename: Path, stage: str) -> dict:
    """输入配置路径与所属阶段，返回字段完整且唯一的参数字典。"""
    config = yaml.load(filename.read_text(), Loader=StrictYamlLoader)
    if not isinstance(config, dict):
        raise ValueError("YAML配置必须是键值映射")
    if config.get("stage") != stage:
        owner = "Data标准化" if stage == "data_standardize" else "Features/Targets"
        raise ValueError(f"此入口需要{owner}配置")
    required = PIPELINE_FIELDS[stage]
    missing = required - config.keys()
    unknown = config.keys() - required
    if missing or unknown:
        raise ValueError(f"配置缺少字段{sorted(missing)}，未知字段{sorted(unknown)}")
    return config
