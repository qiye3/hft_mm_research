"""分阶段YAML配置契约，拒绝重复键、缺项、未知字段及错误入口。"""

from pathlib import Path

import yaml

PIPELINE_FIELDS = {
    "model_comparison": {
        "stage", "start_date", "end_date", "feature_folder", "factor_sources", "label_folder",
        "dataset_output", "output", "train_end", "validation_end", "minimum_train_coverage",
        "families", "training_rows", "validation_rows", "seed", "threads", "chunk_rows",
        "ridge_alphas", "lasso_alphas", "tree", "gru", "read_workers",
    },
    "data_standardize": {
        "stage", "version", "output_root", "dates", "snapshots", "trades",
        "tick_size", "lot_size", "snapshot_silence_ns", "confirmed_snapshot_gaps",
        "snapshot_format", "snapshot_depth",
    },
    "features": {
        "stage", "standard_root", "standard_version", "dates", "feature_groups",
        "feature_root", "feature_version",
        "market_data_latency_ns", "interval_ns", "tick_size", "lot_size",
        "max_book_age_ns", "rolling_min_samples",
    },
    "targets": {
        "stage", "feature_root", "feature_version", "standard_root", "standard_version",
        "dates", "label_root", "label_version", "horizons_ns", "output_interval_ns",
        "source_boundaries", "workers",
    },
    "factors": {
        "stage", "feature_root", "feature_version", "factor_root", "factor_version",
        "dates", "batches", "min_fraction", "tick_size", "chunk_days", "workers", "merge_workers", "write_threads", "output_interval_ns",
        "aggregation_windows_s", "mean_diff_windows_s", "reuse_factor_versions",
    },
    "research": {"stage", "feature_config", "target_config", "factor_config", "evaluation_config", "model_config", "replay_config", "output"},
    "linear_replay": {
        "stage", "date", "book", "trades", "engine_output", "market_data_latency_ns",
        "model_output", "baseline_config", "alpha_config", "output",
    },
    "models": {
        "stage", "feature_root", "feature_version", "factor_root", "factor_version",
        "label_root", "label_version", "evaluation_root", "dates", "train_end", "validation_end",
        "batches", "min_fraction", "factors_per_batch", "realtime_per_batch", "sample_stride_ns",
        "ridge_alphas", "lasso_alphas", "workers", "output", "purge_ns",
    },
    "evaluation": {
        "stage", "factor_root", "factor_version", "label_root", "label_version", "dates",
        "factors", "horizons_ns", "groups", "output", "train_end", "validation_end", "workers", "output_interval_ns", "purge_ns", "factor_batch_size",
        "ols_factors",
        "validation_dev_end", "factor_lag_ns",
    },
}

# 【非核心】可选字段不强制出现，读取后补默认值。
OPTIONAL_FIELDS = {"evaluation": {"ols_factors"}}


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
        owners = {"data_standardize": "Data标准化", "features": "Features", "targets": "Targets",
                  "factors": "Factors", "evaluation": "Evaluation", "models": "Models", "research": "Research", "linear_replay": "LinearReplay"}
        owner = owners[stage]
        raise ValueError(f"此入口需要{owner}配置")
    required = PIPELINE_FIELDS[stage]
    optional = OPTIONAL_FIELDS.get(stage, set())
    missing = (required - optional) - config.keys()
    unknown = config.keys() - required
    if missing or unknown:
        raise ValueError(f"配置缺少字段{sorted(missing)}，未知字段{sorted(unknown)}")
    # 【非核心】可选字段在此补默认值，运行代码无需再判空。
    for field in optional:
        config.setdefault(field, [])
    return config
