"""Gen001生产入口，读取必要Features并按因子独立保存全日期Parquet。"""

from pathlib import Path

import polars as pl

from mm_research.config import load_pipeline_config
from mm_research.data.storage import KEY_COLUMNS, artifact_ready, file_locks, read_metadata, scan_series, write_parquet
from mm_research.factors.gen001 import VERSION, factor_expressions


# 【核心逻辑27】按配置构造Gen001各因子，并在文件锁内原子发布。
def run_factor_production(config_path: Path, rebuild: bool = False) -> None:
    """输入Factors配置路径与重建标记，按需读取依赖并保存所选因子。

    公式基于完整Feature历史计算后才选择输出日期，保留日期范围之前的滚动历史。
    """
    root = Path(__file__).resolve().parents[3]
    config = load_pipeline_config(config_path, "factors")
    features = root / config["feature_root"] / config["feature_version"]
    output = root / config["factor_root"] / config["factor_version"]
    expressions = factor_expressions(config)
    for name in config["factors"]:
        expression = expressions[name]
        columns = sorted(set(expression.meta.root_names()))
        inputs = [features / f"{column}.parquet" for column in columns]
        filename = output / f"{name}.parquet"
        with file_locks(inputs, shared=True), file_locks([filename]):
            table, dataset = scan_series(features, columns)
            if not set(config["dates"]) <= set(dataset["dates"]):
                raise ValueError("Factor配置日期超出Features范围")
            metadata = {
                "kind": "factor", "name": name, "dataset": dataset,
                "definition": {
                    "batch": "gen001", "version": VERSION,
                    "dates": config["dates"],
                    "short_window": config["short_window"], "long_window": config["long_window"],
                    "short_min_samples": config["short_min_samples"], "long_min_samples": config["long_min_samples"],
                    "inputs": {column: read_metadata(features / f"{column}.parquet")["definition"] for column in columns},
                },
            }
            if artifact_ready(filename, metadata, rebuild):
                continue
            values = table.select(*KEY_COLUMNS, expression.alias(name))
            values = values.filter(pl.col("date").is_in(config["dates"]))
            write_parquet(values, filename, metadata)
        print(f"Factor完成 {name}", flush=True)
