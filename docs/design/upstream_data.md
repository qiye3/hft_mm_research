# 上游 Data 层设计

更新日期为2026-11-06。当前使用 `market_v2`，成交直接来自 Binance 官方 USD-M 日档，盘口来自网盘原始25档快照。

## 数据链路

```text
Binance 官方成交 ZIP → 读取CSV、统一字段、重复与质量检查 → trades/日期.parquet
网盘25档快照 CSV.zst → 流式读取、保留25档、盘口质量检查 → snapshots/日期.parquet
```

每个日期保存两份Parquet，分别位于 `workspace/data/standard/market_v2/snapshots/YYYY-MM-DD.parquet` 和 `trades/YYYY-MM-DD.parquet`。来源、行号和质量标签保存在记录中，输入文件身份、单位、时钟和配置定义写入Parquet元数据。原始压缩文件保留供追溯。

## 时间与质量规则

两份表都保存 `exchange_timestamp_ns`，单位为UTC Unix纳秒。研究采样按“交易所时间＋配置的固定行情延迟”生成模拟接收时间，当前研究配置为20ms。

官方成交按交易所时间、编号、源行号排序。`is_buyer_maker=true` 对应主动卖出，false对应主动买入。普通成交口径的编号跳跃直接保留。重复编号保留源文件第一条并标记，内容冲突另加 `content_conflict`，异常字段和越日记录保留原值并标记。

盘口保留全部原始记录、源顺序和双侧25档。快照日期分区对应源文件日期，读取时可按交易所时间与源行号稳定排序。原始接收时间只在生产期间用于源时钟和缺口检查，检查后移除。确认与怀疑中断分别标记在恢复行，实际盘口缺失继续按质量规则处理。

## 代码与配置

| 文件 | 职责 |
| --- | --- |
| `data/sources.py` | 读取官方ZIP和25档快照，统一字段 |
| `data/trades.py` | 官方成交去重、冲突和越日标签、排序 |
| `data/orderbook.py` | 保留完整盘口并标注来源质量 |
| `data/standard.py` | 配置编排、元数据、缓存定义及完整发布 |
| `data/quality.py` | 公共标签表达式 |
| `tests/test_standard_data.py` | 字段转换、异常保留、25档和发布边界 |

上述 `data/` 位于 `research/mm_research/data/`。生产配置为 `config/data_standardize_30d.yaml`，路径相对项目根目录。

```python
from pathlib import Path
from mm_research.data.standard import run_standard_production

summaries = run_standard_production(Path("config/data_standardize_30d.yaml"))
```

相同定义可复用，输入或规则改变需使用新版本。两份文件在临时目录生成并检查后，在版本锁内发布，移动失败时撤回本次已发布文件。

## 与现有研究流程的衔接

研究入口读取market_v2标准数据。100ms基础列属于base组，和衍生组共同保存到 `features/market_groups_v2/日期.parquet`，标签保存到 `labels/market_mid_groups_v2/horizon=周期/日期.parquet`。固定延迟为20ms，跨日读取前日标准表以接回延迟到达记录。

当前Features分为base、book、trades、returns四组，组版本和质量标记内嵌每日特征Parquet。研究读取按所选组及标签的有效性筛选。

完整步骤与YAML解释见 [数据处理笔记](/Users/qiye/Desktop/docs/Projects/HFT_MM/Data/每种数据的处理方式（从原始数据到Features直接使用的上游数据集）.md)。

## 本次三十天产物与核对

2026年7月1—30日共60份Parquet，包含65,244,225条原时点25档快照、88,548,524条官方普通成交，合计3,923,623,231字节，约3.92GB。

三十天每日成交数量与此前独立官方对照一致。7月12日全部成交字段与此前已验收版本一致，全部快照价量、25档列和源行序与原版本一致。所有日期均只有两份标准表，采用schema v2，均使用交易所时间。

盘口仍标记1处已确认中断和420处超过1秒的静默疑点。这些是盘口来源质量信息。统计及核对记录位于 `workspace/results/standard_data/market_v2/validation.json`。
