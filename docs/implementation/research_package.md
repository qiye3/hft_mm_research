# 2026-11-06 — Polars研究包

当前研究只使用market_v2标准行情。盘口来自网盘25档快照，普通成交来自Binance官方ZIP。Polars负责探索性特征和标签，C++保留事件回放与入选实时特征计算能力。

## 唯一生产链路

```text
官方成交ZIP + 网盘25档快照CSV.zst
                    ↓ data/standard.py
标准snapshots/日期.parquet + trades/日期.parquet
                    ↓ features/base.py，固定20ms延迟
          base特征组与各衍生特征组
                    ↓ data/dataset.py
       features/版本/日期.parquet
                    ↓ targets/midprice.py
       labels/版本/horizon=周期/日期.parquet
             ↘             ↙
            按键关联生成X、y
```

| 模块 | 职责 |
| --- | --- |
| `data/sources.py`、`trades.py`、`orderbook.py`、`standard.py` | 官方成交和25档快照标准化 |
| `features/base.py`、`snapshot.py`、`trades.py` | 固定延迟、100ms盘口对齐和成交分桶，构成base组 |
| `data/quality.py` | 标准数据逐记录质量标签的公共表达式 |
| `features/validation.py` | 独立盘口抽查与全网格成交核对 |
| `data/dataset.py` | 版本定义、按日生产、独立特征/标签缓存和X/y加载 |
| `features/book.py`、`trades.py`、`returns.py` | 盘口、主动成交和历史收益 |
| `features/order_flow.py` | 事件OFI公式模块，快照研究中OFI保持未知 |
| `features/transform.py` | 分类公式的组织、输出列顺序和有效性 |
| `features/expressions.py` | 失衡比值、区段内滚动和整数Lot累计 |
| `targets/midprice.py` | 多周期中间价标签，独立管理有效性 |

## 特征组与保存字段

| 组 | 公式版本 | 保存列数 | 模型可选列数 | 有效标记 |
|---|---|---:|---:|---|
| `base` | `v2` | 58 | 47 | `base_feature_valid` |
| `book` | `v1` | 7 | 7 | `book_feature_valid` |
| `trades` | `v1` | 6 | 6 | `trade_feature_valid` |
| `returns` | `v1` | 3 | 3 | `return_feature_valid` |

完整特征文件有81列，包含3个样本键、58个base字段、16个衍生特征及4个组有效标记。默认模型读取16个衍生特征，显式选择base时读取其47个模型可选字段。

book组检查当前盘口，trades组要求完整5秒成交窗口，returns组要求当前及过去5秒盘口连续有效。读取时筛选所选组和标签的有效标记。组定义由`features/transform.py`的`GROUPS`管理。

## 时间与质量

标准表保留交易所时间。研究采样采用“交易所时间＋20ms”作为模拟可见时刻，样本t只读取已可见行情。快照向后对齐，成交窗口为 `(t-100ms,t]`。

20ms根据现有七月30天采集数据选定，快照平均接收延迟2.356ms，成交平均9.846ms。统计包含少量负延迟，说明原采集时钟有偏差。成交有明显长尾，固定20ms用于均值附近的研究假设。统计记录位于 `workspace/results/data_inspection/fixed_latency/summary.json`。

前日标准表存在时，读取前日以接回延迟后跨午夜可见的记录。首个可用日缺少前日输入时，起始未覆盖成交区间标为无效。前日快照也可作为日初已知状态，超过1s仍按盘口年龄规则失效。

盘口缺档、交叉、顺序异常、过期或来源缺口影响盘口连续区段。成交价量、方向或内容冲突异常影响对应成交窗口。编号只作为成交身份。五秒历史特征仍需要足够历史数据，在真实质量中断后重新积累。

价格标签只要求当前至未来价格窗口的盘口连续有效。成交异常不单独使价格标签失效。

## 配置与版本

Data标准化使用 `config/data_standardize_30d.yaml`。Features与Targets生产使用 `config/features_targets_30d.yaml`。

| 配置 | 当前值 |
| --- | --- |
| `standard_version` | market_v2 |
| `market_data_latency_ns` | 20000000，即20ms |
| `feature_groups.base` | v2，与衍生组共同保存 |
| `feature_version` | market_groups_v2 |
| `label_version` | market_mid_groups_v2 |
| `interval_ns` | 100000000，即100ms |
| `horizons_ns` | 100ms、500ms、1s、5s |

```text
workspace/data/
├── standard/market_v2/
│   ├── snapshots/YYYY-MM-DD.parquet
│   └── trades/YYYY-MM-DD.parquet
├── features/market_groups_v2/YYYY-MM-DD.parquet
└── labels/market_mid_groups_v2/horizon=1s/YYYY-MM-DD.parquet
```

其他预测周期分别使用对应horizon目录。基础列与衍生列共同进入每日特征文件，标签独立保存。文件元数据包含组版本、来源、固定延迟及步长，并绑定本日和前日输入身份。修改特征组时复用基础列，修改目标时读取既有基础列生成标签。版本目录共用文件锁，更新单日文件时使用原子替换。

## 生成、校验与读取

以下命令在项目根目录执行。

```sh
.venv/bin/python scripts/build_standard_data.py config/data_standardize_30d.yaml
.venv/bin/python scripts/build_features_targets.py config/features_targets_30d.yaml
.venv/bin/python scripts/validate_research_dataset.py config/features_targets_30d.yaml \
  2026-07-14 workspace/results/m2_standard/market_groups_v2/validation_2026-07-14.json
.venv/bin/python scripts/summarize_research_dataset.py workspace/data/features/market_groups_v2 \
  workspace/results/m2_standard/market_groups_v2/statistics_30d.json \
  --labels workspace/data/labels/market_mid_groups_v2/horizon=1s
```

校验使用独立NumPy数组计算，对盘口按网格抽查，对成交数量、笔数和金额逐网格比较。

新增特征组时配置 `feature_groups`，使用 `--only features`。组公式版本由代码与配置共同声明。修改目标时更新 `label_version` 或周期列表，使用 `--only labels`。`--rebuild` 重建选择的组或标签。改变固定延迟或基础定义时，使用新的特征及标签版本。

```python
from pathlib import Path
from mm_research.data.dataset import load_xy

features, targets, sample_index = load_xy(
    Path("workspace/data/features/market_groups_v2"),
    Path("workspace/data/labels/market_mid_groups_v2/horizon=1s"),
    dates=["2026-07-12", "2026-07-13"],
)
```

加载按 `(symbol,date,timestamp_ns)` 一对一关联，过滤所选组的有效性字段和 `label_valid`，并检查两侧基础定义一致。主标签为bps，同时保存Ticks和USDT辅助标签。

## 分类特征的阅读与修改

`build_features_day()` 复用或生成base组，再通过 `build_feature_group()` 调用
`add_book_features()`、`add_trade_features()` 和 `add_return_features()`。
各文件声明自己的输出列，组定义保存在 `features/transform.py`。
需要查看某类公式时，直接打开对应文件。新增正式研究字段时，在该文件计算公式并更新
类别列清单，再使用新的特征版本生产数据。

公共函数只包含重复使用的运算。`normalized_imbalance()` 接收买卖数量表达式，
调用方决定已知零量窗口的返回值。`rolling_sum_in_segment()` 接收窗口点数，按
`segment_id` 累计，要求完整窗口，可选择先换算整数 Lot。
盘口对齐共用 `features/snapshot.py` 的 `build_snapshot_grid()`，标准逐笔共用
`build_trade_bins()`。具体特征的字段、单位和有效性由各自模块定义。

独立校验从标准文件核对盘口对齐与成交分桶，覆盖跨日及质量中断。事件OFI由独立C++事件路径提供。

## 三类特征模块示例

以下三个模块分别展示快照、逐笔成交和 C++ 事件特征的写法，供后续增加特征时参考。

| 类型 | 模块 | 示例函数 | 特征定义 |
| --- | --- | --- | --- |
| Polars 快照 | `research/mm_research/features/snapshot.py` | `add_snapshot_notional_features()` | 前十档双侧挂单金额及金额失衡 |
| Polars 逐笔 | `research/mm_research/features/trade_events.py` | `build_trade_event_features()` | 主动净成交金额与成交强度 |
| C++ 事件 | `src/mm/features/order_flow.hpp` | `calculate_bbo_ofi()` | 更新前后买卖一价量对应的 OFI |

### 快照模块

`snapshot.py` 接收已按接收时间向后对齐的十档盘口网格。先按 `sum(price * quantity)`
计算双侧挂单金额，再计算双方金额差除以总金额，正值表示买侧金额较大。
金额单位 USDT，失衡无量纲。模块保留输入字段，增加 `SNAPSHOT_EXAMPLE_COLUMNS`
声明的三列，`book_valid` 为假时将示例特征置空。

新增快照特征时，可参考它的字段清单、Polars 表达式和盘口有效性处理。
基础层的 `build_snapshot_grid()` 负责标准快照读取和固定延迟对齐。

### 逐笔模块

`trade_events.py` 接收 `build_trade_bins()` 生成的成交分桶和连续 100ms 网格。
主动净额为买方主动成交金额减去卖方主动成交金额，单位 USDT。
1s 净额累计最近十个网格，成交强度为最近一秒内的成交笔数，单位笔/秒。
窗口使用 `(t-100ms,t]` 的右端点，已知无成交窗口补零。

模块保留 `TRADE_EVENT_EXAMPLE_COLUMNS` 字段清单，使用完整一秒窗口。
只在真实成交异常区间切段，`trade_feature_valid` 为假时将示例特征置空。
新增逐笔特征时，可参考分桶输入、分段窗口和预热处理。

### C++ 模块

`order_flow.hpp` 使用 `BboState` 表示更新前后的买卖一价格数量。
`calculate_bbo_ofi()` 是纯计算函数，返回单位为 BTC、买方压力为正的事件 OFI。
买侧价格上升计入当前数量，下降扣除原数量，相同价格计入数量差，卖侧使用相反方向。

调用方保证两份 BBO 有效且属于连续区段，快照初始化、盘口缺口与累计窗口由调用方管理。
现有 `src/mm/research_sampler.cpp` 已调用该函数，可同时参考模块定义和实际接入位置。
新增 C++ 特征时，可参考输入结构、计算函数及调用方管理状态的分工。

### 公共运算与扩展方式

Polars 公共运算位于 `features/expressions.py`，包括双方失衡比值、分段完整窗口累计
以及可选的整数 Lot 换算。各模块自行定义字段、单位、窗口和有效性。

添加新特征时，在对应 `features` 目录新增具有明确业务名称的模块，声明输出字段并实现计算函数。
进入正式生产时，在`GROUPS`登记公式版本、依赖、输出列、模型列与有效标记，并在`build_feature_group()`接入计算。
当前快照和逐笔示例具有各自的独立字段定义。

## 当前三十天研究结果

2026年7月1—30日共25,920,000个100ms网格。16个快照研究特征与1s标签关联后有25,663,197条可用样本，占99.01%。该版本同时改变了成交来源、时间口径和质量规则。

月初、已知盘口中断日和月末已经使用独立数组路径检查，盘口抽查与逐网格成交笔数、数量和金额一致。独立参考结果见`workspace/results/m2_standard/market_groups_v2/independent_validation_2026-07-14.json`，当前逐日统计见同目录的`statistics_30d.json`。

此前market_v2阶段的清理量及核对记录按历史目录保留。当前数据定义、组版本和来源信息读取Parquet内嵌元数据。
