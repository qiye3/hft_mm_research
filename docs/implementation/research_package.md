# 2026-11-06 — Features、Factors 与 Evaluation 阅读和运行指南

研究对象为Binance Futures BTCUSDT永续合约。当前使用2026年7月1—30日标准行情，每日864,000个100ms网格，共25,920,000行。

## 数据如何流动

```text
官方普通成交ZIP + 网盘25档快照
    → Data标准化，按类型和日期保存
    → Features基础采样，异常研究值置为null
    → 全日期连续计算，每个Feature一个Parquet
    → Gen001，每个Factor一个全日期Parquet
    → Evaluation关联独立标签，计算分布、IC和分组收益
```

基础采样逐源日期读取，以控制同时驻留的原始行情量。基础变量直接追加到最终单列文件的临时行组，滚动公式读取完整日期序列。盘口原始快照保持25档，当前研究读取前10档。

## 配置分别属于谁

所有文件路径相对项目根目录。每个YAML开头给出执行命令、输入和输出。

| 配置 | 负责的阶段 |
| --- | --- |
| `config/data/standardize_30d.yaml` | 原始行情到标准盘口、成交 |
| `config/features/build_30d.yaml` | 基础列、衍生Features与独立标签 |
| `config/factors/gen001.yaml` | Gen001的输入、公式窗口与因子输出 |
| `config/evaluation/gen001.yaml` | 因子、标签、日期、预测周期和评估结果 |
| `config/replay/` | C++行情回放 |
| `config/strategy/` | C++做市委托及风险参数 |

## 当前存储位置

```text
workspace/data/
├── standard/market_v2/
│   ├── snapshots/YYYY-MM-DD.parquet
│   └── trades/YYYY-MM-DD.parquet
├── features/market_columns_v1/
│   ├── mid_price.parquet
│   ├── imbalance_1.parquet
│   ├── trade_imbalance_1s.parquet
│   └── 其他基础变量、质量字段和衍生特征.parquet
├── factors/gen001_v1/
│   └── 因子名.parquet
└── labels/market_mid_columns_v1/
    └── horizon=1s/YYYY-MM-DD.parquet
```

Feature和Factor文件保存全部生产日期，字段为symbol、date、timestamp_ns和对应变量。公式版本、数据来源及窗口参数内嵌文件元数据。标签还支持100ms、500ms、5s及bps、ticks、USDT三种单位。

Parquet时间戳使用差分编码，资产和日期使用字典编码，减少单列布局中的重复时间键开销。相邻`.locks/`目录保存稳定锁文件。不同输出独立写入，读取取得共享锁，发布取得独占锁。

## 缺失值和时间窗口

研究时钟为交易所时间加20ms固定延迟。每个网格只使用截至该时刻已可见的事件。正常无成交窗口的量为零，无法确定的窗口为null。浮点NaN进入滚动公式前转为null。

滚动均值和累计量直接调用Polars。窗口按100ms位置计数，质量中断和午夜都继续向前滑动，缺失位置留在时间网格中。当前1s窗口至少8个有效样本，5s窗口至少40个有效样本。

历史收益使用当前和精确历史时刻的Mid，两端任一缺失则为空。盘口连续区段仅供标签规则使用。当前标签要求当前到未来时刻盘口连续有效、未来时刻精确为t+h，并处于同一UTC日。

## 模块如何分工

| 文件 | 主要职责 |
| --- | --- |
| `data/standard.py`、`sources.py`、`trades.py`、`orderbook.py` | 官方成交与25档快照标准化 |
| `data/storage.py` | 文件锁、元数据、Parquet原子发布和单列关联 |
| `data/dataset.py` | Features生产、增量更新、标签生成及X/y读取 |
| `features/base.py`、`snapshot.py`、`trades.py` | 100ms基础列和缺失值处理 |
| `features/book.py`、`trades.py`、`returns.py` | 按类别组织原生Polars公式 |
| `features/transform.py` | 特征组定义和公式选择 |
| `factors/gen001.py` | 六个压力序列及短长窗口均值差 |
| `factors/dataset.py` | 读取Factor依赖并逐因子发布 |
| `evaluation/dataset.py` | 以时间键关联因子和标签 |
| `evaluation/metrics.py` | 分布、Pearson IC、Rank IC和重复值分组 |
| `evaluation/report.py` | 逐日评估、汇总表格、PNG与HTML报告 |

Feature组用于组织公式和选择计算内容。每个表达式提取实际依赖列，新增衍生组可以复用已发布的基础变量。

Gen001的六个因子沿用用户构造文档的名称及公式，直接使用Polars的rolling_mean(10)减rolling_mean(50)，分别要求8和40个有效值。每个因子只读取所需输入，完整时间序列计算完成后选择输出日期。

## 运行命令

```sh
.venv/bin/python scripts/build_standard_data.py config/data/standardize_30d.yaml
POLARS_MAX_THREADS=6 .venv/bin/python scripts/build_features_targets.py config/features/build_30d.yaml
POLARS_MAX_THREADS=6 .venv/bin/python scripts/build_factors.py config/factors/gen001.yaml
POLARS_MAX_THREADS=6 .venv/bin/python scripts/evaluate_factors.py config/evaluation/gen001.yaml
```

同定义文件直接复用。显式重新计算使用`--rebuild`，只更新Features或标签使用`--only features`或`--only labels`。输入或公式身份变化时需要新版本或显式重建。POLARS_MAX_THREADS在进程启动前设置，控制Polars原生线程池。

读取X/y时明确选择列，保持列顺序。

```python
from pathlib import Path
from mm_research.data.dataset import load_xy

X, y, index = load_xy(
    Path("workspace/data/features/market_columns_v1"),
    Path("workspace/data/labels/market_mid_columns_v1/horizon=1s"),
    dates=["2026-07-12", "2026-07-13"],
    columns=["imbalance_5", "trade_imbalance_1s"],
)
```

## 如何检查结果

```sh
.venv/bin/python scripts/validate_research_dataset.py config/features/build_30d.yaml \
  2026-07-12 workspace/results/features/market_columns_v1/validation_2026-07-12.json
```

独立参考计算检查盘口抽样和逐网格成交数量、笔数及金额。真实跨午夜窗口另外用NumPy整数Lot累计核对。单元测试覆盖缺失值阈值、因果前缀、稳定锁、失败重试、相关系数和重复值分组。

Evaluation输出位于`workspace/results/evaluation/gen001_v1/`，包括summary.parquet、daily_distribution.parquet、daily_ic.parquet、group_returns.parquet、report.html及figures/。结果表元数据包含评估配置和因子定义。

分布统计使用因子自身有效样本，预测指标使用因子与标签共同有效样本。IC按日计算再等权汇总，Rank IC采用平均排名。分组收益采用bps，重复边界合并并记录实际组数，首尾组差只汇总组数完整的日期。
