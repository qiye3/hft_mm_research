# 高频研究链路阅读和运行指南

研究对象为 Binance Futures BTCUSDT 永续合约。当前服务器研究区间为 2025-05-09 至 2026-10-21，共 531 天。研究时钟采用交易所时间加 20ms。Features 每 100ms 一行，Factors 与 Labels 每个 UTC 整秒一行。

## 数据如何流动

```text
官方普通成交 ZIP + 历史盘口快照
    → Data 标准化，按类型和日期保存
    → Features 建立 100ms 网格，每列保存全部日期
    → Factors 使用完整 100ms 历史计算秒级窗口，整秒输出
    → Targets 独立读取 Features，生成五个跨日预测周期
    → Evaluation 评价因子
    → Models 训练与选择线性基线
    → C++ 在真实 L2 与普通成交上回放并核对账本
```

标准盘口保留来源实际的 20 档或 25 档，当前研究观察前 10 档。实时 C++ 盘口使用完整增量事件重建。

## 配置分别属于谁

YAML 用来声明一次生产的输入、参数和输出，路径相对项目根目录。各阶段入口只读取自己所属的配置。

| 配置 | 输入 | 输出 | 入口 |
| --- | --- | --- | --- |
| `config/data/standardize_30d.yaml` | 原始盘口与官方成交 | 标准 snapshots、trades | `scripts/build_standard_data.py` |
| `config/features/build_531d.yaml` | 标准 Parquet | 100ms Features | `scripts/build_features.py` |
| `config/targets/build_1s.yaml` | 100ms Features、标准盘口边界信息 | 五周期 Labels | `scripts/build_labels.py` |
| `config/factors/all_non_ofi.yaml` | 100ms Features | 300 个整秒 Factors | `scripts/build_factors.py` |
| `config/evaluation/all_non_ofi.yaml` | Factors、Labels | IC、分组收益、分布与图表 | `scripts/evaluate_factors.py` |
| `config/models/linear_baseline.yaml` | Features、Factors、Labels、训练期 IC | 冻结线性模型和测试预测 | `scripts/train_linear_baseline.py` |
| `config/replay/linear_test_day.yaml` | 最终测试日真实 L2、官方成交、冻结模型 | 回放、账本及数值对照 | `scripts/run_linear_replay.py` |
| `config/research/overnight.yaml` | 上述各阶段配置路径 | 分阶段日志和完成标记 | `scripts/run_research.py` |

本地 30 天 Features 配置仍可独立使用。服务器总入口复用已有 Features，并补充采用冻结训练阈值的大小单列。

## 存储位置

```text
workspace/data/
├── standard/server_market_v1/{snapshots,trades}/YYYY-MM-DD.parquet
├── features/server_531d_v1/特征名.parquet
├── factors/server_factors_1s_v1/因子名.parquet
└── labels/server_mid_1s_v1/horizon=周期/YYYY-MM-DD.parquet
```

Feature 和 Factor 文件包括 symbol、date、timestamp_ns 与单个变量。每个 Factor 保留全部 45,878,400 个整秒位置，数值使用 Float64，无效值为空。标签按周期和日期独立保存，含目标时间、有效标记及 bps、ticks、USDT 三种单位。质量原因汇总单独保存为小型表。

文件元数据记录共同上游定义、公式版本、采样频率与窗口口径。时间戳用差分编码，资产和日期用字典编码。文件读取取得共享锁，发布取得独占锁，正式文件通过原子替换更新。

## 窗口与缺失

每个样本只使用该时刻已经可见的行情。正常无成交的数量为零，无法确定的数量为空。大小单分桶与时间网格关联时保留网格顺序。

Factors 的外层窗口单位为秒。5s 对应 50 条 100ms 观测，100s 对应 1,000 条，至少要求 80% 有效值。窗口跨午夜连续滑动，缺失位置继续占据时间位置。同组共用输入，滚动完成后才筛选整秒输出。42 天分块带前日历史，分块不改变公式。`return_100ms_bps` 等源特征继续沿用原定义。

Labels 预测未来 1s、10s、60s、180s、300s，目标必须精确落在 t+h。当前到目标时刻的完整 100ms 区间必须通过盘口质量检查。午夜通过标准快照时刻和质量标记判断衔接，2025-10-01 来源切换为显式边界。最后一天无法取得的未来目标为空。

## 代码阅读顺序

| 模块 | 职责 |
| --- | --- |
| `data/standard.py`、`sources.py` | 标准化与显式来源读取 |
| `features/base.py`、`snapshot.py`、`trades.py` | 100ms 盘口和逐笔聚合 |
| `features/transform.py`、`trade_size.py` | 分组公式、冻结阈值大小单桶 |
| `data/dataset.py`、`storage.py` | Features 生产、按列读取、文件锁和发布 |
| `targets/dataset.py`、`midprice.py` | 跨日上下文与五周期标签 |
| `factors/sources.py`、`operators.py`、`gen*.py` | 里层变换、秒窗口与分组公式 |
| `factors/dataset.py` | 分块、并行计算与全日期合并 |
| `evaluation/metrics.py`、`report.py` | 训练冻结分组、逐日 IC 和报告 |
| `models/dataset.py`、`linear.py`、`runner.py` | 分期数据、流式拟合与模型选择 |
| `models/replay.py`、`src/mm/linear_signal.cpp` | Python/C++ 对照与真实回测 |

函数上方的核心逻辑编号为全项目共用的阅读顺序。

## 运行命令

```sh
.venv/bin/python scripts/build_features.py config/features/build_531d.yaml
.venv/bin/python scripts/build_labels.py config/targets/build_1s.yaml
POLARS_MAX_THREADS=5 .venv/bin/python scripts/build_factors.py config/factors/all_non_ofi.yaml
POLARS_MAX_THREADS=2 .venv/bin/python scripts/evaluate_factors.py config/evaluation/all_non_ofi.yaml
POLARS_MAX_THREADS=2 .venv/bin/python scripts/train_linear_baseline.py config/models/linear_baseline.yaml
.venv/bin/python scripts/run_linear_replay.py config/replay/linear_test_day.yaml
```

服务器完整研究入口如下。已有同定义产物直接复用，定义变化须使用新版本或明确重建。

```sh
.venv/bin/python scripts/run_research.py config/research/overnight.yaml
```

Factors 初始使用 2 个进程，每进程 5 个 Polars 线程。Evaluation 按因子批次共享五周期标签读取。输出位于 `workspace/results/evaluation/server_factors_1s_v1/`、`workspace/results/models/linear_1s_v1/` 与 `workspace/results/research/one_second_v1/`。

## 评价与模型

逐日 Pearson、Spearman IC 和分组收益同时报告整秒与按预测周期取点的非重叠结果。1s 两种取点相同。分组阈值仅用训练期拟定，验证和测试冻结。

训练期截至 2026-05-31，验证期为 2026-06-01 至 08-31，测试期为 2026-09-01 至 10-21。前两个分界前统一剔除 300 秒，并检查目标时刻未越界。候选因子根据训练期 IC 选择，标准化只使用训练期数据，模型参数根据验证误差选择。

首版用未来 1s 的 bps 标签训练 OLS、Ridge、Lasso，比较历史收益、原始特征、筛选因子、实时压力四组。C++ 导出使用实时压力组，每 100ms 更新基础状态，整秒更新预测，间隔内使用最近预测，日初预热 120 秒。Python/C++ 使用同一实时输入逐行核对，L2 与研究快照的来源差异单独记录。
