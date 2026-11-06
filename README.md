# 2026-11-06 — 高频做市研究项目

基于 C++20 和 hftbacktest-cpp 的 Binance Futures BTCUSDT 永续合约研究项目，使用官方普通成交与网盘25档快照进行离线研究，使用历史L2事件进行交易仿真。

- **M0 行情回放**支持双时间戳事件调度、盘口重建、前 5／10 档观察及整日性能测试。
- **M1 做市基线**支持固定宽度报价、GTX 委托、在途敞口限制、成交记录和 PnL 核对。
- **M2-A 研究数据**使用官方普通成交与网盘25档快照标准Parquet入口，使用 Polars 构造特征，按独立版本保存特征与多周期 bps 标签。
- Python 用于离线数据转换与实验分析，C++ 负责回放和策略运行。

第三方引擎通过 Git Submodule 固定在 v0.1.0，提交为 `e0d2f7887c68755b220d9e69425d61d27e5b6432`。

## Quickstart

需要 C++20 编译器、Python 3.12+、Git、zlib 和 uv。macOS 可使用 Xcode Command Line Tools，Linux 安装对应编译器与 zlib 开发包。以下命令在项目根目录执行。

### 1. 安装依赖并构建

```sh
git submodule update --init --recursive
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
. .venv/bin/activate
cmake -S . -B workspace/build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build workspace/build/release --parallel 4
```

已有 `.venv` 时跳过创建步骤，继续安装依赖后再构建。

### 2. 运行测试与合成行情

```sh
ctest --test-dir workspace/build/release --output-on-failure
./workspace/build/release/replay_probe config/replay.yaml
```

合成回放报告输出至 `workspace/results/synthetic/replay.json`。

### 3. 准备一天真实行情

样本日为 2025-02-01 UTC。已有标准化数据时可直接进入下一步，转换器要求输出目录为空。

```sh
./scripts/download_tardis.sh
mkdir -p workspace/results/m0/BTCUSDT_2025-02-01
python scripts/convert_tardis.py \
  --book workspace/data/raw/binance-futures_incremental_book_L2_2025-02-01_BTCUSDT.csv.gz \
  --trades workspace/data/raw/binance-futures_trades_2025-02-01_BTCUSDT.csv.gz \
  --date 2025-02-01 --tick-size 0.1 --lot-size 0.001 \
  --output-dir workspace/data/normalized/BTCUSDT_2025-02-01 --shard-rows 250000 \
  > workspace/results/m0/BTCUSDT_2025-02-01/conversion.log 2>&1
```

### 4. 运行回放与做市实验

```sh
./workspace/build/release/replay_probe config/replay_btcusdt.yaml
./workspace/build/release/baseline_mm config/baseline_mm.yaml
```

策略参数位于 `config/baseline_mm.yaml`，默认输出至 `workspace/results/m1/BTCUSDT_2025-02-01/A_width20`。输出目录已存在时，请修改配置中的 output。

运行完整的 A/B/C 实验可指定一组新输出目录。

```sh
mkdir -p workspace/results/m1/BTCUSDT_2025-02-01_repeat
python scripts/analyze_baseline.py --run-experiments \
  --output-root workspace/results/m1/BTCUSDT_2025-02-01_repeat \
  --report-dir docs/reports/m1_repeat_2025-02-01 \
  > workspace/results/m1/BTCUSDT_2025-02-01_repeat/experiments.log 2>&1
```

每组实验保存 `orders.csv`、`fills.csv`、`equity.csv` 和 `summary.json`。实验根目录保存 `configs/`、`analysis.json` 与上述命令的 `experiments.log`。分析已有实验可执行以下命令。

```sh
python scripts/analyze_baseline.py --analyze-existing workspace/results/m1/BTCUSDT_2025-02-01
```

### 5. 运行本地 Polars 研究数据生产

已准备2026-07-01至07-30的market_v2标准行情。研究采样使用统一20ms固定行情延迟，生产独立特征与标签。

```sh
python scripts/build_standard_data.py config/data_standardize_30d.yaml
python scripts/build_features_targets.py config/features_targets_30d.yaml
python scripts/summarize_research_dataset.py workspace/data/features/market_groups_v2 \
  workspace/results/m2_standard/market_groups_v2/statistics_30d.json \
  --labels workspace/data/labels/market_mid_groups_v2/horizon=1s
```

Data 使用 `config/data_standardize_30d.yaml`，Features/Targets 使用 `config/features_targets_30d.yaml`。标准数据按 `standard/版本/{snapshots,trades}/日期.parquet` 保存，特征按 `features/版本/日期.parquet` 保存，标签按 `labels/版本/horizon=周期/日期.parquet` 保存。来源、字段、版本和 X、y 读取见 [研究包指南](docs/implementation/research_package.md)。

## 项目结构

M2-A 数据准备、生成和读取见 [研究数据集指南](docs/implementation/m2_a_dataset.md)。
当前分组研究结果位于 `workspace/results/m2_standard/market_groups_v2/`。

```text
hft_mm_research/
├── src/          C++ 源码
│   ├── apps/     回放与做市命令行入口
│   └── mm/       模块头文件与实现，detail/ 保存内部接入代码
├── research/     Python/Polars 研究包 mm_research
├── config/       回放与策略参数
├── scripts/      下载、转换、校验与实验分析
├── tests/        单元测试、集成测试与 benchmarks
├── third_party/  固定版本的回测引擎
├── docs/
│   ├── reading_guide.md  整体阅读入口
│   ├── implementation/  引擎与做市实现说明
│   └── reports/         按阶段和数据日期保存实验报告
└── workspace/    本地 data、results 与 build
```

运行产物按以下结构保存，全部由 Git 忽略。

```text
workspace/results/
├── synthetic/                  合成行情验证
│   └── replay.json
├── m0/BTCUSDT_2025-02-01/       单日回放结果与日志
│   ├── benchmark.json
│   ├── benchmark.stdout.log
│   ├── benchmark.time.log
│   └── conversion.log
├── m1/BTCUSDT_2025-02-01/       做市实验、配置、汇总与日志
│   ├── configs/
│   ├── analysis.json
│   ├── experiments.log
│   └── A_width20/ …
└── archive/                    已归档的实验附件
```

M0 性能测试及日志保存命令见 [单日性能报告](docs/reports/m0_2025-02-01/report.md#复现命令)。

从 `src/apps/baseline_mm.cpp` 阅读策略入口，随后查看 `src/mm/market_maker.cpp` 的报价与库存控制，以及 `src/mm/trade_recorder.cpp` 的账本和收益计算。

## 实验参数

回放和做市配置使用平铺 YAML 标量，研究配置另有日期和周期列表，相对路径以项目根目录解析。NPZ 保留全量深度，观察器按配置提取前 5 或 10 档。

M1 默认每单 0.001 BTC、最大净仓位 0.01 BTC，Maker/Taker 费率为 2/5 bps，请求/响应延迟为 1 ms/0.5 ms。费率与延迟采用机制实验假设。交易 PnL 覆盖交易现金流、尾仓盯市和手续费，资金费率、保证金与强平留待后续阶段。

## 文档

- [项目阅读指南](docs/reading_guide.md)
- [Polars 研究包与独立版本数据](docs/implementation/research_package.md)
- [30 天快照接入与迁移结果](docs/reports/polars_30d_2026-07/report.md)
- [研究数据的组织、下载、处理与详细概况](docs/implementation/data_requirements.md)
- [服务器推荐配置与费用估算](docs/implementation/server_recommendation.md)
- [GPU、策略组合与 Parquet 存储选择](docs/implementation/gpu_and_parquet.md)
- [M0 引擎接口与回放流程](docs/implementation/engine_notes.md)
- [M0 单日性能结果](docs/reports/m0_2025-02-01/report.md)
- [M1 委托状态、库存与账本](docs/implementation/m1_notes.md)
- [M1 报价宽度、库存上限与更新频率实验](docs/reports/m1_2025-02-01/report.md)
- [历史合约规则获取状态](docs/reports/m0_2025-02-01/report.md#未取得的合约元数据)
