# 2026-11-03 — hft_mm_research / M0 + M1

Binance Futures BTCUSDT 永续合约的 C++20 高频交易研究骨架。已完成 M0 行情回放验证和 M1 固定报价做市基线，支持原生部分成交、在途库存约束及逐笔 PnL 核对。第三方引擎是 `shenron0101/hftbacktest-cpp`，通过 Git Submodule 固定于 v0.1.0 / `e0d2f7887c68755b220d9e69425d61d27e5b6432`。

## 项目结构与入口

```text
hft_mm_research/
├── CMakeLists.txt
├── config/                 # M0 replay*.yaml；M1 baseline_mm.yaml
├── third_party/hftbacktest-cpp/  # 固定版本，源码不修改
├── data/{raw,normalized,samples}/ # 数据忽略提交
├── include/mm/             # market_observer、replay、market_maker、trade_recorder
├── src/
│   ├── market_observer.cpp # 读取原生订单簿、Mid/Spread、逐事件统计、Top-N 采样
│   ├── replay.cpp          # 原生处理器接入、回放编排、配置与 JSON 报告
│   ├── replay_probe.cpp    # 最小观察入口，无订单
│   ├── market_maker.cpp    # M1 固定报价、在途敞口预算与原生回测编排
│   ├── trade_recorder.cpp  # M1 逐笔记录、平均成本 PnL 与原生状态核对
│   ├── trading_engine.hpp  # 原生处理器转发及旧拒绝回报兼容
│   └── strategy/baseline_mm.cpp # M1 可执行入口
├── tools/                  # 单日下载脚本、离线 Tardis 转换器
├── tests/                  # 盘口/双时钟/分片用例及 gzip→NPZ→C++ 集成验证
├── benchmarks/benchmark_replay.cpp
├── research/analyze_baseline.py # M1 A/B/C 实验、CSV 独立对账和 markout
└── docs/                   # 引擎笔记、元数据状态及单日测试结果
```

建议先阅读 `src/replay_probe.cpp`，然后沿 `run_replay → replay_reader → Backtest::close` 阅读，最后研究 `MarketObserver`。引擎具体类、函数、文件和时间语义见 [engine_notes.md](docs/engine_notes.md)。

## 准备构建环境

macOS 需要已安装 Xcode Command Line Tools；Linux 需要 C++20 编译器、zlib 开发包和 Git。Python 只用于离线转换和转换测试，不参与 C++ 回放运行。

```sh
cd /Users/qiye/zxy_hft/hft_mm_research
git submodule update --init --recursive
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
. .venv/bin/activate
```

### 先验收原生框架

```sh
cmake -S third_party/hftbacktest-cpp -B build/upstream -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DHBT_BUILD_TESTS=ON -DHBT_BUILD_EXAMPLES=ON
cmake --build build/upstream --parallel 4
ctest --test-dir build/upstream --output-on-failure
./build/upstream/hbt_market_making
```

上游测试依赖 GoogleTest v1.15.2，首次配置需要联网。主项目构建无需获取 GoogleTest。

### 主项目与 M0 验证

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/release --parallel 4
./build/release/replay_probe config/replay.yaml
ctest --test-dir build/release --output-on-failure
```

`results/synthetic.json` 包含事件数量、首尾时间戳、最终买卖盘口、Mid 和 Spread。合成数据的前四条与第三方示例一致；扩展部分加入 11 档盘口及删除买一，验证深档进入目标深度。预期逻辑事件 25 条，其中深度更新 23、成交 2；最终买一 99.8、卖一 100.1、Mid=99.95、Spread=0.3。

## 一天真实行情

固定样本日为 **2025-02-01 UTC**。免费样本采用 Tardis 完整日文件，只下载 L2 与 trades 两个文件，不扩展其他日期。

```sh
./tools/download_tardis.sh
python tools/convert_tardis.py \
  --book data/raw/binance-futures_incremental_book_L2_2025-02-01_BTCUSDT.csv.gz \
  --trades data/raw/binance-futures_trades_2025-02-01_BTCUSDT.csv.gz \
  --date 2025-02-01 --tick-size 0.1 --lot-size 0.001 \
  --output-dir data/normalized/BTCUSDT_2025-02-01 --shard-rows 250000
./build/release/replay_probe config/replay_btcusdt.yaml
./build/release/benchmark_replay config/replay_btcusdt.yaml
```

可在允许读取系统统计的终端附加 macOS 的 `/usr/bin/time -l` 或 Linux 的 `/usr/bin/time -v`；程序自身已经输出计时和 Peak RSS。两入口使用同一真实引擎；性能结果取独立 `benchmark_replay` 进程。已经准备的标准数据不要原地重新转换；转换器要求输出目录为空，以防混用分片。需要重新准备时，请先选择新的输出目录并修改配置。

### 转换与数据校验

CSV gzip 流完整读取会核对 gzip CRC；检查列、交易所/品种、有限正价格、非负数量、价格/数量网格、本地时间有序、初始快照及完整 UTC 日首尾覆盖。遇到新快照先清空原簿，再写入快照价位；零数量增量交由引擎删除。

事件布局与 C++ `hbt::Event` 一致：`ev,u64; exch_ts,i64; local_ts,i64; px,f64; qty,f64; order_id,u64; ival,i64; fval,f64`，共 64 字节，小端，时间单位为 ns。每个源事件分别生成 EXCH-only 和 LOCAL-only 记录，再按相应可见时间稳定归并，使两个处理器各自有序。相同时间戳优先处理 trades，再处理 L2；不以简单单时钟排序代替双时钟重排。若原始行情存在非正延迟，则对本地时间统一增加最小必要偏移，明确记录在 manifest。

全量深度保留。`depth_levels: 5/10` 仅控制观察器读取的档数，不裁掉深层增量。`snapshot_interval_ns` 仅控制 Top-N 提取频率（默认 100 ms）；事件计数和 BBO 价差仍逐事件计算。最终盘口额外采集一次，并与转换阶段从原始 CSV 得到的最终盘口核对。

转换清单 `manifest.json` 记录原始文件大小、各类事件数、时间范围、最终盘口、路由方式及各分片大小。C++ 引擎共享分片缓存，避免整日数据常驻内存。

### 配置约定

当前 `replay.yaml` 是平铺 YAML 标量，支持注释与单/双引号字符串，不支持嵌套或多行值；路径和标量中不要包含 `#`。C++ 只解析这个明确的配置格式，未知字段报错。所有相对路径相对配置目录的上一级（项目根目录），可从任意目录传入配置文件绝对路径启动。

### 合约规则来源限制

Tardis instruments 接口返回 HTTP 401，Binance 官方 exchangeInfo 在当前网络返回 HTTP 451。历史官方合约元数据尚未取得。`0.1` 价格步长和 `0.001` 数量步长作为研究参数，转换器逐行核对当天数据是否满足；它们不被标注为已认证的历史交易规则。最小名义金额等未验证规则保留为空，详见 [contract_metadata_status.json](docs/contract_metadata_status.json)。M0 不提交订单，规则限制不会被用于模拟下单或宣称真实成交精度。

## 性能结果口径

- 整体耗时包含 `Backtest::close()` 中的加载、解压、调度、Local/Exchange 与观察器，不含下载和离线转换；文件缓存冷热状态不作强制控制。
- `logical_events_per_second` 以本地收到的逻辑事件为分子；`processor_dispatches_per_second` 以双侧实际处理次数为分子；`normalized_rows` 是 NPZ 中包含路由副本的行数。
- `local_step_latency` 系统性地每 1024 个本地事件采一次，范围为 `Local::process + observer`，含计时调用的测量开销，不是完整单事件回放耗时或策略延迟。没有策略计算延迟这一指标。
- Peak RSS 是独立基准进程截至回放完成及最终盘口采集的内存高水位；Release 使用 `-O3 -DNDEBUG`，不启用 native 架构、并行回放或自定义内存管理。

当日数据量、实测结果、机器规格和可复现命令见 `docs/benchmark_2025-02-01.md`，结构化报告与环境文件保存在同一目录。数据、临时构建产物和 `results/` 不纳入 Git。

回放后核对：

```sh
python tools/validate_replay.py data/normalized/BTCUSDT_2025-02-01/manifest.json results/benchmark_BTCUSDT_2025-02-01.json
```

## M1：固定报价做市基线

阅读入口：[M1 引擎与账本笔记](docs/m1_notes.md) · [真实日七组实验报告](docs/m1_2025-02-01/report.md)。

```sh
# 在项目根目录、已激活 .venv 的情况下
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/release --parallel 4
ctest --test-dir build/release -R mm.market_maker --output-on-failure

# 单次 20-tick 半宽基线；输出目录已存在时主动拒绝覆盖
./build/release/baseline_mm config/baseline_mm.yaml

# A/B/C 共七组机制实验，串行回放同一天数据并逐组独立核对
python research/analyze_baseline.py --run-experiments
```

已完成的逐笔记录在 `results/m1/<实验>/`，包含 `orders.csv`、`fills.csv`、`equity.csv` 和 `summary.json`，不纳入 Git。默认 `baseline_mm.yaml` 指向已完成的 `A_width20`，直接重复运行会拒绝覆盖；单次重新实验请修改 baseline_mm.yaml 的 output；重跑整组可使用 `--output-root results/m1_repeat --report-dir results/m1_repeat_report`，无需修改源码。仅重新分析已有逐笔输出可执行：

```sh
python research/analyze_baseline.py --analyze-existing results/m1
```

交易始终调用 `Backtest::submit_buy_order/submit_sell_order/cancel`。实际订单状态、成交和库存由原框架管理；策略只维护目标报价与递增订单 ID。任何新单或撤单在途都继续预留未成交数量，取消请求本身不会释放风险预算。成交从真实回报逐条保存，即使同一周期的买卖成交互相抵消也不会丢失。

默认使用原生 `PartialFillExchange`，每单 0.001 BTC、最大净仓位 0.01 BTC、GTX 仅挂单；每单一个 lot 的真实日没有部分成交，多 lot 部分成交由回归用例验证。Maker/Taker 费率 2/5 bps 与固定请求/响应延迟 1 ms/0.5 ms 是研究假设。历史认证规则仍受限。净 PnL 包含全部已记录交易现金流、尾仓盯市与手续费；不包含资金费、保证金与强平，不制造期末平仓。

M1 只读取 BBO 做决策，不进行 M0 的定期完整 Top-N 提取。因此两阶段运行耗时的观测工作负载不同，M1 的耗时不作为 M0 引擎优化结果。后续阶段固定为：M2 研究特征与 Alpha，M3 研究逆向选择保护、质量归因、延迟/队列敏感性和 C++ 优化；本次未扩展这些模块。
