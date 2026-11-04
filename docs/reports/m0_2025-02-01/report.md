# 2026-11-03 — BTCUSDT 2025-02-01 单日回放基准

运行路径于 2026-11-04 适配当前目录布局。

M0 已通过。使用 Tardis 的 **2025-02-01 UTC 完整日** Binance Futures BTCUSDT 永续行情，经原生 C++ 引擎回放到 EndOfData；本项目没有策略、下单、库存优化或机器学习模块。

## 数据量

| 内容 | 数量 |
| --- | ---: |
| 盘口增量 | 76,536,910 |
| 快照价位记录 | 4,084 |
| 快照前清空事件 | 2 |
| 逐笔成交 | 2,078,447 |
| 单侧逻辑事件合计 | 78,619,443 |
| 标准化路由记录／双侧 dispatch | 157,238,886 |
| NPZ 分片 | 629 |
| L2 原始 csv.gz | 421,757,848 字节 |
| trades 原始 csv.gz | 19,404,936 字节 |
| 原始压缩文件合计 | 441,162,784 字节（441.16 MB） |
| 标准化 NPZ 合计 | 965,343,281 字节（965.34 MB） |
| 离线转换时间 | 228.64 秒（不计入回放耗时） |
| 本地时间统一偏移 | 0 ns |

CSV 保留全部深度，未截掉第十档以外的增量。事件分为单侧 EXCH/LOCAL 路由行，以各自可见时间稳定排序。每个源事件在每侧恰好处理一次；快照开始时新增全簿清空事件。标准化行数不等同于新的逻辑行情数。

原始文件及分片均留在本地 `workspace/data/`，不纳入 Git。逐文件大小、事件分类、时间覆盖及来源文件见 [数据清单](../../../workspace/results/archive/m0_2025-02-01/data_manifest_2025-02-01.json)。

## 机器与构建

Apple M5，10 个逻辑核心，16 GiB 内存，macOS 26.4 arm64；AppleClang 21.0.0。Release 编译参数包含 `-O3 -DNDEBUG -std=c++20`，未启用 `-march=native`、并行回放、自定义分配器。原始编译命令和依赖版本见 [环境记录](../../../workspace/results/archive/m0_2025-02-01/benchmark_environment.json)。

固定引擎提交：`e0d2f7887c68755b220d9e69425d61d27e5b6432`（v0.1.0）；第三方源码未修改。

## 实测结果

| 指标 | 结果 |
| --- | ---: |
| 整体回放耗时 | 253.214654 秒 |
| 单侧逻辑事件吞吐 | 310,485.36 条/秒 |
| 双侧处理器 dispatch 吞吐 | 620,970.72 次/秒 |
| Peak RSS | 74,940,416 字节（71.47 MiB） |
| 局部采样 p50 | 41 ns |
| 局部采样 p95 | 42 ns |
| 局部采样 p99 | 161,625 ns |

整体计时包围 `Backtest::close()`，包括初次加载、NPZ 解压、其余分片 IO、事件调度、Local、Exchange 和观察器。不是纯撮合或纯订单簿微基准。使用共享缓存完整加载并释放了 629 个分片。

Top-10 每 100 ms 提取一次，事件分类及 BBO 价差逐事件统计。局部耗时每 1024 个 Local 事件系统采样一次，共 76,777 个样本；测量范围为 **一次 Local::process + observer**，不含交易所侧、调度器与 IO。分位数采用样本 nearest-rank；计时调用的测量开销未扣除。该范围包含定期 Top-N 提取，不能将这些分位数当作纯引擎完整单事件延迟。当前没有策略，不报告策略自身延迟。

Peak RSS 使用进程内 `getrusage(RUSAGE_SELF)` 的高水位，测量截至回放完成及最终盘口采集；不包含离线转换。外层 `/usr/bin/time -l` 在本执行环境无法读取 `kern.clockrate`，附加统计返回错误；基准程序已完成、JSON 报告正常生成，随后独立数据一致性检查通过。这里使用程序内指标，没有用系统工具失败输出补造数值。

这是单次本机基线；未清空操作系统文件缓存，记录的时间口径包含 IO，但不作为冷盘保证。

## 行情一致性

两侧各处理 78,619,443 条逻辑事件，分类数量与转换清单完全一致。原始 CSV 重建的最终前十档与 C++ 引擎最终盘口逐档一致；首尾本地时间边界也一致。最终买一 100603.5 × 0.712、卖一 100603.6 × 3.197，Mid=100603.55，Spread≈0.1 USDT。

## 复现命令

从 README 完成构建与数据准备后，在项目根目录执行：

```sh
mkdir -p workspace/results/m0/BTCUSDT_2025-02-01
/usr/bin/time -p ./workspace/build/release/benchmark_replay config/replay_btcusdt.yaml \
  > workspace/results/m0/BTCUSDT_2025-02-01/benchmark.stdout.log \
  2> workspace/results/m0/BTCUSDT_2025-02-01/benchmark.time.log
.venv/bin/python scripts/validate_replay.py \
  workspace/data/normalized/BTCUSDT_2025-02-01/manifest.json \
  workspace/results/m0/BTCUSDT_2025-02-01/benchmark.json
```

上述命令将程序输出与系统计时分别保存在同一结果目录，使用 `time -p` 记录 real/user/sys。历史基准的 `time -l` 日志保留原始内容。

样本原始下载地址：

- <https://datasets.tardis.dev/v1/binance-futures/incremental_book_L2/2025/02/01/BTCUSDT.csv.gz>
- <https://datasets.tardis.dev/v1/binance-futures/trades/2025/02/01/BTCUSDT.csv.gz>

原始结构化结果见 [benchmark_2025-02-01.json](../../../workspace/results/archive/m0_2025-02-01/benchmark_2025-02-01.json)。

## 未取得的合约元数据

Tardis instruments 返回 HTTP 401；Binance 官方 exchangeInfo 返回 HTTP 451，未取得认证的历史合约规则。价格步长 0.1、数量步长 0.001 仅作为已通过当天所有源行网格检查的研究参数；最小下单量及最小名义金额等未验证。详见 [规则状态](../../../workspace/results/archive/m0_2025-02-01/contract_metadata_status.json)。M0 不下单，该缺口不改变本次行情读取与回放性能的测量范围。
