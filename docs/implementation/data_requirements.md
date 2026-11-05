# 2026-11-05 — 研究数据的组织、下载、处理与详细概况

## 一、研究需要哪些数据

研究对象是 Binance USDT 本位 BTCUSDT 永续合约。盘口用来观察挂单和流动性变化，逐笔成交用来计算主动买卖量并参与成交模拟，交易规则与费用决定报价、下单和收益计算的口径。

| 类别 | 需要的内容 | 用途 |
| --- | --- | --- |
| L2 增量盘口 | 日初或可衔接的快照、完整价位更新、方向、价格、数量、交易所时间和接收时间 | 盘口重建、OFI、深度特征、回测 |
| 逐笔成交 | 成交编号、时间、价格、数量、主动买卖方向、接收时间 | 成交冲击、订单流特征、成交模拟 |
| 历史合约规则 | 价格/数量步长、最小下单量、最小名义金额、乘数及有效期 | 合法下单和历史参数核对 |
| 账户费率 | Maker/Taker 费率、优惠/返佣条件及有效期 | 成交费用和净收益 |
| 实际资金费率 | 结算费率、实际结算时刻和相关规则 | 持仓对应的资金收付 |
| 标记价格与指数价格 | 价格、时间和来源 | 合约风险、保证金相关分析 |
| 未平仓量、爆仓记录 | 数量、方向、时间和采集口径 | 市场状态与辅助特征 |
| 自身订单与延迟日志 | 请求发送、交易所处理、回报接收、订单状态及执行数量 | 后续执行校准 |

核心特征可以先从前十档盘口和逐笔成交计算，前五档用于对照实验。源文件保留其提供的全部价位，使深层价位能在前方挂单消失后进入观察范围。模型特征和未来价格标签由程序生成，保存对应定义和数据版本。

## 二、目前网盘里有什么

根目录为 `/STUDY/Binance-BTC-Futures`。下面列出已检查的目录与范围，文件清单和行情内容分别核对。

| 目录 | 已知格式与覆盖 | 当前核对程度 |
| --- | --- | --- |
| `L2增量 2020-2026/BTCUSDT/YYYY/MM/DD/` | 日文件 `BTCUSDT.csv.zst`；2025-10-01 至 2026-10-21 共 386 个日期目录 | 核对了日期目录，抽查文件，并完成下述一个整日验证 |
| `BTCUSDT/YYYY-MM-DD/` | 日文件 `BTCUSDT.csv.zst`，已确认包含 Tardis 风格逐笔成交字段 | 检查了同日样本并完成一个整日验证 |
| `BTC-Trades-23-26/` | 日度 ZIP；2025-10-01 至 2026-07-31 共 304 个匹配日文件，约 9.72 GB | 完整核对文件名和日期清单 |
| `25-orderbook 2020-2026/BTCUSDT/YYYY/MM/DD/` | 日文件 `BTCUSDT.csv.zstd`，目录标注 25 档快照 | 已核对目录和样本文件名；实际表头与采样细节需在使用时检查 |
| `HFT-8.1-9.30/` | 2026-08-01 至 09-30，共 61 天、366 个 Parquet、36.56 GB | 核对文件清单，检查了 2026-08-01 的盘口和成交 |
| `增量订单簿-L2/YYYY/` | 另外一套月度 ZIP，2025 年有 12 个月文件 | 目录核对，内部是否与日目录重复需解包比对 |
| `BTC-OI/`、`爆仓数据/` | 未平仓量年度 ZIP、爆仓数据目录 | 目录核对，使用前确认字段、单位与完整性 |

HFT Parquet 中包含 `orderbook`、`trades`、`mark_price`、`ticker`、`open_interest`、`liquidations` 六类，各 61 个日文件。其中盘口合计约 33.09 GB，成交约 2.96 GB，其余约 0.51 GB。

日期目录存在表示有相应目录项；逐日行情连续性还要通过内容检查确认。2026 年 10 月成交需要继续核查其他目录或补齐来源，已核对的 L2 日期目录也只列到 10 月 21 日。

## 三、两套主要格式的字段与时间

### 原始 CSV.zst/.zstd

同源原始数据采用 Tardis 风格。压缩后仍是 CSV，Zstandard 负责压缩，当前转换器已经支持 `.gz`、`.zst` 和 `.zstd`。

盘口表头如下。

```csv
exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount
```

逐笔成交表头如下。

```csv
exchange,symbol,timestamp,local_timestamp,id,side,price,amount
```

`timestamp` 与 `local_timestamp` 均为 UTC Unix 微秒，转换后乘以 1000，统一成引擎使用的纳秒。`exchange` 为 `binance-futures`，`symbol` 为 `BTCUSDT`。价格单位为 USDT/BTC，数量为 BTC。

2026-08-01 的整日成交匹配确认，原始成交 `timestamp` 对应 Parquet `trade_time`；盘口窗口匹配确认，原始盘口 `timestamp` 对应 Parquet `event_time`。两类消息按各自的交易所字段含义处理，接收时间继续使用原始来源。 [接收时间与字段口径诊断](../reports/time_alignment_2026-08-01/report.md)

盘口 `side=bid/ask`，`amount` 表示更新后该价位应有的总数量，零表示删除价位。`is_snapshot=true` 表示快照中的价位，初始快照和重连快照都要先建立完整的一批状态，再继续处理增量。逐笔成交 `side=buy/sell` 表示主动成交方向，`id` 是成交编号。

当前使用价格步长 0.1 USDT、数量步长 0.001 BTC，在转换时逐行核对数据网格。历史最小下单量、最小名义金额和其他规则还需要对应日期的认证资料。已有网格检查只证明数据与研究步长相容。

### HFT Parquet

2026-08-01 盘口字段包括 `received_time`、`event_time`、`transaction_time`、`symbol`、`event_type`、更新编号、`side`、`price`、`quantity` 等。`event_type` 含 `update` 和 `snapshot`，因此可按价位增量处理。

`received_time` 是纳秒；`event_time`、`transaction_time` 和成交文件中的 `trade_time` 是毫秒。价格和数量在所查文件中保存为字符串，需要按字段含义转换。更新编号有些保存为浮点类型，接入时要检查整数精度，并以可空整数保存后续规范化结果。

成交文件含 `trade_id` 和 `is_buyer_maker`。该布尔值为真表示买方为挂单方，对应主动卖出；为假对应主动买入。日文件本身的快照和状态衔接也需要检查。

### 已确认的质量问题

2026-08-01 的成交 Parquet 有 1,087,218 条记录，其中 5,500 条价格和数量同时为零，占约 0.506%。其中 5,499 条标为 `NA`，1 条标为 `INSURANCE_FUND`。接入时保留原始记录，另行分类并核实处理规则，记录排除数量和原因。

当天盘口 Parquet 开头是增量，已找到的快照出现在 UTC 12:27 左右。独立从午夜开始重建时，需要前一天的可衔接状态，或另一个时间与编号匹配的起始快照。快照同批价位还存在不同接收时间，接入时应按快照批次完整性处理。

与原始 CSV 匹配的 7,556 条样本更新中，原始 CSV 的本地接收时间比 Parquet 的接收时间中位数早约 122 毫秒。该差异影响 100 毫秒级研究，应按来源成套使用并记录时间口径。本次完整验证采用原始同源 CSV 盘口和成交。 [抽样检查记录](../../workspace/results/data_inspection/baidu_2026-08-01/checks.json)

## 四、本地和云端怎样组织数据

```text
workspace/
├── data/
│   ├── raw/baidu_2026-07-12/
│   │   ├── book.csv.zst
│   │   └── trades.csv.zst
│   ├── normalized/BTCUSDT_2026-07-12/
│   │   ├── manifest.json
│   │   └── events-00000.npz …
│   └── samples/baidu_2026-08-01/
│       ├── orderbook_first_10000.parquet
│       ├── orderbook_snapshot_rows.parquet
│       ├── raw_l2_first_10000.csv
│       └── BTCUSDT-trades-2026-08-01.parquet
├── results/
│   ├── validation/baidu_2026-07-12/
│   ├── data_inspection/
│   └── research_data/            后续特征、标签等研究表
└── build/                       编译产物和可清理的临时缓存
```

原始文件按来源和日期隔离，下载时避免两个同名 `BTCUSDT.csv.zst` 相互覆盖。来源清单记录网盘原路径、文件大小、下载后位置和耗时。标准事件清单记录格式、时间单位、步长、分类数量、分片和最终盘口。

后续研究表可按 `BTCUSDT/feature_version/date=YYYY-MM-DD/` 分区，标签同时记录预测周期、价格定义和生成版本。模型输出、预测与交易记录用独立实验编号归档，关联数据版本、参数和 Git 提交。数据和结果保存在当前被忽略的目录中，`docs` 保存说明与报告。

云端第一批只预取 7—30 天。标准化和特征生成完成后复用结果，历史原始档案继续保留来源。云端过期缓存清理前确认可从原始档案重建，并保留必要的清单与实验记录。

## 五、下载方式

### 从百度网盘取得原始文件

以本次日期为例，分别取得以下两个原始文件。

```text
/STUDY/Binance-BTC-Futures/L2增量 2020-2026/BTCUSDT/2026/07/12/BTCUSDT.csv.zst
/STUDY/Binance-BTC-Futures/BTCUSDT/2026-07-12/BTCUSDT.csv.zst
```

可以使用百度网盘客户端下载，分别保存为本地 `book.csv.zst` 和 `trades.csv.zst`。本次使用授权文件 API 获取下载信息，按区间续传到 `.part`，长度与网盘元数据一致后改成正式文件名。下载来源记录见下方清单。 [本次来源清单](../../workspace/results/validation/baidu_2026-07-12/source_files.json)

已配置的百度网盘 MCP 用于目录、文件信息和检索。实际下载需要有效账号授权，凭证单独保存在本机私有文件。需要批量下载时，按日期清单预取并保存失败状态，恢复后从已确认的字节位置续传。

### 传入云环境

云服务器和容器可按指南使用文件传输工具或 SSH/SFTP。容器也可以在确认网盘访问和下载吞吐后直接拉取数据。集群的计算节点没有外网，应先将数据和依赖送到允许的存储目录，再提交计算作业。云端开机下载的等待时间也属于运行时长，下载与实验分批安排更容易控制费用。 [服务器配置与费用](server_recommendation.md)

### 补齐其他历史资料

Tardis 可提供连续日期的 `incremental_book_L2`、`trades` 和 `derivative_ticker`，每月 1 日有免费历史 CSV 样本。元数据接口提供部分历史规则变更，需要核对字段完整性。 [行情下载说明](https://docs.tardis.dev/downloadable-csv-files) · [元数据说明](https://docs.tardis.dev/api/instruments-metadata-api)

实际资金费率、分钟标记价格和指数价格可从 Binance 官方归档按月获取；已确认 BTCUSDT 2025 年对应月文件存在，研究 2026 年时按所选月份核对。账户费用结合对应账户条件和历史记录，未来订单延迟与执行数据由自己的程序采集。 [资金费率归档](https://data.binance.vision/?prefix=data/futures/um/monthly/fundingRate/BTCUSDT/) · [标记价格归档](https://data.binance.vision/?prefix=data/futures/um/monthly/markPriceKlines/BTCUSDT/1m/) · [指数价格归档](https://data.binance.vision/?prefix=data/futures/um/monthly/indexPriceKlines/BTCUSDT/1m/)

## 六、处理流程与可复现命令

先按项目 README 准备环境。以下命令在项目根目录执行，转换输出目录应为空，策略实验输出目录应为新的目录。 [环境准备](../../README.md#quickstart)

```sh
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/convert_tardis.py \
  --book workspace/data/raw/baidu_2026-07-12/book.csv.zst \
  --trades workspace/data/raw/baidu_2026-07-12/trades.csv.zst \
  --date 2026-07-12 --tick-size 0.1 --lot-size 0.001 \
  --output-dir workspace/data/normalized/BTCUSDT_2026-07-12 \
  --shard-rows 250000
```

转换器流式读取压缩 CSV，检查字段、品种、时间顺序、价格数量网格和初始快照。它保留完整源深度，将时间统一为纳秒，再分别整理交易所侧和本地侧记录。同一事件在两侧各生成一次记录，保证各侧按自己的时钟处理。分片中的数组成员名为 `data.npy`，每条记录与框架的 64 字节事件结构一致。

随后运行 C++ 回放和独立核对。

```sh
./workspace/build/release/benchmark_replay config/replay_baidu.yaml
.venv/bin/python scripts/validate_replay.py \
  workspace/data/normalized/BTCUSDT_2026-07-12/manifest.json \
  workspace/results/validation/baidu_2026-07-12/replay.json
./workspace/build/release/baseline_mm config/baseline_mm_baidu.yaml
.venv/bin/python workspace/results/validation/baidu_2026-07-12/audit_baseline.py
```

`validate_replay.py` 核对两侧计数、时间边界、分片消费和最终盘口。策略使用现有委托与成交接口，导出 `orders.csv`、`fills.csv`、`equity.csv`、`summary.json`，独立核对脚本重算成交、现金流、费用、仓位和权益。

更换日期时同步修改数据路径、配置日期和结果目录，并重新计算 `quote_stop_timestamp_ns`，它对应当日 UTC 23:59:59。`config/replay_baidu.yaml` 和 `config/baseline_mm_baidu.yaml` 保存本次可复现参数。 [回放配置](../../config/replay_baidu.yaml) · [策略配置](../../config/baseline_mm_baidu.yaml)

## 七、本次完整验证的详细概况

选择 2026-07-12 的配套原始文件完成整日验证。原始盘口 468.64 MB，逐笔成交 10.56 MB，合计 479.20 MB。该日期是完整的小规模验证日，容量和运行时间随市场活跃度变化。

| 项目 | 数量 |
| --- | ---: |
| 盘口增量 | 90,057,673 |
| 快照价位 | 4,976 |
| 清空事件 | 2 |
| 逐笔成交 | 1,548,707 |
| 单侧逻辑事件 | 91,611,358 |
| 双侧标准记录 | 183,222,716 |
| NPZ 分片 | 733 |

交易所侧和本地侧各消费 91,611,358 条逻辑事件。最终前十档与原始 CSV 转换阶段保存的独立盘口对照逐档一致，首尾接收时间和分片消费也一致。本地时间统一偏移为 0 纳秒。转换清单中的完整日标志对应每份源文件进入当日首尾一分钟的检查，中段连续性还应结合更新编号、采集记录和其他流继续检查。

M1 基线成功产生订单和成交，最终净收益为 -153.115508 USDT、费用 106.253608 USDT，最终仓位 0.004 BTC，成交回报数 8,310。本次沿用已有固定报价参数和假设费率，结果用于检查交易和账本链路。 [回放结果](../../workspace/results/validation/baidu_2026-07-12/replay.json) · [独立账本结果](../../workspace/results/validation/baidu_2026-07-12/ledger_audit.json)

## 八、批量研究前的质量和样本安排

每个日期保留文件是否齐全、首尾时间、事件数、快照与重连、异常区间、清洗方式和版本记录。成交编号用于重复检查，盘口更新编号用于增量衔接检查，遇到重建中断时使用完整快照恢复，再经过特征预热。统一规则后按日执行，失败日期单独重跑。

先用 7 天检查批处理，再用 30—90 天建立 Alpha 特征和预测基线，之后在连续的有效覆盖中按时间划分开发、验证和最终测试。已有检查显示原始成交 ZIP 覆盖至 2026-07-31，HFT Parquet 的 8—9 月需要单独解决快照与时间口径，再决定如何进入同一研究链路。

当前归档保留原始压缩文件；研究特征、标签和预测优先使用 Parquet + ZSTD；引擎的 NPZ 分片按运行日期缓存。全部采用 Parquet 的接口改造与压缩实测见单独说明。 [GPU 与 Parquet 选择](gpu_and_parquet.md)

M2-A 的固定 100ms 数据生产、特征与未来 1s 标签定义、七天数据准备和 Python 读取方法见专门指南。 [M2-A 研究数据集](m2_a_dataset.md)
