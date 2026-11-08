# 研究数据的组织、下载、处理与详细概况

更新日期为2026-11-07。Hugging Face 盘口来源与使用方式见第五节。

## 当前生产入口

当前Data输入为Binance官方普通成交ZIP与网盘25档快照。标准数据保存到`workspace/data/standard/market_v2/{snapshots,trades}/YYYY-MM-DD.parquet`。

本次服务器生产区间为2025-05-09—2026-10-21，共531天，标准版本为`server_market_v1`。2025-05-09—09-30使用Hugging Face的20档快照，2025-10-01—2026-10-21使用网盘25档快照，标准表保留来源实际档数，Features观察前10档。成交内容统一采用Binance官方普通成交档案，下载渠道分为官方直连与网盘中的同名ZIP镜像。每份新镜像先与Binance官方`.zip.CHECKSUM`核对，完全一致后才发布到`raw/binance_public/date=日期/`供标准化读取。同一天只使用一份档案。逐份结果保存在服务器运行目录`workspace/results/server_migration/mirror_verification.jsonl`。

531天标准数据、73份全日期Features和四周期共2,124份标签已生成并通过覆盖与独立计算验收。实际数据量、有效率、运行命令与资源记录见[服务器迁移报告](../reports/server_migration_2026-11-07/report.md)。

网盘同名ZIP目录覆盖本区间449天，其中2025-08-24和2025-09-01—09-14的15份档案未通过官方一致性核对，已明确排除。这15天及2026-08-01—10-21的82天采用官方直连，选择清单为运行目录中的`trade_delivery_plan.json`与`trade_delivery_exclusions.json`。校验失败的暂存文件不会进入标准化输入。

Features使用交易所时间加20ms作为模拟可见时刻。base、book、trades、returns四组按特征列保存到`features/market_columns_v1/特征名.parquet`，每份文件包含全部日期。标签保存到`labels/market_mid_columns_v1/horizon=周期/YYYY-MM-DD.parquet`。版本与来源定义内嵌Parquet元数据。

下文网盘内容与整日C++验证保留对应检查日期，当前可执行生产命令见第六节。

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
| `25-orderbook 2020-2026/BTCUSDT/YYYY/MM/DD/` | 25档快照，实际包含 `.csv.zst`、`.csv.zstd`、`.csv.xz` | 服务器下载已核实386天的文件名，按明确压缩格式下载；重叠七天曾与C++核对 |
| `HFT-8.1-9.30/` | 2026-08-01 至 09-30，共 61 天、366 个 Parquet、36.56 GB | 核对文件清单，检查了 2026-08-01 的盘口和成交 |
| `增量订单簿-L2/YYYY/` | 另外一套月度 ZIP，2025 年有 12 个月文件 | 目录核对，内部是否与日目录重复需解包比对 |
| `BTC-OI/`、`爆仓数据/` | 未平仓量年度 ZIP、爆仓数据目录 | 目录核对，使用前确认字段、单位与完整性 |

HFT Parquet 中包含 `orderbook`、`trades`、`mark_price`、`ticker`、`open_interest`、`liquidations` 六类，各 61 个日文件。其中盘口合计约 33.09 GB，成交约 2.96 GB，其余约 0.51 GB。

日期目录存在表示有相应目录项，逐日行情连续性还要通过内容检查确认。2026-11-06 完整目录核对确认，标准化成交 CSV 在目标区间内有 307 个日期，截止 2026-08-03。快照与 L2 各有 386 个日期，截止 2026-10-21。

2026-11-07服务器下载并核实全部386份快照，文件后缀分为341份`.zst`、31份`.xz`、14份`.zstd`。其中2026-08-15—09-04的21份文件是真正XZ，2026-10-12—10-21的10份文件虽以`.xz`结尾，实际编码为Zstandard。其余355份文件也均为Zstandard，核验汇总保存在服务器运行目录的`raw_encoding.json`。

下载入口通过`--source-extension zst|zstd|xz`明确选择目录中的文件名，标准化配置的`snapshots`填写实际后缀。Tardis CSV共用`tardis_csv/25`字段定义。标准化读取六字节文件头，遇到确定的XZ标识时使用标准库LZMA解压并在内存读取单日，其余已核实的Zstandard由Polars原生流式读取。两条路径均保留源行序和全部25档。

Binance 官方 2025-10 至 2026-10 的全部十三个月成交档案已确认可下载。七月三十天已逐条验收，网盘 ZIP 能补足标准化 CSV 缺少的全部普通成交，但 ZIP 没有本地接收时间。七月快照也发现了日内长缺口，成交完整性、接收时钟与盘口覆盖需分别管理。[全区间覆盖与处理调整](../reports/data_coverage_202510_202610/report.md)

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

当前研究目录按以下结构保存。

```text
workspace/data/
├── raw/
├── standard/market_v2/
│   ├── snapshots/YYYY-MM-DD.parquet
│   └── trades/YYYY-MM-DD.parquet
├── features/market_columns_v1/特征名.parquet
├── labels/market_mid_columns_v1/horizon=1s/YYYY-MM-DD.parquet
└── normalized/    C++事件回放的NPZ缓存
```

下面展示此前单日回放验证的原始文件和结果组织。

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

标准行情按版本和日期文件组织，特征按变量保存全日期文件，标签按预测周期和日期保存，记录价格定义和生成版本。模型输出、预测与交易记录用独立实验编号归档，关联数据版本、参数和 Git 提交。数据和结果保存在当前被忽略的目录中，`docs` 保存说明与报告。

云端第一批只预取 7—30 天。标准化和特征生成完成后复用结果，历史原始档案继续保留来源。云端过期缓存清理前确认可从原始档案重建，并保留必要的清单与实验记录。

## 五、下载方式

### 当前成交来源

普通成交直接使用Binance公开日档，地址为`https://data.binance.vision/data/futures/um/daily/trades/BTCUSDT/BTCUSDT-trades-YYYY-MM-DD.zip`。原始文件保留ZIP压缩格式，Data读取其中的CSV并统一字段。

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

### Hugging Face 的 BTCUSDT 永续20档快照

#### 来源与实际覆盖

公开数据集 `delmiron27/cryptolake-binance-futures-btc` 提供从 Crypto Lake 采集后上传的 Binance USDT 本位 BTC 永续行情。盘口流名为 `book`，每条记录包含双侧20档完整快照，文件按日期保存为 Snappy 压缩 Parquet，可直接下载和使用 Polars 读取。该来源适用于前10档盘口特征研究。[数据集与文件目录](https://huggingface.co/datasets/delmiron27/cryptolake-binance-futures-btc)

2026-11-07 检查了实际盘口文件目录，结果如下。这里的日期范围以 `raw/book/` 为准。

| 项目 | 核查结果 |
| --- | --- |
| 合约目录 | `exchange=BINANCE_FUTURES/symbol=BTC-USDT-PERP/` |
| 实际日期范围 | 2025-05-09—2026-06-02 |
| 文件数量 | 379个日期，每日一份 `1.snappy.parquet` |
| 总文件大小 | 32,507,238,204字节，约32.51GB |
| 缺少的日期 | 2026-02-05，以及2026年5月8、11、12、15、20、21、22、27、28、29日 |
| 与目标研究区间的关系 | 包含2025年10月至2026年6月2日的部分日期，2026年7—10月没有盘口文件 |
| 许可 | 数据集标注 `other`，批量采用或再分发前需向上传者确认具体授权 |

日期目录只说明有文件。盘中连续性目前只检查了下一节的单日样本，其余日期使用前还需检查。

#### 已下载样本及质量

本地样本为 `workspace/data/raw/huggingface/cryptolake_btc/book_2025-05-09.parquet`，大小110,820,030字节，共804,364条快照。

| 检查项 | 结果 |
| --- | --- |
| 盘口价量与档位顺序 | 双侧20档价量有效，档位顺序正确，没有买卖盘口交叉或锁定 |
| 交易所时间范围 | UTC 2025-05-08 23:59:59.969 至2025-05-09 23:59:59.868，显示精度为毫秒 |
| 相邻快照间隔 | 原顺序下中位数约102ms，p99约206ms |
| 时间倒序 | 交易所时间和接收时间各35处，交易所时间最大回退约103ms |
| 排序后的采样间断 | 17处间隔超过1秒，最长约2.853秒 |
| 源质量声明 | Parquet元数据标注 `contains_gaps=Yes` |

首条记录的交易所时间落在前一天。标准表保留边界记录及源文件日期，Features按实际交易所时间读取，前日状态也可参与日初盘口采样。盘口序号用于来源诊断，快照间序号增加本身不能证明丢失记录。

#### 下载和直接读取

下面的命令在项目根目录执行，下载已经检查过的单日样本。其余日期先在数据集文件目录确认存在，再替换URL中的 `dt%3D2025-05-09` 和本地文件日期。

```sh
mkdir -p workspace/data/raw/huggingface/cryptolake_btc
curl --fail --location --retry 2 \
  'https://huggingface.co/datasets/delmiron27/cryptolake-binance-futures-btc/resolve/main/raw/book/exchange%3DBINANCE_FUTURES/symbol%3DBTC-USDT-PERP/dt%3D2025-05-09/1.snappy.parquet' \
  --output workspace/data/raw/huggingface/cryptolake_btc/book_2025-05-09.parquet
```

无需解压。使用项目Python环境可以直接读取，下面仅查看买一卖一及时间字段。

```python
import polars as pl

snapshots = pl.scan_parquet(
    "workspace/data/raw/huggingface/cryptolake_btc/book_2025-05-09.parquet"
)
preview = snapshots.select(
    "timestamp", "receipt_timestamp", "sequence_number",
    "bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size",
).head(5).collect()
print(preview)
```

文件元数据包含合约、交易所和日期。样本的 `timestamp`、`receipt_timestamp` 均为UTC Unix纳秒，分别表示交易所时间和采集接收时间。`bid_0`、`ask_0` 是第一档，最大索引19对应第二十档，价格单位为USDT/BTC，数量单位为BTC。

#### 接入标准上游的方式

Hugging Face快照已经接入现有Data生产入口。配置 `config/data/standardize_huggingface_test.yaml` 属于Data阶段，输入是已下载的HF原始Parquet和同日官方成交ZIP，输出是 `workspace/data/standard/cryptolake20_test_v1/{snapshots,trades}/YYYY-MM-DD.parquet`。该测试配置包含2025-05-09与2025-10-01。七月主版本 `market_v2` 继续使用网盘25档CSV。

```sh
.venv/bin/python scripts/build_standard_data.py config/data/standardize_huggingface_test.yaml
```

实际处理顺序如下。

1. `research/mm_research/data/sources.py` 中的 `scan_snapshot_source()` 根据配置读取HF快照。`snapshot_format: cryptolake_parquet` 配合 `snapshot_depth: 20`，网盘CSV使用 `tardis_csv` 与25。HF读取分支核对源元数据中的交易所与合约，统一为 `binance-futures` 和 `BTCUSDT`，保留来源标识及原始行号。
2. 映射字段。`timestamp` 对应 `exchange_timestamp_ns`，无需乘倍数。`receipt_timestamp` 仅用于采集诊断。`bid_0_price`、`bid_0_size` 对应 `bid_price_1`、`bid_quantity_1`，卖侧同理，依次映射全部20档。
3. `data/orderbook.py` 按声明的实际深度检查全部价量与档位顺序。20档标准表只保存来源实际提供的20档，两种来源使用独立的标准数据版本。
4. 对采集接收时间倒序和盘口异常附加逐记录质量标签。标准表保留原始行序和事件时间，研究读取按交易所时间与原始行号稳定排序。采样间断沿用现有盘口时效规则，异常研究值置空，滚动计算通过Polars的 `min_samples` 处理。
5. `data/standard.py` 通过同一发布流程生成两份标准表。元数据声明 `snapshot_depth: 20`，同版本的成交表继续来自Binance官方普通成交。Features读取前10档，沿用交易所时间加20ms的可见时刻规则。

这条路径直接使用源快照，不需要先重建盘口。20档快照可支持当前前10档盘口研究；逐事件挂撤单分析与C++执行仿真需要另外取得相应的完整增量盘口流。

#### 适配结果与跨来源补充试验

2026-11-07完成两个真实交易日的转换，源快照全部80列价量和交易所时间与标准表逐行一致。标准表经过现有100ms基础采样与1s标签计算，并通过独立数组参考计算核对盘口与成交分桶。

| 日期 | 标准快照行数 | 官方普通成交行数 | 原始来源下无效100ms盘口样本 |
| --- | ---: | ---: | ---: |
| 2025-05-09 | 804,364 | 3,180,781 | 76 |
| 2025-10-01 | 784,167 | 3,495,797 | 6,416 |

跨来源核验选择2025-10-01，Tardis每月1日的25档快照可免费下载。下载地址为 `https://datasets.tardis.dev/v1/binance-futures/book_snapshot_25/2025/10/01/BTCUSDT.csv.gz`，本地保存到 `workspace/data/raw/tardis_reference/book25_2025-10-01.csv.gz`。共有1,496,667条快照。[Tardis免费样本说明](https://docs.tardis.dev/downloadable-csv-files/overview)

两种快照有10,648个可直接匹配的唯一事件毫秒时刻，双侧20档价量全部一致，比较的绝对误差门限为1e-9。HF样本纳秒整数相对毫秒网格存在最多192ns偏差，核验键取最近毫秒，转换产物仍保存原始时间。只有这些共同事件时刻做了直接价量对照。

补充试验显式选择HF交易所时间中大于1秒的537个内部间隔，以及日首尾未覆盖区间，将其中真实存在的Tardis快照加入独立试验表。每条新增记录保留Tardis来源与源行号，保留原来源质量标签；参考快照采用与HF相同的前20档。

| 项目 | 实测结果 |
| --- | --- |
| HF最长内部间隔 | UTC 01:10:11.140—01:11:44.756，93.616秒 |
| 该区间内Tardis快照 | 1,661条 |
| HF日尾未覆盖 | 最后时刻为23:59:34.639，距次日零点约25.361秒 |
| 补入的参考快照 | 共19,590条，包含日尾470条 |
| 补充后快照总量 | 803,757条 |
| 无效100ms盘口样本 | 从6,416降至0，时效门限为1秒、固定可见延迟20ms |

无效网格降至0表示本次100ms盘口采样均满足现有有效性规则。补充后仍保留原来源间断标签，未来标签继续遵守现有连续性规则。试验产物位于 `workspace/results/data_inspection/huggingface_adaptation/cross_source_trial/snapshots/2025-10-01.parquet`，跨来源合并尚未加入常规Data生产配置。

另外实测了CryptoHFTData的同日00时与01时免费L2文件。两小时分别有3,791,517条和3,898,443条价位更新，所查小时内部更新编号均连续。01时在上述93.616秒间隔中有1,808批更新，但两份文件均无初始化快照。重建需要另取完整起始状态，HF的20档快照不足以初始化完整增量盘口。下载地址使用 `https://api.cryptohftdata.com/download?file=binance_futures/2025-10-01/01/BTCUSDT_orderbook.parquet`。本次响应内容外层还有Zstandard压缩，解压后才是Parquet。[CryptoHFTData格式和获取方式](https://www.cryptohftdata.com/datasets/binance-orderbook-data)

Tardis免费样本覆盖每月1日，其他日期需要相应权限。CryptoHFTData可继续调查完整起始快照及更多小时文件。其他镜像需核对是否与已有来源同源，盘口缺失区间应依靠另一个真实盘口来源补充。

核验统计保存到 `workspace/results/data_inspection/huggingface_adaptation/conversion.json` 与 `cross_sources.json`。跨来源试验可在原始文件准备完成后复现。

```sh
.venv/bin/python workspace/results/data_inspection/huggingface_adaptation/check_cross_sources.py
```

### 补齐其他历史资料

Tardis 可提供连续日期的 `incremental_book_L2`、`trades` 和 `derivative_ticker`，每月 1 日有免费历史 CSV 样本。元数据接口提供部分历史规则变更，需要核对字段完整性。 [行情下载说明](https://docs.tardis.dev/downloadable-csv-files) · [元数据说明](https://docs.tardis.dev/api/instruments-metadata-api)

实际资金费率、分钟标记价格和指数价格可从 Binance 官方归档按月获取；已确认 BTCUSDT 2025 年对应月文件存在，研究 2026 年时按所选月份核对。账户费用结合对应账户条件和历史记录，未来订单延迟与执行数据由自己的程序采集。 [资金费率归档](https://data.binance.vision/?prefix=data/futures/um/monthly/fundingRate/BTCUSDT/) · [标记价格归档](https://data.binance.vision/?prefix=data/futures/um/monthly/markPriceKlines/BTCUSDT/1m/) · [指数价格归档](https://data.binance.vision/?prefix=data/futures/um/monthly/indexPriceKlines/BTCUSDT/1m/)

## 六、处理流程与可复现命令

在项目根目录执行Data与Features两个阶段。

```sh
.venv/bin/python scripts/build_standard_data.py config/data/standardize_30d.yaml
.venv/bin/python scripts/build_features.py config/features/build_30d.yaml
.venv/bin/python scripts/validate_research_dataset.py \
  config/features/build_30d.yaml 2026-07-14 \
  workspace/results/standard_data/market_v2/validation_2026-07-14.json
```

Data配置引用本地已下载原件。当前官方成交日档位于`workspace/data/raw/binance_public/date=YYYY-MM-DD/BTCUSDT-trades-YYYY-MM-DD.zip`，快照位于`workspace/data/raw/baidu_YYYY-MM-DD/snapshot25.csv.zst`。

标准行情保留25档和逐笔粒度。Features的base组执行100ms采样，再生成衍生组与独立标签。新增组复用已有base列，研究读取按所选组有效标记与标签关联。

C++事件回放使用L2增量与对应逐笔成交，由`scripts/convert_tardis.py`转换成NPZ分片，再交给原生引擎。M0与M1可执行命令见项目README，七月十二日旧验证的来源、实测数据与结果按第七节记录阅读。复现实验时需按所选日期设置输入清单、回放及策略参数和输出位置。

当前分组生产、元数据和X/y读取见 [研究包指南](research_package.md)。

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

当前 Polars 研究包复用基础数据，特征和多周期标签独立保存、分别管理版本，主目标单位为 bps。30 天数据的使用方法见新入口。 [研究包指南](research_package.md)

## 当前整秒研究产物

Features保持100ms，300个非OFI因子和五周期标签每秒输出，标签由独立Targets入口生成。窗口、跨日质量、版本、配置及执行顺序见[研究包指南](research_package.md)。
