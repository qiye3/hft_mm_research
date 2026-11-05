# 2026-11-05 — M2-A 高频研究数据集的生成与使用

## 一、数据怎样流到研究表

本阶段将 BTCUSDT 永续合约的逐事件行情转换为固定时间网格上的研究样本。C++ 复用 M0 的原生双时间戳回放，逐条应用增量盘口和逐笔成交，再每 100ms 保存一次基础信息。Python 使用 Polars 构造历史窗口和未来 1s 标签，保存为按日 Parquet。

```text
同源增量盘口和逐笔成交 → 标准化 NPZ
                              ↓
                    C++ 原生回放与固定网格采样
                              ↓
                 基础 Parquet → Polars → 研究 Parquet
                              ↑
                     网盘 25 档快照逐档核对
```

网盘的 25 档快照只用于校验。生产数据的盘口状态由原生引擎读取全深度增量重建。前十档之外的更新仍然保留，深层价位可以在前方价位消失后进入观察范围。

## 二、每条样本表示什么

所有日期按 UTC 划分。每天网格从 00:00:00.000 到 23:59:59.900，共 864,000 行，时间单位为 Unix 纳秒。样本时刻 t 的盘口包含所有本地接收时间不晚于 t 的事件。同一时间戳的记录即使跨分片，也在完整处理之后采样。

两套来源的约 122ms 接收差异按单独诊断处理，正式研究保留同源原始接收时间。 [时间差诊断](../reports/time_alignment_2026-08-01/report.md)

主动成交和 OFI 的 100ms 累计区间为 `(t-100ms, t]`。没有成交的完整有效区间，成交量为零。盘口使用最近已知状态，同时记录最近盘口更新时间，默认超过 1s 未更新即标为过期。交易所时间只作来源记录和诊断，样本的可用性由本地接收时间决定。

基础表保存每侧十档价格和数量、中间价、价差、最后可见双时间戳、事件数、成交笔数、主动买卖量及成交金额。价格单位为 USDT/BTC，挂单量与成交量为 BTC，金额为 USDT。

OFI 使用买卖一的逐价位更新定义。买价不低于旧买价时加上新买量，买价不高于旧买价时减去旧买量；卖价不高于旧卖价时减去新卖量，卖价不低于旧卖价时加上旧卖量。同价更新自然得到数量差，正值表示买方压力。只对正常盘口更新累计，快照与清空事件不记作订单流。这个定义与仅比较相邻 100ms 快照的数量变化存在区别。 [C++ 采样实现](../../src/mm/research_sampler.cpp)

## 三、特征和标签怎样定义

研究表保留基础字段，增加下列稳定的研究输入。

| 字段类别 | 含义 |
| --- | --- |
| `spread_ticks` | 价差除以价格步长 |
| `imbalance_1/5/10` | 对应深度的 `(买量-卖量)/(买量+卖量)` |
| `bid_depth_10/ask_depth_10` | 当前前十档的买侧或卖侧数量之和 |
| `microprice_minus_mid` | 用对侧挂单量加权的买卖一价格减去 Mid |
| `ofi_100ms/1s/5s` | 过去对应区间的 OFI 累计，单位 BTC |
| `buy_volume_1s/5s`、`sell_volume_1s/5s` | 对应区间的主动成交量 |
| `trade_imbalance_100ms/1s` | `(主动买量-主动卖量)/(主动买量+主动卖量)`，无成交时为零 |
| `return_100ms/1s/5s_bps` | 当前 Mid 相对历史同网格 Mid 的简单收益，乘以 10,000 |

默认标签 `target_mid_change_1s` 为 `Mid(t+1s)-Mid(t)`，单位 USDT/BTC。另保存 `target_mid_return_1s_bps` 和 `label_end_timestamp_ns`。取值来自恰好 t+1s 的网格点，窗口必须在同日、同一有效恢复区段内。日末最后十行的标签为空。

主动成交量先按数量步长换算为整数单位，再进行滚动求和，最后恢复为 BTC。这样，在窗口内已经没有成交时，成交量严格为零，避免浮点累计的微小残差影响失衡比率。

特征窗口只向后取值，标签单独向前取值。清单明确列出进入 X 的 19 个特征，未来价格变化、标签结束时间和数据有效标志均不进入 X。 [Python 特征与标签](../../scripts/build_research_dataset.py)

## 四、快照、缺失与预热怎样处理

`book_valid` 表示采样点具有完整十档、正常双边价差、已结束的快照和未过期盘口。`flow_valid` 表示当前完整 100ms 区间可用于流量统计。发生快照替换、盘口失效或超过 1s 的盘口更新间隔时，区段编号 `segment_id` 改变，流量和滚动关系重新开始。

`feature_valid` 要求最近 5s 的流量区间全部有效、区段内已积累 5s 历史，且进入 X 的字段都有限。`label_valid` 要求从当前到未来 1s 的采样点及区间有效，并且没有跨区段。研究读取入口同时过滤这两个标志。原始无效行继续保留，可以检查剔除原因和数据覆盖。

`trade_sequence_gap` 标记成交编号不连续被发现时所属的区间。跳号只能在下一笔成交收到之后发现，因此标记按当前接收时刻进入样本，不反向改写更早的特征。跳号可能来自特殊记录或采集缺失，当前不补造事件。正式研究排除检测区间并重新预热 5s，跨过疑点的标签也无效。基础表保留引擎的原始区段，研究表的 `engine_segment_id` 保存其来源，研究用 `segment_id` 另计这些质量中断。

超过阈值的静默只能提示数据可能过期。CSV 没有盘口更新编号，无法单凭接收时间确认每一条增量均未遗漏。独立快照比对可以检验采样点的盘口，更新编号与采集日志仍是确认源流连续性的补充依据。正式研究不接受转换阶段自动平移过本地时间的日期，需先完成该日期的时间质量诊断。

## 五、目录与复现命令

生产配置位于 `config/research_dataset.yaml`，首批为 2026-07-12 至 07-18。研究数据在 `workspace/data/research/m2_a_v1/`，每个 `date=YYYY-MM-DD/` 分区保存 `base.parquet`、`research.parquet` 和 `complete.json`。总清单 `manifest.json` 列出日期、特征顺序、标签和配置。

日志与校验报告位于 `workspace/results/m2_a/m2_a_v1/date=YYYY-MM-DD/`。临时 CSV 在成功写出 Parquet 后删除。数据和运行结果均由 Git 忽略。

生产入口逐日执行校验，也会复用同一版本已经通过的校验记录。全部日期通过后才发布带有 `validated=true` 的总清单，X、y 读取入口要求这个标志。

先按 README 安装依赖并构建，随后下载同源原始数据及校验快照。`BAIDU_CREDENTIAL_FILE` 指向本机私有授权文件，文件内容和下载签名链接不会写入项目。

```sh
python scripts/download_baidu.py --kind paired --jobs 3 \
  --dates 2026-07-12 2026-07-13 2026-07-14 2026-07-15 2026-07-16 2026-07-17 2026-07-18 \
  --credential-file "$BAIDU_CREDENTIAL_FILE"
python scripts/download_baidu.py --kind snapshot --jobs 2 \
  --dates 2026-07-12 2026-07-13 2026-07-14 2026-07-15 2026-07-16 2026-07-17 2026-07-18 \
  --credential-file "$BAIDU_CREDENTIAL_FILE"
python scripts/build_research_dataset.py config/research_dataset.yaml
```

缺少 NPZ 时，生产入口调用已有转换器生成标准事件。已有相同配置的完成分区直接复用，改变定义后需升级版本目录或显式使用 `--rebuild`。原始档案和已完成 NPZ 按不可变输入管理，替换源数据后需要重新生成对应版本。

首次生产还会读取同日原始成交文件来检查编号连续性，因为 M0 的已有 NPZ 未保存成交编号。行情状态、OFI 和成交累计仍来自 C++ 回放。质量标记写入 Parquet 后，后续因子研究和 X、y 读取只依赖研究数据集。

需要单独复核某天时，可以直接运行校验入口。基础统计读取已经校验的整份数据集。

```sh
python scripts/validate_research_dataset.py \
  workspace/data/research/m2_a_v1/date=2026-07-12 \
  workspace/data/raw/baidu_2026-07-12/snapshot25.csv.zst \
  workspace/results/m2_a/m2_a_v1/date=2026-07-12/validation.json
python scripts/summarize_research_dataset.py workspace/data/research/m2_a_v1 \
  workspace/results/m2_a/m2_a_v1/statistics.json
```

后续 Python 研究可以调用 `scripts/summarize_research_dataset.py` 中的 `load_research_xy`，直接取得 X、y 和日期时间索引。也可以用 Polars 扫描多个 `research.parquet`，读取 `base.parquet` 则能在已有盘口和基础量足够时构造新特征。

新增依赖逐笔路径的特征时，需要重新回放原始事件。当前缓存支持表内字段及其历史组合。未来训练划分需按时间进行，并按标签的结束时刻隔离训练与验证边界。

## 六、如何验收

小规模测试通过真实 NPZ 读取器进入 C++，手算 OFI、主动成交和深档晋升结果，检查同时间戳、跨分片和快照恢复。Python 测试验证滚动窗口、未来 1s 标签、缺口与日尾处理，并比较追加未来样本前后的既有特征。

真实日期上，网盘独立快照按本地时间向后匹配，逐档检查 C++ 前十档。原始成交另行按时间分桶，核对主动买卖量、金额与笔数。相同输入重复生成时比较数据字段和有效标志，运行耗时等资源记录不参与数据一致性比较。

首批七天已完成校验、X 和 y 读取及基础统计。具体数量、盘口中断和保守筛选影响见验收报告。 [七天数据集验收](../reports/m2_a_2026-07-12_2026-07-18/report.md)
