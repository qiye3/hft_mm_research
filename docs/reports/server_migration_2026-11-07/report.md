# 服务器迁移与531天研究数据构建

本报告保留100ms迁移阶段的原始结果与命令。当前1s因子和跨日标签的入口见[研究包指南](../../implementation/research_package.md)。
2026-11-07完成。服务器项目位于 `/root/shared-nvme/hft_mm_research`，数据和环境保存在持久化共享存储中。全部大文件由服务器直接下载，下载进程已清除代理变量。

## 数据范围与来源

本次覆盖2025-05-09至2026-10-21，共531个连续交易日。

| 日期 | 盘口来源 | 标准盘口保留深度 |
| --- | --- | --- |
| 2025-05-09至2025-09-30，145天 | Hugging Face Crypto Lake，经hf-mirror.com下载 | 20档 |
| 2025-10-01至2026-10-21，386天 | 百度网盘25-orderbook目录 | 25档 |

Features统一观察前10档。成交内容全部采用Binance官方普通成交ZIP，下载渠道包括官方直连与已核对官方CHECKSUM的网盘镜像。

本轮176份新镜像核对一致。2025-08-24及2025-09-01至09-14的15份镜像字节校验不一致，已排除并使用官方直连档案。运行计划显式记录来源，已核对缓存直接复用。

网盘盘口共365份Zstandard、21份真实XZ，其中10份Zstandard误用了xz后缀。读取时按文件头识别已核实的编码。标准盘口保留完整来源深度，读取Features时逐文件选取所需列，再跨日连接。

## 生成结果

| 产物 | 文件数与行数 | 文件总大小 |
| --- | --- | --- |
| 原始行情 | 531天盘口和成交原档及必要来源记录 | 47.318 GiB |
| 标准盘口 | 531份，781,384,099行 | 与成交合计60.189 GiB |
| 标准成交 | 531份，1,906,913,937行 | 计入标准数据合计 |
| Features | 73份，每份458,784,000行 | 38.469 GiB |
| 标签 | 四周期各531份，共2,124份 | 8.329 GiB |

行情、Features与标签合计约154.3 GiB，另需环境、构建目录和临时空间。73份Features包含57个基础变量与16个衍生变量，每份按时间保存全部日期。

相对服务器项目根目录的产物位置如下。

```text
workspace/data/
├── raw/
├── standard/server_market_v1/
│   ├── snapshots/YYYY-MM-DD.parquet
│   └── trades/YYYY-MM-DD.parquet
├── features/server_531d_v1/变量名.parquet
└── labels/server_531d_mid_v1/
    ├── horizon=100ms/YYYY-MM-DD.parquet
    ├── horizon=500ms/YYYY-MM-DD.parquet
    ├── horizon=1s/YYYY-MM-DD.parquet
    └── horizon=5s/YYYY-MM-DD.parquet
```

## 时间口径与质量

研究可见时间统一采用交易所时间加20ms固定延迟，100ms采样只接收已可见状态。盘口超过1秒未更新时判为过期。无效研究值置空，滚动窗口按min_samples处理，跨日继续计算。标签单位包含bps、ticks和USDT。

全区间盘口有效样本457,302,056行，占99.677%。成交窗口有效样本458,783,998行。1秒有效标签456,345,454行，占99.468%。这些比例保留来源异常和标签规则的实际影响。

2026-07-28盘口有效样本664,368行，占当日76.894%，为本轮最低。后续研究读取质量列和null处理该日，日期文件齐全仍须结合日内质量使用。逐日质量明细保存在验收JSON。

标签沿用同一UTC日内的连续盘口规则，日末无法取得未来目标的样本置空。四周期分别为100ms、500ms、1s和5s。

## 验证结果

已核对全部531天标准文件、全部单变量行数与四周期标签的日期覆盖。2025-05-09、2025-09-30、2025-10-01、2026-10-21分别进行了独立盘口采样与全天成交分桶对照，覆盖首日、来源切换边界和末日。

上述四天的标签使用NumPy独立核对有效性、未来目标时刻以及三种单位的数值。正式Python接口成功读取2025-05-09的三个特征与1秒标签，得到863,688行X/y。

C++ Release构建及合成行情入口已在服务器运行。迁移修复的相关回归涵盖流式多列读取、真实XZ与误后缀Zstandard、20档切换25档的午夜可见状态。

## 实际资源与耗时

实例配额为10 vCPU、60 GiB内存，GPU套餐包含RTX3090。本轮使用CPU计算。CPU型号为AMD EPYC 7402，系统Ubuntu24.04.1，Python3.13.18、Polars2.1.0、PyArrow26.0.1。C++使用GCC13.3、C++20、Release `-O3 -DNDEBUG`。

| 完整阶段 | 墙钟耗时 | 进程Peak RSS |
| --- | --- | --- |
| 531天Features | 1小时30分24秒 | 24.620 GiB |
| 四周期标签 | 16分00.54秒 | 0.528 GiB |
| 最终覆盖与质量验收 | 41.44秒 | 1.170 GiB |

Features平均约每秒84,584个100ms样本，此值包含基础采样及逐变量衍生计算。C++事件回放吞吐需单独测量。数据下载和标准化曾因连接超时恢复，相关日志包含缓存复用，未将最后一次重启耗时当作531天首次生产总耗时。

以上三个阶段合计约1.785小时，按本次控制台1.30元/小时计算，计算资源约2.32元，另计下载准备、失败重试、空闲开机和存储。若连续开机7天，计算资源约218.40元。共享存储采用此前指南示例0.00048元/GB/小时，按1000GB保留7天约80.64元，实际以账户存储计费口径为准。

数据和持久化环境位于共享存储，停机后可保留，存储继续计费。当前实例保持运行。

## 复现与运行记录

本次使用运行目录 `workspace/results/server_migration/` 中的显式日期配置。四份Data配置分别为main_data、main_xz、main_zstd和early_data，后者用于早期HF盘口。它们统一输出server_market_v1。features_531d.yaml直接读取标准版本，输出Features和四周期标签。

已有标准数据时，以下命令可复用缓存执行后续链路。

```bash
cd /root/shared-nvme/hft_mm_research
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*'
export POLARS_MAX_THREADS=10 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_root=workspace/results/server_migration
.venv/bin/python scripts/build_features_targets.py "$run_root/features_531d.yaml" --only features
.venv/bin/python scripts/build_features_targets.py "$run_root/features_531d.yaml" --only labels
.venv/bin/python "$run_root/audit_production.py"
```

运行目录还保留下载计划、mirror_verification.jsonl、raw_encoding.json、四天独立对照、final_audit.json及runtime.json。批次专用下载与验收脚本和配置保存在该运行目录，随workspace排除Git；需要重建本批次时同时保留它们。普通生产入口和修复代码在仓库中。

服务器Git基线为65d2e524c7da9cdb2249c2f92b92f84172e7c515，引擎Submodule固定为e0d2f7887c68755b220d9e69425d61d27e5b6432。本次迁移修复保留为未提交修改，等待用户决定提交或回滚。

## 本次文件修改

- `research/mm_research/data/sources.py`增加已核实XZ编码的读取，以及跨来源逐文件选列。
- `research/mm_research/data/storage.py`使用有序流式连接读取多个全日期特征，减少关联阶段内存。
- `research/mm_research/features/snapshot.py`明确只读取前十档及必要时间质量列。
- `research/mm_research/features/validation.py`独立参考按文件选列，支持20档与25档交界验证。
- `research/mm_research/features/base.py`修正文档字符串中的单变量存储说明。
- `scripts/download_baidu.py`增加显式压缩后缀参数，准确定位网盘来源文件。
- `tests/test_standard_data.py`、`tests/test_storage.py`、`tests/test_research_versions.py`覆盖对应格式、流式连接与跨来源边界，并补齐样本配置。
- `AGENTS.md`、`docs/implementation/data_requirements.md`、`docs/implementation/server_recommendation.md`更新来源、下载网络要求、实例实测与报告入口。
- `docs/design/hft_alpha_annotations_ui_v5/`中的批注JSON和HTML内嵌批注同步修复及完成状态。
- 新增本报告。运行数据和批次记录保留在workspace，既有其他目录保持原状。
