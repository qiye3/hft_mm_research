# 2026-11-05 — 30 天快照接入与 Polars 研究包迁移

> 版本说明。本文记录三十天同源CSV快照研究与迁移的历史版本，其中18,329,997条可用样本对应旧来源及质量规则。当前使用官方普通成交和分组特征，日期命名的Parquet路径、组有效性及运行方式见[研究包指南](../../implementation/research_package.md)。


本次先提交已有 M2-A 与带批注设计，基线提交为 `dcc8256`。随后下载 2026-07-01 至 07-30 的 25 档快照和同源逐笔成交，共 60 份原始文件、2,414,859,260 字节，约 2.415 GB。已有七天原始文件直接复用。

研究包位于 `research/mm_research/`，采用数据、质量、特征与标签的职责划分。探索性计算使用 Polars，命令脚本保留原调用方式。新增包安装配置、独立版本生产配置和相关回归验证。C++ 继续提供事件状态与已有事件基础表，模型确定后再实现入选实时特征。

原七天组合表的全部字段与迁移后计算结果逐列、逐值一致。独立存储的十九个事件特征与原表一致，1s bps 和 USDT 标签也与原标签一致。原始转换工具的 gzip/Zstandard、NPZ 及双时钟回放检查继续通过。

30 天快照入口共生成 25,920,000 个 100ms 网格样本。当前快照特征版本有 16 个字段，精确事件 OFI 继续显式缺失。以 1s bps 为目标，在质量和预热筛选后关联得到 18,329,997 条可用样本。

| 标签周期 | 独立有效标签行数 | Parquet 大小 |
| --- | ---: | ---: |
| 100ms | 25,345,704 | 180.02 MB |
| 500ms | 24,680,810 | 185.15 MB |
| 1s | 23,910,456 | 184.31 MB |
| 5s | 19,174,482 | 185.67 MB |

特征与标签各自保存完整网格和有效标志，因此有效标签数与最终可用样本数不同。主标签单位为 bps，每个周期另有 Ticks 和 USDT 辅助列。更新特征与更新标签相互独立，读取时核验基础版本与时钟，并按唯一时间键关联。

基础快照 Parquet 为 1,344.33 MB，16 个特征的 Parquet 为 1,053.96 MB，四周期标签合计 735.14 MB，三类合计约 3.133 GB。容量均按十进制计算。

与已有 C++ 基础表重叠的七天完成了前十档与成交窗口的逐项核对，公共有效区间没有差异。其余日期完成源文件时序、首尾覆盖、数值与编号检查，以及采样和标签边界验证。30 天并未新增全深度增量回放参考，快照质量仍需按来源记录和已知缺口解释。

默认快照年龄阈值为 1s，特征预热为 5s。年龄门限用于保守剔除，静默不能直接解释为采集断线。7 月 14 日已有中断记录在快照基础表中继续生效，成交跳号在被观察到时标记并重新预热。快照差分与完整事件 OFI 保持不同的字段含义。

实际读取 7 月 12—13 日的两个日期，得到 `X.shape=(1306097,16)`、`y.shape=(1306097,)`，主目标为 bps，矩阵数值均有限且样本索引唯一。月度统计采用按日计算，全部 30 天的统计已保存。

验证覆盖三种标签单位、多周期跨日边界、缺失 Mid、独立缓存复用、版本错配拒绝，以及真实压缩快照与成交的时间对齐。迁移前后的七天对照和独立版本对照分别保存为 `migration.json` 与 `split_parity.json`。 [迁移对照](../../../workspace/results/m2_polars/migration.json) · [独立版本对照](../../../workspace/results/m2_polars/split_parity.json) · [快照与 C++ 对照](../../../workspace/results/m2_polars/snapshot_parity.json)

30 天的下载来源与文件数量记录位于 `workspace/results/m2_polars/download_30d.json`，统计和矩阵读取结果分别位于 `statistics_30d.json` 与 `xy_check.json`。复现命令和模块职责见研究包指南。 [下载记录](../../../workspace/results/m2_polars/download_30d.json) · [研究包与使用方法](../../implementation/research_package.md)

| 本轮文件 | 修改内容 |
| --- | --- |
| `research/mm_research/data/base.py`、`data/quality.py`、`data/dataset.py` | 基础源适配、迁移质检、独立版本生产、关联和统计 |
| `research/mm_research/features/transform.py`、`targets/midprice.py` 及包初始化文件 | 候选特征、独立多周期标签和包导入 |
| `scripts/build_research_dataset.py`、`validate_research_dataset.py`、`summarize_research_dataset.py` | 命令行参数保留，业务调用研究包 |
| `scripts/convert_tardis.py`、`scripts/download_baidu.py` | 复用压缩读取，增加只下载同源成交的类型 |
| `pyproject.toml`、`requirements.txt`、`.gitignore` | 安装研究包、声明 Python 与依赖版本、忽略研究产物 |
| `config/research_versions.yaml`、`config/research_local_30d.yaml` | 七天事件来源和三十天快照来源的版本参数 |
| `tests/test_research_dataset.py`、`tests/test_research_versions.py`、`CMakeLists.txt` | 更新导入，检查独立缓存、单位、日界、缺失值和快照时间对齐 |
| README、阅读指南、数据说明、M2-A 指南和本次指南/报告 | 当前结构、命令和验收依据 |
| 指定 HTML 及批注/定位 JSON、`docs/design/` 同步副本 | 更新研究包路径、实施状态和验证批注，保留可收起的笔记面板 |
