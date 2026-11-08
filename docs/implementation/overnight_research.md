# 夜间因子与模型实验

常驻入口是 `scripts/run_overnight_server.sh`，配置属于 `config/research/overnight_server.yaml`。脚本运行在服务器自身，SSH 断开后继续工作。主配置声明各任务的输入模板、覆盖参数、输出、依赖、线程及内存预算，临时分组 YAML 自动生成到运行目录。

```sh
nohup bash scripts/run_overnight_server.sh config/research/overnight_server.yaml \
  > workspace/results/overnight_2026-11-07/supervisor.log 2>&1 < /dev/null &
```

## 今晚的队列

原有 300 因子的五周期评价、线性模型和 C++ 验证继续运行。Gen011—Gen020 增加 360 因子，Gen021—Gen029 增加 324 因子。Gen031—Gen037再增加252因子。Gen030、Gen038、Gen039、Gen040分别等待现货盘口、标记/指数价格、强平事件和ETH行情。

Gen023 的九个 100ms 桶内规模统计由 `features/trade_distribution.py` 从标准官方逐笔生成。Gen014 与 Gen021 先携带历史状态，再计算外层窗口。Gen012、Gen025、Gen026 先共享里层计算，减少多窗口重复开销。Gen028 的两秒失败突破需要逐位置状态转换，使用 CPU Numba 内核，其余窗口继续使用 Polars。

新增批次可原子追加到 `config/research/overnight_queue/`，监督器运行中自动载入。

额外 CPU 重任务同时最多一个，GPU 队列同时最多一个。每个因子组和评价组单独启动。组失败时记录日志，其他独立组继续。缺少必需上游的任务标为跳过。扩展模型明确记录未完整发布而排除的因子组。

## 内存控制

服务器容器上限为 60 GiB。启动新任务时预留 8 GiB，并为已运行任务尚未达到的预计峰值保留额度。实际可用量包含可回收的干净文件缓存，脏页与共享内存不会被当作立即可回收空间。

每个新任务及其后代继承 `RLIMIT_DATA`，限制堆与匿名映射。因子任务分配硬上限为 30 GiB，逐笔分布为 8 GiB，评价为 12 GiB，模型队列为 16—18 GiB。监督器每五秒检查任务进程树的匿名内存。容器实际余量低于 6 GiB 时，结束额外任务并记录内存停止状态。原有基础链路单独监测。

研究矩阵的文件映射与匿名堆内存分别统计。GPU 单实验最多使用 80% 显存。模型实验采用独立子进程，失败后进程退出并释放 CUDA 上下文，再运行下一个候选。

## 模型实验

使用 MLP、时间线性、GRU、LSTM、TCN、Transformer 与 LightGBM，分别预测未来 1s、10s、60s、180s、300s。先运行小步数筛查，再扩大历史长度、隐藏宽度、训练步数并改变随机种子。每项实验最多训练十分钟。

神经训练从整个训练区间的合法整秒位置均匀抽取批次，记录实际训练步数与抽样次数。树模型明确限制训练抽样数量。验证集使用固定时间位置抽样筛查，各模型在同一目标上采用相同验证位置。按验证误差选择每个目标的模型后，统一计算完整有效测试秒点的预测。

训练截至 2026-05-31，验证截至 2026-08-31，测试截至 2026-10-21。训练与验证末尾隔离 300 秒，并检查目标时间。候选列先按组覆盖选取，再仅用训练期 IC 排序。标准化仅使用训练期。神经输入缺失使用训练均值，并附加独立缺失通道。

Factor 正式文件保持 Float64。模型共享矩阵使用 Float32，GPU 训练采用混合精度。服务器建模环境为独立 `.venv-models`，复用镜像已有的 NVIDIA PyTorch，采用与之兼容的 NumPy 1.26.4、Arrow 19.0.1，Polars 仍为 2.1.0。数据生产环境 `.venv` 保持原有依赖。实际环境随实验保存。

原有因子组的候选训练截止凌晨 2 点，扩展组截止 6 点 15 分，预留最终整秒测试时间。模型比较按验证集选择，测试结果单独报告。实时 C++ 路径继续核对已移植的压力线性模型。

## 查看结果

运行目录为 `workspace/results/overnight_2026-11-07/`。

| 文件 | 内容 |
| --- | --- |
| `status.json` | 每项任务状态、开始结束时间和峰值匿名内存 |
| `resources.jsonl` | 容器内存、可回收缓存和 CPU 采样 |
| `任务名/run.log` | 该项任务输出、异常和 GNU time 记录 |
| `任务名/exit_code.txt` | 独立退出状态 |
| `report.md` | 可直接阅读的汇总 |

原有和扩展模型分别保存在 `workspace/results/models/overnight_legacy_v1/`、`overnight_expanded_v1/`。逐实验 `training.log` 和 `report.json` 保存错误或验证结果，`best.pt`、`best.txt` 保存模型，`frozen_selection.json` 保存冻结选择，`test_predictions_周期.parquet` 保存整秒测试预测。

北京时间 08:00 生成当时进度报告。数据任务如仍未结束，继续计算，整条队列结束后更新最终报告。监督进程异常退出时，外层脚本最多恢复三次，并接管状态中尚在运行的任务。

## 因子内存修复与补算

本轮补算配置为 `config/research/factor_recovery.yaml`，追加任务位于 `config/research/recovery_queue/`。输入继续使用 `server_531d_v1` 的100ms Features，补齐三个已有1s因子版本及其五周期评价。实际数值定义、Float64精度和80%有效比例保持原口径。

```bash
bash scripts/run_overnight_server.sh config/research/factor_recovery.yaml
```

普通组采用7天分块，Gen025、Gen034采用1天分块。每块和每个合并任务完成后退出进程，避免分配器跨块保留内存。Gen014、Gen021携带上一原始位置、状态锚点及120秒尾部，输出子区间时先计算完整事前状态。Gen020的逐日读取也放入独立进程。Gen022先物化九个盘口墙状态供各窗口复用。Gen034先共享状态计数，再只在整秒计算信息熵。Gen025和Gen034的有界里层使用300秒事前输入，覆盖最长100秒里层和100秒外层，避免每次重算完整前日。

队列最多并行三组因子，每组四个Polars线程；评价使用四个单线程进程。长任务依据实测耗时优先排队。单组因子匿名分配上限24GiB，评价12GiB，同时保留全局准入和紧急停止规则。实际能启动多少任务还取决于可用内存。

状态和日志位于 `workspace/results/factor_recovery_2026-11-08/`。旧失败及试跑失败保留原始记录，修复后的恢复任务使用独立编号。已完成因子的汇总报告位于 `docs/reports/factor_review_2026-11-08/report.md`，服务器完整对照表位于 `workspace/results/factor_review_2026-11-08/comparison.parquet`。
