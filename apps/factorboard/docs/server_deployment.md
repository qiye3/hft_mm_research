# 服务器看板

看板位于服务器 `/root/shared-nvme/hft_mm_research/apps/factorboard`。入口 `scripts/serve_results.py`，评估清单 `deploy/results.json`。它读取四个生产批次的已完成评估，每分钟刷新索引，刷新网页后更新列表。只收录有 `completed.json` 的目录，与研究写入共用共享锁。新增批次时明确加入清单并重启服务。

## 使用

2026-11-09已在新实例核验原JuiceFS共享盘，项目路径保持 `/root/shared-nvme/hft_mm_research`。实例资源为10vCPU、60GiB内存、RTX3090 24GiB显存，两个Python环境均可使用。看板已恢复，当前收录1,236个完成评测因子。

当前连接主机为 `qiye-rtx3090-10vcpu`，SSH地址为 `ssh.zw1.paratera.com:2222`。本机运行 `bash apps/factorboard/deploy/connect.sh`，保持终端打开，然后访问 http://127.0.0.1:8765/ 。服务器服务只监听回环地址，通过已有SSH账号访问。保持原浏览器和地址可继续使用本机分类、笔记和偏好。

服务器上首次启动，或容器重新开机后启动

```bash
cd /root/shared-nvme/hft_mm_research
bash apps/factorboard/deploy/start_server.sh
```

服务由独立 Supervisor 管理，终端断开继续运行，进程退出自动拉起。它没有修改容器平台的启动配置。容器重启后需运行上面的启动命令。

查看、重启、停止

```bash
/base/mambaforge/bin/supervisorctl -c /root/shared-nvme/hft_mm_research/apps/factorboard/deploy/supervisord.conf status
/base/mambaforge/bin/supervisorctl -c /root/shared-nvme/hft_mm_research/apps/factorboard/deploy/supervisord.conf restart factorboard
/base/mambaforge/bin/supervisorctl -c /root/shared-nvme/hft_mm_research/apps/factorboard/deploy/supervisord.conf shutdown
```

日志保存在 `workspace/results/factorboard/service/`，自动轮转。Polars两线程、OpenBLAS单线程，单进程串行响应，不运行全量因子扫描、回放或评估任务。

## 现有评估口径

覆盖2025-05-09至2026-10-21，共531日。因子每秒输出。预测周期1s、10s、60s、180s、300s，支持全部整秒和非重叠采样。训练截至2026-05-31，验证截至2026-08-31。列表和筛选固定用测试期1s全部整秒指标，详情可切换。

生产结果是训练期冻结五分组，重复分位阈值使用负、零、正三组。按原结果展示，不从五组推算十二组。高低组差来自上游完整分组日统计，端点缺失时保持空值。IC、分组收益沿用日期等权，零值率按因子有限样本数加权。

成本开关提供 `0.5×方向×高低组差−2×单边费率` 的双侧等权手续费情景，默认单边5bps，不自动选方向。原始分组收益维持毛收益显示。

资料库组织、状态、收藏、筛选方案和笔记沿用原前端。新增 `production_data.js`、`production_evaluation.js` 由生产HTTP入口显式加载，本机预览继续使用原入口。现有评估未保存秒级价格联动和因子值相关矩阵，时序页展示日级因子统计，对比页展示同口径指标。

## 本次架构批注

2026-11-08，部署节点增加只读评估适配。研究代码及结果的生产流程保持独立，评测完成标记作为收录边界，界面分组、周期、样本范围与已有产物一致。批注保存在看板目录，以避免并行修改主项目架构文件。

## 多维诊断

详情中的“多维诊断”按四个视角组织，每个视角四张图。图表直接使用已有评测统计，日指标的均值按有效日期等权。

- 稳定性。日IC直方图只统计有限值，六个等宽区间，常数序列单独一组。七日均值采用真实日历窗口，至少五个有效日。月度IC和正IC日比例按当月实际有效日计算，零IC归入非正日。
- 预测周期。月份×周期热图使用各月份的日IC均值，颜色在当前图内采用对称尺度，悬浮显示有效天数。训练／验证／测试图采用当前预测周期和采样方式的已保存汇总。
- 分组结构。样本占比的分母是所有组的样本合计。相邻组变化为各组日均收益的差，两组有效日期可能不同。逐日高低组差要求当天两端点组同时存在，否则保留缺失。诊断图维持毛收益，手续费情景仍在总览与对比表中查看。
- 数据质量。以日期关联因子分布与当前周期的日IC。共同覆盖率分母是当前采样的网格数，缺失率和零值率沿用分布表自身口径，三者分母不同。

因子对比新增IC日间波动与共同覆盖率散点，每个点代表一个选中因子。IC日间波动采用日IC样本标准差，不年化。所有散点可悬浮读取，键盘聚焦后用左右方向键切换观测。

接口新增 `daily_groups`、`daily_horizons`、`split_summary`，从原有锁保护下的统计表读取。相关派生计算验证可运行 `node apps/factorboard/tests/test_production_diagnostics.cjs`。
