# Gen041长周期微观结构：开发期实验结果

36个因子已完成531天生产，四组输入对应的8个方向模型、开发收益回测和同样本预测评价均已完成。本轮精简12列优于其余新增因子组合，但没有达到30%账户年化目标，不替换现有基线，也不进入历史复核。

## 固定实验条件

历史范围2025-05-09至2026-10-21；训练截至2026-03-22，内部拟合截至03-07，03-08起用于内部校准。开发区间2026-03-23至05-14，共53天。历史复核本轮未评价。新因子读取决策前一完整秒，均值与标准差仅用内部fit估计，缺失填补后保留缺失标记。

分类目标为1800秒后按买卖一价、双边各5bps计算的获利概率，两个方向分别训练。XGBoost采用seed17、max_depth=4、max_bin=64、min_child_weight=50、eta=0.05、subsample和colsample_bytree均0.8、lambda=10，最多800轮、内部校准集50轮早停，再做Platt概率校准。旧146列、现货16列和慢价格状态18列保留。

收益筛选采用到达时刻十档快照逐档执行，每笔0.10BTC、最长持仓7200秒、初始本金10000USDT、最高1倍总敞口，Taker单边5bps及实际资金费。该表是快照执行筛选；原生Maker事件回放单独报告。

## 费用后收益

| 新增输入 | 往返数 | 净收益USDT | 53天账户收益 | 年化 | 最大回撤 | 去最好周USDT |
|---|---:|---:|---:|---:|---:|---:|
| 27列长期水平 | 62 | -105.06 | -1.05% | -7.02% | 4.04% | -303.25 |
| 9列短长背离 | 41 | 17.34 | 0.17% | 1.20% | 3.60% | -123.55 |
| 全部36列 | 60 | -12.45 | -0.12% | -0.85% | 5.44% | -207.39 |
| 训练期筛选12列 | 26 | 110.85 | 1.11% | 7.89% | 3.14% | 4.18 |

E1毛收益348.11USDT、手续费459.30USDT，交易成本超过毛收益。E3毛收益426.26USDT、手续费444.10USDT，同样被成本吞没。E4降至26笔往返，手续费187.98USDT，获得110.85USDT净收益；但最好一周贡献约96.2%的净收益，最差月亏损16.84USDT。减少冗余输入改善了这一组交易结果，仍未解决收益集中。

## 预测指标及解释

所有组、每个方向使用同一批3719562条有效开发标签，不因因子缺失删除样本。E4空头AUC由基线0.58166升至0.58565，LogLoss由0.599768降至0.599554；多头AUC由0.59127降至0.59071，LogLoss由0.608532升至0.609069。E1—E3也表现为部分空头指标改善、多头概率误差增加。

各组LogLoss和Brier的新旧误差差值，以8个UTC周为块重采样1000次，其95%区间全部跨零。因此本轮没有显示稳定的预测增量。分类目标为30分钟，策略固定持仓2小时，两者也不是同一收益目标；后续优先拆分方向及持仓期限的贡献，不继续盲目增加窗口。

## E4冻结的12列

- `cross_trade_book_sign_agree_mean_1800s`
- `cross_trade_book_sign_agree_mean_900s`
- `book_depth_notional_p10_900s`
- `book_depth_notional_p10_1800s`
- `trade_small_qtyimb_1800s`
- `trade_aggr_qtyimb_1800s`
- `trade_aggr_qtyimb_900s`
- `trade_small_qtyimb_900s`
- `book_bid_depth_down_share_diff_60s_3600s`
- `book_net_depth_direction_signed_ratio_diff_60s_1800s`
- `book_ask_depth_down_share_diff_60s_3600s`
- `book_ask_depth_down_share_900s`

筛选仅使用内部fit的多头费用标签：每日样本至少100、列覆盖至少80%、有效日比例至少80%、相关同号日比例至少60%，最多12列且每类最多2列。两个方向模型共用此清单。

## 文件与复现

公式：`research/mm_research/factors/gen041.py`；全历史生产与缓存：`models/gen041.py`；精简输入：`models/gen041_selection.py`；同样本评价：`evaluation/fee_models.py`。后台批次为`config/research/maker_native_queue/gen041*.yaml`。

```bash
PYTHONPATH=research .venv-models/bin/python scripts/run_gen041.py config/models/gen041_selection_20261114.yaml select
PYTHONPATH=research .venv-models/bin/python scripts/run_cost_classifier.py config/models/cost_gen041_e4_20261114.yaml train --horizon 1800 --side 1 --seed 17
PYTHONPATH=research .venv-models/bin/python scripts/run_cost_classifier.py config/models/cost_gen041_e4_20261114.yaml train --horizon 1800 --side -1 --seed 17
PYTHONPATH=research .venv-models/bin/python scripts/run_account_sizing.py config/strategy/account_gen041_e4_20261114.yaml screen --group e4
PYTHONPATH=research .venv-models/bin/python scripts/evaluate_fee_models.py config/evaluation/gen041_e4_20261114.yaml
```

明细归属`workspace/results/gen041_evaluation_20261114_v1`、`gen041_e4_evaluation_20261114_v1`、`gen041_selected_inputs_20261114_v1`及两组`account_gen041*`目录。文档库附带`Gen041实验结果明细_20261114.zip`。
