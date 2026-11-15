# E4完整研究过程与结果｜2026-11-15

## 一、先认识E4，以及这条研究最终得到什么

E4是一组基于长周期微观结构信息的费用后获利分类模型。它在原有180列数值输入上增加12列Gen041因子，分别预测做多、做空并持有1800秒后，按当时买卖一价和双边费用估计是否获利。策略把两个方向的概率换算成bps净价值，再决定是否入场。

E4最初是输入选择方案的名称。后续实验沿用这组输入和模型权重，依次调整持有期、波动风险退出、仓位规模和Maker执行。文中的各组结果都列出对应日期和交易设置。

最终原生候选使用0.125 BTC固定委托、最长3600秒持仓、事前波动止损与移动止盈，开平仓优先Maker，退出最多等待30秒再转IOC。它在开发期2026-03-23至05-14取得410.8111 USDT净收益、31.9510%年化；冻结历史复核2026-05-15至07-06取得291.0373 USDT净收益、21.8438%年化，最高敞口0.8802倍。历史复核未达到预设30%年化门槛。

本文以2026-11-14保存的实验配置、完成记录和账户汇总为结果依据，结合项目代码解释实际计算。公开附件保存复核所需的小型汇总和日末权益。后文的开发和复核各为53个完整日历日。历史复核区间曾被其他实验查看，因此在本项目中称为冻结历史复核。

## 二、完整数据链路与时间安排

### 2.1 从原始行情到模型，再到交易账本

完整链路如下。

```text
Binance官方普通成交 + 历史盘口快照
    ↓ 统一时间、字段与质量标签
标准 trades.parquet / snapshots.parquet
    ↓ 100ms特征计算
盘口、成交、小单及历史价格Features
    ↓ 秒级聚合及长窗口统计
Gen041的36列候选因子
    ↓ 仅在内部fit筛选
12列E4因子 + 原146列 + 现货16列 + 慢价格状态18列
    ↓ fit预处理、多空独立XGBoost、tune概率校准
每秒多头/空头净价值预测
    ↓ 策略规则与风险状态
快照交易筛选 → 冻结原生Maker候选
    ↓ 真实L2 + 官方成交进入hftbacktest-cpp
订单、实际成交、手续费、资金费、连续账户权益
```

研究范围为2025-05-09至2026-10-21，共531天。研究行情可见时间为交易所时间加20ms，所有日期按UTC解释。早期盘口使用已适配的Crypto Lake / HuggingFace来源，2025-10-01起切换到网盘来源；标准化时保留质量与来源标记。盘口研究读取前10档，原生交易执行另外使用真实逐事件L2及官方普通成交。

基础Features每100ms计算。Gen041每秒输出，所有36列都覆盖完整时间轴，每列45,878,400行，无效位置为空，Float64落盘。模型输入另读取决策前一完整秒的新增因子值。既有Features、因子和模型缓存分别管理。

### 2.2 日期分工与边界隔离

| 区间 | 日期 | 实际用途 |
| --- | --- | --- |
| 全部研究行情 | 2025-05-09—2026-10-21 | 基础数据、因子生产 |
| 内部fit前缀 | 2025-05-09—2026-03-07 | 小单阈值、Gen041选列、预处理和模型拟合，合法行以冻结行集为准 |
| 内部tune | 2026-03-08起，位于截至03-22的Train内部 | 早停和Platt概率校准 |
| Train结束 | 2026-03-22 | 训练阶段上界 |
| val_dev | 2026-03-23—2026-05-14 | 比较输入、退出与规模，冻结执行候选 |
| val_select | 2026-05-15—2026-07-06 | 同参数历史复核 |

因子选择和模型训练沿用原有fit/tune行集。1800秒标签要求`当前行+1800 < 下一分区起始行`。fit的目标不能进入tune，tune的目标不能越过Train结束；预测评价同样要求目标落在本分区。E4选择记录的`fit_last_row=26263495`、`label_end_before_row=26265300`，是原全局秒网格的行号。

交易信号保留完整开发与复核时钟。未来标签是否有效只影响训练与评价，交易决策使用截至当前可见的输入、预测和风险，不用未来收益筛选可交易时刻。

## 三、36列Gen041怎样构造，12列E4怎样选出

### 3.1 从100ms压成一秒时，各字段采用不同聚合

秒时间戳记为t，对应右闭合的一秒，即`t-1s < 观测时间 ≤ t`。盘口价格和数量取t时刻已可见的秒末状态。OBI先在100ms位置计算，再取该秒均值，10个位置中至少8个有效。主动买卖量和小单量要求10个100ms位置全部完整才累计；已知没有成交记0，未知缺口保留为空。

一档盘口失衡为`(买一数量−卖一数量)/(买一数量+卖一数量)`。十档买深度B和卖深度A分别为各侧数量之和。相邻秒计算`ΔB`、`ΔA`和`Δ(B−A)`，时间不连续、同日质量段变化及2025-10-01来源切换处，差分置空。十档深度变化同时受到价位进出前十档影响。

### 3.2 九类基础信号与36列候选

| 家族 | 窗口内实际计算 | 长期窗口秒 |
| --- | --- | --- |
| 一档盘口压力 | 秒内OBI的窗口均值 | 300、900、1800 |
| 净深度方向 | ΣΔ(B−A) / Σ\|Δ(B−A)\| | 300、900、1800 |
| 买侧深度下降占比 | Σmax(−ΔB,0) / Σ\|ΔB\| | 900、1800、3600 |
| 卖侧深度下降占比 | Σmax(−ΔA,0) / Σ\|ΔA\| | 900、1800、3600 |
| 主动成交量失衡 | Σ(主动买量−主动卖量) / Σ(主动买量+主动卖量) | 300、900、1800 |
| 小单成交量失衡 | 同上，仅累计成交数量严格小于fit期Q50的记录 | 300、900、1800 |
| 成交与盘口方向一致性 | mean(sign(秒内主动买量−主动卖量)×sign(秒内OBI)) | 300、900、1800 |
| 25,000 USDT理论吃盘成本差 | 买侧吃盘成本bps减卖侧吃盘成本bps，再求窗口均值 | 900、1800、3600 |
| 十档名义深度低分位 | 两侧Σ(价格×数量)的窗口10%分位数，线性插值 | 900、1800、3600 |

九类各有三个长期值，共27列；各类再产生一个60秒与最长窗口的背离，共9列。一般背离为`60秒统计−最长窗口统计`，名义深度低分位背离为`P10_60s/P10_3600s−1`。

外层滚动按真实秒位置继续滑动，至少`ceil(窗口秒数×0.8)`个有效值。缺口占据时间位置，跨午夜连续。比值先汇总分子和分母再相除；已知零活动且分母为零时记0。成本计算按当前Mid将25,000 USDT换成BTC目标量，再在十档盘口逐档计算VWAP，深度不足时为空。

小单阈值重新用2025-05-09至2026-03-07的1,116,064,239笔普通成交拟定，Q50=0.003 BTC、Q90=0.08 BTC。新Q50与已有列一致，因此复用其数值，并记录本轮合法的阈值估计来源。实际小单规则是数量严格小于0.003 BTC。

按日生产时回读前3601秒，覆盖最长3600秒窗口及一次差分；完成后按时间顺序合成每因子一份全日期Parquet。代码入口为`factors/gen041.py`和`models/gen041.py`。

### 3.3 E0—E4的输入对照

| 组别 | 在原180列上新增什么 | 数值列数 |
| --- | --- | ---: |
| E0 | 原146列、现货16列、慢价格状态18列 | 180 |
| E1 | 27列长期水平 | 207 |
| E2 | 9列短长背离 | 189 |
| E3 | 全部36列 | 216 |
| E4 | fit期筛选出的12列 | 192 |

模型使用192列数值特征，另外保留缺失标记。旧146列和现货16列沿用已有冻结缓存，18列慢状态由60/300/900/1800/3600/14400秒六档历史窗口各生成收益、收益波动、价格z-score。

新增因子只用fit样本估计均值和标准差，缺失以fit均值填补，标准化后对应均值位置为0，同时保留缺失标记。E4从E3已生成的矩阵中按冻结顺序抽列，并复制对应预处理参数。旧列、样本行集和预处理口径沿用原缓存。

### 3.4 E4只在fit中筛选

选列目标是1800秒多头费用后收益是否大于0。每天计算候选列与二分类标签的Spearman相关，要求如下。

1. 每天至少100个标签样本，该列有效覆盖至少80%。
2. 该列可计算日占fit全部日期至少80%。
3. 日相关与平均相关同号的比例至少60%，平均相关非零。
4. 按`abs(平均日相关)×同号日比例`降序排列，同分按名称排序。
5. 每个家族最多2列，总计最多12列。

两方向模型共用这12列。fit中排名第三的同一家族可能因家族上限落选。筛选结果先落盘，再复制开发和复核输入。

### 3.5 冻结12列、含义及fit统计

下面按冻结输入顺序列出。日相关针对多头费用标签，其正负号描述fit关联方向。

| 顺序 | 因子 | 计算含义 | fit平均日相关 | 同号日比例 | 有效日 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `cross_trade_book_sign_agree_mean_1800s` | 过去1800秒盘口与成交方向一致程度 | 0.07459558 | 74.0132% | 304 |
| 2 | `cross_trade_book_sign_agree_mean_900s` | 过去900秒盘口与成交方向一致程度 | 0.06401135 | 72.6974% | 304 |
| 3 | `book_depth_notional_p10_900s` | 过去900秒两侧十档名义深度P10 | -0.04570499 | 62.5000% | 304 |
| 4 | `book_depth_notional_p10_1800s` | 过去1800秒两侧十档名义深度P10 | -0.04274754 | 62.1711% | 304 |
| 5 | `trade_small_qtyimb_1800s` | 过去1800秒小单主动买卖数量失衡 | -0.03888521 | 65.1316% | 304 |
| 6 | `trade_aggr_qtyimb_1800s` | 过去1800秒全部主动成交数量失衡 | -0.04025608 | 60.5263% | 304 |
| 7 | `trade_aggr_qtyimb_900s` | 过去900秒全部主动成交数量失衡 | -0.03151895 | 65.1316% | 304 |
| 8 | `trade_small_qtyimb_900s` | 过去900秒小单主动成交数量失衡 | -0.03002901 | 65.7895% | 304 |
| 9 | `book_bid_depth_down_share_diff_60s_3600s` | 买侧深度下降占比60秒减3600秒 | -0.00766507 | 82.2368% | 304 |
| 10 | `book_net_depth_direction_signed_ratio_diff_60s_1800s` | 净深度方向比值60秒减1800秒 | 0.00747581 | 81.2500% | 304 |
| 11 | `book_ask_depth_down_share_diff_60s_3600s` | 卖侧深度下降占比60秒减3600秒 | 0.00724809 | 81.9079% | 304 |
| 12 | `book_ask_depth_down_share_900s` | 过去900秒卖侧深度下降占比 | 0.00776437 | 60.5263% | 304 |

三类量失衡与两类盘口变化之间保留了多种关系。成交—盘口方向一致性强调两种压力是否同向；名义深度P10强调过去一段时间的较弱流动性；小单和总成交失衡描述不同成交群体的方向；深度下降和短长背离描述盘口状态的近期变化。经济含义需结合非线性树和其他输入阅读，负相关列仍可进入模型。

### 3.6 全部192列输入及384通道的实际排列

E4的输入由6列基础Features、35族各4列历史因子、16列现货变量、18列慢价格状态和12列Gen041组成。原146列来自已冻结的旧研究矩阵，历史因子按家族均匀抽取4列，使用`np.linspace(0, 家族列数−1, 4, dtype=int)`选位置。该步骤不读取开发或复核期IC。E4保留这一清单及次序。

原146列保留3、5、10、20、50、100秒等旧窗口，准确窗口以列名为准。当前生产配置中的10/60/180/600秒四档属于后续版本。历史缓存的解释和重建应使用冻结清单。

| 通道索引，零起始 | 输入内容 | 数量 |
| --- | --- | ---: |
| 0—145 / 146—291 | 原146列数值 / 对应缺失标记 | 146+146 |
| 292—307 / 308—323 | 现货数值 / 对应缺失标记 | 16+16 |
| 324—341 / 342—359 | 慢状态数值 / 对应缺失标记 | 18+18 |
| 360—371 / 372—383 | Gen041精选数值 / 对应缺失标记 | 12+12 |

每个缺失标记取0或1。各块先放数值再放本块缺失标记，三份缓存矩阵按`models/intraday.py::input_parts()`拼接。完整逐列索引、计算模块和表达式依赖见[192列输入清单](evidence/input_columns.csv)。下表补齐原180列的计算说明。

#### 六列基础Features

- `imbalance_1`、`imbalance_5`、`imbalance_10`分别使用前1、5、10档两侧BTC挂单量，计算`(买侧量−卖侧量)/(买侧量+卖侧量)`。
- `trade_imbalance_100ms`、`trade_imbalance_1s`分别使用过去100ms、1s主动买卖BTC成交量，计算`(主动买量−主动卖量)/(主动买量+主动卖量)`。已知零成交记0，未知位置保留为空。
- `return_1s_bps`是`(Mid_t/Mid_t−1s−1)×10000`，采用普通收益。代码按100ms网格回看10个位置并核对精确时间差，端点缺失则为空。

#### 35族历史因子，共140列

每行列出E4确实读取的4列。外层窗口按完整100ms位置计算，整秒输出，通常要求80%有效。内层年龄、历史基准和完整秒聚合按各家族实现计算，公式可从对应模块及清单中的表达式依赖定位。

| 家族 | 计算内容 | 冻结的4列 |
| --- | --- | --- |
| [Gen001](../../../research/mm_research/factors/gen001.py) | 盘口与成交压力的滚动均值 | `book_obi_l1_mean_3s`<br>`book_obi_l5_mean_100s`<br>`trade_aggr_imb_100ms_mean_100s`<br>`cross_trade_book_gap_mean_100s` |
| [Gen002](../../../research/mm_research/factors/gen002.py) | 短窗口均值减长窗口均值 | `book_obi_l1_mean_diff_3s_10s`<br>`book_obi_l5_mean_diff_20s_100s`<br>`trade_aggr_imb_100ms_mean_diff_20s_100s`<br>`cross_trade_book_gap_mean_diff_20s_100s` |
| [Gen003](../../../research/mm_research/factors/gen003.py) | 近端深度占比、数量加权档位距离及双侧差 | `book_bid_share_l1_l10_mean_5s`<br>`book_bid_weighted_dist_l10_mean_50s`<br>`book_ask_weighted_dist_l10_mean_50s`<br>`book_dist_asym_l10_mean_50s` |
| [Gen004](../../../research/mm_research/factors/gen004.py) | 成交笔数、数量和金额的滚动和 | `trade_count_100ms_sum_5s`<br>`trade_sell_qty_100ms_sum_50s`<br>`trade_buy_notional_100ms_sum_50s`<br>`trade_total_notional_100ms_sum_50s` |
| [Gen005](../../../research/mm_research/factors/gen005.py) | 收益、正收益、正收益平方及方向切换计数 | `px_ret_100ms_sum_5s`<br>`px_pos_ret_100ms_sum_50s`<br>`px_pos_sq_ret_100ms_sum_50s`<br>`px_turn_flag_100ms_sum_50s` |
| [Gen006](../../../research/mm_research/factors/gen006.py) | 相邻100ms排行深度变化的滚动和 | `book_bid_depth_l1_delta_100ms_sum_5s`<br>`book_net_depth_l1_delta_100ms_sum_50s`<br>`book_net_depth_l5_delta_100ms_sum_50s`<br>`book_net_depth_l10_delta_100ms_sum_50s` |
| [Gen008](../../../research/mm_research/factors/gen008.py) | 成交压力与同桶收益的同向、反向和吸收关系 | `impact_trade_qty_aligned_100ms_mean_5s`<br>`impact_trade_qty_absorb_100ms_mean_50s`<br>`impact_trade_imb_aligned_100ms_mean_50s`<br>`impact_trade_imb_opposed_100ms_mean_50s` |
| [Gen009](../../../research/mm_research/factors/gen009.py) | 成交失衡、成交加速度与盘口OBI之差 | `cross_trade_aggr_imb_100ms_book_obi_l1_gap_mean_5s`<br>`cross_trade_aggr_imb_100ms_book_obi_l5_gap_mean_50s`<br>`cross_trade_aggr_imb_accel_100ms_book_obi_l1_gap_mean_50s`<br>`cross_trade_aggr_imb_accel_100ms_book_obi_l10_gap_mean_50s` |
| [Gen010](../../../research/mm_research/factors/gen010.py) | 按fit数量阈值划分的小中大单数量和失衡 | `trade_small_buy_qty_100ms_mean_5s`<br>`trade_small_imb_100ms_mean_50s`<br>`trade_mid_imb_100ms_mean_50s`<br>`trade_large_imb_100ms_mean_50s` |
| [Gen011](../../../research/mm_research/factors/gen011.py) | 价差水平、扩大、收缩和翻转 | `spread_level_ticks_mean_5s`<br>`spread_wide_2_flag_mean_100s`<br>`spread_contract_ticks_100ms_mean_100s`<br>`spread_flip_flag_100ms_mean_100s` |
| [Gen012](../../../research/mm_research/factors/gen012.py) | 前十档内5千、2.5万、10万USDT理论执行成本 | `book_exec_buy_cost_5k_mean_5s`<br>`book_exec_gap_cost_5k_mean_100s`<br>`book_exec_gap_cost_25k_mean_100s`<br>`book_exec_gap_cost_100k_mean_100s` |
| [Gen013](../../../research/mm_research/factors/gen013.py) | 成交活跃、密集、笔数跳变和活跃转静默 | `trade_active_flag_100ms_mean_5s`<br>`trade_dense_flag_100ms_mean_100s`<br>`trade_count_jump_up_100ms_mean_100s`<br>`trade_active_to_idle_flag_100ms_mean_100s` |
| [Gen014](../../../research/mm_research/factors/gen014.py) | 报价、挂单数量和成交状态持续年龄，单位秒 | `book_bid_px_age_sec_100ms_mean_5s`<br>`book_mid_px_age_sec_100ms_mean_100s`<br>`book_ask_q1_age_sec_100ms_mean_100s`<br>`trade_sell_age_sec_100ms_mean_100s` |
| [Gen015](../../../research/mm_research/factors/gen015.py) | 买卖报价上涨、变价及双侧协同 | `quote_bid_up_ticks_100ms_mean_5s`<br>`quote_ask_up_ticks_100ms_mean_100s`<br>`quote_ask_delta_ticks_100ms_mean_100s`<br>`quote_bid_ask_comove_100ms_mean_100s` |
| [Gen016](../../../research/mm_research/factors/gen016.py) | 成交VWAP相对桶末Mid、上桶Mid及买卖VWAP差 | `trade_buy_vwap_mid_dev_bps_100ms_mean_5s`<br>`trade_all_vwap_mid_dev_bps_100ms_mean_100s`<br>`trade_buy_sell_vwap_gap_bps_100ms_mean_100s`<br>`trade_all_vwap_prev_mid_dev_bps_100ms_mean_100s` |
| [Gen017](../../../research/mm_research/factors/gen017.py) | 成交量相对一、五、十档挂单深度 | `trade_buy_flow_rel_depth_l1_100ms_mean_5s`<br>`trade_net_flow_rel_depth_l1_100ms_mean_100s`<br>`trade_net_flow_rel_depth_l5_100ms_mean_100s`<br>`trade_net_flow_rel_depth_l10_100ms_mean_100s` |
| [Gen018](../../../research/mm_research/factors/gen018.py) | 历史滞后压力与100ms收益的Pearson相关，lag单位为100ms桶 | `lead_trade_aggr_imb_100ms_ret_lag1_corr_ret_5s`<br>`lead_trade_aggr_imb_100ms_ret_lag10_corr_ret_100s`<br>`lead_trade_aggr_imb_1s_ret_lag10_corr_ret_100s`<br>`lead_trade_net_flow_rel_depth_l5_100ms_ret_lag10_corr_ret_100s` |
| [Gen019](../../../research/mm_research/factors/gen019.py) | 绝对压力超过0.6的极端状态及退出计数 | `book_obi_l1_extreme_state_thr06_sum_5s`<br>`book_obi_l1_extreme_exit_thr06_sum_100s`<br>`book_obi_l5_extreme_exit_thr06_sum_100s`<br>`trade_aggr_imb_1s_extreme_exit_thr06_sum_100s` |
| [Gen020](../../../research/mm_research/factors/gen020.py) | 同UTC半小时时段过去20个完整日参考的均值比和Q90异常 | `utcslot_spread_level_ticks_ratio_20d_mean_5s`<br>`utcslot_spread_level_ticks_q90flag_20d_mean_100s`<br>`utcslot_trade_count_100ms_q90flag_20d_mean_100s`<br>`utcslot_book_depth_total_l10_q90flag_20d_mean_100s` |
| [Gen021](../../../research/mm_research/factors/gen021.py) | 成交方向连续长度、切换及相邻非零同向 | `trade_dir_buy_run_100ms_mean_5s`<br>`trade_dir_signed_run_100ms_mean_100s`<br>`trade_dir_switch_flag_100ms_mean_100s`<br>`trade_dir_same_nonzero_100ms_mean_100s` |
| [Gen022](../../../research/mm_research/factors/gen022.py) | 最大挂单占比与相邻价格空洞 | `book_bid_max_share_l10_mean_5s`<br>`book_bid_max_gap_ticks_l10_mean_100s`<br>`book_ask_max_gap_ticks_l10_mean_100s`<br>`book_bid_ask_max_gap_ticks_gap_l10_mean_100s` |
| [Gen023](../../../research/mm_research/factors/gen023.py) | 100ms桶内成交规模中位数和集中度HHI | `trade_size_buy_median_qty_100ms_mean_5s`<br>`trade_size_buy_hhi_100ms_mean_100s`<br>`trade_size_sell_hhi_100ms_mean_100s`<br>`trade_size_all_hhi_100ms_mean_100s` |
| [Gen024](../../../research/mm_research/factors/gen024.py) | 两侧深度绝对变化及毛变动减净变动 | `book_depth_bid_abs_l1_100ms_mean_5s`<br>`book_depth_gross_minus_net_l1_100ms_mean_100s`<br>`book_depth_gross_minus_net_l5_100ms_mean_100s`<br>`book_depth_gross_minus_net_l10_100ms_mean_100s` |
| [Gen025](../../../research/mm_research/factors/gen025.py) | 价差和深度冲击、恢复年龄 | `liq_shock_spread_active_100ms_mean_5s`<br>`liq_shock_spread_age_sec_100ms_mean_100s`<br>`liq_shock_bid_depth_age_sec_100ms_mean_100s`<br>`liq_shock_ask_depth_age_sec_100ms_mean_100s` |
| [Gen026](../../../research/mm_research/factors/gen026.py) | 收益强度与价差、2.5万USDT执行成本的交互 | `cross_vol_abs_x_spread_100ms_mean_5s`<br>`cross_vol_abs_x_exec_100ms_mean_100s`<br>`cross_vol_down_x_exec_100ms_mean_100s`<br>`cross_vol_square_x_exec_100ms_mean_100s` |
| [Gen027](../../../research/mm_research/factors/gen027.py) | 过去1秒VWAP锚点偏离及价格位于锚点上方的标志 | `trade_buy_vwap1s_mid_gap_bps_mean_5s`<br>`trade_buy_vwap1s_above_flag_mean_100s`<br>`trade_sell_vwap1s_above_flag_mean_100s`<br>`trade_all_vwap1s_above_flag_mean_100s` |
| [Gen028](../../../research/mm_research/factors/gen028.py) | 相对过去5秒价格区间的突破及2秒确认后的失败突破 | `px_range_up_flag_100ms_mean_5s`<br>`px_range_up_down_flag_gap_100ms_mean_100s`<br>`px_range_up_down_dist_ticks_gap_100ms_mean_100s`<br>`px_range_up_down_fail_flag_gap_100ms_mean_100s` |
| [Gen029](../../../research/mm_research/factors/gen029.py) | 最近1秒与过去5秒状态的均值、方差和CUSUM变化 | `regime_spread_mean_shift_1s5s_mean_5s`<br>`regime_spread_cusum_shift_1s5s_mean_100s`<br>`regime_activity_cusum_shift_1s5s_mean_100s`<br>`regime_depth_cusum_shift_1s5s_mean_100s` |
| [Gen031](../../../research/mm_research/factors/gen031.py) | 三种盘口隐含中心相对Mid的Tick偏离 | `book_micro_l1_crossqty_delta_ticks_100ms_mean_5s`<br>`book_micro_l1_touchsign_delta_ticks_100ms_mean_100s`<br>`book_micro_l5_touchsign_delta_ticks_100ms_mean_100s`<br>`book_micro_l10_touchsign_delta_ticks_100ms_mean_100s` |
| [Gen032](../../../research/mm_research/factors/gen032.py) | 报价方向连续性和有向变价频率 | `px_mid_transition_cont_100ms_mean_5s`<br>`px_mid_transition_signed_100ms_mean_100s`<br>`px_bid_transition_signed_100ms_mean_100s`<br>`px_ask_transition_signed_100ms_mean_100s` |
| [Gen033](../../../research/mm_research/factors/gen033.py) | 滞后收益乘积及反向抵消幅度 | `noise_mid_ret_lag1_prod_100ms_mean_5s`<br>`noise_mid_reversal_overlap_100ms_mean_100s`<br>`noise_bid_reversal_overlap_100ms_mean_100s`<br>`noise_ask_reversal_overlap_100ms_mean_100s` |
| [Gen034](../../../research/mm_research/factors/gen034.py) | 离散状态熵及lag2互信息 | `info_mid_move_sign_100ms_entropy_5s`<br>`info_mid_move_sign_100ms_mi_lag2_100s`<br>`info_trade_aggr_sign_100ms_mi_lag2_100s`<br>`info_book_obi_l5_sign_100ms_mi_lag2_100s` |
| [Gen035](../../../research/mm_research/factors/gen035.py) | 一至二、三至五、六至十档数量除以价差Tick的边际密度和双侧不对称 | `book_bid_marginal_density_l1_2_mean_5s`<br>`book_marginal_density_asym_l1_2_mean_100s`<br>`book_marginal_density_asym_l3_5_mean_100s`<br>`book_marginal_density_asym_l6_10_mean_100s` |
| [Gen036](../../../research/mm_research/factors/gen036.py) | 历史上桶压力对当前100ms收益的OLS斜率及正负斜率差 | `book_obi_l5_lag_beta_5s`<br>`book_obi_l5_lag_asym_beta_100s`<br>`trade_aggr_imb_100ms_lag_asym_beta_100s`<br>`trade_aggr_imb_1s_lag_asym_beta_100s` |
| [Gen037](../../../research/mm_research/factors/gen037.py) | 完整1秒内十个100ms桶的早晚活跃差、HHI，再按秒滚动 | `intrasec_count_late_early_1s_sec_mean_5s`<br>`intrasec_count_hhi_1s_sec_mean_100s`<br>`intrasec_qty_hhi_1s_sec_mean_100s`<br>`intrasec_absret_hhi_1s_sec_mean_100s` |

#### 现货16列

来源为Binance官方`spot/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-YYYY-MM.zip`。每个10/60/180/600秒窗口依次生成以下4列，共16列。

| 列名前缀 | 计算 |
| --- | --- |
| `spot_imbalance` | `(2Σ主动买BTC量−Σ总BTC量)/Σ总BTC量`，零总量为空 |
| `spot_return` | `ln(现货收盘价_t/现货收盘价_t−窗口)×10000` |
| `spot_close_perp_mid_gap` | `mean_窗口(ln(现货收盘价/永续Mid)×10000)` |
| `spot_perp_return_gap` | 现货窗口对数收益bps减永续窗口对数收益bps |

滚动统计至少要求`int(窗口秒数×0.8)`个有效位置。秒线结束后增加20ms可见延迟，再向上取整到决策秒，读取代码将开盘时间加2秒作为可见整秒。永续Mid使用决策前一秒状态。月块携带前1202条历史。E4读取现货分组的16列。

#### 慢价格状态18列

窗口依次为60/300/900/1800/3600/14400秒，每窗口按收益、波动、价格z-score排列。令`p=ln(决策前一秒Mid)`、`r=p.diff()`。

| 列名前缀 | 计算 |
| --- | --- |
| `return` | `(p_t−p_t−窗口)×10000` |
| `volatility` | `rolling_std_窗口(r)×10000`，即1秒收益标准差 |
| `price_zscore` | `(p_t−rolling_mean_窗口(p))/rolling_std_窗口(p)` |

滚动至少要求`int(窗口秒数×0.8)`个有效位置，标准差使用Polars默认`ddof=1`，零价格标准差时z-score为空。最长历史为14400秒。这些字段由`models/intraday.py::prepare_intraday()`直接读取既有市场矩阵计算。

#### 时间与预处理

旧146列按决策前一完整秒对齐，午夜携带上一秒。现货按上述已结束秒线的可见时刻对齐，慢状态先将永续Mid后移一秒再计算，Gen041读取`values[rows−1]`。100ms上游仅接纳`交易所时间+20ms ≤ 网格时刻`的记录，特征计算延迟另记1ms。

各缓存仅用fit行拟定均值和标准差。非有限输入用fit均值填补，随后标准化并裁剪到`[-20,20]`，再追加Float32缺失标记。归一化标准差使用NumPy的`ddof=0`，小于`1e−8`时设1。新增12列从E3复制对应数值、缺失通道和预处理参数。缺失样本保留在共同的合法行集中。

## 四、标签、模型训练与概率转交易价值

### 4.1 1800秒费用后标签的精确公式

设入场报价为P0、1800秒后退出报价为P1，r=P1/P0，方向s取多头+1、空头−1。

- 多头P0取当前Ask，P1取未来Bid。
- 空头P0取当前Bid，P1取未来Ask。
- 净收益标签为`Y_bps = s×(r−1)×10000 − 5×(1+r)`。
- 分类标签为`Y_bps > 0`。

两次费用都按成交名义金额计算，因此第二笔费用随r变化。标签已经包含买卖价差与双边Taker各5bps。标签不含资金费；实际交易账本另计资金费。

当前和未来报价必须为正且有限，两个端点通过质量检查并拥有相同的连续质量段标记。边界再按1800秒目标时刻剔除。该标签实现使用端点和质量段条件，没有在这个函数里重新逐条遍历未来区间。

### 4.2 两个独立的XGBoost模型

| 项目 | 设置 |
| --- | --- |
| 模型数量 | 多头、空头各一份 |
| 目标 | binary:logistic |
| 优化评价 | LogLoss |
| 树方法与设备 | hist，CUDA |
| seed | 17 |
| max_depth / max_bin | 4 / 64 |
| min_child_weight | 50 |
| eta | 0.05 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| lambda | 10 |
| 训练轮数 | 最多800轮，tune连续50轮未改善则早停，保留最佳迭代模型 |
| 每方向合法fit样本 | 3,734,431 |
| 近期fit截断 | 未启用 |
| 样本加权 | 普通二分类，未启用盈亏幅度加权 |
| GPU并发 | 文件锁串行占用GPU，分批拼接输入 |

本轮同步的完成摘要保留训练参数、样本数和校准结果，未记录最佳迭代轮数。模型权重位于各方向的`model.ubj`，精确树数需从保存的权重读取。

### 4.3 Platt校准及净价值映射

模型先输出原始概率p，裁剪到[10⁻⁶,1−10⁻⁶]。tune拟定两个参数，得到`p_cal = sigmoid(a×logit(p)+b)`，优化LogLoss并对a施加`10⁻⁴×a²`轻微正则。

fit分别计算盈利标签平均收益W、非盈利标签平均收益L，映射为`edge_bps = p_cal×W + (1−p_cal)×L`。W与L在后续区间冻结。

| 方向 | a | b | fit盈利均值W bps | fit非盈利均值L bps | fit正例比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.4957050237 | 0.3118906234 | 22.2453616144 | -25.2623079474 | 0.3193648510 |
| -1 | 0.7562328106 | -0.1108605845 | 22.8346239424 | -25.1113488807 | 0.3165708511 |

多头edge大于0所需校准概率约53.17%，空头约52.38%，由`−L/(W−L)`确定。fit中多头、空头的正收益样本比例分别约31.94%和31.66%；入场概率门槛由盈利与亏损幅度计算。

每秒比较多空edge。至少一个大于0时选择较大者，相等优先多头；输出多头为正值、空头为负值，否则输出0。配置中的0.9/0.99/0.999分位另用于其他信号实验；本报告最终原生E4的`entry_bps=0`、`cost_filter=0`，准入直接依赖正净价值及风险有效性。

### 4.4 同样本预测结果

所有E0/E4模型每方向共用3,719,562个合法开发标签，不因新增因子缺失删除样本。

| group | side | count | positive_rate | logloss | brier | auc |
| --- | --- | --- | --- | --- | --- | --- |
| e0 | 1 | 3719562 | 0.3105266157682007 | 0.6085315860453903 | 0.20944389888738207 | 0.5912654486944724 |
| e4 | 1 | 3719562 | 0.3105266157682007 | 0.6090692284300205 | 0.2096095485508937 | 0.5907073312933323 |
| e0 | -1 | 3719562 | 0.29466318883782555 | 0.599767629574987 | 0.20547845567251588 | 0.5816604314487056 |
| e4 | -1 | 3719562 | 0.29466318883782555 | 0.5995539329622477 | 0.20530744811845308 | 0.5856506993283148 |

LogLoss与Brier越低越好，AUC越高越好。E4空头略改善，多头略退化。按8个UTC周为块重采样1000次，误差差值定义为E4减E0，得到以下95%区间。

| 方向 | 指标 | E4−E0 | 95%下界 | 95%上界 |
| --- | --- | ---: | ---: | ---: |
| -1 | logloss | -0.000213697 | -0.000999854 | 0.000644372 |
| -1 | brier | -0.000171008 | -0.000551126 | 0.000217417 |
| 1 | logloss | 0.000537642 | -0.000038482 | 0.001058364 |
| 1 | brier | 0.000165650 | -0.000068268 | 0.000390026 |

四个区间均跨0。后续策略收益改善主要发生在退出与执行层，同时保留这个概率评价结果。

## 五、开发期怎样一步步走到最终候选

### 5.1 第一步，固定2小时持仓比较E1—E4

统一每笔0.10 BTC、固定7200秒、本金10,000 USDT、最高1倍总敞口；按委托到达时十档盘口逐档执行，Taker每侧5bps并计实际资金费。

| 新增输入 | 往返 | 净收益USDT | 年化 | 最大回撤 | 去最好周USDT |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1长期27列 | 62 | −105.06 | −7.02% | 4.04% | −303.25 |
| E2背离9列 | 41 | 17.34 | 1.20% | 3.60% | −123.55 |
| E3全部36列 | 60 | −12.45 | −0.85% | 5.44% | −207.39 |
| E4精选12列 | 26 | 110.85 | 7.89% | 3.14% | 4.18 |

E4的26笔都是多头。毛收益295.6031 USDT、手续费187.9763 USDT、资金费收入3.2231 USDT，得到净收益110.8499 USDT。最好周贡献约96.2%的净收益，最差月−16.84 USDT。E4在四组新增输入中净收益最高，主要盈利来自最好一周。

### 5.2 第二步，保持模型不变，调整持有与退出

完整批次还对旧E0和E3做方向消融。E4自身四种规则如下，每笔仍为0.10 BTC。

| E4规则 | 净收益USDT | 年化 | 回撤 | 去最好周USDT |
| --- | ---: | ---: | ---: | ---: |
| 固定1800秒 | 109.38 | 7.78% | 2.15% | −1.05 |
| 固定3600秒 | 176.79 | 12.83% | 2.33% | 20.76 |
| 固定7200秒，沿用上一步 | 110.85 | 7.89% | 3.14% | 4.18 |
| policy5风险退出，最长3600秒 | 274.67 | 20.52% | 1.59% | 147.35 |

风险退出版本22笔往返，毛收益约434.90 USDT、手续费161.46 USDT、资金费收入1.23 USDT；最高敞口约0.794倍。按开发期月份分布，3月52.41、4月157.39、5月64.87 USDT。3月与5月为部分月份。

### 5.3 第三步，风险退出下比较数量和最长持有

以下六例每一例都重新执行快照成交和账户现金逻辑。

| 最长持有秒 | 数量BTC | 往返 | 净收益USDT | 年化 | 回撤 | 最高敞口 | 去最好周USDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1800 | 0.125 | 25 | 278.55 | 20.83% | 1.54% | 0.994 | 119.54 |
| 1800 | 0.120 | 25 | 267.41 | 19.93% | 1.48% | 0.955 | 114.76 |
| 3600 | 0.125 | 22 | 343.33 | 26.17% | 1.98% | 0.987 | 184.18 |
| 3600 | 0.120 | 22 | 329.60 | 25.02% | 1.90% | 0.948 | 176.82 |
| 7200 | 0.125 | 19 | 80.27 | 5.66% | 2.90% | 0.973 | −14.07 |
| 7200 | 0.120 | 21 | 160.90 | 11.62% | 2.78% | 0.965 | 70.33 |

3600秒、0.125 BTC版本无入场保证金拒绝，3/4/5月净收益65.51、196.74、81.09 USDT。7200秒、0.125 BTC版本有10次保证金拒绝，0.120 BTC版本无拒绝；两档数量改变了实际交易机会。

本阶段冻结0.125 BTC与最长3600秒，随后研究原生Maker执行。开发Taker版本26.17%的年化尚未达到30%。

### 5.4 E4的其他分支实验

| 分支 | 变更 | E4开发结果 |
| --- | --- | --- |
| 持续确认 | 要求信号持续5/15/60秒后入场 | 净收益235.98 / 169.87 / 138.42 USDT |
| 信号失效退出 | policy4，至少持有300/900秒后按同向信号失效退出，最长3600秒 | 净收益−423.10 / −247.27 USDT |
| 短周期方向确认 | 冻结180秒家族均衡XGBoost同向才准入，保留policy5 | 18往返，净收益175.47 USDT、年化12.73%、去最好周57.96 USDT |

上述均为0.125 BTC、完整53日开发、快照Taker执行及实际资金费。300秒信号失效退出的最高敞口为1.00036倍，同时未满足风险门槛。最终原生E4沿用原净价值准入和policy5。

seed42、seed73、depth2、depth6和compact6属于同系列模型对照。compact6采用另外的精简输入，开发表现和其历史复核记录单列；这里的E4冻结复核始终使用12列、seed17、depth4。

## 六、风险退出和原生订单规则的实际细节

### 6.1 风险尺度怎样算

用决策时刻之前连续600个1秒对数收益计算

`vol_600s_bps = sqrt(Σ r_i²) × 10000`。

该量是600秒已实现波动尺度。代码先滚动600个收益平方求和，再`shift(1)`，要求600个收益完整，覆盖的601个价格在同一质量段。缺失或跨段时风险无效，该秒不准入新仓。

入场风险尺度为`R=max(20,2×vol_600s_bps)`，单位bps。Python将风险序列和冻结净价值预测按同一整秒时间键导出。C++在开仓指令阶段记录待用R，在实际仓位由0变为非0时冻结它，后续持仓不随每秒波动更新R。

### 6.2 何时请求退出

C++用独立账本的实际平均成交成本P0，本地可见Bid作为多头退出估值、Ask作为空头退出估值。设r=可见退出价格/P0，使用

`mark_bps = s×(r−1)×10000 − taker_fee×10000×(1+r)`。

本轮`taker_fee=0.0005`，所以触发估值扣除双边各5bps。实际成交手续费仍按Maker/Taker真实类型记入最终账本。

持仓期间记录`best_mark_bps=max(历史最高mark_bps,当前mark_bps)`，满足任一条件便锁定减仓意图。

1. 止损，`mark_bps ≤ −2R`。
2. 移动止盈，历史最高`mark_bps ≥ 2R`，且当前`mark_bps ≤ 最高mark_bps−R`。
3. 实际持仓达到最长3600秒。
4. 报价窗口结束，进入终端清仓流程。

例如R=20bps，触发止损的位置为−40bps；曾达到至少+40bps后，回撤20bps触发移动止盈。触发基于当时可见的价格及真实成本；退出成交价格随后由订单和市场路径决定。

### 6.3 怎样入场、等待成交与处理退出

空仓、盘口新鲜且非交叉、预测有限、风险有限时，正信号在本地买一挂买单，负信号在卖一挂卖单，GTX避免主动穿价。委托数量0.125 BTC，最大仓位同为0.125 BTC。持仓后停止新开仓，不因预测短暂消失直接结束持仓。

下单前检查所需名义金额及开平仓费用预算与账户权益。活动单、在途请求、部分成交剩余量继续由原有MarketMaker、ExecutionPolicy和hftbacktest-cpp管理。

触发减仓后，多头在卖一挂Maker卖单，空头在买一挂Maker买单。持续使用实际剩余仓位，退出最多等待一个30秒TTL；超时先撤销剩余挂单，再以IOC保护限价清仓，Taker滑点保护为1 Tick。终端清仓使用独立强制退出路径。部分成交不会解除剩余仓位的减仓意图。

### 6.4 配置参数与频率

| 参数 | 值 |
| --- | --- |
| 执行模式 | directional_maker_risk |
| 队列模型 | risk_adverse |
| Tick / Lot | 0.1 USDT / 0.001 BTC |
| order_quantity / max_position | 0.125 / 0.125 BTC |
| 策略与风控检查 | 100ms |
| 冻结预测序列 | 1s |
| 预测延迟 | 1ms |
| 委托到达 / 回报延迟 | 1ms / 0.5ms |
| 行情可见延迟 | 交易所时间加20ms |
| TTL | 30秒 |
| max_hold | 3600秒 |
| Maker / Taker费率 | 单侧2bps / 5bps |
| entry_bps / exit_bps | 0 / 0 |
| cost_filter | 0 |
| inventory_skew_bps / offset_ticks | 0 / 0 |
| minimum_notional_usdt | 100 |
| taker_slippage_ticks | 1 |

本轮预测由Python离线生成，C++按时钟读取带预测延迟的信号，并逐事件模拟市场、订单和实际成交。本次C++回放读取Python预先生成的预测。C++重算192列特征并运行XGBoost的实时推理尚未验证。

## 七、原生开发期结果

### 7.1 账户汇总

| 指标 | val_dev |
| --- | ---: |
| 完整区间 | 2026-03-23—2026-05-14，53日 |
| 起始本金 | 10,000 USDT |
| 费用前收益 | 488.2287 USDT |
| 手续费 | 78.9585 USDT |
| 资金费净收入 | 1.5408 USDT |
| 费用后净收益 | 410.8111 USDT |
| 期末权益 | 10,410.8111 USDT |
| 53日账户收益 | 4.1081% |
| 复合年化 | 31.9510% |
| 最大回撤 | 1.8168%，对应184.4178 USDT |
| 峰值总敞口 | 0.980414倍 |
| 实际成交 / 往返交易 | 104 / 22 |
| Maker / Taker成交金额 | 394,792.3703 / 0 USDT |
| 去最好一天净收益 | 303.2311 USDT |
| 去最好周净收益 | 238.7002 USDT |
| 原生逻辑事件 | 3,121,250,191 |

账本满足`净收益=费用前收益−手续费−资金费支付`。资金费支付为负表示净收入。全部7个区块结束时仓位为0且账本一致。

快照同规模、同风险退出的净收益343.33 USDT；原生Maker为410.81 USDT。两条执行路径都实际重算，各自依据成交、费用和资金费形成结果，差异还包含成交时机和订单路径变化。

### 7.2 七个连续回放区块

| id | first_date | last_date | initial_account_equity_usdt | ending_account_equity_usdt | net_pnl_usdt | fills | final_position_btc | ledger_reconciled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_dev_00 | 2026-03-23 | 2026-03-24 | 10000.0 | 10057.611165 | 57.611165000001094 | 6 | 0 | true |
| val_dev_01 | 2026-03-27 | 2026-03-30 | 10057.611165 | 10091.15448532 | 33.543320319999474 | 14 | 0 | true |
| val_dev_02 | 2026-04-02 | 2026-04-03 | 10091.15448532 | 10108.87420282 | 17.719717500000726 | 14 | 0 | true |
| val_dev_03 | 2026-04-12 | 2026-04-15 | 10108.87420282 | 10141.070623853246 | 32.19642103324684 | 17 | 0 | true |
| val_dev_04 | 2026-04-17 | 2026-04-24 | 10141.070623853246 | 10389.38805752524 | 248.317433671995 | 31 | 0 | true |
| val_dev_05 | 2026-04-27 | 2026-04-28 | 10389.38805752524 | 10318.32922002524 | -71.05883750000037 | 2 | 0 | true |
| val_dev_06 | 2026-05-05 | 2026-05-07 | 10318.32922002524 | 10410.811057525241 | 92.48183750000109 | 20 | 0 | true |

这些区块覆盖25个必要行情日。全无潜在入场、且此前仓位已平的日期可跳过事件重放，仍保留在53日日历收益分母。区块由全部非零准入信号及最长持有后的必要日期决定，不读取交易利润挑选日期。

块内事件连续处理，块间空仓并续接现金。区间规划对分区开始的前120秒和末尾最长持有加20秒限制开仓。最大持有3600秒时，信号日后的必要日期也纳入行情准备。

### 7.3 月度分布

| split | month | net_pnl_usdt |
| --- | --- | --- |
| val_dev | 2026-03 | 91.15448531999937 |
| val_dev | 2026-04 | 227.17473470524055 |
| val_dev | 2026-05 | 92.48183750000135 |

4月贡献约55%的开发净收益。4月27—28日区块亏损71.06 USDT已计入。开发年化超过30%、峰值敞口小于1、去最好周收益为正，满足继续冻结历史复核的条件。

## 八、参数冻结后的历史复核

### 8.1 进入复核前冻结了什么

复核从10,000 USDT重新起算，固定2026-05-15至07-06共53日。保留12列及顺序、fit预处理、多空保存权重、概率校准及净价值映射、净价值准入、0.125 BTC、同一C++风险退出实现、3600秒持有上限、30秒Maker退出等待、费用及资金费口径。

该复核不重训模型，也不按复核收益更改阈值。策略需要覆盖完整账户日期，原生区块均结束、仓位为0、独立账本一致，门槛为年化>30%、最高敞口≤1、去最好周收益为正，并单独检查月份集中度。

### 8.2 最终账户结果

| 指标 | val_select |
| --- | ---: |
| 费用前收益 | 365.3303 USDT |
| 手续费 | 74.4436 USDT |
| 资金费净收入 | 0.1505 USDT |
| 费用后净收益 | 291.0373 USDT |
| 期末权益 | 10,291.0373 USDT |
| 53日账户收益 | 2.9104% |
| 复合年化 | 21.8438% |
| 最大回撤 | 2.3922%，对应242.3411 USDT |
| 峰值总敞口 | 0.880202倍 |
| 实际成交 / 往返交易 | 86 / 26 |
| Maker / Taker成交金额 | 372,217.7629 / 0 USDT |
| 去最好一天净收益 | 167.5530 USDT |
| 去最好周净收益 | 74.9703 USDT |
| 原生逻辑事件 | 3,730,703,886 |
| 缺失区块 | 0 |
| 账本与期末仓位 | 全部一致、全部0 |

### 8.3 四个区块、月份和周度收益

| id | first_date | last_date | initial_account_equity_usdt | ending_account_equity_usdt | net_pnl_usdt | fills | final_position_btc | ledger_reconciled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_select_00 | 2026-06-02 | 2026-06-10 | 10000.0 | 10248.26778508385 | 248.2677850838506 | 62 | 0 | true |
| val_select_01 | 2026-06-17 | 2026-06-18 | 10248.26778508385 | 10246.23984508385 | -2.02794 | 3 | 0 | true |
| val_select_02 | 2026-06-24 | 2026-06-30 | 10246.23984508385 | 10343.390717583849 | 97.15087249999873 | 19 | 0 | true |
| val_select_03 | 2026-07-06 | 2026-07-06 | 10343.390717583849 | 10291.037275083849 | -52.353442499999815 | 2 | 0 | true |

| split | month | net_pnl_usdt |
| --- | --- | --- |
| val_select | 2026-05 | 0.0 |
| val_select | 2026-06 | 343.3907175838485 |
| val_select | 2026-07 | -52.3534424999998 |

5月无入场，6月盈利343.3907 USDT，7月亏损52.3534 USDT。五月及七月为53日区间中的部分月份。复核只需要19个行情日重放，完整53日均纳入账户汇总。

| split | week | net_pnl_usdt |
| --- | --- | --- |
| val_select | 2026-05-11 | 0.0 |
| val_select | 2026-05-18 | 0.0 |
| val_select | 2026-05-25 | 0.0 |
| val_select | 2026-06-01 | 32.20080802476514 |
| val_select | 2026-06-08 | 216.06697705908482 |
| val_select | 2026-06-15 | -2.027939999999944 |
| val_select | 2026-06-22 | 68.99851499999932 |
| val_select | 2026-06-29 | 28.152357499999198 |
| val_select | 2026-07-06 | -52.3534424999998 |

最大盈利周从2026-06-08开始，贡献216.0670 USDT，约占总净收益74.2%。去掉该周仍盈利74.9703 USDT。最大盈利日为6月26日，+123.4842 USDT；7月6日−52.3534 USDT。

最终判定是正的费用后收益，最高敞口与对账通过，年化21.84%未达到30%目标。收益集中在6月，7月转负。该冻结候选保留为有完整原生账本的研究结果。

## 九、收益、回撤与执行验证的统一口径

年化使用`(期末权益/起始权益)^(365/53)−1`，无成交日纳入53日分母。年化是该区间的复合折算值。

风险汇总使用保存的原生权益路径并计实际资金费。最大回撤比例按权益峰值计算，因此USDT回撤除以10,000与回撤百分比可能有差异。峰值总敞口按当时持仓名义价值与权益计算。去最好日、去最好周根据完整日级账户净收益计算。

原生成交按独立账本核对持仓、现金、已实现盈亏、平均成本、累计费用和账户状态。区块汇总净收益续接前一区块现金；资金费由实际持仓跨结算时刻，结合官方费率和标记价格计算，随后反映到账户权益和风险汇总。

回放启用了risk_adverse队列及配置延迟，实验中的Maker成交金额为模拟结果。最终开发和复核都未出现实际Taker成交，虽然超时IOC机制已经配置。结果同时依赖冻结信号与队列、费用、延迟等执行假设。

## 十、阅读与复现入口

### 10.1 关键代码

| 环节 | 文件及函数 |
| --- | --- |
| 秒级统计与36列公式 | `research/mm_research/factors/gen041.py`，`second_statistics`、`attach_book_statistics`、`factor_expressions` |
| 阈值、生产、输入缓存 | `research/mm_research/models/gen041.py`，`prepare_small_threshold`、`produce_factors`、`cache_model_inputs` |
| 12列选择 | `research/mm_research/models/gen041_selection.py`，`prepare_selected_inputs`、`select_from_daily` |
| 标签、训练、校准 | `research/mm_research/models/cost_classifier.py`，`net_labels`、`train_cost_classifier`、`probability_calibration` |
| 同样本模型评价 | `research/mm_research/evaluation/fee_models.py`，`evaluate_fee_models` |
| 快照账户实验 | `research/mm_research/strategies/account_sizing.py`及`multihorizon/inventory.py` |
| 事前600秒风险 | `research/mm_research/strategies/stable_strategy.py`，`visible_management` |
| 原生信号、区块和报告 | `research/mm_research/strategies/cost_native.py`，`prepare_risk_signals`、`plan_native`、`replay_native_block`、`report_native` |
| 原生风险和订单决策 | `src/mm/strategies/execution_policy.cpp`，`risk_exit_trigger`、`ExecutionPolicy::decide`、`maker_close` |

### 10.2 YAML配置分别负责什么

每份YAML声明本阶段的路径和规则，由对应Python入口读取。在项目根目录执行时，相对路径以项目根解析。

| YAML | 输入 | 输出与用途 |
| --- | --- | --- |
| `config/factors/gen041_20261114.yaml` | 标准成交、100ms Features、日期行集 | 阈值、36列全日期因子及完成标记 |
| `config/models/gen041_inputs_20261114.yaml` | 因子、原慢状态与冻结行集 | E1/E2/E3标准化输入缓存 |
| `config/models/gen041_selection_20261114.yaml` | E3缓存、fit标签 | 12列清单、fit排名、E4输入缓存 |
| `config/models/cost_gen041_e4_20261114.yaml` | E4缓存、原146列、现货16列、市场矩阵 | 多空权重、校准参数、逐秒净价值 |
| `config/evaluation/gen041_e4_20261114.yaml` | E0/E4预测、同样本标签 | 概率指标、逐日配对误差与周bootstrap |
| `config/strategy/account_gen041_e4_20261114.yaml` | E4净价值、十档快照 | 固定7200秒开发账本 |
| `config/strategy/gen041_holding_20261114.yaml` | 冻结净价值、快照、风险 | 方向与持有期对照 |
| `config/strategy/gen041_risk_size_20261114.yaml` | 同上 | 两档数量、三档持有上限的开发账本 |
| `config/strategy/gen041_native_risk_20261114.yaml` | E4净价值、风险、真实L2、官方成交 | 原生开发区块、成交与账户汇总 |
| `config/strategy/gen041_native_review_20261114.yaml` | 同模型和规则的复核输入 | 原生冻结历史复核账本 |
| `config/research/gen041_native_risk_20261114.yaml`及对应review文件 | 上述阶段配置与任务依赖 | 后台队列、资源约束、失败状态与日志 |

### 10.3 复现命令

以下在服务器项目根执行，复用项目环境与已准备的上游数据。原生数据准备需要有效下载授权；历史任务使用过的私有凭证已清理。完整重跑应设置独立输出目录，现有完成标记会使生产入口复用完成产物。

```bash
PYTHONPATH=research .venv-models/bin/python scripts/run_gen041.py config/factors/gen041_20261114.yaml threshold
PYTHONPATH=research .venv-models/bin/python scripts/run_gen041.py config/factors/gen041_20261114.yaml factors --workers 2
PYTHONPATH=research .venv-models/bin/python scripts/run_gen041.py config/models/gen041_inputs_20261114.yaml cache
PYTHONPATH=research .venv-models/bin/python scripts/run_gen041.py config/models/gen041_selection_20261114.yaml select
PYTHONPATH=research .venv-models/bin/python scripts/run_cost_classifier.py config/models/cost_gen041_e4_20261114.yaml train --horizon 1800 --side 1 --seed 17
PYTHONPATH=research .venv-models/bin/python scripts/run_cost_classifier.py config/models/cost_gen041_e4_20261114.yaml train --horizon 1800 --side -1 --seed 17
PYTHONPATH=research .venv-models/bin/python scripts/evaluate_fee_models.py config/evaluation/gen041_e4_20261114.yaml
PYTHONPATH=research .venv-models/bin/python scripts/run_account_sizing.py config/strategy/account_gen041_e4_20261114.yaml screen --group e4
PYTHONPATH=research .venv-models/bin/python scripts/run_research_background.py config/research/gen041_holding_20261114.yaml
PYTHONPATH=research .venv-models/bin/python scripts/run_research_background.py config/research/gen041_risk_size_20261114.yaml
PYTHONPATH=research .venv-models/bin/python scripts/run_research_background.py config/research/gen041_native_risk_20261114.yaml
PYTHONPATH=research .venv-models/bin/python scripts/run_research_background.py config/research/gen041_native_review_20261114.yaml
```

原生后台配置依次执行风险信号准备、区块规划、构建、行情准备、顺序回放和报告。行情日期可并行准备，账户区块按现金续接顺序执行，单项失败由队列记录并重试。原生风险退出没有使用预先冻结的退出成交时间表。

### 10.4 原始资料与结果定位

- [本地代码仓库](<../../..>)
- [gen041_selected_inputs_20261114_v1 / selection.json](<evidence/gen041_selected_inputs_20261114_v1/selection.json>)
- [gen041_selected_inputs_20261114_v1 / fit_ranking.csv](<evidence/gen041_selected_inputs_20261114_v1/fit_ranking.csv>)
- [cost_gen041_e4_20261114_v1 / config.json](<evidence/cost_gen041_e4_20261114_v1/config.json>)
- [gen041_e4_evaluation_20261114_v1 / completed.json](<evidence/gen041_e4_evaluation_20261114_v1/completed.json>)
- [gen041_native_risk_20261114_v1 / summary/native_account.json](<evidence/gen041_native_risk_20261114_v1/summary/native_account.json>)
- [gen041_native_review_20261114_v1 / summary/native_account.json](<evidence/gen041_native_review_20261114_v1/summary/native_account.json>)
- [gen041_native_review_20261114_v1 / summary/blocks.csv](<evidence/gen041_native_review_20261114_v1/summary/blocks.csv>)
- [19_Gen041长周期微观结构实施_20261114](../gen041_20261114/implementation.md)
- [20_Gen041模型与收益结果_20261114](../gen041_20261114/results.md)
- [21_Gen041方向与持仓实验_20261114](../gen041_20261114/holding_plan.md)
- [22_Gen041风险退出规模试验_20261114](../gen041_20261114/risk_size.md)
- [26_模型输出与失败场景分析_20261114](../gen041_20261114/model_failure_research.md)
- [32_信号失效退出对照_20261114](../gen041_20261114/signal_exit.md)
- [34_短周期新模型确认长周期入场_20261114](../gen041_20261114/short_confirmation.md)
- [37_E4原生开发结果与复核冻结](../gen041_20261114/native_risk_result.md)
- [39_E4风险退出冻结历史复核最终结果_20261114](../gen041_20261114/native_risk_result.md)

原生结果目录下的`summary/native_account.json`是完整账户结果，`native_risk.json`是风险汇总，`totals.csv`拆分毛收益、手续费和资金费，`blocks.csv`用于区块现金续接及对账，`native_daily.csv`、`weekly.csv`、`monthly.csv`用于日历和集中度。订单与成交明细留在服务器对应`runs/`、`blocks/`和`block_results/`。本地快照同步了配置与小型结果，完整逐笔流水从服务器原目录读取。

## 十一、从整个过程可以得出的结论

E4的输入筛选减少了新增因子冗余，固定持有对照中费用后收益优于其余三组，但同样本概率误差增量的周抽样区间跨零。退出规则把原0.10 BTC版本净收益从固定2小时的110.85 USDT提高到风险退出1小时的274.67 USDT，并降低收益对最好周的依赖。

0.125 BTC原生Maker开发验证获得410.81 USDT净收益、31.95%年化，冻结历史复核为291.04 USDT、21.84%，回撤由1.82%升到2.39%。两阶段账本完整，复核收益集中在6月。最终候选的复核年化低于预设30%门槛。

E4已完成数据准备、因子筛选、模型训练、信号生成及原生交易回放，并保存了成交和账户记录。实时推理、队列和延迟敏感性仍需另做实验。

## 第三方引擎署名

原生回放使用shenron0101维护的hftbacktest-cpp v0.1.0，固定提交为`e0d2f7887c68755b220d9e69425d61d27e5b6432`。该项目是nkaz001及其贡献者开发的HftBacktest的独立C++20移植，采用MIT许可。订单簿、事件调度、延迟、队列、成交和订单状态由引擎处理。本项目编写数据与因子计算、模型训练、策略决策、风险退出和独立账本。[hftbacktest-cpp](https://github.com/shenron0101/hftbacktest-cpp) · [HftBacktest](https://github.com/nkaz001/hftbacktest) · [许可证](https://github.com/shenron0101/hftbacktest-cpp/blob/e0d2f7887c68755b220d9e69425d61d27e5b6432/LICENSE) · [项目来源](https://github.com/shenron0101/hftbacktest-cpp/blob/e0d2f7887c68755b220d9e69425d61d27e5b6432/NOTICE)
