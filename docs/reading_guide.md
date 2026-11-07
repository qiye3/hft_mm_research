# 2026-11-05 — 项目阅读指南

## 一、先理解这个项目要回答什么

这个项目研究 Binance 的 BTCUSDT 永续合约。BTC 是交易的资产，USDT 用来表示价格和收益。永续合约可以持有多头或空头，买入会增加净持仓，卖出会减少净持仓。这里的“库存”就是做市策略当前持有的净仓位，正数表示多头，负数表示空头。

做市的基本动作是在市场两边挂出限价单。假设当前价格在 60000 USDT 附近，策略挂出 59998 的买单和 60002 的卖单，希望先后成交后获得价格差。但是，买单成交之后，市场可能继续下跌，卖单也可能一直无法成交。因此，阅读这个项目时，可以始终围绕一个问题展开。**从看到行情、决定报价，到收到成交和计算收益，程序到底发生了什么。**

历史回测把过去发生的行情按时间重新交给程序。程序在当时能够看到的信息下作出决策，回测框架根据历史行情、排队规则和延迟假设模拟订单的后续变化。历史数据提供市场已经发生的事情，策略决定自己的动作，框架负责推进时间和处理这些动作。

项目目前有两条运行流程。M0 把行情读进来，检查盘口是否正确，并测量回放速度。M1 在同一批行情上加入固定宽度的双边报价、成交记录、仓位限制和收益核对。后续的价格预测和报价改进，可以建立在这两条流程之上。

## 二、认识目录，再开始看代码

先打开项目根目录的 README，完成构建和合成行情运行。`CMakeLists.txt` 说明哪些源码组成可执行程序，构建后的 `replay_probe` 用于观察行情，`baseline_mm` 用于运行做市基线，`benchmark_replay` 用于测量回放性能。安装依赖和运行命令集中在 README 的 Quickstart 部分。 [项目介绍与 Quickstart](../README.md)

`src/apps/` 保存这两个主要程序的启动文件。它们读取配置、调用运行函数，并把运行错误显示出来。`src/mm/` 保存行情观察、回放、报价和交易记录的实现。相同名字的 `.hpp` 和 `.cpp` 放在一起，前者介绍这个模块提供哪些类型和函数，后者给出具体实现。阅读一个模块时，先看头文件中的类和函数，再沿着实际调用去看实现，会比较容易建立整体认识。

`config/` 保存参数，例如数据从哪里读取、报价距离中间价多远、每单数量和最大持仓。`scripts/` 保存数据下载、转换、检查和实验分析脚本。`tests/` 通过小规模、结果明确的例子检查程序行为。`third_party/hftbacktest-cpp/` 是固定版本的第三方框架，本项目通过它维护盘口、处理委托并模拟成交。

`docs/reading_guide.md` 是整体阅读入口，`docs/implementation/` 解释引擎和做市代码如何工作，`docs/reports/` 按阶段和数据日期保存实测报告。可以先读本指南，再按需要查看实现细节和实验结果。

`workspace/` 保存本机运行时产生的内容。其中 `data/` 放原始行情和转换后的行情，`build/` 放编译产物，`results/` 放回放结果与交易记录。`results/synthetic/` 对应合成行情，`results/m0/BTCUSDT_2025-02-01/` 对应单日回放，`results/m1/BTCUSDT_2025-02-01/` 对应做市实验，`archive/` 保存历史附件。运行数据和结果保存在本地，Git 保存源码、配置和项目文档。

## 三、第一遍阅读，从一段合成行情走到盘口

建议先打开 `src/apps/replay_probe.cpp`。这个文件很短，主流程依次调用 `load_config`、`run_replay` 和 `write_report`。它对应三个容易理解的动作，读取参数、运行行情回放、保存结果。此时可以对照 `config/replay/synthetic.yaml`，其中 `source: synthetic` 表示使用程序自带的少量合成行情。 [回放入口](../src/apps/replay_probe.cpp) · [合成行情配置](../config/replay/synthetic.yaml)

接着打开 `src/mm/replay.hpp`。`ReplayConfig` 保存输入参数，`ReplayResult` 保存回放结果。然后到 `replay.cpp` 找到 `synthetic_events`，看这组小行情包含哪些价格、数量和时间，再从 `run_replay` 进入 `run_memory_replay` 和内部的 `replay_reader`。后者把行情读取器、本地处理器和交易所处理器交给 `hbt::Backtest`，由框架推进整段行情。 [回放接口](../src/mm/replay.hpp) · [回放实现](../src/mm/replay.cpp)

这里需要先理解“盘口”。盘口记录不同价格上有多少买单和卖单。买方最高的价格叫买一，卖方最低的价格叫卖一。两者的平均值叫中间价，也就是代码中的 `mid_price`。卖一减去买一叫价差，也就是 `spread`。例如买一为 59999.9、卖一为 60000.1，中间价就是 60000，价差就是 0.2 USDT。前五档或前十档，是从各侧最优价格向外取出的五个或十个价位。

再读 `MarketObserver::on_event` 和 `MarketObserver::capture`。框架先根据行情更新盘口，观察器再读取更新后的状态。`on_event` 统计盘口更新和成交等事件，`capture` 提取当前前五档或前十档，并计算中间价和价差。这样，行情的输入、盘口的变化和结果中的统计就连起来了。 [观察器接口](../src/mm/market_observer.hpp) · [观察器实现](../src/mm/market_observer.cpp)

默认每 100 毫秒提取一次完整的前十档视图，事件计数仍然逐条进行。更深的价格档位也会保留在框架的完整盘口中，因为前面的挂单消失后，原来的第十一档可能进入前十档。理解这一点后，可以阅读 `test_depth_promotion_and_configuration`，看测试如何验证这种变化。 [行情观察测试](../tests/test_market_observer.cpp)

## 四、再看真实行情如何进入同一套回放流程

数据说明整理了网盘文件、原始字段、下载转换方式和整日验证结果。服务器配置、费用，以及 GPU 与 Parquet 的选择分别保存在另外两份说明中。 [研究数据说明](implementation/data_requirements.md) · [服务器配置与费用](implementation/server_recommendation.md) · [GPU 与 Parquet 选择](implementation/gpu_and_parquet.md)

M0的历史样本来自Tardis。盘口文件包含快照和增量更新。快照描述某个时刻的盘口，增量描述之后某个价位的数量变化，数量为零表示删除该价位。逐笔成交文件记录市场中已经发生的买卖成交。转换脚本将这些信息整理成框架能够读取的事件。

阅读 `scripts/convert_tardis.py` 时，可以先看 `convert` 的整体顺序，再进入 `append_source` 和 `write_routed_shards`。前者读取原始 CSV 记录，检查价格和数量步长，处理快照与更新。后者整理两侧的时间顺序，并把事件写成多个 NPZ 文件。NPZ 是保存数组的压缩文件，拆成多个分片方便框架逐批读取。旁边的 `manifest.json` 是数据清单，记录分片位置、事件数量和检查结果。 [数据转换脚本](../scripts/convert_tardis.py)

每条行情有两个重要时间。`exch_ts` 表示事件在交易所发生的时间，`local_ts` 表示采集端收到消息的时间。假设交易所在某一秒的第 100 毫秒发生价格变化，本地在第 102 毫秒才收到消息，那么这两毫秒之内，策略仍然只能使用之前看到的行情。

框架里的 Exchange Processor 根据交易所时间更新交易所侧状态，Local Processor 根据本地接收时间更新策略可以看到的状态。转换器为这两侧分别安排记录顺序，所以同一个市场事件会产生分别送往两侧的记录。报告中的“标准化行数”与“逻辑事件数量”因此可能不同。框架随后分别处理这两种记录，使策略按本地时钟获得信息。

有了真实数据后，`config/replay/btcusdt.yaml` 将 `source` 指向数据清单。程序从文件建立读取器，之后仍然进入前面读过的 `replay_reader`。可以把真实行情和合成行情理解成同一条回放流程的两种输入。更多接口名称和时间处理细节集中在引擎笔记中。 [真实行情配置](../config/replay/btcusdt.yaml) · [引擎接口与事件流程](implementation/engine_notes.md)

当前探索性研究使用`research/mm_research/`。先从`data/standard.py`看官方成交与25档快照的标准化，再从`features/base.py`、`snapshot.py`、`trades.py`看100ms对齐。随后查看`features/transform.py`的分组定义、`targets/midprice.py`的未来标签，以及`data/dataset.py`的增量生产和按组读取。 [研究包指南](implementation/research_package.md)

## 五、第二遍阅读，跟着一笔委托理解做市

如果当前重点是因子研究，可以先转到M2-A。当前离线入口读取标准Parquet，在Features内部建立100ms网格，生成分组特征和独立多周期标签。先读数据集指南，再从`build_features_targets.py`进入`run_dataset_production()`。C++事件OFI路径可从`extract_research`跟到`ResearchSampler`。 [M2-A 数据集指南](implementation/m2_a_dataset.md)

先打开 `src/apps/baseline_mm.cpp`，它读取策略配置后调用 `run_baseline`。再对照 `config/strategy/baseline_mm.yaml` 阅读 `TradingConfig`，认识策略使用的参数。默认价格步长 `tick_size` 为 0.1 USDT，表示一 tick 对应 0.1 USDT。报价半宽 `half_width_ticks` 为 20，表示买价和卖价各距离中间价约 2 USDT。每笔订单为 0.001 BTC，最大净持仓绝对值为 0.01 BTC。 [做市入口](../src/apps/baseline_mm.cpp) · [做市配置](../config/strategy/baseline_mm.yaml) · [参数定义](../src/mm/trade_recorder.hpp)

然后阅读 `src/mm/market_maker.cpp` 中接收行情读取器的那个 `run_baseline`。它先建立交易记录器、回测环境和报价对象，接着反复推进模拟时间并调用 `MarketMaker::on_decision`。默认决策间隔是 100 毫秒。在这段时间内发生的行情和订单回报，由框架依次处理，策略在下一个决策时刻使用更新后的本地状态。 [报价与回测主流程](../src/mm/market_maker.cpp)

`fixed_quotes` 负责根据中间价计算双边目标价格。价格要落在合法的 tick 上，因此买价向下取整，卖价向上取整。`on_decision` 负责判断本次是否需要更新目标报价。默认模式按固定时间检查，也可以在参考中间价累计变化达到指定 tick 数时更新。`quote_side` 则逐侧检查已有订单，决定继续保留、请求撤销或提交新订单。

这里使用的 GTX 表示订单只允许挂在盘口上等待成交，也就是只做 Maker。如果订单到达交易所时已经可以直接与对手盘成交，框架会让它失效。这样可以观察主动提供买卖报价的效果。订单在队列中的位置如何影响成交，由框架选定的排队模型处理。

提交买单之后，先有请求发往交易所，再有接受、成交或其他状态变化的消息返回本地。配置中的 `entry_latency_ns` 和 `response_latency_ns` 分别控制这两个方向的模拟延迟，单位是纳秒。默认分别为 1 毫秒和 0.5 毫秒。框架通过这些时间安排，模拟发出请求与收到结果之间的等待过程。

撤单也需要等待。假设当前已持有 0.009 BTC，还有一笔剩余数量为 0.001 BTC 的买单，那么它完全成交后，持仓就会达到 0.01 BTC 上限。即使策略已经发出撤单请求，在收到结果前，这笔买单仍可能成交，因此仍然要占用买方的仓位预算。`reservations` 统计的就是这些尚未终结订单的剩余数量。读 `quote_side` 时，可以重点追踪“已有订单是什么状态、还可能成交多少、什么时候允许补单”这三个问题。

理解了策略动作后，再读 `src/mm/detail/trading_engine.hpp`。其中 `TradingEngine` 把第三方框架的本地处理器、交易所处理器和订单通信队列组装到一起。`RecordingLocal` 在本地收到回报时保存原始消息，交给框架更新状态，再逐笔交给交易记录器。这里还处理了固定版本框架中一种撤单拒绝消息携带旧成交字段的情况，细节和对应测试见 M1 笔记。 [引擎接入代码](../src/mm/detail/trading_engine.hpp) · [M1 实现笔记](implementation/m1_notes.md)

## 六、第三遍阅读，理解成交如何变成收益

现在可以打开 `src/mm/trade_recorder.cpp`。从 `response` 进入 `apply_fill`，跟随一条真实成交回报，看持仓、买卖现金流和手续费如何变化。买入 0.001 BTC，净持仓增加 0.001；卖出相同数量，净持仓减少 0.001。记录器使用平均持仓成本，计算减仓时已经确定的收益，以及剩余持仓随着市场价格变化产生的浮动盈亏。 [交易记录与收益计算](../src/mm/trade_recorder.cpp)

沿用 59998 买入、60002 卖出的例子，数量为 0.001 BTC。买卖一轮的毛收益是 4 × 0.001，即 0.004 USDT。按项目默认的 Maker 费率 0.0002 计算，买入手续费为 0.0119996 USDT，卖出手续费为 0.0120004 USDT，两次合计 0.024 USDT。因此，即使顺利赚到了这段买卖价差，扣费后的结果仍为 -0.020 USDT。这个例子可以帮助理解报价宽度、成交次数和成本为什么要放在一起观察。

如果只有买单成交，剩余持仓就要按当前中间价估值。此时账户收益同时受已完成交易和剩余仓位价格变化影响。代码使用以下两种方式表达同一笔净收益，`PnL` 就是 profit and loss，也就是盈亏。

```text
净收益 = 累计卖出收入 − 累计买入支出 + 当前持仓 × 当前中间价 − 累计手续费
净收益 = 已实现毛收益 + 剩余持仓浮动盈亏 − 累计手续费
账户权益 = 初始资金 + 净收益
```

`reconcile` 把记录器逐笔累加的持仓、费用和成交量等，与框架维护的账户状态核对。`checkpoint` 在决策点保存权益观察值，成交发生时也会保存一次。`finish` 汇总整个实验。行情结束前，策略会停止新报价并请求撤单，随后处理剩余回报，最后把仍然持有的仓位按最后有效中间价计入收益。

查看一个实验结果时，建议先打开 `summary.json`，了解最终持仓、费用和净收益。然后从 `fills.csv` 找一笔成交，用订单编号到 `orders.csv` 中追踪提交、撤单请求和回报，再到 `equity.csv` 查看当时的持仓和权益。这样读，文件中的数字就能与前面读过的函数逐一对应。文件位于 `workspace/results/m1/BTCUSDT_2025-02-01/<实验名称>/`。

## 七、用小测试和单日实验巩固理解

测试里的行情规模小，预期结果明确，适合边读边手算。先读 `tests/test_market_observer.cpp`，观察买一卖一如何变化、深层价格如何进入前十档，以及两个时间顺序如何分别处理。然后读 `tests/test_market_maker.cpp` 中的 `test_quotes_and_round_trip`，追踪一次买入和卖出的收益。最后看 `test_partial_fills_and_limits` 和 `test_cancel_in_flight`，理解一笔订单分多次成交，以及撤单等待期间为什么仍需保留风险预算。 [行情测试](../tests/test_market_observer.cpp) · [做市测试](../tests/test_market_maker.cpp)

理解这些小例子后，再读 `scripts/analyze_baseline.py`。`CASES` 定义了单日对照实验，A 改变报价宽度，B 改变持仓上限，C 改变报价更新规则。`run_experiments` 依次调用同一个 C++ 程序，`audit_orders`、`audit_fills` 和 `audit_equity` 从导出的文件重新计算并核对交易结果，`analyze_run` 汇总每组指标。 [实验运行与分析脚本](../scripts/analyze_baseline.py)

报告中的一秒 markout，描述成交后约一秒的中间价相对成交价格移动了多少，并按买卖方向统一正负。对买单来说，之后的中间价高于买入价，对这个指标有利；对卖单来说，之后的中间价低于卖出价更有利。项目从本地收到成交回报的时刻开始计时，使用之后的权益观察点近似观察价格。它帮助检查成交后的价格走势，最终交易收益仍要结合后续成交、库存和手续费计算。实际实验比较与详细口径见单日报告。 [M1 单日实验报告](reports/m1_2025-02-01/report.md)

另外，M0 性能报告衡量的是程序回放行情的速度。每秒处理事件数越高，表示同样的计算时间能够处理更多行情。p50、p95 和 p99 表示采样耗时的分位数，例如 p99 表示约 99% 的样本耗时处于该数值以内。阅读时要同时看测量范围，报告中的局部耗时包含本地行情处理和观察，整体回放耗时还包含文件读取、解压和其他处理。 [M0 单日性能报告](reports/m0_2025-02-01/report.md)

## 八、最后再进入第三方框架

读到这里，再进入 `third_party/hftbacktest-cpp/` 会更容易把接口和实际用途联系起来。可以先看 `include/hbt/types.hpp` 的事件与订单类型，再看 `depth/hashmap_depth.hpp` 如何保存价格档位。接着从 `backtest/backtest.hpp` 的 `elapse` 和 `close` 进入时间推进过程，结合 `backtest/local.hpp` 和 `backtest/exchange.hpp` 阅读本地状态、交易所状态和成交处理。这些头文件都位于框架的 `include/hbt/` 下。

如果想理解“挂在同一价格的订单，为什么有的先成交、有的后成交”，再读 `models/queue.hpp`。如果想理解成交后仓位、现金流和费用如何变化，再读 `backtest/state.hpp`。每次带着项目中已经遇到的一个问题进入框架，可以逐步把策略调用、框架处理和输出结果连起来。具体源码位置和接口索引见引擎笔记。 [引擎源码阅读索引](implementation/engine_notes.md)

完整阅读可以分成三轮。第一轮跟随合成行情，读懂盘口和两个时间。第二轮跟随一笔委托，读懂报价、成交、撤单等待和库存变化。第三轮从成交记录重算收益，再比较单日实验。当前实验采用配置中的费率和延迟假设；评估完整的合约账户收益时，还需要结合资金费率、保证金和强平规则。相关参数与历史交易规则的获取情况记录在现有项目文档中。 [实验参数](../config/strategy/baseline_mm.yaml) · [历史合约规则获取情况](reports/m0_2025-02-01/report.md#未取得的合约元数据)

## 当前研究数据入口

Data使用`config/data/standardize_30d.yaml`，入口为`scripts/build_standard_data.py`。Features/Targets使用`config/features/build_30d.yaml`，入口为`scripts/build_features_targets.py`。

标准文件保存到`workspace/data/standard/market_v2/{snapshots,trades}/YYYY-MM-DD.parquet`。base与衍生组共同写入`features/market_columns_v1/特征名.parquet`，标签位于`labels/market_mid_columns_v1/horizon=周期/YYYY-MM-DD.parquet`。

`features/transform.py`注册base、book、trades、returns四组，发布时每个变量保存为一个全日期文件。当前可见时间为交易所时间加20ms，滚动窗口跨午夜及缺失位置连续计算。研究读取使用`load_samples()`或`load_xy()`，明确选择特征列与日期。

Gen001从`factors/gen001.py`读取六个因子的公式，经`factors/dataset.py`保存到`workspace/data/factors/gen001_v1/`。评估入口是`scripts/evaluate_factors.py`，指标与报告分别位于`evaluation/metrics.py`和`evaluation/report.py`。报告保存到`workspace/results/evaluation/gen001_v1/report.html`。

完整命令见 [研究包指南](implementation/research_package.md)。历史报告按对应的数据版本、时钟和质量规则阅读。
