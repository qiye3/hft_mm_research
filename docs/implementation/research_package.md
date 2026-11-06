# 特征模块阅读与扩展示例

研究包通过 `pip install -r requirements.txt` 安装。Python 代码位于 `research/mm_research/`。

## 分类特征的阅读与修改

`build_feature_table()` 先调用 `prepare_quality_state()`，然后依次调用
`add_book_features()`、`add_order_flow_features()`、`add_trade_features()` 和
`add_return_features()`。各文件声明自己的输出列，入口按类别拼接成既有十九列顺序。
需要查看某类公式时，直接打开对应文件。新增正式研究字段时，在该文件计算公式并更新
类别列清单，再使用新的特征版本生产数据。

公共函数只包含重复使用的运算。`normalized_imbalance()` 接收买卖数量表达式，
调用方决定已知零量窗口的返回值。`rolling_sum_in_segment()` 接收窗口点数，按
`segment_id` 累计，要求完整窗口，可选择先换算整数 Lot。
盘口对齐共用 `data/base.py` 的 `build_snapshot_grid()`，原始成交共用
`build_trade_bins()`。具体特征的字段、单位和有效性由各自模块定义。

本轮分类拆分已使用 2026-07-14 的事件基础表和快照基础表与拆分前逐行对照，
全部十九个计算字段及 `feature_valid` 精确一致，覆盖当天已知缺口与重新预热。

## 三类特征模块示例

以下三个模块分别展示快照、逐笔成交和 C++ 事件特征的写法，供后续增加特征时参考。

| 类型 | 模块 | 示例函数 | 特征定义 |
| --- | --- | --- | --- |
| Polars 快照 | `research/mm_research/features/snapshot.py` | `add_snapshot_notional_features()` | 前十档双侧挂单金额及金额失衡 |
| Polars 逐笔 | `research/mm_research/features/trade_events.py` | `build_trade_event_features()` | 主动净成交金额与成交强度 |
| C++ 事件 | `src/mm/features/order_flow.hpp` | `calculate_bbo_ofi()` | 更新前后买卖一价量对应的 OFI |

### 快照模块

`snapshot.py` 接收已按接收时间向后对齐的十档盘口网格。先按 `sum(price * quantity)`
计算双侧挂单金额，再计算双方金额差除以总金额，正值表示买侧金额较大。
金额单位 USDT，失衡无量纲。模块保留输入字段，增加 `SNAPSHOT_EXAMPLE_COLUMNS`
声明的三列，`book_valid` 为假时将示例特征置空。

新增快照特征时，可参考它的字段清单、Polars 表达式和盘口有效性处理。
基础层的 `build_snapshot_grid()` 负责原始快照读取和时间对齐。

### 逐笔模块

`trade_events.py` 接收 `build_trade_bins()` 生成的成交分桶和连续 100ms 网格。
主动净额为买方主动成交金额减去卖方主动成交金额，单位 USDT。
1s 净额累计最近十个网格，成交强度为最近一秒内的成交笔数，单位笔/秒。
窗口使用 `(t-100ms,t]` 的右端点，已知无成交窗口补零。

模块保留 `TRADE_EVENT_EXAMPLE_COLUMNS` 字段清单，使用独立五秒预热。
成交跳号在首次收到时切段，随后重新预热，`trade_feature_valid` 为假时将示例特征置空。
新增逐笔特征时，可参考分桶输入、分段窗口和预热处理。

### C++ 模块

`order_flow.hpp` 使用 `BboState` 表示更新前后的买卖一价格数量。
`calculate_bbo_ofi()` 是纯计算函数，返回单位为 BTC、买方压力为正的事件 OFI。
买侧价格上升计入当前数量，下降扣除原数量，相同价格计入数量差，卖侧使用相反方向。

调用方保证两份 BBO 有效且属于连续区段，快照初始化、盘口缺口与累计窗口由调用方管理。
现有 `src/mm/research_sampler.cpp` 已调用该函数，可同时参考模块定义和实际接入位置。
新增 C++ 特征时，可参考输入结构、计算函数及调用方管理状态的分工。

### 公共运算与扩展方式

Polars 公共运算位于 `features/expressions.py`，包括双方失衡比值、分段完整窗口累计
以及可选的整数 Lot 换算。各模块自行定义字段、单位、窗口和有效性。

添加新特征时，在对应 `features` 目录新增具有明确业务名称的模块，声明输出字段并实现计算函数。
需要进入正式数据版本时，再由 `transform.py` 组织调用并更新特征列清单和版本。
当前快照和逐笔示例具有各自的独立字段定义。
