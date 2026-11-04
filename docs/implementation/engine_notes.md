# 2026-11-03 — M0 引擎集成笔记

依赖：`shenron0101/hftbacktest-cpp` v0.1.0，提交 `e0d2f7887c68755b220d9e69425d61d27e5b6432`。通过 Git Submodule 固定；第三方源码不作修改。它是 nkaz001/HftBacktest 的独立 C++20 移植，目前仅支持 L2，尚未稳定 API。

## 公开接口与运行链路

以下路径相对 `third_party/hftbacktest-cpp/`。

| 环节 | 源码位置及接口 | 本项目用法 |
| --- | --- | --- |
| 构建 | `CMakeLists.txt`：`hbt::hbt`、`hbt::io` | `add_subdirectory` 后直接链接；无需复制头文件 |
| 事件布局 | `include/hbt/types.hpp`：`Event`、`event_is` | 64 字节、8 个 8 字节字段；时间统一为 ns |
| NumPy IO | `src/npy.cpp`、`include/hbt/data/npy.hpp`：`load_events`、`load_npz` | 标准事件必须装入 NPZ 的 `data.npy`；实际便捷入口不会按扩展名自动读取裸 NPY |
| 文件清单 | `include/hbt/data/reader.hpp`：`event_files_from_manifest` | 从 JSON `shards` 下依次读取 `file` 字段，相对清单路径解析 |
| 分片加载 | 同上：`EventReaderCache::acquire/release`、`EventReader::next_chunk` | Local/Exchange 共用一个 cache，consumer_count=2；预读下一片，双方用完后释放 |
| 逐处理器游标 | `include/hbt/backtest/backtest.hpp`：`ProcessorState::advance` | 跳过未路由给该处理器的记录，跨分片寻找下一事件 |
| 事件调度 | `include/hbt/backtest/event_set.hpp`：`EventSet::next` | 从 LocalData、LocalOrder、ExchData、ExchOrder 四类意图选最早时间 |
| 回放入口 | `include/hbt/backtest/backtest.hpp`：`Backtest::add_asset/close/do_goto` | 构造 Local、Exchange、共享 reader；`close()` 使用真实调度循环消耗到 EndOfData |
| 本地处理 | `include/hbt/backtest/local.hpp`：`Local::process/depth/feed_latency` | 根据 local_ts 更新本地可见订单簿、保存行情时间戳；M0 的 last_trades 容量设为 0，避免无限累积 |
| 交易所处理 | `include/hbt/backtest/exchange.hpp`：`NoPartialFillExchange::process` | 根据 exch_ts 重建交易所盘口，使用原有队列与成交模型；M0 不下单 |
| 订单簿 | `include/hbt/depth/hashmap_depth.hpp`：`HashMapMarketDepth::update_bid_depth/update_ask_depth` | 原有哈希订单簿执行更新、数量为 0 时删除，并维护 BBO |
| 盘口只读访问 | 同上：`best_bid/best_ask/best_bid_qty/best_ask_qty/bid_depth_map/ask_depth_map` | `MarketObserver` 提取排序后的前 5/10 档，不建立第二套订单簿 |
| 延迟与订单总线 | `include/hbt/models/latency.hpp`、`include/hbt/backtest/order_bus.hpp`：`ConstantLatency`、`make_order_buses` | 保留原生订单请求/响应延迟；M0 无订单，不影响行情回放 |

Local Processor 表示策略当时能看到的本地状态；Exchange Processor 表示交易所已经发生的真实状态。`exch_ts` 是撮合相关事件的交易所时间，`local_ts` 是采集端收到消息的时间。行情延迟为 `local_ts - exch_ts`。两侧分别按自己的时间戳处理，不能把本地已知状态提前暴露给策略。

当两个时钟的事件顺序不一致时，不能仅按某一个时间戳排序并同时标记两侧。转换器分别排序交易所路由与本地路由后，按各自调度时间归并，生成 EXCH-only / LOCAL-only 记录。这会增加标准化行数，但每个源事件在每侧只处理一次。首日和重连快照前加入两侧清空事件，防止已失效的旧价位残留。

## 观察器适配边界

`Local` 是 final，`Backtest` 不提供逐事件 callback，`wait_next_feed` 会处理同一时间戳的一组事件。因此，不能将调用次数当作事件数量。本项目 `src/mm/replay.cpp` 的 `ObservedLocal` 实现原生 `LocalProcessor` 接口，只将所有调用转发给实际 `Local`，在其 `process` 返回后通知 `MarketObserver`。`CountedExchange` 同样转发给实际 Exchange，仅增加计数及时间顺序检查。

事件计数与 BBO 价差统计逐事件进行。完整前 N 档按 `snapshot_interval_ns` 采样，结束时额外采集最终盘口。默认 100 ms，设为 0 可逐事件提取。采样是读取频率，不影响输入数据和引擎盘口完整性。Top-N 使用原始 map 的只读视图及 `partial_sort_copy`，与上游 snapshot 一样过滤越过当前 BBO 的旧价位。

## 性能口径

整体回放计时包围 `Backtest::close()`，包含首次加载、NPZ 解压、后续分片 IO、双侧处理、调度、观察器。报告同时给出单侧逻辑事件吞吐与双侧处理器 dispatch 吞吐。逻辑事件包括快照价位、深度增量、成交和转换器新增的清空事件。

`local_step_latency` 的抽样范围仅为一次 `Local::process + observer`，不包含 Exchange、调度器和 IO，也不是策略延迟；M0 没有策略。p50/p95/p99 按实际样本 nearest-rank 计算，不从整批耗时除以事件数伪造分位数。Peak RSS 使用 `getrusage(RUSAGE_SELF)`，macOS 的单位为字节，Linux 为 KiB 并换算成字节；记录独立基准进程截至回放完成及最终盘口采集的内存高水位。

## M0 验收

已独立构建并通过上游原有测试及 `hbt_market_making`。主项目 Release 构建通过，`replay_probe` 使用上游四条合成事件及扩展深度用例。项目测试覆盖 BBO 数量、Mid/Spread、5/10 档配置、第 11 档升入前十档、快照重置、双时钟重排和跨分片释放。观察器回放不提交订单，持仓为零。
