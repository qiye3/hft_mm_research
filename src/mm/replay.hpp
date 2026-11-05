// pragma once 是 C++ 中的预处理指令，用于防止头文件被重复包含。
#pragma once

/** M0 回放配置与报告接口，真实文件输入和内存夹具使用同一原生引擎。 */

#include <filesystem>
#include <string>
#include <vector>

#include "mm/market_observer.hpp"

namespace mm {

// 在原生 Local 应用行情的前后接收通知，研究统计只读取本地可见状态。
class ReplayEventSink {
public:
    virtual ~ReplayEventSink() = default;
    virtual void before_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) = 0;
    virtual void after_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) = 0;
    virtual void finish(const hbt::HashMapMarketDepth& depth) = 0;
};

// 回放参数的时间统一使用纳秒，source 在 synthetic 和清单文件两种来源间选择。
struct ReplayConfig {
    std::string source = "synthetic"; // 数据来源，synthetic 表示合成数据，清单文件表示真实数据。
    std::string date = "synthetic"; // 数据日期，格式为 YYYY-MM-DD。
    std::string symbol = "BTCUSDT"; // 交易对，默认 BTCUSDT。
    double tick_size = 0.1; // 价格最小变动单位，默认 0.1。
    double lot_size = 0.001; // 最小交易量，默认 0.001。
    std::size_t depth_levels = 10; // 深度级别，默认 10。
    std::int64_t snapshot_interval_ns = 100'000'000; // 快照间隔时间，默认 100ms。仅控制 Top-N 提取频率，零值逐事件提取。
    std::size_t latency_sample_every = 0; // 本地事件采样间隔，正值表示每 N 个本地事件系统取样，零值关闭计时。
    std::filesystem::path output; // 输出报告的文件路径。
};

// 完整回放后的观察与性能结果，局部计时样本仅覆盖 Local 和观察器调用。
struct ReplayResult {
    EventStatistics events; // 事件统计信息。
    BookSnapshot final_book; // 最终盘口。
    std::uint64_t exchange_events = 0; // 交易所事件数。
    std::uint64_t normalized_rows = 0; // 输入物理记录数，双路由记录的交集只计算一次。
    double elapsed_seconds = 0.0; // 回放时间。
    std::uint64_t peak_rss_bytes = 0; // 峰值内存使用量。
    std::vector<double> local_step_samples_ns; // 本地事件采样时间。
    std::size_t shards_loaded = 0; // 加载的分片数。
    std::size_t shards_released = 0; // 释放的分片数。
};

// 从配置文件取得观察参数，文件位置相对 config 的上一级解析。
ReplayConfig load_config(const std::filesystem::path& filename);
// 返回用于检查 BBO、分类计数和深档升入前十档的确定性事件。
std::vector<hbt::Event> synthetic_events();
// 根据 source 选择数据来源，返回完整原生回放后的观察结果。
ReplayResult run_replay(const ReplayConfig& config, ReplayEventSink* sink = nullptr);
// 接管内存分片进行双时钟回放，供合成测试检查路由顺序与缓存释放。
ReplayResult run_memory_replay(const ReplayConfig& config,
                               std::vector<std::vector<hbt::Event>> chunks,
                               ReplayEventSink* sink = nullptr);
// 将回放汇总序列化为 JSON，分别报告逻辑事件吞吐和处理器 dispatch 吞吐。
void write_report(const ReplayConfig& config, const ReplayResult& result,
                  const std::filesystem::path& output);

} // namespace mm
