#pragma once

/** M0 行情观察的数据结构与接口，数量按源行情单位表达，BTCUSDT 场景使用 BTC。 */

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <hbt/hbt.hpp>

namespace mm {

// 一个有效价位的实际价格与挂单数量。
struct PriceLevel {
    double price = 0.0;
    double quantity = 0.0;
};

// 一次提取的盘口视图，固定数组中的有效内容以各侧 count 字段为准。
struct BookSnapshot {
    std::int64_t exchange_timestamp = 0;
    std::int64_t local_timestamp = 0;
    std::array<PriceLevel, 10> bids{};
    std::array<PriceLevel, 10> asks{};
    std::size_t bid_count = 0;
    std::size_t ask_count = 0;
    double mid_price = std::numeric_limits<double>::quiet_NaN();
    double spread = std::numeric_limits<double>::quiet_NaN();
};

// 按本地接收顺序累计的事件统计，时间戳单位统一为纳秒。
struct EventStatistics {
    std::uint64_t total = 0;
    std::uint64_t depth_updates = 0;
    std::uint64_t snapshot_levels = 0;
    std::uint64_t clears = 0;
    std::uint64_t trades = 0;
    std::uint64_t valid_bbo_events = 0;
    std::uint64_t snapshots_observed = 0;
    std::int64_t first_exchange_timestamp = 0;
    std::int64_t last_exchange_timestamp = 0;
    std::int64_t first_local_timestamp = 0;
    std::int64_t last_local_timestamp = 0;
    double spread_sum = 0.0; // 有效 BBO 事件上的价差之和，用于事件加权均值。
};

// 从原生订单簿读取状态，逐条统计本地行情，并按配置提取深度快照。
class MarketObserver {
public:
    // 【核心逻辑164】以 5 或 10 档和非负纳秒间隔初始化观察状态。
    explicit MarketObserver(std::size_t depth_levels, std::int64_t snapshot_interval_ns);
    // 【核心逻辑176】接收已经应用到 depth 的本地事件，累计计数并按间隔刷新快照。
    void on_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth);
    // 【核心逻辑171】主动保存当前盘口，回放结束时用它补齐最后一个观察点。
    void capture(const hbt::Event& event, const hbt::HashMapMarketDepth& depth);
    // 【非核心】取得统计的只读引用，首尾时间表示本地接收顺序中的事件边界。
    [[nodiscard]] const EventStatistics& statistics() const noexcept;
    // 【非核心】取得最新 Top-N 视图，只有 count 指定的数组前缀有效。
    [[nodiscard]] const BookSnapshot& latest_snapshot() const noexcept;
    // 【非核心】取得当前观察档数，返回值为构造时校验过的 5 或 10。
    [[nodiscard]] std::size_t depth_levels() const noexcept;

private:
    std::size_t depth_levels_;
    std::int64_t snapshot_interval_ns_;
    EventStatistics statistics_;
    BookSnapshot latest_;
    std::optional<std::int64_t> last_snapshot_timestamp_;
};

} // namespace mm
