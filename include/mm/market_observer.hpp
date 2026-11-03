#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <hbt/hbt.hpp>

namespace mm {

struct PriceLevel {
    double price = 0.0;
    double quantity = 0.0;
};

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
    double spread_sum = 0.0;
};

// 只读取引擎订单簿，不维护第二套盘口；统计逐条事件，深度快照按配置采样。
class MarketObserver {
public:
    explicit MarketObserver(std::size_t depth_levels, std::int64_t snapshot_interval_ns);
    void on_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth);
    void capture(const hbt::Event& event, const hbt::HashMapMarketDepth& depth);
    [[nodiscard]] const EventStatistics& statistics() const noexcept;
    [[nodiscard]] const BookSnapshot& latest_snapshot() const noexcept;
    [[nodiscard]] std::size_t depth_levels() const noexcept;

private:
    std::size_t depth_levels_;
    std::int64_t snapshot_interval_ns_;
    EventStatistics statistics_;
    BookSnapshot latest_;
    std::optional<std::int64_t> last_snapshot_timestamp_;
};

} // namespace mm
