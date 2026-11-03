#include "mm/market_observer.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace mm {
namespace {

std::size_t read_levels(const std::unordered_map<std::int64_t, double>& levels,
                        std::int64_t best_tick, double tick_size, bool is_bid,
                        std::size_t count, std::array<PriceLevel, 10>& output) {
    // 与上游 do_snapshot 一致，忽略已越过当前 BBO 的旧价位。
    auto visible = levels | std::views::filter([=](const auto& level) {
        return level.second > 0.0 && (is_bid ? level.first <= best_tick : level.first >= best_tick);
    });
    std::array<std::pair<std::int64_t, double>, 10> selected{};
    const auto selected_end = std::partial_sort_copy(
        visible.begin(), visible.end(), selected.begin(), selected.begin() + count,
        [is_bid](const auto& left, const auto& right) {
            return is_bid ? left.first > right.first : left.first < right.first;
        });
    const auto actual_count = static_cast<std::size_t>(selected_end - selected.begin());
    for (std::size_t index = 0; index < actual_count; ++index) {
        output[index] = {static_cast<double>(selected[index].first) * tick_size, selected[index].second};
    }
    return actual_count;
}

} // namespace

MarketObserver::MarketObserver(std::size_t depth_levels, std::int64_t snapshot_interval_ns)
    : depth_levels_(depth_levels), snapshot_interval_ns_(snapshot_interval_ns) {
    if (depth_levels != 5 && depth_levels != 10) {
        throw std::invalid_argument("depth_levels must be 5 or 10");
    }
    if (snapshot_interval_ns < 0) {
        throw std::invalid_argument("snapshot_interval_ns must be nonnegative");
    }
}

void MarketObserver::on_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) {
    if (statistics_.total == 0) {
        statistics_.first_exchange_timestamp = event.exch_ts;
        statistics_.first_local_timestamp = event.local_ts;
    } else if (event.local_ts < statistics_.last_local_timestamp) {
        throw std::runtime_error("local event timestamps moved backwards");
    }
    ++statistics_.total;
    statistics_.last_exchange_timestamp = event.exch_ts;
    statistics_.last_local_timestamp = event.local_ts;
    switch (event.ev & 0xffU) {
    case hbt::DEPTH_EVENT:
        ++statistics_.depth_updates;
        break;
    case hbt::DEPTH_SNAPSHOT_EVENT:
        ++statistics_.snapshot_levels;
        break;
    case hbt::DEPTH_CLEAR_EVENT:
        ++statistics_.clears;
        break;
    case hbt::TRADE_EVENT:
        ++statistics_.trades;
        break;
    default:
        throw std::runtime_error("unsupported L2 event kind");
    }
    if (std::isfinite(depth.best_bid()) && std::isfinite(depth.best_ask())) {
        ++statistics_.valid_bbo_events;
        statistics_.spread_sum += depth.best_ask() - depth.best_bid();
    }
    if (!last_snapshot_timestamp_ || event.local_ts - *last_snapshot_timestamp_ >= snapshot_interval_ns_) {
        capture(event, depth);
    }
}

void MarketObserver::capture(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) {
    latest_ = BookSnapshot{};
    latest_.exchange_timestamp = event.exch_ts;
    latest_.local_timestamp = event.local_ts;
    latest_.bid_count = read_levels(depth.bid_depth_map(), depth.best_bid_tick(), depth.tick_size(),
                                    true, depth_levels_, latest_.bids);
    latest_.ask_count = read_levels(depth.ask_depth_map(), depth.best_ask_tick(), depth.tick_size(),
                                    false, depth_levels_, latest_.asks);
    latest_.mid_price = (depth.best_bid() + depth.best_ask()) * 0.5;
    latest_.spread = depth.best_ask() - depth.best_bid();
    last_snapshot_timestamp_ = event.local_ts;
    ++statistics_.snapshots_observed;
}

const EventStatistics& MarketObserver::statistics() const noexcept { return statistics_; }
const BookSnapshot& MarketObserver::latest_snapshot() const noexcept { return latest_; }
std::size_t MarketObserver::depth_levels() const noexcept { return depth_levels_; }

} // namespace mm
