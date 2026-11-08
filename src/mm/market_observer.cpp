/**
 * M0 行情观察器，从原生完整订单簿提取 Top-N，并对每条本地行情累计统计。
 * 输入是处理器已应用的事件和订单簿，输出包括最新盘口快照、BBO 价差和分类计数。
 */
#include "mm/market_observer.hpp"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <stdexcept>

namespace mm {
namespace {

// 【核心逻辑155】从完整单侧盘口中选出当前最优的 N 档，并转换为实际价格与数量。
std::size_t read_levels(const std::unordered_map<std::int64_t, double>& levels,
                        std::int64_t best_tick, double tick_size, bool is_bid,
                        std::size_t count, std::array<PriceLevel, 10>& output) {
    /** levels 的键和 best_tick 都是整数价格档位，is_bid 决定排序方向。
     * count 必须不超过 output 容量，返回实际档数，调用方只读取对应前缀。
     * partial_sort_copy 只保留目标档数，深层价位仍由原生完整盘口维护。
     */
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

// 【核心逻辑147】建立前五档或前十档观察器，设定本地快照提取的最短时间间隔。
MarketObserver::MarketObserver(std::size_t depth_levels, std::int64_t snapshot_interval_ns)
    : depth_levels_(depth_levels), snapshot_interval_ns_(snapshot_interval_ns) {
    /** 输入 depth_levels 只接受 5 或 10，snapshot_interval_ns 为非负纳秒数。
     * 构造后计数与快照为空，间隔为零时每条本地事件都会触发一次提取。
     */
    if (depth_levels != 5 && depth_levels != 10) {
        throw std::invalid_argument("depth_levels must be 5 or 10");
    }
    if (snapshot_interval_ns < 0) {
        throw std::invalid_argument("snapshot_interval_ns must be nonnegative");
    }
}

// 【核心逻辑159】在本地盘口更新后累计事件与价差统计，并按本地时钟触发快照采样。
void MarketObserver::on_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) {
    /** 输入 event 必须按 local_ts 非递减到达，depth 是应用该事件后的原生状态。
     * 输出体现为统计和可选的最新快照更新，未知 L2 类型或本地时间倒退会抛出异常。
     * 记录的首尾 exch_ts 属于首尾本地事件，交易所时间在本地接收顺序中可以逆序。
     */
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
        // 累积的是事件加权价差，后续除以有效 BBO 事件数得到均值。
        ++statistics_.valid_bbo_events;
        statistics_.spread_sum += depth.best_ask() - depth.best_bid();
    }
    if (!last_snapshot_timestamp_ || event.local_ts - *last_snapshot_timestamp_ >= snapshot_interval_ns_) {
        capture(event, depth);
    }
}

// 【核心逻辑154】保存指定事件之后的双边 Top-N、Mid 和 Spread，供定期观察及最终输出使用。
void MarketObserver::capture(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) {
    /** event 提供双时间戳，depth 提供当前完整盘口，输出保存在 latest_。
     * bid_count/ask_count 描述数组中的有效前缀，空簿的价格指标可为 NaN。
     * 回测收尾可主动调用一次，以便输出最后一个采样间隔内的最终状态。
     */
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

// 【非核心】返回逐事件累计统计的只读引用，供回放汇总和测量采样使用。
const EventStatistics& MarketObserver::statistics() const noexcept { return statistics_; }
// 【非核心】返回最近一次采集的盘口视图，有效价位数量由快照中的 count 字段给出。
const BookSnapshot& MarketObserver::latest_snapshot() const noexcept { return latest_; }
// 【非核心】返回已验证的目标观察档数，用于解释快照数组的有效容量。
std::size_t MarketObserver::depth_levels() const noexcept { return depth_levels_; }

} // namespace mm
