/** 在原生 Local 处理边界累计 BBO OFI 和主动成交，按 UTC 固定网格输出研究基础表。 */
#include "mm/research_sampler.hpp"
#include "mm/features/order_flow.hpp"

#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace mm {

// 【核心逻辑38】建立固定采样网格与输出列，浮点数保留可往返的十进制精度。
ResearchSampler::ResearchSampler(const std::filesystem::path& output, std::int64_t start,
                                 std::int64_t end, std::int64_t interval,
                                 std::int64_t max_book_age, std::size_t depth_levels)
    : output_(output), observer_(depth_levels, interval), next_sample_(start), end_(end),
      interval_(interval), max_book_age_(max_book_age) {
    /** 输入为已有父目录中的 CSV 文件和纳秒网格参数，输出为流式研究表。
     * 网格必须整除研究区间，空盘口输出 nan，后续 Python 统一为 null。
     */
    if (!output_ || interval <= 0 || end <= start || (end - start) % interval != 0 || max_book_age <= 0) {
        throw std::invalid_argument("invalid research output or sampling range");
    }
    output_ << std::setprecision(17);
    output_ << "timestamp_ns,last_local_timestamp_ns,last_exchange_timestamp_ns,last_book_timestamp_ns,"
               "segment_id,segment_start_ns,book_valid,flow_valid,event_count,trade_count,"
               "ofi_bbo,buy_volume,sell_volume,buy_notional,sell_notional,mid_price,spread";
    for (std::size_t level = 0; level < depth_levels; ++level) {
        output_ << ",bid_price_" << level + 1 << ",bid_quantity_" << level + 1
                << ",ask_price_" << level + 1 << ",ask_quantity_" << level + 1;
    }
    output_ << '\n';
}

// 【核心逻辑44】在下一条事件应用前补齐更早的网格点，同时间戳的全部事件均归入右端点。
void ResearchSampler::before_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) {
    /** 输入为尚未应用的事件和旧盘口，输出为严格早于 event.local_ts 的样本。
     * 原生调度器保留同时间戳顺序，因此跨分片也不会提前写出半批快照。
     */
    if (snapshot_active_ && event.local_ts > last_local_) {
        snapshot_active_ = false;
        initialized_ = previous_bbo_valid_;
    }
    while (next_sample_ < event.local_ts && next_sample_ < end_) {
        write_sample(depth, true);
    }
}

// 【核心逻辑49】将快照恢复或行情中断标为新的研究区段，滚动窗口随后重新预热。
void ResearchSampler::reset_segment(std::int64_t timestamp) {
    /** timestamp 是中断或恢复的本地时刻，输出为新 segment 与失效区间标记。
     * 本区间仍保留事件计数，但 OFI 与成交累计不跨区段合并。
     */
    ++segment_;
    segment_start_ = timestamp;
    flow_valid_ = false;
    ofi_ = 0;
    buy_volume_ = 0;
    sell_volume_ = 0;
    buy_notional_ = 0;
    sell_notional_ = 0;
}

// 【核心逻辑48】从原生更新后的最优价位计算 OFI，快照恢复和无效 BBO 不产生订单流。
void ResearchSampler::after_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) {
    /** 输入为已应用的本地事件，输出更新当前区间统计及最后可见双时间戳。
     * OFI 使用逐价位事件的买卖一价格比较定义，单位 BTC，正值表示买方压力。
     */
    const auto kind = event.ev & 0xffU;
    const bool book_event = kind != hbt::TRADE_EVENT;
    if (kind == hbt::DEPTH_CLEAR_EVENT) {
        reset_segment(event.local_ts);
        initialized_ = false;
        snapshot_active_ = true;
    } else if (book_event && last_book_local_ > 0 && event.local_ts - last_book_local_ > max_book_age_) {
        reset_segment(event.local_ts);
    }
    if (kind == hbt::DEPTH_SNAPSHOT_EVENT) {
        snapshot_active_ = true;
    }
    const double bid = depth.best_bid();
    const double ask = depth.best_ask();
    const bool valid_bbo = std::isfinite(bid) && std::isfinite(ask) && bid < ask;
    const double bid_quantity = valid_bbo ? depth.bid_qty_at_tick(depth.best_bid_tick()) : 0;
    const double ask_quantity = valid_bbo ? depth.ask_qty_at_tick(depth.best_ask_tick()) : 0;
    if (kind == hbt::DEPTH_EVENT && initialized_ && previous_bbo_valid_ && valid_bbo) {
        const BboState previous{previous_bid_, previous_ask_, previous_bid_quantity_, previous_ask_quantity_};
        const BboState current{bid, ask, bid_quantity, ask_quantity};
        ofi_ += calculate_bbo_ofi(previous, current);
    }
    if (!valid_bbo && previous_bbo_valid_ && !snapshot_active_) {
        reset_segment(event.local_ts);
    }
    if (kind == hbt::DEPTH_EVENT && !snapshot_active_) {
        initialized_ = valid_bbo;
    }
    if (kind == hbt::TRADE_EVENT) {
        ++trade_count_;
        if (hbt::event_is(event, hbt::BUY_EVENT)) {
            buy_volume_ += event.qty;
            buy_notional_ += event.qty * event.px;
        } else {
            sell_volume_ += event.qty;
            sell_notional_ += event.qty * event.px;
        }
    }
    ++event_count_;
    last_local_ = event.local_ts;
    last_exchange_ = event.exch_ts;
    if (book_event) {
        last_book_local_ = event.local_ts;
    }
    previous_bid_ = bid;
    previous_ask_ = ask;
    previous_bid_quantity_ = bid_quantity;
    previous_ask_quantity_ = ask_quantity;
    previous_bbo_valid_ = valid_bbo;
}

// 【核心逻辑45】保存完整网格行，缺档价格和数量使用 NaN，空窗成交量保留为零。
void ResearchSampler::write_sample(const hbt::HashMapMarketDepth& depth, bool covered) {
    /** 输入为当前原生订单簿和源流覆盖标记，输出一行及清空区间统计。
     * 盘口超过 max_book_age 后无效，流量恢复需经过一个完整有效采样区间。
     */
    hbt::Event sample_event{};
    sample_event.local_ts = last_local_;
    sample_event.exch_ts = last_exchange_;
    observer_.capture(sample_event, depth);
    const auto& book = observer_.latest_snapshot();
    const bool fresh = last_book_local_ > 0 && next_sample_ - last_book_local_ <= max_book_age_;
    const bool valid = covered && initialized_ && !snapshot_active_ && fresh && previous_bbo_valid_
                       && book.bid_count == observer_.depth_levels() && book.ask_count == observer_.depth_levels();
    output_ << next_sample_ << ',' << last_local_ << ',' << last_exchange_ << ',' << last_book_local_
            << ',' << segment_ << ',' << segment_start_ << ',' << valid << ',' << (valid && flow_valid_)
            << ',' << event_count_ << ',' << trade_count_ << ',' << ofi_ << ',' << buy_volume_
            << ',' << sell_volume_ << ',' << buy_notional_ << ',' << sell_notional_
            << ',' << book.mid_price << ',' << book.spread;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t level = 0; level < observer_.depth_levels(); ++level) {
        output_ << ',' << (level < book.bid_count ? book.bids[level].price : missing)
                << ',' << (level < book.bid_count ? book.bids[level].quantity : missing)
                << ',' << (level < book.ask_count ? book.asks[level].price : missing)
                << ',' << (level < book.ask_count ? book.asks[level].quantity : missing);
    }
    output_ << '\n';
    if (!output_) {
        throw std::runtime_error("failed writing research samples");
    }
    event_count_ = 0;
    trade_count_ = 0;
    ofi_ = 0;
    buy_volume_ = 0;
    sell_volume_ = 0;
    buy_notional_ = 0;
    sell_notional_ = 0;
    flow_valid_ = valid;
    next_sample_ += interval_;
}

// 【核心逻辑52】补齐日末网格，源流结束之后的点保留为无效样本。
void ResearchSampler::finish(const hbt::HashMapMarketDepth& depth) {
    /** 输入为回放最终盘口，输出剩余 [next_sample, end) 的样本。
     * 文件末尾若恰好停在快照内，无法确认快照完整，保持无效。
     */
    while (next_sample_ < end_) {
        write_sample(depth, next_sample_ <= last_local_);
    }
    output_.flush();
    if (!output_) {
        throw std::runtime_error("failed flushing research samples");
    }
}

} // namespace mm
