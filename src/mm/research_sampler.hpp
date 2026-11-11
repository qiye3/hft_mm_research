#pragma once

/** M2-A 固定网格采样器，只消费原生回放交付的本地事件，不提交订单。 */

#include <fstream>
#include "mm/replay.hpp"

namespace mm {

// 每个网格输出截至该时刻的盘口，以及左开右闭区间内的事件累计量。
class ResearchSampler final : public ReplayEventSink {
public:
    // 【核心逻辑163】建立 [start, end) 的固定网格，max_book_age 限制允许沿用旧盘口的时长。
    ResearchSampler(const std::filesystem::path& output, std::int64_t start, std::int64_t end,
                    std::int64_t interval, std::int64_t max_book_age, std::size_t depth_levels);
    // 【核心逻辑169】用事件应用前的盘口补齐严格早于当前事件的网格点。
    void before_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) override;
    // 【核心逻辑173】用事件应用后的盘口更新OFI、主动成交与连续区段。
    void after_event(const hbt::Event& event, const hbt::HashMapMarketDepth& depth) override;
    // 【核心逻辑177】补齐回放结束后的剩余网格，并标记超出源流覆盖的样本。
    void finish(const hbt::HashMapMarketDepth& depth) override;

private:
    // 【核心逻辑170】写出一个网格点，并清零已输出区间的事件量。
    void write_sample(const hbt::HashMapMarketDepth& depth, bool covered);
    // 【核心逻辑174】切断快照、失效盘口或长时间缺口前后的流量累计关系。
    void reset_segment(std::int64_t timestamp);

    std::ofstream output_;
    MarketObserver observer_;
    std::int64_t next_sample_;
    std::int64_t end_;
    std::int64_t interval_;
    std::int64_t max_book_age_;
    std::int64_t last_local_ = 0;
    std::int64_t last_exchange_ = 0;
    std::int64_t last_book_local_ = 0;
    std::int64_t segment_start_ = 0;
    std::uint64_t segment_ = 0;
    std::uint64_t event_count_ = 0;
    std::uint64_t trade_count_ = 0;
    double buy_volume_ = 0;
    double sell_volume_ = 0;
    double buy_notional_ = 0;
    double sell_notional_ = 0;
    double ofi_ = 0;
    double previous_bid_ = 0;
    double previous_ask_ = 0;
    double previous_bid_quantity_ = 0;
    double previous_ask_quantity_ = 0;
    bool previous_bbo_valid_ = false;
    bool snapshot_active_ = false;
    bool initialized_ = false;
    bool flow_valid_ = false;
};

} // namespace mm
