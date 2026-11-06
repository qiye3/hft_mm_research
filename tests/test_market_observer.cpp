#include "mm/replay.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

// 【非核心】在测试条件不成立时用指定信息终止当前用例。
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// 【非核心】按测试容差检查盘口统计的浮点结果。
void require_near(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1e-9, message);
}

// 【非核心】核对上游合成行情的事件计数、最优价量和双时间戳边界。
void test_upstream_synthetic_feed() {
    auto events = mm::synthetic_events();
    events.resize(4);
    const auto result = mm::run_memory_replay(mm::ReplayConfig{}, {events});
    require(result.events.total == 4 && result.exchange_events == 4, "original feed count");
    require(result.events.depth_updates == 2 && result.events.trades == 2, "original event kinds");
    require_near(result.final_book.bids[0].price, 99.9, "original best bid");
    require_near(result.final_book.asks[0].quantity, 5.0, "original best ask quantity");
    require_near(result.final_book.mid_price, 100.0, "original mid");
    require_near(result.final_book.spread, 0.2, "original spread");
    require(result.events.first_exchange_timestamp == 1'000'000, "first exchange timestamp");
    require(result.events.last_local_timestamp == 4'010'000, "last local timestamp");
}

// 【非核心】检查五档和十档观察配置，以及删除买一后的深档晋升。
void test_depth_promotion_and_configuration() {
    for (const auto depth_levels : {5U, 10U}) {
        mm::ReplayConfig config;
        config.depth_levels = depth_levels;
        config.snapshot_interval_ns = 0;
        config.latency_sample_every = 1;
        const auto result = mm::run_memory_replay(config, {mm::synthetic_events()});
        require(result.events.total == 25 && result.normalized_rows == 25, "extended feed count");
        require(result.events.depth_updates == 23 && result.events.trades == 2, "extended kinds");
        require(result.final_book.bid_count == depth_levels && result.final_book.ask_count == depth_levels,
                "configured top N");
        require_near(result.final_book.bids[0].price, 99.8, "delete best bid");
        require_near(result.final_book.bids[0].quantity, 2.0, "promoted quantity");
        require_near(result.final_book.bids[depth_levels - 1].price,
                     99.9 - static_cast<double>(depth_levels) * 0.1, "deep level promotion");
        require_near(result.final_book.mid_price, 99.95, "updated mid");
        require_near(result.final_book.spread, 0.3, "updated spread");
        require(result.local_step_samples_ns.size() == 25, "local step sampling");
    }
    bool rejected = false;
    try {
        mm::MarketObserver observer{6, 0};
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid depth must fail");
}

// 【非核心】检查快照替换旧盘口与双时钟跨分片回放结果。
void test_snapshot_clear_and_split_clocks() {
    const auto both = hbt::EXCH_EVENT | hbt::LOCAL_EVENT;
    const std::vector<hbt::Event> events{
        {both | hbt::BUY_EVENT | hbt::DEPTH_EVENT, 10, 20, 99, 1, 0, 0, 0},
        {both | hbt::SELL_EVENT | hbt::DEPTH_EVENT, 10, 20, 101, 1, 0, 0, 0},
        {both | hbt::DEPTH_CLEAR_EVENT, 30, 40, 0, 0, 0, 0, 0},
        {both | hbt::BUY_EVENT | hbt::DEPTH_SNAPSHOT_EVENT, 30, 40, 98, 2, 0, 0, 0},
        {both | hbt::SELL_EVENT | hbt::DEPTH_SNAPSHOT_EVENT, 30, 40, 102, 3, 0, 0, 0},
    };
    const auto result = mm::run_memory_replay(mm::ReplayConfig{}, {events});
    require(result.events.clears == 1 && result.events.snapshot_levels == 2, "clear/snapshot count");
    require_near(result.final_book.bids[0].price, 98, "snapshot replaced old book");
    const auto bid = hbt::BUY_EVENT | hbt::DEPTH_EVENT;
    const auto ask = hbt::SELL_EVENT | hbt::DEPTH_EVENT;
    const std::vector<std::vector<hbt::Event>> chunks{
        {{hbt::EXCH_EVENT | ask, 10, 50, 101, 2, 0, 0, 0}},
        {{both | bid, 20, 40, 99, 1, 0, 0, 0}},
        {{hbt::LOCAL_EVENT | ask, 10, 50, 101, 2, 0, 0, 0}},
    };
    const auto split = mm::run_memory_replay(mm::ReplayConfig{}, chunks);
    require(split.events.total == 2 && split.exchange_events == 2 && split.normalized_rows == 3,
            "split route counts");
    require(split.shards_loaded == 3 && split.shards_released == 3, "shared reader releases shards");
    require_near(split.final_book.mid_price, 100, "split clock book");
}

} // namespace

// 【非核心】顺序运行观察器用例，并把断言失败转换为进程失败状态。
int main() {
    try {
        test_upstream_synthetic_feed();
        test_depth_promotion_and_configuration();
        test_snapshot_clear_and_split_clocks();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
