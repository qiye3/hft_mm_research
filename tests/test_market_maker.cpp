#include "mm/market_maker.hpp"
#include "mm/detail/trading_engine.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

namespace {

// 【非核心】在测试条件不成立时用指定信息终止当前用例。
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
// 【非核心】按测试容差比较库存、费用与收益的浮点结果。
void near(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1e-8, message);
}

// 【非核心】构造具有独立输出目录与固定延迟的做市测试配置。
mm::TradingConfig config_for(const std::filesystem::path& root, const std::string& name) {
    mm::TradingConfig config;
    config.name = name;
    config.source = "memory";
    config.output = root / name;
    config.tick_size = 1;
    config.lot_size = 1;
    config.order_quantity = 1;
    config.max_position = 3;
    config.initial_capital = 1000;
    config.half_width_ticks = 1;
    config.entry_latency_ns = 100;
    config.response_latency_ns = 50;
    config.decision_interval_ns = 100;
    config.quote_stop_timestamp_ns = 3500;
    config.maker_fee = 0.001;
    return config;
}

// 【非核心】构造同时路由到交易所和本地的确定性测试事件。
hbt::Event event(std::uint64_t kind, std::int64_t timestamp, double price, double quantity) {
    return {kind | hbt::EXCH_EVENT | hbt::LOCAL_EVENT, timestamp, timestamp + 50,
            price, quantity, 0, 0, 0};
}

// 【非核心】构造买一99、卖一101的初始盘口夹具。
std::vector<hbt::Event> book() {
    return {
        event(hbt::BUY_EVENT | hbt::DEPTH_EVENT, 1000, 99, 1),
        event(hbt::SELL_EVENT | hbt::DEPTH_EVENT, 1000, 101, 1),
    };
}

// 【非核心】追加尾部行情，使测试引擎能够推进到停止报价之后。
void add_end(std::vector<hbt::Event>& events) {
    events.push_back(event(hbt::SELL_EVENT | hbt::DEPTH_EVENT, 5000, 101, 1));
}

// 【非核心】检查做市配置缺少必填参数时明确失败。
void test_required_trading_config(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    const auto filename = root / "incomplete.yaml";
    std::ofstream(filename) << "source: memory\noutput: unused\n";
    bool rejected = false;
    try {
        (void)mm::load_trading_config(filename);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("missing required config field") != std::string::npos;
    }
    require(rejected, "missing trading parameters must not use defaults");
    auto config = config_for(root, "invalid_quantity");
    config.order_quantity = 0;
    rejected = false;
    try {
        (void)mm::run_baseline(config, hbt::EventReaderCache::from_chunks({book()}, 2));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "replay entry must validate directly constructed configuration");
    require(!std::filesystem::exists(config.output), "invalid configuration must fail before creating outputs");
}

// 【非核心】核对报价Tick取整、完整买卖回合及负费率返佣。
void test_quotes_and_round_trip(const std::filesystem::path& root) {
    const auto quotes = mm::fixed_quotes(599999, 600002, 20);
    require(quotes.bid == 599980 && quotes.ask == 600021, "outward tick rounding");
    auto config = config_for(root, "round_trip");
    auto events = book();
    events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, 2000, 98, 5));
    events.push_back(event(hbt::BUY_EVENT | hbt::TRADE_EVENT, 3000, 102, 5));
    add_end(events);
    auto result = mm::run_baseline(config, hbt::EventReaderCache::from_chunks({events}, 2));
    require(result.fills == 2 && result.cancellations > 0, "native fills and cancel acks");
    near(result.position, 0, "flat round trip");
    near(result.realized_gross, 2, "gross spread capture");
    near(result.fees, 0.2, "native notional fees");
    near(result.net_pnl, 1.8, "net PnL");
    config.name = "rebate";
    config.output = root / config.name;
    config.maker_fee = -0.0001;
    result = mm::run_baseline(config, hbt::EventReaderCache::from_chunks({events}, 2));
    near(result.net_pnl, 2.02, "negative maker fee rebate");
}

// 【非核心】核对部分成交新增量，以及严格和宽松仓位上限的实际约束。
void test_partial_fills_and_limits(const std::filesystem::path& root) {
    auto config = config_for(root, "partial");
    config.order_quantity = 3;
    auto events = book();
    events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, 2000, 99, 2));
    events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, 2300, 99, 1));
    events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, 2600, 99, 1));
    add_end(events);
    const auto partial = mm::run_baseline(config, hbt::EventReaderCache::from_chunks({events}, 2));
    require(partial.fills == 3 && partial.partial_fills == 2, "incremental partial fill recording");
    near(partial.position, 3, "partial inventory");
    near(partial.max_abs_position, 3, "maximum inventory");
    events = book();
    for (const auto timestamp : {2000, 2500, 3000}) {
        events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, timestamp, 98, 5));
    }
    add_end(events);
    config.order_quantity = 1;
    config.max_position = 1;
    config.output = root / "strict";
    const auto strict = mm::run_baseline(config, hbt::EventReaderCache::from_chunks({events}, 2));
    require(strict.fills == 1, "strict cap prevents extra buy fills");
    config.max_position = 3;
    config.output = root / "loose";
    const auto loose = mm::run_baseline(config, hbt::EventReaderCache::from_chunks({events}, 2));
    require(loose.fills == 3, "loose safety cap");
}

// 【非核心】检查撤单在途仍预留库存，确认到达后才允许补单。
void test_cancel_in_flight(const std::filesystem::path& root) {
    auto config = config_for(root, "pending_cancel");
    config.max_position = 1;
    auto events = book();
    add_end(events);
    mm::TradeRecorder recorder{config};
    mm::detail::TradingEngine engine{config, recorder, hbt::EventReaderCache::from_chunks({events}, 2)};
    mm::MarketMaker maker{config, recorder};
    auto& backtest = engine.backtest;
    backtest.elapse(100);
    maker.on_decision(backtest);
    backtest.elapse(500);
    maker.on_decision(backtest, true);
    maker.on_decision(backtest);
    const auto [buys, sells] = maker.reservations(backtest);
    near(buys, 1, "cancel-in-flight buy reservation");
    near(sells, 1, "cancel-in-flight sell reservation");
    require(recorder.summary().submissions == 2, "no replace before cancel ack");
    backtest.elapse(300);
    maker.on_decision(backtest);
    require(recorder.summary().submissions == 4, "replace after cancel ack");
}

// 【非核心】已成交订单的撤单拒绝只解除在途请求，旧成交状态不得再次记账。
void test_partial_cancel_race_regression(const std::filesystem::path& root) {
    auto config = config_for(root, "partial_cancel_race");
    config.entry_latency_ns = 200;
    config.order_quantity = 3;
    auto events = book();
    events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, 2000, 99, 2));
    events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, 2150, 98, 5));
    add_end(events);
    mm::TradeRecorder recorder{config};
    mm::detail::TradingEngine engine{config, recorder, hbt::EventReaderCache::from_chunks({events}, 2)};
    auto& backtest = engine.backtest;
    backtest.elapse(100);
    backtest.submit_buy_order(0, 1, 99, 3, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    recorder.action("submit", backtest.orders(0).at(1), backtest.current_timestamp());
    backtest.elapse(950);
    near(backtest.position(0), 1, "partial before cancellation");
    backtest.cancel(0, 1);
    recorder.action("cancel_request", backtest.orders(0).at(1), backtest.current_timestamp());
    backtest.elapse(300);
    near(backtest.position(0), 3, "cancel rejection must not duplicate filled inventory");
    require(recorder.summary().fills == 2, "cancel rejection must not add another fill");
    require(recorder.summary().rejected == 1, "cancel rejection remains in lifecycle statistics");
    require(backtest.orders(0).at(1).req == hbt::Status::None, "rejected cancel clears in-flight request");
    recorder.reconcile(backtest.state_values(0));

}

// 【非核心】核对GTX穿价失效，以及同一回报时刻双边成交的逐笔记账。
void test_post_only_and_same_timestamp_fills(const std::filesystem::path& root) {
    auto config = config_for(root, "post_only");
    auto events = book();
    events.push_back(event(hbt::SELL_EVENT | hbt::TRADE_EVENT, 2000, 98, 5));
    events.push_back(event(hbt::BUY_EVENT | hbt::TRADE_EVENT, 2000, 102, 5));
    add_end(events);
    mm::TradeRecorder recorder{config};
    mm::detail::TradingEngine engine{config, recorder, hbt::EventReaderCache::from_chunks({events}, 2)};
    auto& backtest = engine.backtest;
    backtest.elapse(100);
    backtest.submit_buy_order(0, 1, 101, 1, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    backtest.elapse(300);
    require(backtest.orders(0).at(1).status == hbt::Status::Expired, "GTX must not take liquidity");
    require(recorder.summary().fills == 0, "post-only rejection is not a trade");
    backtest.submit_buy_order(0, 2, 99, 1, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    backtest.submit_sell_order(0, 3, 101, 1, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    backtest.elapse(1000);
    require(recorder.summary().fills == 2, "both fills in one response batch recorded");
    near(backtest.position(0), 0, "offsetting fills must not disappear from ledger");
}

} // namespace

// 【非核心】运行做市测试用例，成功后清理本次生成的临时实验目录。
int main() {
    const auto root = std::filesystem::temp_directory_path() / ("hft_mm_m1_tests_" + std::to_string(getpid()));
    try {
        test_required_trading_config(root);
        test_quotes_and_round_trip(root);
        test_partial_fills_and_limits(root);
        test_cancel_in_flight(root);
        test_partial_cancel_race_regression(root);
        test_post_only_and_same_timestamp_fills(root);
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << " artifacts=" << root << '\n';
        return 1;
    }
}
