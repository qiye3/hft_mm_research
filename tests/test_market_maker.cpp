#include "mm/market_maker.hpp"
#include "trading_engine.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
void near(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1e-8, message);
}

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

hbt::Event event(std::uint64_t kind, std::int64_t timestamp, double price, double quantity) {
    return {kind | hbt::EXCH_EVENT | hbt::LOCAL_EVENT, timestamp, timestamp + 50,
            price, quantity, 0, 0, 0};
}

std::vector<hbt::Event> book() {
    return {
        event(hbt::BUY_EVENT | hbt::DEPTH_EVENT, 1000, 99, 1),
        event(hbt::SELL_EVENT | hbt::DEPTH_EVENT, 1000, 101, 1),
    };
}

void add_end(std::vector<hbt::Event>& events) {
    events.push_back(event(hbt::SELL_EVENT | hbt::DEPTH_EVENT, 5000, 101, 1));
}

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
    near(backtest.position(0), 3, "late cancel rejection must not duplicate old partial fill");
    require(recorder.summary().fills == 2 && recorder.summary().normalized_rejections == 1,
            "two real fills and one non-economic rejected cancel");
    recorder.reconcile(backtest.state_values(0));

    // 直接复现固定上游版本的缺陷：没有任何真实成交也会计入旧拒绝 payload。
    hbt::OrderBus<> requests;
    hbt::OrderBus<> responses;
    hbt::Order<> stale{42, 99, 1, 3, hbt::Side::Buy, hbt::OrdType::Limit, hbt::TimeInForce::GTX};
    stale.status = hbt::Status::PartiallyFilled;
    stale.req = hbt::Status::Rejected;
    stale.exec_qty = 1;
    stale.exec_price_tick = 99;
    responses.append(stale, 300);
    mm::detail::Local stock_local{mm::detail::Depth{1, 1},
        hbt::State<mm::detail::Asset, mm::detail::Fees>{mm::detail::Asset{1}, mm::detail::Fees{hbt::CommonFees{0, 0}}},
        0, hbt::LocalToExch<mm::detail::Latency>{requests, responses, mm::detail::Latency{100, 50}}};
    stock_local.process_recv_order(300, std::nullopt);
    near(stock_local.position(), 1, "pinned upstream stale reject reproduction");
}

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

int main() {
    const auto root = std::filesystem::temp_directory_path() / ("hft_mm_m1_tests_" + std::to_string(getpid()));
    try {
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
