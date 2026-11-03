#include "mm/market_maker.hpp"
#include "trading_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace mm {
namespace {

bool reserves_inventory(const hbt::Order<>& order) {
    return order.active() || (order.status == hbt::Status::None && order.req == hbt::Status::New);
}

std::string strip(std::string value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    return value.substr(start, value.find_last_not_of(" \t\r\n") - start + 1);
}

std::map<std::string, std::string> read_flat_config(const std::filesystem::path& filename) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("cannot open trading config: " + filename.string());
    }
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(input, line)) {
        line = strip(line.substr(0, line.find('#')));
        if (line.empty()) {
            continue;
        }
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            throw std::runtime_error("expected scalar key: value");
        }
        const auto key = strip(line.substr(0, separator));
        auto value = strip(line.substr(separator + 1));
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        if (!fields.emplace(key, value).second) {
            throw std::runtime_error("duplicate config key: " + key);
        }
    }
    return fields;
}

} // namespace

QuoteTicks fixed_quotes(std::int64_t best_bid, std::int64_t best_ask, std::int64_t half_width) {
    if (best_bid >= best_ask || half_width <= 0) {
        throw std::invalid_argument("need noncrossed book and positive half width");
    }
    const double mid_tick = (static_cast<double>(best_bid) + static_cast<double>(best_ask)) * 0.5;
    return {static_cast<std::int64_t>(std::floor(mid_tick - static_cast<double>(half_width))),
            static_cast<std::int64_t>(std::ceil(mid_tick + static_cast<double>(half_width)))};
}

void validate_trading_config(const TradingConfig& config) {
    for (const double value : {config.tick_size, config.lot_size, config.order_quantity,
                               config.max_position, config.initial_capital}) {
        if (!std::isfinite(value) || value <= 0) {
            throw std::invalid_argument("size, capital and tick parameters must be positive finite");
        }
    }
    for (const double value : {config.order_quantity, config.max_position}) {
        if (std::abs(value / config.lot_size - std::round(value / config.lot_size)) > 1e-8) {
            throw std::invalid_argument("quantity must be an integer number of lots");
        }
    }
    if (config.order_quantity > config.max_position || config.half_width_ticks <= 0 ||
        config.decision_interval_ns <= 0 || config.price_threshold_ticks <= 0 ||
        config.entry_latency_ns < 0 || config.response_latency_ns < 0 ||
        !std::isfinite(config.maker_fee) || !std::isfinite(config.taker_fee) ||
        (config.update_mode != "time" && config.update_mode != "price") ||
        config.quote_stop_timestamp_ns <= 0 || config.source.empty() || config.output.empty()) {
        throw std::invalid_argument("invalid baseline trading configuration");
    }
}

TradingConfig load_trading_config(const std::filesystem::path& filename) {
    TradingConfig config;
    for (const auto& [key, value] : read_flat_config(filename)) {
        if (key == "name") {
            config.name = value;
        } else if (key == "date") {
            config.date = value;
        } else if (key == "source") {
            config.source = value;
        } else if (key == "output") {
            config.output = value;
        } else if (key == "update_mode") {
            config.update_mode = value;
        } else if (key == "tick_size") {
            config.tick_size = std::stod(value);
        } else if (key == "lot_size") {
            config.lot_size = std::stod(value);
        } else if (key == "order_quantity") {
            config.order_quantity = std::stod(value);
        } else if (key == "max_position") {
            config.max_position = std::stod(value);
        } else if (key == "initial_capital") {
            config.initial_capital = std::stod(value);
        } else if (key == "maker_fee") {
            config.maker_fee = std::stod(value);
        } else if (key == "taker_fee") {
            config.taker_fee = std::stod(value);
        } else if (key == "half_width_ticks") {
            config.half_width_ticks = std::stoll(value);
        } else if (key == "decision_interval_ns") {
            config.decision_interval_ns = std::stoll(value);
        } else if (key == "price_threshold_ticks") {
            config.price_threshold_ticks = std::stoll(value);
        } else if (key == "entry_latency_ns") {
            config.entry_latency_ns = std::stoll(value);
        } else if (key == "response_latency_ns") {
            config.response_latency_ns = std::stoll(value);
        } else if (key == "quote_stop_timestamp_ns") {
            config.quote_stop_timestamp_ns = std::stoll(value);
        } else {
            throw std::runtime_error("unknown trading field: " + key);
        }
    }
    const auto root = std::filesystem::absolute(filename).parent_path().parent_path();
    config.source = root / config.source;
    config.output = root / config.output;
    validate_trading_config(config);
    return config;
}

MarketMaker::MarketMaker(const TradingConfig& config, TradeRecorder& recorder)
    : config_(config), recorder_(recorder) { validate_trading_config(config); }

std::pair<double, double> MarketMaker::reservations(const hbt::Backtest& backtest) const {
    double buys = 0.0;
    double sells = 0.0;
    for (const auto& [order_id, order] : backtest.orders(0)) {
        (void)order_id;
        if (!reserves_inventory(order)) {
            continue;
        }
        if (order.side == hbt::Side::Buy) {
            buys += order.leaves_qty;
        } else {
            sells += order.leaves_qty;
        }
    }
    return {buys, sells};
}

void MarketMaker::quote_side(hbt::Backtest& backtest, hbt::Side side, std::int64_t target_tick, bool stop) {
    const hbt::Order<>* existing = nullptr;
    for (const auto& [order_id, order] : backtest.orders(0)) {
        (void)order_id;
        if (order.side == side && reserves_inventory(order)) {
            if (existing != nullptr) {
                throw std::runtime_error("more than one working order on a side");
            }
            existing = &order;
        }
    }
    const double sign = hbt::side_sign(side);
    const double capacity = config_.max_position - sign * backtest.position(0);
    const auto available_lots = static_cast<std::int64_t>(std::floor(capacity / config_.lot_size + 1e-8));
    const auto order_lots = static_cast<std::int64_t>(std::llround(config_.order_quantity / config_.lot_size));
    const double quantity = static_cast<double>(std::max<std::int64_t>(0, std::min(order_lots, available_lots))) * config_.lot_size;
    if (existing != nullptr) {
        const bool replace = stop || quantity == 0.0 || existing->price_tick != target_tick;
        if (replace && existing->cancellable()) {
            const auto order_id = existing->order_id;
            backtest.cancel(0, order_id);
            recorder_.action("cancel_request", backtest.orders(0).at(order_id), backtest.current_timestamp());
        }
        // 撤单在途仍预留全部剩余数量；确认前不补新单、不释放预算。
        return;
    }
    if (stop || quantity == 0.0) {
        return;
    }
    const auto order_id = next_order_id_++;
    const double price = static_cast<double>(target_tick) * config_.tick_size;
    if (side == hbt::Side::Buy) {
        backtest.submit_buy_order(0, order_id, price, quantity, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    } else {
        backtest.submit_sell_order(0, order_id, price, quantity, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    }
    recorder_.action("submit", backtest.orders(0).at(order_id), backtest.current_timestamp());
}

void MarketMaker::on_decision(hbt::Backtest& backtest, bool stop) {
    backtest.clear_inactive_orders(0);
    const auto best_bid = backtest.best_bid_tick(0);
    const auto best_ask = backtest.best_ask_tick(0);
    stop = stop || !std::isfinite(backtest.mid(0));
    if (!stop) {
        const double mid_tick = (static_cast<double>(best_bid) + static_cast<double>(best_ask)) * 0.5;
        if (!reference_tick_ || config_.update_mode == "time" ||
            std::abs(mid_tick - *reference_tick_) >= static_cast<double>(config_.price_threshold_ticks)) {
            targets_ = fixed_quotes(best_bid, best_ask, config_.half_width_ticks);
            reference_tick_ = mid_tick;
        }
    }
    quote_side(backtest, hbt::Side::Buy, targets_.bid, stop);
    quote_side(backtest, hbt::Side::Sell, targets_.ask, stop);
    const auto [buys, sells] = reservations(backtest);
    if (backtest.position(0) + buys > config_.max_position + 1e-10 ||
        backtest.position(0) - sells < -config_.max_position - 1e-10) {
        throw std::runtime_error("working and pending orders exceed inventory budget");
    }
}

TradingSummary run_baseline(const TradingConfig& config, std::shared_ptr<hbt::EventReaderCache> reader) {
    validate_trading_config(config);
    TradeRecorder recorder{config};
    detail::TradingEngine engine{config, recorder, std::move(reader)};
    auto& backtest = engine.backtest;
    MarketMaker maker{config, recorder};
    const auto start = std::chrono::steady_clock::now();
    auto status = backtest.elapse(0);
    while (status != hbt::ElapseResult::EndOfData) {
        const auto now = backtest.current_timestamp();
        maker.on_decision(backtest, now >= config.quote_stop_timestamp_ns);
        if (std::isfinite(engine.local->last_mid)) {
            const auto [buys, sells] = maker.reservations(backtest);
            recorder.checkpoint(now, engine.local->last_mid, backtest.state_values(0), buys, sells);
        }
        const auto next = (now / config.decision_interval_ns + 1) * config.decision_interval_ns;
        status = backtest.elapse(next - now);
    }
    // 数据结束也只撤单和排空回报，不捏造平仓成交；尾仓按最后本地中间价计价。
    maker.on_decision(backtest, true);
    backtest.close();
    backtest.clear_inactive_orders(0);
    if (!backtest.orders(0).empty()) {
        throw std::runtime_error("unresolved orders at end of data; increase stop buffer");
    }
    recorder.checkpoint(backtest.current_timestamp(), engine.local->last_mid, backtest.state_values(0), 0, 0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    recorder.finish(backtest.state_values(0), engine.local->last_mid, engine.local->event_count,
                    engine.exchange->event_count, seconds);
    const auto result = recorder.summary();
    std::cout << config.name << " fills=" << result.fills << " pnl=" << result.net_pnl
              << " fees=" << result.fees << " max_inventory=" << result.max_abs_position
              << " seconds=" << seconds << '\n';
    return result;
}

TradingSummary run_baseline(const TradingConfig& config) {
    return run_baseline(config, hbt::event_reader_from_files(hbt::event_files_from_manifest(config.source.string())));
}

} // namespace mm
