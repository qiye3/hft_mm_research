/** 曲线决策与Maker执行共享既有本地/交易所状态，未成交目标不写入实际库存。 */
#include "mm/strategies/curve_policy.hpp"
#include "mm/detail/config.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mm {

// 【核心逻辑280】计划收益、风险、换仓和窗口末端平仓统一为每风险单位bps。
double curve_inventory_target(const ReturnCurve& curve, const ReturnCurve& variances,
        double risk_lambda, double cost_bps, double actual_units) {
    std::array<double, 5> values{};
    std::array<int, 5> first_choice{};
    for (int state = 0; state < 5; ++state) {
        const double units = state - 2;
        values[state] = units * curve[0] - .5 * risk_lambda * variances[0] * units * units
                      - cost_bps * std::abs(units - actual_units);
        first_choice[state] = state;
    }
    for (std::size_t step = 1; step < curve_size; ++step) {
        std::array<double, 5> next{};
        std::array<int, 5> next_first{};
        const double increment = curve[step] - curve[step - 1];
        for (int current = 0; current < 5; ++current) {
            double best = -std::numeric_limits<double>::infinity();
            int predecessor = 0;
            for (int previous = 0; previous < 5; ++previous) {
                const double candidate = values[previous] - cost_bps * std::abs(current - previous);
                if (candidate > best) {
                    best = candidate;
                    predecessor = previous;
                }
            }
            const double units = current - 2;
            next[current] = best + units * increment - .5 * risk_lambda * variances[step] * units * units;
            next_first[current] = first_choice[predecessor];
        }
        values = next;
        first_choice = next_first;
    }
    double best = -std::numeric_limits<double>::infinity();
    int selected = 2;
    for (int state = 0; state < 5; ++state) {
        const double value = values[state] - cost_bps * std::abs(state - 2);
        if (value > best) {
            best = value;
            selected = first_choice[state];
        }
    }
    return selected - 2;
}

// 【非核心】分别载入模式、仓位单位、决策间隔和独立订单机会参数。
CurveParameters load_curve_parameters(const std::filesystem::path& filename) {
    const auto fields = detail::read_flat_config(filename, {"mode", "risk_lambda", "cost_buffer_bps",
        "risk_unit_btc", "horizon_index", "rebalance_interval_ns", "opportunity_interval_ns",
        "opportunity_side", "edge_minimum_bps", "minimum_notional_usdt"});
    CurveParameters parameters{
        fields.at("mode"), detail::config_number<double>(fields, "risk_lambda"),
        detail::config_number<double>(fields, "cost_buffer_bps"), detail::config_number<double>(fields, "risk_unit_btc"),
        detail::config_number<std::size_t>(fields, "horizon_index"),
        detail::config_number<std::int64_t>(fields, "rebalance_interval_ns"),
        detail::config_number<std::int64_t>(fields, "opportunity_interval_ns"),
        detail::config_number<int>(fields, "opportunity_side"), detail::config_number<double>(fields, "edge_minimum_bps"),
        detail::config_number<double>(fields, "minimum_notional_usdt")};
    const bool mode_known = parameters.mode == "fixed" || parameters.mode == "single" || parameters.mode == "curve"
        || parameters.mode == "curve_edge" || parameters.mode == "opportunity";
    if (!mode_known || parameters.risk_lambda <= 0 || parameters.risk_unit_btc <= 0
        || parameters.horizon_index >= curve_size || parameters.rebalance_interval_ns < 1'000'000'000
        || parameters.opportunity_interval_ns < 400'000'000'000
        || std::abs(parameters.opportunity_side) != 1 || parameters.minimum_notional_usdt < 0) {
        throw std::invalid_argument("invalid curve parameters");
    }
    return parameters;
}

namespace {
struct CurveRow {
    std::int64_t timestamp;
    ReturnCurve prediction;
    double edge_buy_bps;
    double edge_sell_bps;
};

// 【非核心】读取唯一约定的timestamp、16期限和执行价值列，缺失预测以NaN保留。
std::vector<CurveRow> read_curve_rows(const std::filesystem::path& filename) {
    std::ifstream stream(filename);
    std::string line;
    if (!stream || !std::getline(stream, line)
        || line != "timestamp_ns,h1,h2,h3,h5,h8,h10,h15,h20,h30,h45,h60,h90,h120,h180,h240,h300,edge_buy_bps,edge_sell_bps") {
        throw std::invalid_argument("curve CSV header mismatch");
    }
    std::vector<CurveRow> rows;
    while (std::getline(stream, line)) {
        std::istringstream fields(line);
        std::string value;
        std::getline(fields, value, ',');
        CurveRow row{};
        row.timestamp = std::stoll(value);
        for (double& prediction : row.prediction) {
            std::getline(fields, value, ',');
            prediction = std::stod(value);
        }
        std::getline(fields, value, ',');
        row.edge_buy_bps = std::stod(value);
        std::getline(fields, value, ',');
        row.edge_sell_bps = std::stod(value);
        if (row.timestamp % 1'000'000'000 != 0 || (!rows.empty() && row.timestamp - rows.back().timestamp != 1'000'000'000)) {
            throw std::invalid_argument("curve predictions require complete UTC seconds");
        }
        rows.push_back(row);
    }
    if (rows.empty()) {
        throw std::invalid_argument("curve prediction file empty");
    }
    return rows;
}

// 【非核心】风险文件以两行保存分段方差和累计方差，单位均为bps平方。
std::pair<ReturnCurve, ReturnCurve> read_risk(const std::filesystem::path& filename) {
    std::ifstream stream(filename);
    ReturnCurve increments{};
    ReturnCurve cumulative{};
    for (double& value : increments) {
        if (!(stream >> value) || !std::isfinite(value) || value <= 0) {
            throw std::invalid_argument("invalid increment risk");
        }
    }
    for (double& value : cumulative) {
        if (!(stream >> value) || !std::isfinite(value) || value <= 0) {
            throw std::invalid_argument("invalid cumulative risk");
        }
    }
    return {increments, cumulative};
}
} // namespace

// 【核心逻辑281】曲线只在推理完成后可见，每100ms用实际库存和原生在途订单生成目标。
TradingSummary run_curve_replay(const TradingConfig& config, const ExecutionParameters& execution,
        const CurveParameters& parameters, const std::filesystem::path& predictions_file,
        const std::filesystem::path& risk_file) {
    validate_trading_config(config);
    const auto rows = read_curve_rows(predictions_file);
    const auto [increment_risk, cumulative_risk] = read_risk(risk_file);
    TradeRecorder recorder{config};
    const auto queue = execution.queue_model == "risk_adverse"
        ? detail::QueueAssumption::RiskAdverse : detail::QueueAssumption::LogProbability;
    detail::TradingEngine engine{config, recorder,
        hbt::event_reader_from_files(hbt::event_files_from_manifest(config.source.string())), true, queue};
    ExecutionPolicy orders{config, execution, recorder, engine.local->depth()};
    auto& backtest = engine.backtest;
    std::ofstream liquidation(config.output / "liquidation_equity.csv");
    std::ofstream decisions(config.output / "curve_decisions.csv");
    liquidation << std::setprecision(17)
        << "local_timestamp_ns,mid,bid,ask,position,net_pnl_liquidated,book_fresh\n";
    decisions << "local_timestamp_ns,target_units,actual_units,reserved_buy,reserved_sell\n";
    std::size_t cursor = 0;
    std::int64_t last_rebalance = 0;
    std::int64_t opened_at = 0;
    std::int64_t opportunity_slot = -1;
    std::int64_t opportunity_started = 0;
    bool opportunity_submitted = false;
    double target_units = 0;
    int previous_sign = 0;
    bool reducing = false;
    const auto started = std::chrono::steady_clock::now();
    auto status = backtest.elapse(0);
    while (status != hbt::ElapseResult::EndOfData) {
        const auto now = backtest.current_timestamp();
        while (cursor + 1 < rows.size() && rows[cursor + 1].timestamp + execution.prediction_latency_ns <= now) {
            ++cursor;
        }
        const auto& row = rows[cursor];
        const double position = backtest.position(0);
        const int sign = position > config.lot_size * .5 ? 1 : (position < -config.lot_size * .5 ? -1 : 0);
        if (sign != previous_sign) {
            opened_at = sign == 0 ? 0 : now;
            previous_sign = sign;
            if (sign == 0) {
                reducing = false;
            }
        }
        const bool fresh = engine.local->last_book_timestamp_ns > 0
            && now - engine.local->last_book_timestamp_ns <= 1'000'000'000
            && backtest.best_bid_tick(0) < backtest.best_ask_tick(0)
            && std::isfinite(backtest.mid(0)) && backtest.best_bid(0) > 0;
        const bool visible = row.timestamp + execution.prediction_latency_ns <= now
            && now - row.timestamp - execution.prediction_latency_ns < 1'000'000'000;
        const bool final_exit = now >= config.quote_stop_timestamp_ns;
        if (sign != 0 && now - opened_at >= execution.max_hold_ns) {
            reducing = true;
        }
        const auto slot = now / parameters.opportunity_interval_ns;
        if (parameters.mode == "opportunity" && slot != opportunity_slot) {
            opportunity_slot = slot;
            opportunity_started = now;
            opportunity_submitted = false;
        }
        std::pair<double, double> reserved;
        if (final_exit || reducing) {
            reserved = orders.liquidate(backtest, final_exit, fresh);
        } else if (!fresh || !visible || now < config.quote_start_timestamp_ns) {
            reserved = orders.post(backtest, std::nullopt, std::nullopt, 0, 0);
        } else {
            const double actual_units = position / parameters.risk_unit_btc;
            const double mid = backtest.mid(0);
            const double cost = config.maker_fee * 10000 + (backtest.best_ask(0) - backtest.best_bid(0)) / mid * 5000
                              + parameters.cost_buffer_bps;
            if (parameters.mode == "opportunity") {
                target_units = parameters.opportunity_side;
            } else if (now - last_rebalance >= parameters.rebalance_interval_ns) {
                const double alpha = row.prediction[parameters.horizon_index];
                if (parameters.mode == "single" && std::isfinite(alpha)) {
                    const double raw = std::clamp(std::round(alpha / (parameters.risk_lambda * cumulative_risk[parameters.horizon_index])), -2., 2.);
                    const double benefit = (raw - actual_units) * alpha
                        - .5 * parameters.risk_lambda * cumulative_risk[parameters.horizon_index] * (raw * raw - actual_units * actual_units);
                    if (benefit > cost * std::abs(raw - actual_units)) {
                        target_units = raw;
                    }
                } else if (parameters.mode != "fixed"
                    && std::all_of(row.prediction.begin(), row.prediction.end(), [](double value) { return std::isfinite(value); })) {
                    target_units = curve_inventory_target(row.prediction, increment_risk, parameters.risk_lambda, cost, actual_units);
                }
                last_rebalance = now;
            }
            const double change = target_units * parameters.risk_unit_btc - position;
            double buy_budget = std::max(0., change);
            double sell_budget = std::max(0., -change);
            if (parameters.mode == "fixed") {
                buy_budget = config.order_quantity;
                sell_budget = config.order_quantity;
            }
            if (position * target_units < 0) {
                buy_budget = position < 0 ? -position : 0;
                sell_budget = position > 0 ? position : 0;
            }
            std::optional<std::int64_t> bid;
            std::optional<std::int64_t> ask;
            bool allow_entry = true;
            if (parameters.mode == "curve_edge" && std::abs(position) < config.lot_size * .5) {
                const double edge = target_units > 0 ? row.edge_buy_bps : row.edge_sell_bps;
                allow_entry = std::isfinite(edge) && edge > parameters.edge_minimum_bps;
            }
            if (parameters.mode == "opportunity") {
                allow_entry = !opportunity_submitted && now - opportunity_started < execution.ttl_ns;
                if (opportunity_submitted && std::abs(position) < config.lot_size * .5) {
                    buy_budget = 0;
                    sell_budget = 0;
                }
            }
            if (allow_entry && buy_budget >= config.lot_size * .5) {
                bid = backtest.best_bid_tick(0) - execution.offset_ticks;
            }
            if (allow_entry && sell_budget >= config.lot_size * .5) {
                ask = backtest.best_ask_tick(0) + execution.offset_ticks;
            }
            if (parameters.mode == "opportunity" && opportunity_submitted) {
                for (const auto& [identifier, order] : backtest.orders(0)) {
                    (void)identifier;
                    if (order.active() && now - opportunity_started < execution.ttl_ns) {
                        if (order.side == hbt::Side::Buy) {
                            bid = order.price_tick;
                            buy_budget = order.leaves_qty;
                        } else {
                            ask = order.price_tick;
                            sell_budget = order.leaves_qty;
                        }
                    }
                }
            }
            if (parameters.mode != "opportunity") {
                buy_budget = std::min(buy_budget, config.max_position - position);
                sell_budget = std::min(sell_budget, config.max_position + position);
                if (position >= 0 && buy_budget * backtest.best_bid(0) < parameters.minimum_notional_usdt) {
                    bid.reset();
                    buy_budget = 0;
                }
                if (position <= 0 && sell_budget * backtest.best_ask(0) < parameters.minimum_notional_usdt) {
                    ask.reset();
                    sell_budget = 0;
                }
            }
            reserved = orders.post(backtest, bid, ask, buy_budget, sell_budget);
            if (parameters.mode == "opportunity" && !opportunity_submitted
                && (reserved.first > 0 || reserved.second > 0)) {
                opportunity_started = now;
                opportunity_submitted = true;
            }
        }
        if (std::isfinite(engine.local->last_mid)) {
            recorder.checkpoint(now, engine.local->last_mid, backtest.state_values(0), reserved.first, reserved.second);
            const auto& account = backtest.state_values(0);
            const double price = account.position >= 0 ? backtest.best_bid(0) : backtest.best_ask(0);
            const double value = account.position * price;
            liquidation << now << ',' << engine.local->last_mid << ',' << backtest.best_bid(0) << ','
                        << backtest.best_ask(0) << ',' << account.position << ','
                        << account.balance + value - account.fee - std::abs(value) * config.taker_fee << ',' << fresh << '\n';
            decisions << now << ',' << target_units << ',' << position / parameters.risk_unit_btc << ','
                      << reserved.first << ',' << reserved.second << '\n';
        }
        const auto next = (now / config.decision_interval_ns + 1) * config.decision_interval_ns;
        status = backtest.elapse(next - now);
    }
    backtest.close();
    backtest.clear_inactive_orders(0);
    if (!backtest.orders(0).empty() || std::abs(backtest.position(0)) > 1e-10) {
        throw std::runtime_error("curve replay ended with unresolved orders or inventory");
    }
    recorder.checkpoint(backtest.current_timestamp(), engine.local->last_mid, backtest.state_values(0), 0, 0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    recorder.finish(backtest.state_values(0), engine.local->last_mid, engine.local->event_count, engine.exchange->event_count, seconds);
    liquidation.flush();
    decisions.flush();
    if (!liquidation || !decisions) {
        throw std::runtime_error("failed writing curve replay diagnostics");
    }
    return recorder.summary();
}
} // namespace mm
