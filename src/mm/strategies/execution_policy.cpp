/** 预测驱动的挂单、报价偏移、目标库存及Maker/Taker混合策略。 */
#include "mm/strategies/execution_policy.hpp"
#include "mm/detail/config.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mm {

// 【非核心】逐项解析本阶段唯一的策略参数格式，非法枚举或不合理时长明确失败。
ExecutionParameters load_execution_parameters(const std::filesystem::path& filename) {
    const auto fields = detail::read_flat_config(filename, {"mode", "queue_model", "entry_bps", "exit_bps",
        "scale_bps", "max_alpha_bps", "inventory_skew_bps", "cost_filter", "offset_ticks", "ttl_ns",
        "max_hold_ns", "prediction_latency_ns", "taker_slippage_ticks", "quote_half_width_bps", "minimum_notional_usdt"});
    ExecutionParameters parameters{
        fields.at("mode"), fields.at("queue_model"), detail::config_number<double>(fields, "entry_bps"),
        detail::config_number<double>(fields, "exit_bps"), detail::config_number<double>(fields, "scale_bps"),
        detail::config_number<double>(fields, "max_alpha_bps"), detail::config_number<double>(fields, "inventory_skew_bps"),
        detail::config_number<int>(fields, "cost_filter") != 0, detail::config_number<std::int64_t>(fields, "offset_ticks"),
        detail::config_number<std::int64_t>(fields, "ttl_ns"), detail::config_number<std::int64_t>(fields, "max_hold_ns"),
        detail::config_number<std::int64_t>(fields, "prediction_latency_ns"),
        detail::config_number<std::int64_t>(fields, "taker_slippage_ticks"),
        detail::config_number<double>(fields, "quote_half_width_bps"),
        detail::config_number<double>(fields, "minimum_notional_usdt")};
    const bool known_mode = parameters.mode == "fixed" || parameters.mode == "passive"
        || parameters.mode == "skew" || parameters.mode == "inventory" || parameters.mode == "hybrid"
        || parameters.mode == "directional" || parameters.mode == "defensive";
    if (!known_mode || (parameters.queue_model != "risk_adverse" && parameters.queue_model != "log_probability")
        || parameters.entry_bps < 0 || parameters.exit_bps < 0 || parameters.exit_bps > parameters.entry_bps
        || parameters.scale_bps <= 0 || parameters.max_alpha_bps <= 0 || parameters.inventory_skew_bps < 0
        || parameters.offset_ticks < 0 || parameters.ttl_ns <= 0 || parameters.max_hold_ns <= 0
        || parameters.prediction_latency_ns < 0 || parameters.taker_slippage_ticks < 0
        || !std::isfinite(parameters.quote_half_width_bps) || parameters.quote_half_width_bps < 0
        || !std::isfinite(parameters.minimum_notional_usdt) || parameters.minimum_notional_usdt < 0) {
        throw std::invalid_argument("invalid execution parameters");
    }
    return parameters;
}

// 【核心逻辑236】预测CSV仅接受时间键和bps预测两列，整秒间距保证交易网格完整。
PredictionSeries::PredictionSeries(const std::filesystem::path& filename) {
    std::ifstream input(filename);
    std::string line;
    if (!input || !std::getline(input, line) || line != "timestamp_ns,prediction_bps") {
        throw std::invalid_argument("prediction CSV requires timestamp_ns,prediction_bps");
    }
    while (std::getline(input, line)) {
        const auto separator = line.find(',');
        const auto timestamp = std::stoll(line.substr(0, separator));
        const double value = std::stod(line.substr(separator + 1));
        if (timestamp % 1'000'000'000 != 0 || (!values_.empty()
            && timestamp - values_.back().first != 1'000'000'000)) {
            throw std::invalid_argument("prediction grid must be complete UTC seconds");
        }
        values_.emplace_back(timestamp, value);
    }
    if (values_.empty()) {
        throw std::invalid_argument("prediction file is empty");
    }
}

// 【核心逻辑239】推理延迟加在决策时间键之后，等待期间保留尚未过期的上一秒预测。
double PredictionSeries::at(std::int64_t timestamp, std::int64_t latency) {
    while (cursor_ + 1 < values_.size() && values_[cursor_ + 1].first + latency <= timestamp) {
        ++cursor_;
    }
    const auto ready_at = values_[cursor_].first + latency;
    if (ready_at > timestamp || timestamp - ready_at >= 1'000'000'000) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return values_[cursor_].second;
}

// 【非核心】订单ID、目标库存及持有时间只服务于策略决策，订单状态从Backtest读取。
ExecutionPolicy::ExecutionPolicy(const TradingConfig& config, const ExecutionParameters& parameters,
                                 TradeRecorder& recorder, const hbt::HashMapMarketDepth& depth)
    : config_(config), parameters_(parameters), recorder_(recorder), depth_(depth), maker_(config, recorder),
      ioc_actions_(config.output / "ioc_actions.csv") {
    if (!ioc_actions_) {
        throw std::runtime_error("cannot create IOC purpose log");
    }
    ioc_actions_ << "order_id,local_timestamp_ns,purpose\n";
}

// 【核心逻辑242】TTL从原生本地下单时刻计算，撤单在途期间仍由现有模块保留风险预算。
std::pair<double, double> ExecutionPolicy::post(hbt::Backtest& backtest,
        std::optional<std::int64_t> bid, std::optional<std::int64_t> ask,
        double buy_budget, double sell_budget) {
    for (const auto& [identifier, order] : backtest.orders(0)) {
        (void)identifier;
        if (order.cancellable() && backtest.current_timestamp() - order.local_timestamp >= parameters_.ttl_ns) {
            if (order.side == hbt::Side::Buy) {
                bid.reset();
            } else {
                ask.reset();
            }
        }
    }
    // 开仓和增加风险须满足最小名义金额；纯减仓仍按实际剩余仓位退出。
    const double position = backtest.position(0);
    const double buy_quantity = std::floor(std::min({buy_budget, config_.order_quantity,
        config_.max_position - position}) / config_.lot_size + 1e-8) * config_.lot_size;
    const double sell_quantity = std::floor(std::min({sell_budget, config_.order_quantity,
        config_.max_position + position}) / config_.lot_size + 1e-8) * config_.lot_size;
    if (bid && buy_quantity > std::max(0., -position) + 1e-10
        && buy_quantity * *bid * config_.tick_size < parameters_.minimum_notional_usdt) {
        bid.reset();
    }
    if (ask && sell_quantity > std::max(0., position) + 1e-10
        && sell_quantity * *ask * config_.tick_size < parameters_.minimum_notional_usdt) {
        ask.reset();
    }
    return maker_.quote_targets(backtest, bid, ask, buy_budget, sell_budget);
}

// 【核心逻辑243】每次IOC至多一个最小下单批次，并限制在本地对手买卖一的显示数量内。
void ExecutionPolicy::take(hbt::Backtest& backtest, hbt::Side side, double quantity, const char* purpose) {
    backtest.clear_inactive_orders(0);
    if (!backtest.orders(0).empty()) {
        return;
    }
    const auto touch = side == hbt::Side::Buy ? backtest.best_ask_tick(0) : backtest.best_bid_tick(0);
    const double displayed = side == hbt::Side::Buy ? depth_.ask_qty_at_tick(touch) : depth_.bid_qty_at_tick(touch);
    const auto lots = static_cast<std::int64_t>(std::floor(
        std::min({quantity, config_.order_quantity, displayed}) / config_.lot_size + 1e-8));
    if (lots == 0) {
        return;
    }
    const double size = static_cast<double>(lots) * config_.lot_size;
    if (std::string{purpose} == "alpha_taker"
        && size * touch * config_.tick_size < parameters_.minimum_notional_usdt) {
        return;
    }
    const auto limit_tick = touch + (side == hbt::Side::Buy ? 1 : -1) * parameters_.taker_slippage_ticks;
    const auto identifier = next_taker_id_++;
    if (side == hbt::Side::Buy) {
        backtest.submit_buy_order(0, identifier, limit_tick * config_.tick_size, size,
                                  hbt::TimeInForce::IOC, hbt::OrdType::Limit, false);
    } else {
        backtest.submit_sell_order(0, identifier, limit_tick * config_.tick_size, size,
                                   hbt::TimeInForce::IOC, hbt::OrdType::Limit, false);
    }
    recorder_.action("submit", backtest.orders(0).at(identifier), backtest.current_timestamp());
    ioc_actions_ << identifier << ',' << backtest.current_timestamp() << ',' << purpose << '\n';
    ioc_actions_.flush();
    if (!ioc_actions_) {
        throw std::runtime_error("failed writing IOC purpose");
    }
}

// 【核心逻辑279】先等待Maker撤单终结，再按实际仓位逐批IOC减仓，复用现有在途风险预算。
std::pair<double, double> ExecutionPolicy::liquidate(hbt::Backtest& backtest, bool final_exit, bool book_fresh) {
    const auto reserved = maker_.quote_targets(backtest, std::nullopt, std::nullopt, 0, 0);
    const double position = backtest.position(0);
    if (book_fresh && std::abs(position) >= config_.lot_size * .5
        && reserved.first == 0 && reserved.second == 0) {
        take(backtest, position > 0 ? hbt::Side::Sell : hbt::Side::Buy, std::abs(position),
             final_exit ? "final_liquidation" : "max_hold");
    }
    return maker_.reservations(backtest);
}

// 【核心逻辑240】分离开仓信号、库存目标与风险退出，所有委托仍通过原生Backtest执行。
std::pair<double, double> ExecutionPolicy::decide(hbt::Backtest& backtest, double alpha,
        bool quotes_allowed, bool final_exit, bool book_fresh) {
    const auto now = backtest.current_timestamp();
    const double position = backtest.position(0);
    const int sign = position > config_.lot_size * .5 ? 1 : (position < -config_.lot_size * .5 ? -1 : 0);
    if (sign != position_sign_) {
        opened_at_ = sign == 0 ? 0 : now;
        position_sign_ = sign;
        if (sign == 0) {
            reducing_ = false;
        }
    }
    if (sign != 0 && now - opened_at_ >= parameters_.max_hold_ns) {
        reducing_ = true;
    }
    const bool valid_book = book_fresh && std::isfinite(backtest.mid(0))
        && backtest.best_bid_tick(0) < backtest.best_ask_tick(0);
    if (final_exit || reducing_) {
        return liquidate(backtest, final_exit, valid_book);
    }
    if (!quotes_allowed || !valid_book || !std::isfinite(alpha)) {
        return maker_.quote_targets(backtest, std::nullopt, std::nullopt, 0, 0);
    }
    const double mid = backtest.mid(0);
    const double spread_bps = (backtest.best_ask(0) - backtest.best_bid(0)) / mid * 10000;
    const double entry_gate = parameters_.cost_filter
        ? std::max(parameters_.entry_bps, 2 * config_.maker_fee * 10000 + spread_bps)
        : parameters_.entry_bps;
    // 强反向Alpha触发实际库存减仓，避免宽报价在持续行情中长期积累风险。
    if (parameters_.mode == "defensive"
        && ((sign > 0 && alpha <= -parameters_.exit_bps)
            || (sign < 0 && alpha >= parameters_.exit_bps))) {
        reducing_ = true;
        return liquidate(backtest, false, valid_book);
    }
    // 固定持有实验在空仓时按信号IOC开仓，持仓超时沿用上方原生减仓路径。
    if (parameters_.mode == "directional") {
        if (sign == 0 && alpha != 0 && std::abs(alpha) >= entry_gate) {
            take(backtest, alpha > 0 ? hbt::Side::Buy : hbt::Side::Sell,
                 config_.order_quantity, "alpha_taker");
        }
        return maker_.reservations(backtest);
    }
    const double shifted = std::clamp(alpha - parameters_.inventory_skew_bps * position / config_.max_position,
                                      -parameters_.max_alpha_bps, parameters_.max_alpha_bps);
    double shift_bps = shifted;
    if (parameters_.mode == "fixed" || parameters_.mode == "passive" || parameters_.mode == "defensive") {
        shift_bps = 0;
    }
    // 模型输出为对数收益bps，先转换参考价格，再按当前价格计算报价半宽。
    const double center_tick = mid / config_.tick_size * std::exp(shift_bps * .0001);
    const double half_width = mid / config_.tick_size * parameters_.quote_half_width_bps * .0001;
    const auto maker_bid = std::min(backtest.best_bid_tick(0),
        static_cast<std::int64_t>(std::floor(center_tick - half_width))) - parameters_.offset_ticks;
    const auto maker_ask = std::max(backtest.best_ask_tick(0),
        static_cast<std::int64_t>(std::ceil(center_tick + half_width))) + parameters_.offset_ticks;
    std::optional<std::int64_t> bid = maker_bid;
    std::optional<std::int64_t> ask = maker_ask;
    double buy_budget = config_.order_quantity;
    double sell_budget = config_.order_quantity;
    if (parameters_.mode == "passive") {
        bid.reset();
        ask.reset();
        if (sign == 0 && alpha >= entry_gate) {
            bid = maker_bid;
        } else if (sign == 0 && alpha <= -entry_gate) {
            ask = maker_ask;
        } else if (sign > 0 && alpha < parameters_.exit_bps) {
            ask = backtest.best_ask_tick(0) + parameters_.offset_ticks;
            sell_budget = std::min(config_.order_quantity, position);
        } else if (sign < 0 && alpha > -parameters_.exit_bps) {
            bid = backtest.best_bid_tick(0) - parameters_.offset_ticks;
            buy_budget = std::min(config_.order_quantity, -position);
        }
    } else if (parameters_.mode == "defensive") {
        if (alpha >= entry_gate) {
            ask.reset();
        } else if (alpha <= -entry_gate) {
            bid.reset();
        }
    } else if (parameters_.mode == "skew") {
        if (alpha >= entry_gate && sign <= 0) {
            ask.reset();
        } else if (alpha <= -entry_gate && sign >= 0) {
            bid.reset();
        }
        if (alpha >= entry_gate && sign > 0) {
            sell_budget = std::min(sell_budget, position);
        } else if (alpha <= -entry_gate && sign < 0) {
            buy_budget = std::min(buy_budget, -position);
        }
    } else if (parameters_.mode == "inventory" || parameters_.mode == "hybrid") {
        if (std::abs(alpha) >= entry_gate) {
            target_position_ = config_.max_position * std::clamp(alpha / parameters_.scale_bps, -1.0, 1.0);
            target_position_ = std::trunc(target_position_ / config_.lot_size) * config_.lot_size;
        } else if (std::abs(alpha) <= parameters_.exit_bps) {
            target_position_ = 0;
        }
        double change = target_position_ - position;
        if (position * target_position_ < 0) {
            change = -position;
        }
        buy_budget = std::max(0.0, change);
        sell_budget = std::max(0.0, -change);
        if (buy_budget < config_.lot_size * .5) {
            bid.reset();
        }
        if (sell_budget < config_.lot_size * .5) {
            ask.reset();
        }
        const double taker_gate = (config_.taker_fee + config_.maker_fee) * 10000 + spread_bps
                                 + parameters_.taker_slippage_ticks * config_.tick_size / mid * 10000;
        if (parameters_.mode == "hybrid" && std::abs(alpha) >= std::max(entry_gate, taker_gate)
            && alpha * change > 0 && std::abs(change) >= config_.lot_size * .5) {
            maker_.quote_targets(backtest, std::nullopt, std::nullopt, 0, 0);
            take(backtest, change > 0 ? hbt::Side::Buy : hbt::Side::Sell, std::abs(change), "alpha_taker");
            return maker_.reservations(backtest);
        }
    }
    return post(backtest, bid, ask, buy_budget, sell_budget);
}

// 【核心逻辑237】在100ms本地时钟决策、事件级成交，日末预留20秒等待撤单及IOC平仓。
TradingSummary run_execution_replay(const TradingConfig& config, const ExecutionParameters& parameters,
        PredictionSeries& predictions, std::shared_ptr<hbt::EventReaderCache> reader) {
    validate_trading_config(config);
    TradeRecorder recorder{config};
    std::ofstream liquidation(config.output / "liquidation_equity.csv");
    liquidation << std::setprecision(17);
    liquidation << "local_timestamp_ns,mid,bid,ask,position,net_pnl_liquidated,book_fresh\n";
    const auto queue = parameters.queue_model == "risk_adverse"
        ? detail::QueueAssumption::RiskAdverse : detail::QueueAssumption::LogProbability;
    detail::TradingEngine engine{config, recorder, std::move(reader), true, queue};
    auto& backtest = engine.backtest;
    ExecutionPolicy policy{config, parameters, recorder, engine.local->depth()};
    const auto started = std::chrono::steady_clock::now();
    auto status = backtest.elapse(0);
    while (status != hbt::ElapseResult::EndOfData) {
        const auto now = backtest.current_timestamp();
        const bool fresh = engine.local->last_book_timestamp_ns > 0
            && now - engine.local->last_book_timestamp_ns <= 1'000'000'000;
        const double alpha = parameters.mode == "fixed" ? 0 : predictions.at(now, parameters.prediction_latency_ns);
        const auto reserved = policy.decide(backtest, alpha, now >= config.quote_start_timestamp_ns,
                                            now >= config.quote_stop_timestamp_ns, fresh);
        if (std::isfinite(engine.local->last_mid)) {
            recorder.checkpoint(now, engine.local->last_mid, backtest.state_values(0), reserved.first, reserved.second);
            const auto& state = backtest.state_values(0);
            const double liquidation_price = state.position >= 0 ? backtest.best_bid(0) : backtest.best_ask(0);
            const double value = state.position * liquidation_price;
            const double pnl = state.balance + value - state.fee - std::abs(value) * config.taker_fee;
            liquidation << now << ',' << engine.local->last_mid << ',' << backtest.best_bid(0) << ','
                        << backtest.best_ask(0) << ',' << state.position << ',' << pnl << ',' << fresh << '\n';
        }
        const auto next = (now / config.decision_interval_ns + 1) * config.decision_interval_ns;
        status = backtest.elapse(next - now);
    }
    backtest.close();
    backtest.clear_inactive_orders(0);
    if (!backtest.orders(0).empty() || std::abs(backtest.position(0)) >= config.lot_size * .5) {
        throw std::runtime_error("execution replay ended with unresolved orders or unliquidated inventory");
    }
    recorder.checkpoint(backtest.current_timestamp(), engine.local->last_mid, backtest.state_values(0), 0, 0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    recorder.finish(backtest.state_values(0), engine.local->last_mid, engine.local->event_count,
                     engine.exchange->event_count, seconds);
    liquidation.flush();
    if (!liquidation) {
        throw std::runtime_error("failed writing execution diagnostics");
    }
    std::cout << config.name << " net_pnl=" << recorder.summary().net_pnl
              << " fills=" << recorder.summary().fills << " seconds=" << seconds << '\n';
    return recorder.summary();
}

} // namespace mm
