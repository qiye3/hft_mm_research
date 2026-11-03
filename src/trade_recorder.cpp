#include "mm/trade_recorder.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace mm {
namespace {

void check_close(double actual, double expected, const char* field) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-7) {
        throw std::runtime_error(std::string("ledger mismatch: ") + field);
    }
}

std::ofstream csv_file(const std::filesystem::path& filename) {
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("cannot create output: " + filename.string());
    }
    output << std::setprecision(17);
    return output;
}

} // namespace

std::string order_status(hbt::Status status) {
    switch (status) {
    case hbt::Status::None: return "none";
    case hbt::Status::New: return "new";
    case hbt::Status::Expired: return "expired";
    case hbt::Status::Filled: return "filled";
    case hbt::Status::PartiallyFilled: return "partial";
    case hbt::Status::Canceled: return "canceled";
    case hbt::Status::Rejected: return "rejected";
    case hbt::Status::Replaced: return "replaced";
    default: throw std::runtime_error("unknown order status");
    }
}

TradeRecorder::TradeRecorder(const TradingConfig& config)
    : config_(config), peak_equity_(config.initial_capital) {
    if (std::filesystem::exists(config.output)) {
        throw std::runtime_error("output directory already exists: " + config.output.string());
    }
    std::filesystem::create_directories(config.output);
    orders_ = csv_file(config.output / "orders.csv");
    fills_ = csv_file(config.output / "fills.csv");
    equity_ = csv_file(config.output / "equity.csv");
    orders_ << "kind,exchange_timestamp_ns,local_timestamp_ns,order_id,side,price,quantity,leaves_quantity,status,request,exec_price,exec_quantity,maker\n";
    fills_ << "fill_id,order_id,exchange_timestamp_ns,local_timestamp_ns,side,price,quantity,maker,fee,position,balance,average_cost,realized_gross,mid,net_pnl\n";
    equity_ << "local_timestamp_ns,mid,position,balance,fees,inventory_value,realized_gross,unrealized,net_pnl,equity,reserved_buy,reserved_sell,running_max_drawdown\n";
}

void TradeRecorder::write_order(const std::string& kind, const hbt::Order<>& order, std::int64_t timestamp) {
    orders_ << kind << ',' << order.exch_timestamp << ',' << timestamp << ',' << order.order_id << ','
            << static_cast<int>(order.side) << ',' << order.price() << ',' << order.qty << ','
            << order.leaves_qty << ',' << order_status(order.status) << ',' << order_status(order.req)
            << ',' << order.exec_price() << ',' << order.exec_qty << ',' << order.maker << '\n';
}

void TradeRecorder::action(const std::string& kind, const hbt::Order<>& order, std::int64_t local_timestamp) {
    if (kind == "submit") {
        ++summary_.submissions;
    } else if (kind == "cancel_request") {
        ++summary_.cancel_requests;
    } else {
        throw std::invalid_argument("unknown order action");
    }
    write_order(kind, order, local_timestamp);
}

void TradeRecorder::apply_fill(const hbt::Order<>& order, std::int64_t timestamp, double mid) {
    const double quantity = order.exec_qty;
    const double price = order.exec_price();
    const double signed_quantity = hbt::side_sign(order.side) * quantity;
    const double old_position = summary_.position;
    const double new_position = old_position + signed_quantity;
    if (old_position == 0.0 || old_position * signed_quantity > 0) {
        summary_.average_cost = (std::abs(old_position) * summary_.average_cost + quantity * price)
                                 / (std::abs(old_position) + quantity);
    } else {
        const double closed_quantity = std::min(std::abs(old_position), quantity);
        const double old_sign = old_position > 0 ? 1.0 : -1.0;
        summary_.realized_gross += closed_quantity * (price - summary_.average_cost) * old_sign;
        if (std::abs(new_position) < 1e-12) {
            summary_.average_cost = 0.0;
        } else if (old_position * new_position < 0) {
            summary_.average_cost = price;
        }
    }
    summary_.position = std::abs(new_position) < 1e-12 ? 0.0 : new_position;
    summary_.balance -= signed_quantity * price;
    const double fee = price * quantity * (order.maker ? config_.maker_fee : config_.taker_fee);
    summary_.fees += fee;
    summary_.volume += quantity;
    summary_.turnover += price * quantity;
    ++summary_.fills;
    if (order.status == hbt::Status::PartiallyFilled) {
        ++summary_.partial_fills;
    }
    const double pnl = summary_.balance + summary_.position * mid - summary_.fees;
    fills_ << summary_.fills << ',' << order.order_id << ',' << order.exch_timestamp << ',' << timestamp
           << ',' << static_cast<int>(order.side) << ',' << price << ',' << quantity << ',' << order.maker
           << ',' << fee << ',' << summary_.position << ',' << summary_.balance << ',' << summary_.average_cost
           << ',' << summary_.realized_gross << ',' << mid << ',' << pnl << '\n';
    if (std::abs(summary_.position) > config_.max_position + 1e-10) {
        throw std::runtime_error("inventory limit violated by a received fill");
    }
    // 成交也进入权益序列；不会漏掉两个决策点之间已经发生的库存变化。
    record_equity(timestamp, mid, std::nullopt, std::nullopt);
}

void TradeRecorder::response(const hbt::Order<>& order, std::int64_t local_timestamp, double mid) {
    write_order("response", order, local_timestamp);
    if (order.req == hbt::Status::Rejected) {
        ++summary_.rejected;
        return;
    }
    if (order.status == hbt::Status::Canceled) {
        ++summary_.cancellations;
    } else if (order.status == hbt::Status::Expired) {
        ++summary_.expired;
    }
    if ((order.status == hbt::Status::Filled || order.status == hbt::Status::PartiallyFilled)
        && order.exec_qty > 0.0) {
        apply_fill(order, local_timestamp, mid);
    }
}

void TradeRecorder::reconcile(const hbt::StateValues& state) const {
    check_close(state.position, summary_.position, "position");
    check_close(state.balance, summary_.balance, "balance");
    check_close(state.fee, summary_.fees, "fees");
    check_close(state.trading_volume, summary_.volume, "volume");
    check_close(state.trading_value, summary_.turnover, "turnover");
    if (state.num_trades != static_cast<std::int64_t>(summary_.fills)) {
        throw std::runtime_error("ledger mismatch: fill count");
    }
}

void TradeRecorder::record_equity(std::int64_t timestamp, double mid, std::optional<double> reserved_buy,
                                   std::optional<double> reserved_sell) {
    if (!std::isfinite(mid)) {
        throw std::runtime_error("cannot mark inventory without a finite mid-price");
    }
    const double inventory_value = summary_.position * mid;
    const double unrealized = summary_.position * (mid - summary_.average_cost);
    summary_.net_pnl = summary_.balance + inventory_value - summary_.fees;
    check_close(summary_.net_pnl, summary_.realized_gross + unrealized - summary_.fees, "PnL decomposition");
    const double equity = config_.initial_capital + summary_.net_pnl;
    peak_equity_ = std::max(peak_equity_, equity);
    summary_.max_drawdown = std::max(summary_.max_drawdown, peak_equity_ - equity);
    summary_.max_abs_position = std::max(summary_.max_abs_position, std::abs(summary_.position));
    equity_ << timestamp << ',' << mid << ',' << summary_.position << ',' << summary_.balance << ','
            << summary_.fees << ',' << inventory_value << ',' << summary_.realized_gross << ',' << unrealized
            << ',' << summary_.net_pnl << ',' << equity << ',';
    if (reserved_buy) {
        equity_ << *reserved_buy;
    }
    equity_ << ',';
    if (reserved_sell) {
        equity_ << *reserved_sell;
    }
    equity_ << ',' << summary_.max_drawdown << '\n';
}

void TradeRecorder::checkpoint(std::int64_t timestamp, double mid, const hbt::StateValues& state,
                               double reserved_buy, double reserved_sell) {
    reconcile(state);
    if (state.position + reserved_buy > config_.max_position + 1e-10 ||
        state.position - reserved_sell < -config_.max_position - 1e-10) {
        throw std::runtime_error("outstanding order exposure exceeds inventory limit");
    }
    record_equity(timestamp, mid, reserved_buy, reserved_sell);
}

void TradeRecorder::normalized_rejection() { ++summary_.normalized_rejections; }
const TradingSummary& TradeRecorder::summary() const noexcept { return summary_; }

void TradeRecorder::finish(const hbt::StateValues& state, double mid, std::uint64_t local_events,
                           std::uint64_t exchange_events, double elapsed_seconds) {
    reconcile(state);
    summary_.final_mid = mid;
    summary_.local_events = local_events;
    summary_.exchange_events = exchange_events;
    summary_.elapsed_seconds = elapsed_seconds;
    write_summary();
    orders_.flush();
    fills_.flush();
    equity_.flush();
    if (!orders_ || !fills_ || !equity_) {
        throw std::runtime_error("failed while writing trading records");
    }
}

void TradeRecorder::write_summary() {
    auto output = csv_file(config_.output / "summary.json");
    const auto& totals = summary_;
    output << "{\n\"name\":" << std::quoted(config_.name) << ",\n\"date\":" << std::quoted(config_.date)
           << ",\n\"initial_capital\":" << config_.initial_capital
           << ",\n\"maker_fee\":" << config_.maker_fee << ",\n\"taker_fee\":" << config_.taker_fee
           << ",\n\"half_width_ticks\":" << config_.half_width_ticks << ",\n\"max_position\":" << config_.max_position
           << ",\n\"order_quantity\":" << config_.order_quantity
           << ",\n\"update_mode\":" << std::quoted(config_.update_mode)
           << ",\n\"decision_interval_ns\":" << config_.decision_interval_ns
           << ",\n\"price_threshold_ticks\":" << config_.price_threshold_ticks
           << ",\n\"entry_latency_ns\":" << config_.entry_latency_ns
           << ",\n\"response_latency_ns\":" << config_.response_latency_ns
           << ",\n\"submissions\":" << totals.submissions << ",\n\"cancel_requests\":" << totals.cancel_requests
           << ",\n\"cancellations\":" << totals.cancellations << ",\n\"expired\":" << totals.expired
           << ",\n\"rejected\":" << totals.rejected << ",\n\"fills\":" << totals.fills
           << ",\n\"partial_fills\":" << totals.partial_fills
           << ",\n\"normalized_rejections\":" << totals.normalized_rejections
           << ",\n\"position\":" << totals.position << ",\n\"balance\":" << totals.balance
           << ",\n\"fees\":" << totals.fees << ",\n\"volume\":" << totals.volume
           << ",\n\"turnover\":" << totals.turnover << ",\n\"average_cost\":" << totals.average_cost
           << ",\n\"realized_gross\":" << totals.realized_gross
           << ",\n\"unrealized\":" << totals.position * (totals.final_mid - totals.average_cost)
           << ",\n\"inventory_value\":" << totals.position * totals.final_mid
           << ",\n\"final_mid\":" << totals.final_mid << ",\n\"net_pnl\":" << totals.net_pnl
           << ",\n\"equity\":" << config_.initial_capital + totals.net_pnl
           << ",\n\"max_abs_position\":" << totals.max_abs_position
           << ",\n\"max_drawdown\":" << totals.max_drawdown
           << ",\n\"local_events\":" << totals.local_events << ",\n\"exchange_events\":" << totals.exchange_events
           << ",\n\"elapsed_seconds\":" << totals.elapsed_seconds
           << ",\n\"reconciled\":true,\n\"final_working_orders\":0,\n\"funding_included\":false\n}\n";
    output.flush();
    if (!output) {
        throw std::runtime_error("failed to write summary");
    }
}

} // namespace mm
