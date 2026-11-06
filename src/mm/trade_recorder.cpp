/**
 * M1 交易记录与独立账本，从原生订单回报累计现金流、费用和平均成本收益。
 * TradeRecorder 在回报批次结束时与 StateValues 对账，输出逐笔 CSV 和 summary.json。
 * 本文件统一使用 BTC 数量、USDT 金额和本地接收时刻的行情标记价格。
 */
#include "mm/trade_recorder.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

namespace mm {
namespace {

// 【非核心】用绝对误差核对账本字段，出现非有限值或超差时报告具体字段。
void check_close(double actual, double expected, const char* field) {
    /** 输入 actual 为待核对数值，expected 为独立计算结果，field 用于错误定位。
     * 正常返回表示两值误差不超过 1e-7，失败时抛出 runtime_error。
     */
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-7) {
        throw std::runtime_error(std::string("ledger mismatch: ") + field);
    }
}

// 【非核心】打开结果文件并设置足以用于跨语言账本核对的浮点输出精度。
std::ofstream csv_file(const std::filesystem::path& filename) {
    /** 输入为已有父目录内的文件名，输出可移动的文件流，失败时抛出异常。
     * 17 位有效数字用于保留 double 的往返精度，CSV 与 JSON 输出共用此约定。
     */
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("cannot create output: " + filename.string());
    }
    output << std::setprecision(17);
    return output;
}

} // namespace

// 【非核心】把框架状态映射成订单流水中约定的稳定文本值。
std::string order_status(hbt::Status status) {
    /** 输入为框架订单状态或请求状态，输出供 Python 分析器识别的字符串。
     * 未知枚举值抛出异常，防止导出无法解释的生命周期记录。
     */
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

// 【非核心】为一组全新实验建立订单、成交和权益三份逐笔输出文件。
TradeRecorder::TradeRecorder(const TradingConfig& config)
    : config_(config), peak_equity_(config.initial_capital) {
    /** 输入 config 提供输出目录、初始资金和费率，构造后各文件已写入列名。
     * 输出目录必须尚未存在，以便每次实验有明确归属，已有目录会触发异常。
     */
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

// 【非核心】把本地操作或原始订单回报写为一行生命周期记录。
void TradeRecorder::write_order(const std::string& kind, const hbt::Order<>& order, std::int64_t timestamp) {
    /** kind 区分请求与响应，order 提供当时的原生字段，timestamp 是本地记录时刻。
     * 输出至 orders.csv，回报的 exchange_timestamp 保留交易所处理时刻。
     * 请求上的 exchange_timestamp 只能表示已知的原订单时间，确认时间由后续回报给出。
     */
    orders_ << kind << ',' << order.exch_timestamp << ',' << timestamp << ',' << order.order_id << ','
            << static_cast<int>(order.side) << ',' << order.price() << ',' << order.qty << ','
            << order.leaves_qty << ',' << order_status(order.status) << ',' << order_status(order.req)
            << ',' << order.exec_price() << ',' << order.exec_qty << ',' << order.maker << '\n';
}

// 【核心逻辑56】在原生请求提交成功后记录下单或撤单动作，并累计请求次数。
void TradeRecorder::action(const std::string& kind, const hbt::Order<>& order, std::int64_t local_timestamp) {
    /** 输入 kind 取 submit 或 cancel_request，order 为框架请求后的订单视图。
     * 输出为请求计数与订单流水，实际接受或撤销结果由 response 接口记录。
     */
    if (kind == "submit") {
        ++summary_.submissions;
    } else if (kind == "cancel_request") {
        ++summary_.cancel_requests;
    } else {
        throw std::invalid_argument("unknown order action");
    }
    write_order(kind, order, local_timestamp);
}

// 【核心逻辑62】依据一笔新增成交更新独立账本，写出成交后的成本、库存和收益。
void TradeRecorder::apply_fill(const hbt::Order<>& order, std::int64_t timestamp, double mid) {
    /** 输入 order.exec_qty 是本次回报新增成交量，timestamp 为接收时刻，mid 为可见标记价格。
     * 输出为独立账本、fills.csv 和成交时刻权益记录，数量为 BTC，收益为 USDT。
     * 采用平均成本法处理加仓与平仓，翻向时以本笔成交价建立剩余仓位成本。
     */
    if (!std::isfinite(mid)) {
        throw std::runtime_error("cannot record a fill without a current finite mid-price");
    }
    const double quantity = order.exec_qty;
    const double price = order.exec_price();
    const double signed_quantity = hbt::side_sign(order.side) * quantity;
    const double old_position = summary_.position;
    const double new_position = old_position + signed_quantity;
    if (old_position == 0.0 || old_position * signed_quantity > 0) {
        // 同向成交按绝对数量加权，空头平均成本同样保持为正的价格。
        summary_.average_cost = (std::abs(old_position) * summary_.average_cost + quantity * price)
                                 / (std::abs(old_position) + quantity);
    } else {
        // 先对冲旧仓位实现收益，超出旧仓位的数量才建立相反方向库存。
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
    // balance 累计交易现金流，初始资金在计算账户权益时另行加入。
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
    // 同一批回报可能含多笔成交，中间成交点的预留数量留空。
    // 决策点会使用回报处理完成后的原生订单状态补充真实预留数量。
    record_equity(timestamp, mid, std::nullopt, std::nullopt);
}

// 【核心逻辑61】区分生命周期确认与真实成交，只对有效成交回报增加经济账本。
void TradeRecorder::response(const hbt::Order<>& order, std::int64_t local_timestamp, double mid) {
    /** 输入为队列中保存的原始回报、本地实际接收时间及当前标记价格。
     * 每条回报都会导出，拒绝请求只计拒绝次数，成交状态且新增量为正才调用 apply_fill。
     * 接入层已经拒绝状态异常的回报，此处按当前回报语义记录单次成交。
     */
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

// 【非核心】将原生仓位、现金、费用和成交统计逐项与独立账本核对。
void TradeRecorder::reconcile(const hbt::StateValues& state) const {
    /** 输入为同一批回报处理结束后的 StateValues，正常返回表示账本一致。
     * 持仓或成交数抵消无法代表逐笔正确，因而同时核对成交量和成交额。
     * 任一字段超差都会中止回测，调用方应保留输出用于定位对应回报。
     */
    check_close(state.position, summary_.position, "position");
    check_close(state.balance, summary_.balance, "balance");
    check_close(state.fee, summary_.fees, "fees");
    check_close(state.trading_volume, summary_.volume, "volume");
    check_close(state.trading_value, summary_.turnover, "turnover");
    if (state.num_trades != static_cast<std::int64_t>(summary_.fills)) {
        throw std::runtime_error("ledger mismatch: fill count");
    }
}

// 【核心逻辑59】在指定本地时刻盯市，并更新观察点上的权益峰值和金额回撤。
void TradeRecorder::record_equity(std::int64_t timestamp, double mid, std::optional<double> reserved_buy,
                                   std::optional<double> reserved_sell) {
    /** 输入 mid 必须有限，reserved_buy/sell 在决策点为真实剩余风险量，成交点可为空。
     * 输出权益 CSV，并校验现金流表达与已实现、未实现收益分解相等。
     * 最大回撤使用初始资金和已记录观察点计算，单位为 USDT。
     */
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

// 【核心逻辑58】在决策点核对最坏单侧成交风险，并记录完整权益观察值。
void TradeRecorder::checkpoint(std::int64_t timestamp, double mid, const hbt::StateValues& state,
                               double reserved_buy, double reserved_sell) {
    /** 输入为同一时刻的原生账户状态、标记价格和买卖侧预留 BTC 数量。
     * 输出一个带风险预留字段的权益点，任一方向可能突破净仓位上限时抛出异常。
     */
    if (state.position + reserved_buy > config_.max_position + 1e-10 ||
        state.position - reserved_sell < -config_.max_position - 1e-10) {
        throw std::runtime_error("outstanding order exposure exceeds inventory limit");
    }
    record_equity(timestamp, mid, reserved_buy, reserved_sell);
}

// 【非核心】返回当前累积汇总的只读视图，引用随记录器存活。
const TradingSummary& TradeRecorder::summary() const noexcept { return summary_; }

// 【非核心】在引擎排空后完成最终对账，保存汇总并确认逐笔文件写入成功。
void TradeRecorder::finish(const hbt::StateValues& state, double mid, std::uint64_t local_events,
                           std::uint64_t exchange_events, double elapsed_seconds) {
    /** 输入 state 为最终原生状态，mid 为最终标记，事件数按两个处理器分别统计。
     * 调用方应先执行最终 checkpoint，使 net_pnl 与此 mid 对齐。
     * 输出 summary.json 并刷新三个 CSV，写盘失败会抛出异常。
     */
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

// 【非核心】将策略参数与已核对的最终交易指标写入可独立读取的 JSON 汇总。
void TradeRecorder::write_summary() {
    /** 输入来自 config_ 和 summary_，输出位于本组实验目录中的 summary.json。
     * final_working_orders=0 由 run_baseline 收尾校验提供前提，资金费口径写入报告字段。
     */
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
