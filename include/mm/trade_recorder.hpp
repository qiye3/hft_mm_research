#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <optional>
#include <hbt/hbt.hpp>

namespace mm {

struct TradingConfig {
    std::string name = "baseline";
    std::string date = "2025-02-01";
    std::filesystem::path source;
    std::filesystem::path output;
    double tick_size = 0.1;
    double lot_size = 0.001;
    double order_quantity = 0.001;
    double max_position = 0.01;
    double initial_capital = 10000.0;
    double maker_fee = 0.0002;
    double taker_fee = 0.0005;
    std::int64_t half_width_ticks = 20;
    std::int64_t decision_interval_ns = 100'000'000;
    std::int64_t price_threshold_ticks = 20;
    std::int64_t entry_latency_ns = 1'000'000;
    std::int64_t response_latency_ns = 500'000;
    std::int64_t quote_stop_timestamp_ns = 1738454399000000000;
    std::string update_mode = "time";
};

struct TradingSummary {
    std::uint64_t submissions = 0;
    std::uint64_t cancel_requests = 0;
    std::uint64_t cancellations = 0;
    std::uint64_t expired = 0;
    std::uint64_t rejected = 0;
    std::uint64_t partial_fills = 0;
    std::uint64_t fills = 0;
    std::uint64_t normalized_rejections = 0;
    double position = 0.0;
    double balance = 0.0;
    double fees = 0.0;
    double volume = 0.0;
    double turnover = 0.0;
    double average_cost = 0.0;
    double realized_gross = 0.0;
    double max_abs_position = 0.0;
    double max_drawdown = 0.0;
    double final_mid = 0.0;
    double net_pnl = 0.0;
    std::uint64_t local_events = 0;
    std::uint64_t exchange_events = 0;
    double elapsed_seconds = 0.0;
};

// 账本仅用于记录/独立核对；策略风险判断始终读取框架原生库存与活动订单。
class TradeRecorder {
public:
    explicit TradeRecorder(const TradingConfig& config);
    void action(const std::string& kind, const hbt::Order<>& order, std::int64_t local_timestamp);
    void response(const hbt::Order<>& order, std::int64_t local_timestamp, double mid);
    void reconcile(const hbt::StateValues& state) const;
    void checkpoint(std::int64_t timestamp, double mid, const hbt::StateValues& state,
                    double reserved_buy, double reserved_sell);
    void finish(const hbt::StateValues& state, double mid, std::uint64_t local_events,
                std::uint64_t exchange_events, double elapsed_seconds);
    void normalized_rejection();
    [[nodiscard]] const TradingSummary& summary() const noexcept;

private:
    void write_order(const std::string& kind, const hbt::Order<>& order, std::int64_t timestamp);
    void apply_fill(const hbt::Order<>& order, std::int64_t timestamp, double mid);
    void record_equity(std::int64_t timestamp, double mid, std::optional<double> reserved_buy,
                       std::optional<double> reserved_sell);
    void write_summary();
    TradingConfig config_;
    TradingSummary summary_;
    double peak_equity_;
    std::ofstream orders_;
    std::ofstream fills_;
    std::ofstream equity_;
};

std::string order_status(hbt::Status status);

} // namespace mm
