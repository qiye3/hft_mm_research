#pragma once

#include <optional>
#include <hbt/hbt.hpp>
#include "mm/trade_recorder.hpp"

namespace mm {

struct QuoteTicks {
    std::int64_t bid;
    std::int64_t ask;
};

QuoteTicks fixed_quotes(std::int64_t best_bid, std::int64_t best_ask, std::int64_t half_width);
TradingConfig load_trading_config(const std::filesystem::path& filename);
void validate_trading_config(const TradingConfig& config);

class MarketMaker {
public:
    MarketMaker(const TradingConfig& config, TradeRecorder& recorder);
    void on_decision(hbt::Backtest& backtest, bool stop = false);
    [[nodiscard]] std::pair<double, double> reservations(const hbt::Backtest& backtest) const;

private:
    void quote_side(hbt::Backtest& backtest, hbt::Side side, std::int64_t target_tick, bool stop);
    TradingConfig config_;
    TradeRecorder& recorder_;
    hbt::OrderId next_order_id_ = 1;
    std::optional<double> reference_tick_;
    QuoteTicks targets_{};
};

TradingSummary run_baseline(const TradingConfig& config,
                            std::shared_ptr<hbt::EventReaderCache> reader);
TradingSummary run_baseline(const TradingConfig& config);

} // namespace mm
