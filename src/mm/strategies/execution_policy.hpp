#pragma once

/** 固定预测流与四类订单决策，库存、排队和回报仍由已有原生引擎管理。 */
#include "mm/market_maker.hpp"
#include "mm/detail/trading_engine.hpp"
#include <vector>

namespace mm {

struct ExecutionParameters {
    std::string mode;
    std::string queue_model;
    double entry_bps;
    double exit_bps;
    double scale_bps;
    double max_alpha_bps;
    double inventory_skew_bps;
    bool cost_filter;
    std::int64_t offset_ticks;
    std::int64_t ttl_ns;
    std::int64_t max_hold_ns;
    std::int64_t prediction_latency_ns;
    std::int64_t taker_slippage_ticks;
    double quote_half_width_bps = 0;
    double minimum_notional_usdt = 0;
};

// 【非核心】读取明确属于Strategies阶段的参数，输入输出路径由独立交易配置提供。
ExecutionParameters load_execution_parameters(const std::filesystem::path& filename);

class PredictionSeries {
public:
    // 【核心逻辑236】读取完整整秒预测，严格递增时间键不包含标签有效性筛选。
    explicit PredictionSeries(const std::filesystem::path& filename);
    // 【核心逻辑239】只返回已完成推理的最近一秒预测，缺失或过期返回NaN。
    double at(std::int64_t timestamp, std::int64_t latency);
private:
    std::vector<std::pair<std::int64_t, double>> values_;
    std::size_t cursor_ = 0;
};

class ExecutionPolicy {
public:
    // 【非核心】关联既有挂撤单模块与账本，保存该次实验的固定决策参数。
    ExecutionPolicy(const TradingConfig& config, const ExecutionParameters& parameters, TradeRecorder& recorder,
                    const hbt::HashMapMarketDepth& depth);
    // 【核心逻辑240】根据预测、实际库存和订单在途状态作决策，收尾阶段优先真实平仓。
    std::pair<double, double> decide(hbt::Backtest& backtest, double alpha, bool quotes_allowed,
                                    bool final_exit, bool book_fresh);
    // 【核心逻辑242】到期报价先发撤单并等待确认，Maker价格始终位于本地对手盘之外。
    std::pair<double, double> post(hbt::Backtest& backtest,
        std::optional<std::int64_t> bid, std::optional<std::int64_t> ask,
        double buy_budget, double sell_budget);
    // 【核心逻辑279】目标库存策略复用同一撤单等待和IOC减仓，最后退出与持仓超时分别记录。
    std::pair<double, double> liquidate(hbt::Backtest& backtest, bool final_exit, bool book_fresh);
private:
    // 【核心逻辑243】风险退出和Hybrid主动调整均复用原生IOC，发单前先等待已有风险订单终结。
    void take(hbt::Backtest& backtest, hbt::Side side, double quantity, const char* purpose);
    TradingConfig config_;
    ExecutionParameters parameters_;
    TradeRecorder& recorder_;
    const hbt::HashMapMarketDepth& depth_;
    MarketMaker maker_;
    std::ofstream ioc_actions_;
    hbt::OrderId next_taker_id_ = 1'000'000'000'000ULL;
    std::int64_t opened_at_ = 0;
    int position_sign_ = 0;
    bool reducing_ = false;
    double target_position_ = 0;
};

// 【核心逻辑237】将冻结预测连接到原生历史事件回放，输出可核对且最终平仓的交易账本。
TradingSummary run_execution_replay(const TradingConfig& config, const ExecutionParameters& parameters,
                                   PredictionSeries& predictions,
                                   std::shared_ptr<hbt::EventReaderCache> reader);

} // namespace mm
