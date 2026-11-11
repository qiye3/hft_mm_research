#pragma once

/** 多期限收益曲线的目标库存决策，订单仍由现有ExecutionPolicy和原生引擎管理。 */
#include "mm/strategies/execution_policy.hpp"
#include <array>

namespace mm {

constexpr std::size_t curve_size = 16;
using ReturnCurve = std::array<double, curve_size>;

struct CurveParameters {
    std::string mode;
    double risk_lambda;
    double cost_buffer_bps;
    double risk_unit_btc;
    std::size_t horizon_index;
    std::int64_t rebalance_interval_ns;
    std::int64_t opportunity_interval_ns;
    int opportunity_side;
    double edge_minimum_bps;
    double minimum_notional_usdt;
};

// 【核心逻辑280】使用五档实际库存与分段风险动态规划，返回当前时刻的第一段目标。
double curve_inventory_target(const ReturnCurve& curve, const ReturnCurve& variances,
                              double risk_lambda, double cost_bps, double actual_units);

// 【非核心】曲线参数独立声明，公开接口不隐式识别旧版配置。
CurveParameters load_curve_parameters(const std::filesystem::path& filename);

// 【核心逻辑281】以完整曲线读取原生事件，输出报价、在途库存、实际成交和清算权益。
TradingSummary run_curve_replay(const TradingConfig& config, const ExecutionParameters& execution,
                               const CurveParameters& curve_parameters,
                               const std::filesystem::path& predictions_file,
                               const std::filesystem::path& risk_file);

} // namespace mm
