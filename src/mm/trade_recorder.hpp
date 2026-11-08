#pragma once

/** M1 交易参数、汇总字段和逐笔记录接口，统一以 BTC、USDT 和纳秒表达。 */

#include <filesystem>
#include <fstream>
#include <string>
#include <optional>
#include <hbt/hbt.hpp>

namespace mm {

// 一组可复现实验的参数，价格和数量步长分别用于报价与下单网格。
struct TradingConfig {
    std::string name = "baseline";
    std::string date = "2025-02-01";
    std::filesystem::path source; // 标准事件 manifest.json，分片由原生 reader 载入。
    std::filesystem::path output; // 每次实验专属目录，记录器要求该目录尚未存在。
    double tick_size = 0.1;
    double lot_size = 0.001;
    double order_quantity = 0.001;
    double max_position = 0.01; // 净仓位绝对上限，风控同时覆盖未成交数量。
    double initial_capital = 10000.0; // 加入账户权益的初始 USDT 资金。
    double maker_fee = 0.0002; // 成交额乘数，负值表示返佣，0.0002 对应 2 bps。
    double taker_fee = 0.0005;
    std::int64_t half_width_ticks = 20; // 相对本地中间价的单侧距离。
    std::int64_t decision_interval_ns = 100'000'000;
    std::int64_t price_threshold_ticks = 20;
    std::int64_t entry_latency_ns = 1'000'000;
    std::int64_t response_latency_ns = 500'000;
    std::int64_t quote_start_timestamp_ns = 0; // 新实验可预留日初历史窗口，双基线使用相同起点。
    std::int64_t quote_stop_timestamp_ns = 1738454399000000000; // 停止新报价的绝对 UTC 纳秒时刻。
    std::string update_mode = "time"; // time 每次决策刷新目标，price 按锚点变动阈值刷新。
};

// 已核对账本的累计量，成交数统计回报笔数，成交量与成交额分别累计 BTC 和 USDT。
struct TradingSummary {
    std::uint64_t submissions = 0;
    std::uint64_t cancel_requests = 0;
    std::uint64_t cancellations = 0;
    std::uint64_t expired = 0;
    std::uint64_t rejected = 0;
    std::uint64_t partial_fills = 0;
    std::uint64_t fills = 0;
    double position = 0.0;
    double balance = 0.0; // 累计买卖现金流，初始资金单独计入 equity。
    double fees = 0.0;
    double volume = 0.0;
    double turnover = 0.0;
    double average_cost = 0.0;
    double realized_gross = 0.0; // 平均成本法累计已实现收益，手续费另列。
    double max_abs_position = 0.0;
    double max_drawdown = 0.0; // 决策点及成交点上观测到的最大金额回撤。
    double final_mid = 0.0;
    double net_pnl = 0.0;
    std::uint64_t local_events = 0;
    std::uint64_t exchange_events = 0;
    double elapsed_seconds = 0.0;
};

/**
 * 逐笔保存请求、原始回报、真实成交和权益观察值，并与原生 StateValues 对账。
 * 输入来自同一回测会话，输出位于 config.output，实例应覆盖全部处理器的存活期。
 * 内部平均成本账本支持收益分解，策略风险检查从原生账户与订单读取实际状态。
 */
class TradeRecorder {
public:
    // 【非核心】为全新输出目录建立三个 CSV，拒绝覆盖已有实验。
    explicit TradeRecorder(const TradingConfig& config);
    // 【核心逻辑175】记录原生 submit 或 cancel_request 操作及当时的订单视图。
    void action(const std::string& kind, const hbt::Order<>& order, std::int64_t local_timestamp);
    // 【核心逻辑180】保存一条原始回报，依据状态与新增执行量累计成交账本。
    void response(const hbt::Order<>& order, std::int64_t local_timestamp, double mid);
    // 【非核心】在同批回报处理完成后，对齐原生现金流、费用、仓位与成交统计。
    void reconcile(const hbt::StateValues& state) const;
    // 【核心逻辑177】集中核对决策点最坏方向敞口，并记录带预留数量的权益值。
    void checkpoint(std::int64_t timestamp, double mid, const hbt::StateValues& state,
                    double reserved_buy, double reserved_sell);
    // 【非核心】依据最终 checkpoint 的账本保存 JSON 汇总，并检查 CSV 刷新结果。
    void finish(const hbt::StateValues& state, double mid, std::uint64_t local_events,
                std::uint64_t exchange_events, double elapsed_seconds);
    // 【非核心】返回当前汇总的只读引用，用于回测结束时形成返回结果。
    [[nodiscard]] const TradingSummary& summary() const noexcept;

private:
    // 【非核心】序列化订单动作和原始回报共用的生命周期字段。
    void write_order(const std::string& kind, const hbt::Order<>& order, std::int64_t timestamp);
    // 【核心逻辑181】用本次新增执行量更新现金、费用、平均成本和仓位。
    void apply_fill(const hbt::Order<>& order, std::int64_t timestamp, double mid);
    // 【核心逻辑178】盯市并更新权益峰值，空的预留数量在 CSV 中保留为空字段。
    void record_equity(std::int64_t timestamp, double mid, std::optional<double> reserved_buy,
                       std::optional<double> reserved_sell);
    // 【非核心】输出可供离线独立核对的参数、收益分解和最终统计。
    void write_summary();
    TradingConfig config_;
    TradingSummary summary_;
    double peak_equity_;
    std::ofstream orders_;
    std::ofstream fills_;
    std::ofstream equity_;
};

// 【非核心】将订单状态或请求状态映射成 CSV 使用的协议文本。
std::string order_status(hbt::Status status);

} // namespace mm
