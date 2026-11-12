#include "mm/strategies/execution_policy.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

// 【非核心】隔离预测可见性和原生平仓回归，失败直接携带业务原因。
void require_execution(bool value, const char* reason) {
    if (!value) {
        throw std::runtime_error(reason);
    }
}

// 【非核心】以完整整秒预测验证推理完成前无法提前读取当前信号。
void check_prediction_clock(const std::filesystem::path& root) {
    const auto file = root / "clock.csv";
    std::ofstream(file) << "timestamp_ns,prediction_bps\n0,1\n1000000000,-2\n2000000000,3\n";
    mm::PredictionSeries values{file};
    require_execution(std::isnan(values.at(0, 1000)), "signal visible before inference completes");
    require_execution(values.at(1000, 1000) == 1, "first ready prediction");
    require_execution(values.at(1'000'000'000, 1000) == 1, "future second leaked");
    require_execution(values.at(1'000'001'000, 1000) == -2, "new second not visible at completion");
    require_execution(std::isnan(values.at(3'000'001'000, 1000)), "stale prediction was reused");
}

// 【非核心】以真实原生处理器检查Maker成交、IOC退出和两种排队假设的最终平仓。
void check_native_exit(const std::filesystem::path& root, const std::string& queue,
                       const std::string& mode = "passive", bool cost_filter = false,
                       double width_bps = 0, double minimum_notional = 0, bool zero_signal = false) {
    mm::TradingConfig config;
    config.name = queue + mode + std::to_string(cost_filter) + std::to_string(width_bps)
                  + std::to_string(minimum_notional) + std::to_string(zero_signal);
    config.source = "memory";
    config.output = root / config.name;
    config.tick_size = 1;
    config.lot_size = 1;
    config.order_quantity = 1;
    config.max_position = 3;
    config.initial_capital = 1000;
    config.maker_fee = .001;
    config.taker_fee = .002;
    config.half_width_ticks = 1;
    config.entry_latency_ns = 100;
    config.response_latency_ns = 50;
    config.decision_interval_ns = 100'000'000;
    config.quote_stop_timestamp_ns = 1'500'000'000;
    const auto signals = root / (queue + ".csv");
    double later_alpha = 10;
    if (mode == "directional" || mode == "directional_maker") {
        later_alpha = 0;
    } else if (mode == "defensive") {
        later_alpha = -10;
    }
    std::ofstream(signals) << "timestamp_ns,prediction_bps\n0," << (zero_signal ? 0 : 10) << "\n1000000000,"
                          << later_alpha << "\n2000000000," << later_alpha << "\n";
    mm::PredictionSeries prediction{signals};
    mm::ExecutionParameters parameters{mode, queue, 1, .5, 10, 10, 0, cost_filter,
                                       0, 1'000'000'000, 500'000'000, 1000, 1};
    parameters.quote_half_width_bps = width_bps;
    parameters.minimum_notional_usdt = minimum_notional;
    if (zero_signal) {
        parameters.entry_bps = 0;
        parameters.exit_bps = 0;
    }
    if (mode == "directional" || mode == "directional_maker") {
        parameters.max_hold_ns = 1'100'000'000;
    }
    std::vector<hbt::Event> events;
    for (const auto [kind, timestamp, price, quantity] : std::vector<std::tuple<std::uint64_t, std::int64_t, double, double>>{
            {hbt::DEPTH_EVENT | hbt::BUY_EVENT, 1'000'000, 99, 2},
            {hbt::DEPTH_EVENT | hbt::SELL_EVENT, 1'000'000, 101, 2},
            {hbt::TRADE_EVENT | hbt::SELL_EVENT, 600'000'000, 98, 5},
            {hbt::DEPTH_EVENT | hbt::SELL_EVENT, 1'000'000'000, 101, 2},
            {hbt::DEPTH_EVENT | hbt::SELL_EVENT, 1'400'000'000, 101, 2},
            {hbt::DEPTH_EVENT | hbt::SELL_EVENT, 1'800'000'000, 101, 2},
            {hbt::DEPTH_EVENT | hbt::SELL_EVENT, 2'500'000'000, 101, 2}}) {
        events.push_back({kind | hbt::EXCH_EVENT | hbt::LOCAL_EVENT, timestamp, timestamp + 50,
                           price, quantity, 0, 0, 0});
    }
    const auto result = mm::run_execution_replay(config, parameters, prediction,
        hbt::EventReaderCache::from_chunks({events}, 2));
    const bool no_fill = zero_signal || cost_filter || width_bps >= 1000 || minimum_notional >= 1000;
    require_execution(result.fills == (no_fill ? 0 : 2), "cost gate or quote width entry/exit mismatch");
    require_execution(result.position == 0, "native exit left inventory");
    const double expected_fees = mode == "directional" ? .4 : .297;
    require_execution(std::abs(result.fees - (no_fill ? 0 : expected_fees)) < 1e-9, "Maker/Taker fees inconsistent");
    require_execution(result.max_abs_position <= 3, "risk budget exceeded");
}

// 【非核心】直接驱动原生队列验证Maker退出、部分退出和超时IOC，不用假定成交替代撮合。
void check_maker_hold_exit(const std::filesystem::path& root, bool closing_trade, bool partial) {
    mm::TradingConfig config;
    config.name = "maker_hold_" + std::to_string(closing_trade) + std::to_string(partial);
    config.source = "memory";
    config.output = root / config.name;
    config.tick_size = 1;
    config.lot_size = 1;
    config.order_quantity = partial ? 2 : 1;
    config.max_position = config.order_quantity;
    config.initial_capital = 1000;
    config.maker_fee = .001;
    config.taker_fee = .002;
    config.half_width_ticks = 1;
    config.entry_latency_ns = 100;
    config.response_latency_ns = 50;
    config.decision_interval_ns = 100'000'000;
    config.quote_stop_timestamp_ns = 2'500'000'000;
    const auto signals = root / (config.name + ".csv");
    std::ofstream(signals) << "timestamp_ns,prediction_bps\n0,1\n1000000000,0\n2000000000,0\n3000000000,0\n";
    mm::PredictionSeries prediction{signals};
    mm::ExecutionParameters parameters{"directional_maker", "risk_adverse", 0, 0, 1, 10, 0, false,
        0, 300'000'000, 500'000'000, 1000, 1};
    std::vector<hbt::Event> events;
    // 【非核心】合成事件共用明确的交易所/本地时间差，直接进入原生事件读取器。
    const auto add = [&](std::uint64_t kind, std::int64_t timestamp, double price, double quantity) {
        events.push_back({kind | hbt::EXCH_EVENT | hbt::LOCAL_EVENT, timestamp, timestamp + 50,
                          price, quantity, 0, 0, 0});
    };
    add(hbt::DEPTH_EVENT | hbt::BUY_EVENT, 1'000'000, 99, 2);
    add(hbt::DEPTH_EVENT | hbt::SELL_EVENT, 1'000'000, 101, 2);
    add(hbt::TRADE_EVENT | hbt::SELL_EVENT, 600'000'000, 98, 8);
    add(hbt::DEPTH_EVENT | hbt::SELL_EVENT, 1'000'000'000, 101, 2);
    if (closing_trade) {
        add(hbt::TRADE_EVENT | hbt::BUY_EVENT, 1'300'000'000, partial ? 101 : 102, partial ? 3 : 8);
    }
    add(hbt::DEPTH_EVENT | hbt::SELL_EVENT, 1'400'000'000, 101, 2);
    add(hbt::DEPTH_EVENT | hbt::SELL_EVENT, 2'500'000'000, 101, 2);
    add(hbt::DEPTH_EVENT | hbt::SELL_EVENT, 3'100'000'000, 101, 2);
    const auto result = mm::run_execution_replay(config, parameters, prediction,
        hbt::EventReaderCache::from_chunks({events}, 2));
    require_execution(result.position == 0, "maker hold exit left position");
    require_execution(result.max_abs_position <= config.max_position, "maker hold risk exceeded");
    require_execution(result.fills == (partial ? 3 : 2), "maker hold fill lifecycle mismatch");
    double expected_fee = .297;
    if (partial) {
        expected_fee = .497;
    } else if (closing_trade) {
        expected_fee = .2;
    }
    require_execution(std::abs(result.fees - expected_fee) < 1e-9, "maker hold actual maker/taker mix mismatch");
}

// 【非核心】测试输出位于独立临时目录，断言通过后释放夹具文件。
int main() {
    const auto root = std::filesystem::temp_directory_path() / ("mm_execution_" + std::to_string(getpid()));
    std::filesystem::create_directories(root);
    check_prediction_clock(root);
    check_native_exit(root, "risk_adverse");
    check_native_exit(root, "log_probability");
    check_native_exit(root, "risk_adverse", "passive", true);
    check_native_exit(root, "risk_adverse", "inventory", true);
    check_native_exit(root, "risk_adverse", "hybrid", true);
    check_native_exit(root, "risk_adverse", "passive", false, 1000);
    check_native_exit(root, "risk_adverse", "passive", false, 0, 1000);
    check_native_exit(root, "risk_adverse", "directional");
    check_native_exit(root, "risk_adverse", "directional", false, 0, 1000);
    check_native_exit(root, "risk_adverse", "directional", false, 0, 0, true);
    check_native_exit(root, "risk_adverse", "defensive");
    check_maker_hold_exit(root, true, false);
    check_maker_hold_exit(root, false, false);
    check_maker_hold_exit(root, true, true);
    check_native_exit(root, "risk_adverse", "directional_maker", false, 0, 0, true);
    std::filesystem::remove_all(root);
}
