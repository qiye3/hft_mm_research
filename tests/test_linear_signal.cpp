/** 独立数组参考核对实时压力窗口、缺失处理及线性推理。 */
#include "mm/linear_signal.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

namespace {
// 【非核心】按数组切片独立计算窗口，检查80%有效阈值。
double reference_mean(const std::vector<double>& values, int window) {
    double sum = 0;
    int valid = 0;
    const auto start = values.size() > static_cast<std::size_t>(window) ? values.size() - window : 0;
    for (auto index = start; index < values.size(); ++index) {
        if (std::isfinite(values[index])) {
            sum += values[index];
            ++valid;
        }
    }
    if (valid < static_cast<int>(std::ceil(window * 0.8))) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / valid;
}
} // namespace

// 【非核心】构造真实原生深度，逐点对照窗口与导出的原单位系数。
int main() {
    const auto root = std::filesystem::temp_directory_path() / ("mm-linear-" + std::to_string(getpid()));
    try {
        std::filesystem::create_directories(root);
        {
            std::ofstream model{root / "model.txt"};
            model << "intercept 0.1\nmin_fraction 0.8\nstate_interval_ns 100000000\ninference_interval_ns 1000000000\nbook_obi_l1_mean_5s 0.7\nbook_obi_l1_mean_100s 0.05\ntrade_aggr_imb_100ms_mean_diff_3s_10s -0.2\nreturn_1s_bps 0.5\n";
        }
        mm::LinearSignal signal{root / "model.txt", root / "samples.csv"};
        hbt::HashMapMarketDepth depth{0.1, 0.001};
        std::vector<double> book_history;
        std::vector<double> trade_history;
        std::vector<double> mids;
        double held_prediction = std::numeric_limits<double>::quiet_NaN();
        for (int index = 0; index < 1300; ++index) {
            const std::int64_t timestamp = (index + 1) * 100'000'000LL;
            const double bid_quantity = (index % 8 + 1) * 0.001;
            for (int level = 1; level <= 10; ++level) {
                depth.update_bid_depth(100 - level * 0.1, level == 1 ? bid_quantity : 0.01, timestamp);
                depth.update_ask_depth(100 + level * 0.1, 0.01, timestamp);
            }
            const bool book_valid = index != 50 && index != 105 && index != 1005;
            const double missing = std::numeric_limits<double>::quiet_NaN();
            const double buy = index % 9 == 0 ? 0 : (index % 4 + 1) * 0.001;
            const double sell = index % 9 == 0 ? 0 : 0.003;
            book_history.push_back(book_valid ? (bid_quantity - 0.01) / (bid_quantity + 0.01) : missing);
            double trade_imbalance = 0;
            if (index == 0) {
                trade_imbalance = missing;
            } else if (buy + sell > 0) {
                trade_imbalance = (buy - sell) / (buy + sell);
            }
            trade_history.push_back(trade_imbalance);
            mids.push_back(book_valid ? 100 : missing);
            double expected = 0.1 + 0.7 * reference_mean(book_history, 50)
                              + 0.05 * reference_mean(book_history, 1000)
                              - 0.2 * (reference_mean(trade_history, 30) - reference_mean(trade_history, 100));
            if (index < 10 || !std::isfinite(mids[index]) || !std::isfinite(mids[index - 10])) {
                expected = missing;
            }
            if (timestamp % 1'000'000'000 == 0) {
                held_prediction = expected;
            }
            expected = book_valid && index > 0 ? held_prediction : missing;
            const auto last_book = book_valid ? timestamp : timestamp - 2'000'000'000LL;
            const double actual = signal.sample(timestamp, depth, last_book, buy, sell);
            if (std::isfinite(expected) != std::isfinite(actual)
                || (std::isfinite(expected) && std::abs(expected - actual) > 1e-10)) {
                throw std::runtime_error("realtime pressure window or prediction differs from independent reference");
            }
        }
        std::filesystem::remove_all(root);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
