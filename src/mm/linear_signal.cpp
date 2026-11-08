/** 压力特征只依赖已收到的原生盘口与成交，模型系数在进入测试前冻结。 */
#include "mm/linear_signal.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace mm {
namespace {
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
constexpr std::array<const char*, 6> raw_names = {
    "imbalance_1", "imbalance_5", "imbalance_10", "trade_imbalance_100ms", "trade_imbalance_1s", "return_1s_bps"
};
constexpr std::array<const char*, 6> pressure_names = {
    "book_obi_l1", "book_obi_l5", "book_obi_l10", "trade_aggr_imb_100ms", "trade_aggr_imb_1s", "cross_trade_book_gap"
};

// 【非核心】已知无成交返回零，缺失输入保持非有限值。
double imbalance(double buy, double sell) {
    if (!std::isfinite(buy) || !std::isfinite(sell)) {
        return missing;
    }
    return buy + sell == 0 ? 0.0 : (buy - sell) / (buy + sell);
}
} // namespace

// 【非核心】识别模型导出文件支持的原始列与Gen001、Gen002压力窗口。
LinearSignal::Term LinearSignal::parse_term(const std::string& name, double coefficient) {
    for (int index = 0; index < 6; ++index) {
        if (name == raw_names[index]) {
            return {name, coefficient, index, 0, 0, true};
        }
    }
    for (int source = 0; source < 6; ++source) {
        const std::string prefix = std::string(pressure_names[source]) + "_mean_";
        if (name.starts_with(prefix + "diff_")) {
            const auto windows = name.substr(prefix.size() + 5);
            const auto separator = windows.find('_');
            return {name, coefficient, source, std::stoi(windows.substr(0, separator)) * 10,
                    std::stoi(windows.substr(separator + 1)) * 10, false};
        }
        if (name.starts_with(prefix)) {
            return {name, coefficient, source, std::stoi(name.substr(prefix.size())) * 10, 0, false};
        }
    }
    throw std::invalid_argument("unsupported realtime model column: " + name);
}

// 【核心逻辑161】保存模型系数与可供独立公式对照的100ms输入文件。
LinearSignal::LinearSignal(const std::filesystem::path& model, const std::filesystem::path& samples) {
    std::ifstream input{model};
    std::string name;
    if (!(input >> name >> intercept_) || name != "intercept") {
        throw std::runtime_error("invalid linear model header");
    }
    if (!(input >> name >> min_fraction_) || name != "min_fraction" || min_fraction_ <= 0 || min_fraction_ > 1) {
        throw std::runtime_error("invalid model min_fraction");
    }
    if (!(input >> name >> state_interval_ns_) || name != "state_interval_ns" || state_interval_ns_ != 100'000'000) {
        throw std::runtime_error("model state cadence must be 100ms");
    }
    if (!(input >> name >> inference_interval_ns_) || name != "inference_interval_ns" || inference_interval_ns_ != 1'000'000'000) {
        throw std::runtime_error("model inference cadence must be 1s");
    }
    double coefficient = 0;
    while (input >> name >> coefficient) {
        terms_.push_back(parse_term(name, coefficient));
    }
    if (!input.eof() || terms_.empty()) {
        throw std::runtime_error("invalid linear model coefficients");
    }
    std::filesystem::create_directories(samples.parent_path());
    samples_.open(samples);
    if (!samples_) {
        throw std::runtime_error("cannot create realtime parity samples");
    }
    samples_ << std::setprecision(17);
    samples_ << "timestamp_ns,mid_price,buy_volume,sell_volume,book_valid,flow_valid";
    for (const auto side : {"bid", "ask"}) {
        for (const auto depth : {1, 5, 10}) {
            samples_ << ',' << side << "_depth_" << depth;
        }
    }
    for (const auto raw : raw_names) {
        samples_ << ',' << raw;
    }
    for (const auto& term : terms_) {
        samples_ << ",input_" << term.name;
    }
    samples_ << ",inference_update,prediction_bps\n";
}

// 【核心逻辑170】按固定采样位置统计有效压力，跨缺失和午夜保持滑动。
double LinearSignal::pressure_mean(int source, int window) const {
    const auto& history = pressure_[source];
    const auto count = std::min(static_cast<int>(history.size()), window);
    double sum = 0;
    int valid = 0;
    for (int index = 0; index < count; ++index) {
        const auto value = history[history.size() - 1 - static_cast<std::size_t>(index)];
        if (std::isfinite(value)) {
            sum += value;
            ++valid;
        }
    }
    return valid >= static_cast<int>(std::ceil(window * min_fraction_)) ? sum / valid : missing;
}

// 【核心逻辑169】读取框架实际盘口，更新源窗口，再以冻结系数进行当前时刻推理。
double LinearSignal::sample(std::int64_t timestamp, hbt::HashMapMarketDepth& depth,
                            std::int64_t last_book_timestamp, double buy_quantity, double sell_quantity) {
    if (first_sample_ == 0) {
        first_sample_ = timestamp;
    }
    const bool flow_valid = timestamp >= first_sample_ + 100'000'000;
    if (!flow_valid) {
        buy_quantity = missing;
        sell_quantity = missing;
    }
    const hbt::Event observation{0, 0, timestamp, 0, 0, 0, 0, 0};
    observer_.capture(observation, depth);
    const auto& book = observer_.latest_snapshot();
    const bool valid = book.bid_count == 10 && book.ask_count == 10 && book.spread > 0
                       && last_book_timestamp > 0 && timestamp - last_book_timestamp <= 1'000'000'000;
    std::array<double, 3> bids{};
    std::array<double, 3> asks{};
    if (valid) {
        for (int level = 0; level < 10; ++level) {
            for (int index = 0; index < 3; ++index) {
                constexpr std::array<int, 3> depths{1, 5, 10};
                const int selected_depth = depths[index];
                if (level < selected_depth) {
                    bids[index] += book.bids[level].quantity;
                    asks[index] += book.asks[level].quantity;
                }
            }
        }
    } else {
        bids.fill(missing);
        asks.fill(missing);
    }
    const double mid = valid ? book.mid_price : missing;
    flow_.push_back({buy_quantity, sell_quantity});
    if (flow_.size() > 10) {
        flow_.pop_front();
    }
    mid_.push_back(mid);
    if (mid_.size() > 11) {
        mid_.pop_front();
    }
    double buy_sum = 0;
    double sell_sum = 0;
    int flow_samples = 0;
    for (const auto& quantities : flow_) {
        if (std::isfinite(quantities[0]) && std::isfinite(quantities[1])) {
            buy_sum += quantities[0];
            sell_sum += quantities[1];
            ++flow_samples;
        }
    }
    const double trade_1s = flow_samples >= 8 ? imbalance(buy_sum, sell_sum) : missing;
    const double return_1s = mid_.size() == 11 && std::isfinite(mid_.front()) ? (mid / mid_.front() - 1) * 10'000 : missing;
    std::array<double, 6> raw{imbalance(bids[0], asks[0]), imbalance(bids[1], asks[1]), imbalance(bids[2], asks[2]),
                              imbalance(buy_quantity, sell_quantity), trade_1s, return_1s};
    std::array<double, 6> pressure = raw;
    pressure[5] = trade_1s - raw[1];
    for (int source = 0; source < 6; ++source) {
        pressure_[source].push_back(pressure[source]);
        if (pressure_[source].size() > 1000) {
            pressure_[source].pop_front();
        }
    }
    samples_ << timestamp << ',' << mid << ',' << buy_quantity << ',' << sell_quantity << ',' << valid << ',' << flow_valid;
    for (const auto& quantities : {bids, asks}) {
        for (const auto value : quantities) {
            samples_ << ',' << value;
        }
    }
    for (const auto value : raw) {
        samples_ << ',' << value;
    }
    const bool update = timestamp % inference_interval_ns_ == 0;
    double prediction = intercept_;
    for (const auto& term : terms_) {
        double value = missing;
        if (update) {
            value = term.raw ? raw[term.source] : pressure_mean(term.source, term.short_window);
            if (term.long_window > 0) {
                value -= pressure_mean(term.source, term.long_window);
            }
            prediction += value * term.coefficient;
        }
        samples_ << ',' << value;
    }
    if (update) {
        last_prediction_ = prediction;
    }
    // 预测在下一整秒更新，当前盘口或成交覆盖失效时暂停报价。
    const double effective_prediction = valid && flow_valid ? last_prediction_ : missing;
    samples_ << ',' << update << ',' << effective_prediction << '\n';
    return effective_prediction;
}
// 【非核心】确保整日实时记录完整写出。
void LinearSignal::finish() {
    samples_.flush();
    if (!samples_) {
        throw std::runtime_error("failed writing realtime parity samples");
    }
}
} // namespace mm
