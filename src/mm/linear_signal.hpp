#pragma once

/** 线性压力模型的实时特征和推理。滚动窗口与Polars压力因子共用定义。 */

#include <array>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "mm/market_observer.hpp"

namespace mm {

class LinearSignal {
public:
    // 【核心逻辑161】读取冻结的原单位系数，并建立实时与Python对照的输入记录。
    LinearSignal(const std::filesystem::path& model, const std::filesystem::path& samples);
    // 【核心逻辑169】在100ms网格更新原生盘口与窗口，每1s更新预测，其间返回最近有效预测。
    double sample(std::int64_t timestamp, hbt::HashMapMarketDepth& depth,
                  std::int64_t last_book_timestamp, double buy_quantity, double sell_quantity);

    // 【非核心】刷新并检查实时对照文件，磁盘错误通过回测入口报告。
    void finish();

private:
    struct Term {
        std::string name;
        double coefficient;
        int source;
        int short_window;
        int long_window;
        bool raw;
    };
    // 【非核心】把已知压力因子名映射到实时源及窗口，未知定义明确失败。
    static Term parse_term(const std::string& name, double coefficient);
    // 【核心逻辑170】按窗口位置计数，达到模型声明的有效比例才计算均值。
    double pressure_mean(int source, int window) const;

    double intercept_ = 0;
    double min_fraction_ = 0.8;
    std::int64_t state_interval_ns_ = 100'000'000;
    std::int64_t inference_interval_ns_ = 1'000'000'000;
    double last_prediction_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<Term> terms_;
    std::array<std::deque<double>, 6> pressure_;
    std::deque<std::array<double, 2>> flow_;
    std::deque<double> mid_;
    std::int64_t first_sample_ = 0;
    MarketObserver observer_{10, 0};
    std::ofstream samples_;
};

} // namespace mm
