/** 冻结线性模型的实时回测入口，盘口、订单和成交复用原生引擎及已有M1代码。 */
#include "mm/linear_signal.hpp"
#include "mm/market_maker.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>

// 【非核心】连接冻结模型与交易参数，保存实时特征对照和完整订单账本。
int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument("usage: linear_mm trading.yaml realtime_model.txt samples.csv");
        }
        const auto config = mm::load_trading_config(argv[1]);
        if (config.decision_interval_ns != 100'000'000) {
            throw std::invalid_argument("linear baseline requires a 100ms decision grid");
        }
        mm::LinearSignal signal{argv[2], argv[3]};
        auto reader = hbt::event_reader_from_files(hbt::event_files_from_manifest(config.source.string()));
        mm::run_baseline(config, std::move(reader), &signal);
        signal.finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "linear_mm: " << error.what() << '\n';
        return 1;
    }
}
