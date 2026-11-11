/** Strategies独立入口，分别接收交易参数、策略规则和完整整秒预测。 */
#include "mm/strategies/execution_policy.hpp"
#include <iostream>
#include <stdexcept>

// 【非核心】解析本次回放的三个显式输入，将异常交给运行队列保留失败日志。
int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument("usage: strategy_replay trading.yaml policy.yaml predictions.csv");
        }
        const auto config = mm::load_trading_config(argv[1]);
        const auto parameters = mm::load_execution_parameters(argv[2]);
        mm::PredictionSeries predictions{argv[3]};
        mm::run_execution_replay(config, parameters, predictions,
            hbt::event_reader_from_files(hbt::event_files_from_manifest(config.source.string())));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "strategy_replay: " << error.what() << '\n';
        return 1;
    }
}
