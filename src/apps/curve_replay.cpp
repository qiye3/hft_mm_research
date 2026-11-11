/** 多期限预测、风险和委托规则的原生回放入口。 */
#include "mm/strategies/curve_policy.hpp"
#include <iostream>

// 【非核心】五个输入分别属于交易、执行、曲线规则、冻结预测和训练OOF风险。
int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument("usage: curve_replay trading.yaml execution.yaml curve.yaml predictions.csv risk.txt");
        }
        mm::run_curve_replay(mm::load_trading_config(argv[1]), mm::load_execution_parameters(argv[2]),
            mm::load_curve_parameters(argv[3]), argv[4], argv[5]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "curve_replay: " << error.what() << '\n';
        return 1;
    }
}
