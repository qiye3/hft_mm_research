/** M1 命令行入口，加载一组策略参数并启动完整做市与逐笔记录流程。 */
#include "mm/market_maker.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>

// 【非核心】将策略配置交给统一回测入口，并把配置或运行失败转换成进程失败状态。
int main(int argc, char** argv) {
    /** argv 必须提供一个配置文件，成功时结果写入其中的 output 目录并返回 0。
     * 参数错误、原生回测异常或对账失败会输出错误信息并返回 1，供实验脚本判断。
     */
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: baseline_mm config/baseline_mm.yaml");
        }
        mm::run_baseline(mm::load_trading_config(argv[1]));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "baseline_mm: " << error.what() << '\n';
        return 1;
    }
}
