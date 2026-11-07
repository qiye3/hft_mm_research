/** M0 Release 性能测量入口，整体计时与局部采样口径由共享回放实现统一定义。 */
#include "mm/replay.hpp"

#include <exception>
#include <iostream>

// 【非核心】在 Release 构建中按指定配置回放，输出吞吐、局部耗时分位数和内存高水位。
int main(int argc, char** argv) {
    /** argv 必须提供一份回放配置，报告写到其中的 output，成功返回 0。
     * 调试构建或运行异常返回 1，防止把不同编译优化条件的结果混作同一基准。
     */
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: benchmark_replay config/replay/btcusdt.yaml");
        }
#ifdef NDEBUG
        const auto config = mm::load_config(argv[1]);
        const auto result = mm::run_replay(config);
        mm::write_report(config, result, config.output);
        return 0;
#else
        throw std::runtime_error("benchmark requires a Release build");
#endif
    } catch (const std::exception& error) {
        std::cerr << "benchmark_replay: " << error.what() << '\n';
        return 1;
    }
}
