/** M0 最小观察入口，用同一套公开回放接口检查合成或真实行情。 */
#include "mm/replay.hpp"

#include <exception>
#include <iostream>

// 从可选配置启动观察回放，并将结果写入配置指定的 JSON 文件。
int main(int argc, char** argv) {
    /** argv 可提供一个配置文件，省略时使用 config/replay.yaml。
     * 成功输出观察报告并返回 0，参数或回放错误写到标准错误并返回 1。
     */
    try {
        if (argc > 2) {
            throw std::invalid_argument("usage: replay_probe [config/replay.yaml]");
        }
        // 读取参数
        const auto config = mm::load_config(argc == 2 ? argv[1] : "config/replay.yaml");
        // 运行行情回放
        const auto result = mm::run_replay(config);
        // 写入报告
        mm::write_report(config, result, config.output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "replay_probe: " << error.what() << '\n';
        return 1;
    }
}
