/** M2-A 离线提取入口，复用原生双时间戳回放并导出固定时间网格基础表。 */
#include "mm/research_sampler.hpp"

#include <iostream>
#include <stdexcept>

// 【非核心】将日期范围和输出位置连接到已有回放配置，异常以失败状态返回批处理程序。
int main(int argc, char** argv) {
    /** 输入依次为 replay.yaml、起止纳秒、最大盘口年龄纳秒和 CSV 输出位置。
     * 输出为研究基础表，回放仍完整消费原生事件，不产生委托。
     */
    try {
        if (argc != 6) {
            throw std::invalid_argument("usage: extract_research replay.yaml start_ns end_ns max_book_age_ns samples.csv");
        }
        auto config = mm::load_config(argv[1]);
        mm::ResearchSampler sampler{argv[5], std::stoll(argv[2]), std::stoll(argv[3]),
                                    config.snapshot_interval_ns, std::stoll(argv[4]), config.depth_levels};
        // M0 观察器只保留最终盘口，研究采样由独立的固定网格负责。
        config.snapshot_interval_ns = std::numeric_limits<std::int64_t>::max();
        const auto result = mm::run_replay(config, &sampler);
        std::cout << "local_events=" << result.events.total << " seconds=" << result.elapsed_seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
