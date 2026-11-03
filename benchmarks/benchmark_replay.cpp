#include "mm/replay.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: benchmark_replay config/replay_btcusdt.yaml");
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
