#include "mm/replay.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc > 2) {
            throw std::invalid_argument("usage: replay_probe [config/replay.yaml]");
        }
        const auto config = mm::load_config(argc == 2 ? argv[1] : "config/replay.yaml");
        const auto result = mm::run_replay(config);
        mm::write_report(config, result, config.output);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "replay_probe: " << error.what() << '\n';
        return 1;
    }
}
