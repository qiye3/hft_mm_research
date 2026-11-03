#include "mm/market_maker.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
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
