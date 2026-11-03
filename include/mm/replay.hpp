#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "mm/market_observer.hpp"

namespace mm {

struct ReplayConfig {
    std::string source = "synthetic";
    std::string date = "synthetic";
    std::string symbol = "BTCUSDT";
    double tick_size = 0.1;
    double lot_size = 0.001;
    std::size_t depth_levels = 10;
    std::int64_t snapshot_interval_ns = 100'000'000;
    std::size_t latency_sample_every = 0;
    std::filesystem::path output;
};

struct ReplayResult {
    EventStatistics events;
    BookSnapshot final_book;
    std::uint64_t exchange_events = 0;
    std::uint64_t normalized_rows = 0;
    double elapsed_seconds = 0.0;
    std::uint64_t peak_rss_bytes = 0;
    std::vector<double> local_step_samples_ns;
    std::size_t shards_loaded = 0;
    std::size_t shards_released = 0;
};

ReplayConfig load_config(const std::filesystem::path& filename);
std::vector<hbt::Event> synthetic_events();
ReplayResult run_replay(const ReplayConfig& config);
// 允许测试使用具有乱序交易所时间戳的多分片夹具，仍由同一引擎回放。
ReplayResult run_memory_replay(const ReplayConfig& config,
                               std::vector<std::vector<hbt::Event>> chunks);
void write_report(const ReplayConfig& config, const ReplayResult& result,
                  const std::filesystem::path& output);

} // namespace mm
