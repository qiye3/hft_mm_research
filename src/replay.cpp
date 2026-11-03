#include "mm/replay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>

namespace mm {
namespace {

using Asset = hbt::LinearAsset;
using Latency = hbt::ConstantLatency;
using Depth = hbt::HashMapMarketDepth;
using Queue = hbt::RiskAdverseQueueModel;
using Fees = hbt::TradingValueFeeModel<hbt::CommonFees>;
using Local = hbt::Local<Asset, Latency, Depth, Fees>;
using Exchange = hbt::NoPartialFillExchange<Asset, Latency, Queue, Depth, Fees>;
using Clock = std::chrono::steady_clock;

// 上游 Local 是 final 且 Backtest 没有逐事件回调，因此只在公开处理器边界装饰它。
// 所有状态和订单操作仍转发给原生 Local；不维护独立盘口或订单生命周期。
class ObservedLocal final : public hbt::LocalProcessor {
public:
    ObservedLocal(std::unique_ptr<Local> local, MarketObserver& observer, std::size_t sample_every)
        : local_(std::move(local)), observer_(observer), sample_every_(sample_every) {}

    void process(const hbt::Event& event) override {
        const bool sample = sample_every_ > 0 && observer_.statistics().total % sample_every_ == 0;
        const auto start = sample ? Clock::now() : Clock::time_point{};
        local_->process(event);
        observer_.on_event(event, local_->depth());
        if (sample) {
            samples_ns.push_back(std::chrono::duration<double, std::nano>(Clock::now() - start).count());
        }
        if (hbt::event_is(event, hbt::EXCH_EVENT)) {
            ++combined_local_rows;
        }
        last_event = event;
    }

    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return local_->event_seen_timestamp(event);
    }
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        return local_->process_recv_order(timestamp, order_id);
    }
    std::int64_t earliest_recv_order_timestamp() const noexcept override {
        return local_->earliest_recv_order_timestamp();
    }
    std::int64_t earliest_send_order_timestamp() const noexcept override {
        return local_->earliest_send_order_timestamp();
    }
    void submit_order(hbt::OrderId order_id, hbt::Side side, double price, double quantity,
                      hbt::OrdType order_type, hbt::TimeInForce time_in_force, std::int64_t timestamp) override {
        local_->submit_order(order_id, side, price, quantity, order_type, time_in_force, timestamp);
    }
    void modify(hbt::OrderId order_id, double price, double quantity, std::int64_t timestamp) override {
        local_->modify(order_id, price, quantity, timestamp);
    }
    void cancel(hbt::OrderId order_id, std::int64_t timestamp) override {
        local_->cancel(order_id, timestamp);
    }
    void clear_inactive_orders() override { local_->clear_inactive_orders(); }
    double position() const noexcept override { return local_->position(); }
    const hbt::StateValues& state_values() const noexcept override { return local_->state_values(); }
    const std::unordered_map<hbt::OrderId, hbt::Order<>>& orders() const noexcept override {
        return local_->orders();
    }
    const std::vector<hbt::Event>& last_trades() const noexcept override { return local_->last_trades(); }
    void clear_last_trades() noexcept override { local_->clear_last_trades(); }
    std::optional<std::pair<std::int64_t, std::int64_t>> feed_latency() const noexcept override {
        return local_->feed_latency();
    }
    std::optional<std::tuple<std::int64_t, std::int64_t, std::int64_t>> order_latency() const noexcept override {
        return local_->order_latency();
    }
    double best_bid() const noexcept override { return local_->best_bid(); }
    double best_ask() const noexcept override { return local_->best_ask(); }
    std::int64_t best_bid_tick() const noexcept override { return local_->best_bid_tick(); }
    std::int64_t best_ask_tick() const noexcept override { return local_->best_ask_tick(); }
    double tick_size() const noexcept override { return local_->tick_size(); }
    double lot_size() const noexcept override { return local_->lot_size(); }
    const Depth& depth() const noexcept { return local_->depth(); }

    std::uint64_t combined_local_rows = 0;
    std::vector<double> samples_ns;
    hbt::Event last_event{};

private:
    std::unique_ptr<Local> local_;
    MarketObserver& observer_;
    std::size_t sample_every_;
};

class CountedExchange final : public hbt::Processor {
public:
    explicit CountedExchange(std::unique_ptr<Exchange> exchange) : exchange_(std::move(exchange)) {}
    void process(const hbt::Event& event) override {
        if (count > 0 && event.exch_ts < last_timestamp_) {
            throw std::runtime_error("exchange event timestamps moved backwards");
        }
        exchange_->process(event);
        last_timestamp_ = event.exch_ts;
        ++count;
    }
    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return exchange_->event_seen_timestamp(event);
    }
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        return exchange_->process_recv_order(timestamp, order_id);
    }
    std::int64_t earliest_recv_order_timestamp() const noexcept override {
        return exchange_->earliest_recv_order_timestamp();
    }
    std::int64_t earliest_send_order_timestamp() const noexcept override {
        return exchange_->earliest_send_order_timestamp();
    }
    std::uint64_t count = 0;

private:
    std::unique_ptr<Exchange> exchange_;
    std::int64_t last_timestamp_ = 0;
};

std::uint64_t peak_rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
#ifdef __APPLE__
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

ReplayResult replay_reader(const ReplayConfig& config, std::shared_ptr<hbt::EventReaderCache> reader) {
    MarketObserver observer{config.depth_levels, config.snapshot_interval_ns};
    auto [exchange_bus, local_bus] = hbt::make_order_buses<Latency>(Latency{100'000, 50'000});
    auto native_local = std::make_unique<Local>(
        Depth{config.tick_size, config.lot_size},
        hbt::State<Asset, Fees>{Asset{1.0}, Fees{hbt::CommonFees{0.0, 0.0}}}, 0, std::move(local_bus));
    auto native_exchange = std::make_unique<Exchange>(
        Depth{config.tick_size, config.lot_size},
        hbt::State<Asset, Fees, Queue::OrderState>{Asset{1.0}, Fees{hbt::CommonFees{0.0, 0.0}}},
        Queue{}, std::move(exchange_bus));
    auto local = std::make_unique<ObservedLocal>(std::move(native_local), observer, config.latency_sample_every);
    auto exchange = std::make_unique<CountedExchange>(std::move(native_exchange));
    auto* local_view = local.get();
    auto* exchange_view = exchange.get();
    hbt::Backtest backtest;
    backtest.add_asset(std::move(local), std::move(exchange), reader);

    const auto start = Clock::now();
    const auto status = backtest.close();
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    if (status != hbt::ElapseResult::EndOfData || observer.statistics().total == 0) {
        throw std::runtime_error("replay did not consume a nonempty stream to EndOfData");
    }
    observer.capture(local_view->last_event, local_view->depth());
    if (!backtest.orders(0).empty() || backtest.position(0) != 0.0) {
        throw std::runtime_error("M0 must not submit orders or hold positions");
    }
    ReplayResult result;
    result.events = observer.statistics();
    result.final_book = observer.latest_snapshot();
    result.exchange_events = exchange_view->count;
    result.normalized_rows = result.events.total + result.exchange_events - local_view->combined_local_rows;
    result.elapsed_seconds = elapsed;
    result.peak_rss_bytes = peak_rss_bytes();
    result.local_step_samples_ns = std::move(local_view->samples_ns);
    result.shards_loaded = reader->load_count();
    result.shards_released = reader->released_shards();
    return result;
}

std::string trim(std::string value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

void write_number(std::ostream& output, double value) {
    if (std::isfinite(value)) {
        output << value;
    } else {
        output << "null";
    }
}

void write_levels(std::ostream& output, const std::array<PriceLevel, 10>& levels, std::size_t count) {
    output << '[';
    for (std::size_t index = 0; index < count; ++index) {
        if (index > 0) {
            output << ',';
        }
        output << '[' << levels[index].price << ',' << levels[index].quantity << ']';
    }
    output << ']';
}

} // namespace

ReplayConfig load_config(const std::filesystem::path& filename) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("cannot open replay config: " + filename.string());
    }
    ReplayConfig config;
    std::string line;
    // 当前配置约定为平铺 YAML 标量，遇到未知字段直接报错。
    while (std::getline(input, line)) {
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) {
            continue;
        }
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            throw std::runtime_error("expected key: value in replay config");
        }
        const auto key = trim(line.substr(0, separator));
        auto value = trim(line.substr(separator + 1));
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        if (key == "source") {
            config.source = value;
        } else if (key == "date") {
            config.date = value;
        } else if (key == "symbol") {
            config.symbol = value;
        } else if (key == "tick_size") {
            config.tick_size = std::stod(value);
        } else if (key == "lot_size") {
            config.lot_size = std::stod(value);
        } else if (key == "depth_levels") {
            config.depth_levels = std::stoull(value);
        } else if (key == "snapshot_interval_ns") {
            config.snapshot_interval_ns = std::stoll(value);
        } else if (key == "latency_sample_every") {
            config.latency_sample_every = std::stoull(value);
        } else if (key == "output") {
            config.output = value;
        } else {
            throw std::runtime_error("unknown replay config field: " + key);
        }
    }
    if (!std::isfinite(config.tick_size) || config.tick_size <= 0 ||
        !std::isfinite(config.lot_size) || config.lot_size <= 0 || config.symbol != "BTCUSDT") {
        throw std::runtime_error("expected BTCUSDT and positive finite tick_size/lot_size");
    }
    const auto root = std::filesystem::absolute(filename).parent_path().parent_path();
    if (config.source != "synthetic") {
        config.source = (root / config.source).string();
    }
    if (config.output.empty()) {
        throw std::runtime_error("output must be configured");
    }
    config.output = root / config.output;
    return config;
}

std::vector<hbt::Event> synthetic_events() {
    // 前四条来自上游 examples/market_making.cpp；追加 11 档及删除买一用例。
    const auto both = hbt::EXCH_EVENT | hbt::LOCAL_EVENT;
    std::vector<hbt::Event> events{
        {both | hbt::BUY_EVENT | hbt::DEPTH_EVENT, 1'000'000, 1'010'000, 99.9, 5.0, 0, 0, 0},
        {both | hbt::SELL_EVENT | hbt::DEPTH_EVENT, 1'000'000, 1'010'000, 100.1, 5.0, 0, 0, 0},
        {both | hbt::SELL_EVENT | hbt::TRADE_EVENT, 2'000'000, 2'010'000, 99.8, 5.0, 0, 0, 0},
        {both | hbt::BUY_EVENT | hbt::TRADE_EVENT, 4'000'000, 4'010'000, 100.2, 5.0, 0, 0, 0},
    };
    for (std::int64_t level = 1; level <= 10; ++level) {
        const auto exchange_timestamp = 5'000'000 + level * 100'000;
        events.push_back({both | hbt::BUY_EVENT | hbt::DEPTH_EVENT, exchange_timestamp,
                          exchange_timestamp + 10'000, 99.9 - static_cast<double>(level) * 0.1,
                          static_cast<double>(level + 1), 0, 0, 0});
        events.push_back({both | hbt::SELL_EVENT | hbt::DEPTH_EVENT, exchange_timestamp,
                          exchange_timestamp + 10'000, 100.1 + static_cast<double>(level) * 0.1,
                          static_cast<double>(level + 2), 0, 0, 0});
    }
    events.push_back({both | hbt::BUY_EVENT | hbt::DEPTH_EVENT, 7'000'000, 7'010'000, 99.9, 0, 0, 0, 0});
    return events;
}

ReplayResult run_memory_replay(const ReplayConfig& config, std::vector<std::vector<hbt::Event>> chunks) {
    return replay_reader(config, hbt::EventReaderCache::from_chunks(std::move(chunks), 2));
}

ReplayResult run_replay(const ReplayConfig& config) {
    if (config.source == "synthetic") {
        return run_memory_replay(config, {synthetic_events()});
    }
    return replay_reader(config, hbt::event_reader_from_files(hbt::event_files_from_manifest(config.source)));
}

void write_report(const ReplayConfig& config, const ReplayResult& result, const std::filesystem::path& output) {
    std::filesystem::create_directories(output.parent_path());
    std::ofstream report(output);
    if (!report) {
        throw std::runtime_error("cannot write report: " + output.string());
    }
    report << std::setprecision(16);
    const auto& counts = result.events;
    report << "{\n  \"date\": " << std::quoted(config.date)
           << ",\n  \"symbol\": " << std::quoted(config.symbol)
           << ",\n  \"source\": " << std::quoted(config.source)
           << ",\n  \"compiler\": " << std::quoted(__VERSION__)
           << ",\n  \"depth_levels\": " << config.depth_levels
           << ",\n  \"snapshot_interval_ns\": " << config.snapshot_interval_ns
           << ",\n  \"normalized_rows\": " << result.normalized_rows
           << ",\n  \"local_events\": " << counts.total
           << ",\n  \"exchange_events\": " << result.exchange_events
           << ",\n  \"depth_updates\": " << counts.depth_updates
           << ",\n  \"snapshot_levels\": " << counts.snapshot_levels
           << ",\n  \"clear_events\": " << counts.clears
           << ",\n  \"trade_events\": " << counts.trades
           << ",\n  \"first_exchange_timestamp_ns\": " << counts.first_exchange_timestamp
           << ",\n  \"last_exchange_timestamp_ns\": " << counts.last_exchange_timestamp
           << ",\n  \"first_local_timestamp_ns\": " << counts.first_local_timestamp
           << ",\n  \"last_local_timestamp_ns\": " << counts.last_local_timestamp
           << ",\n  \"elapsed_seconds\": " << result.elapsed_seconds
           << ",\n  \"logical_events_per_second\": " << static_cast<double>(counts.total) / result.elapsed_seconds
           << ",\n  \"processor_dispatches_per_second\": "
           << static_cast<double>(counts.total + result.exchange_events) / result.elapsed_seconds
           << ",\n  \"peak_rss_bytes\": " << result.peak_rss_bytes
           << ",\n  \"shards_loaded\": " << result.shards_loaded
           << ",\n  \"shards_released\": " << result.shards_released
           << ",\n  \"valid_bbo_events\": " << counts.valid_bbo_events
           << ",\n  \"event_weighted_mean_spread\": ";
    write_number(report, counts.valid_bbo_events > 0 ? counts.spread_sum / static_cast<double>(counts.valid_bbo_events)
                                                   : std::numeric_limits<double>::quiet_NaN());
    report << ",\n  \"final_mid_price\": ";
    write_number(report, result.final_book.mid_price);
    report << ",\n  \"final_spread\": ";
    write_number(report, result.final_book.spread);
    report << ",\n  \"final_bids\": ";
    write_levels(report, result.final_book.bids, result.final_book.bid_count);
    report << ",\n  \"final_asks\": ";
    write_levels(report, result.final_book.asks, result.final_book.ask_count);
    auto samples = result.local_step_samples_ns;
    std::sort(samples.begin(), samples.end());
    report << ",\n  \"local_step_latency\": {\"scope\": \"Local::process + observer; excludes exchange, scheduler and IO; not strategy latency\","
           << " \"sample_every\": " << config.latency_sample_every << ", \"samples\": " << samples.size();
    for (const auto percentile : {50, 95, 99}) {
        report << ", \"p" << percentile << "_ns\": ";
        if (samples.empty()) {
            report << "null";
        } else {
            const auto index = static_cast<std::size_t>(std::ceil(static_cast<double>(percentile) / 100 * static_cast<double>(samples.size()))) - 1;
            write_number(report, samples[index]);
        }
    }
    report << "}\n}\n";
    if (!report) {
        throw std::runtime_error("failed while writing report");
    }
    std::cout << "local_events=" << counts.total << " exchange_events=" << result.exchange_events
              << " seconds=" << result.elapsed_seconds << " events/s="
              << static_cast<double>(counts.total) / result.elapsed_seconds
              << " peak_rss_bytes=" << result.peak_rss_bytes << " report=" << output << '\n';
}

} // namespace mm
