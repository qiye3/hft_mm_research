/**
 * M0 原生回放接入与性能报告，将共享事件缓存交给 Backtest 的双时钟调度器。
 * 处理器适配层增加观察和计数，run_replay 选择数据来源，write_report 输出 JSON。
 * 行情加载、订单簿更新和事件顺序由依赖引擎执行。
 */
#include "mm/replay.hpp"
#include "mm/detail/config.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
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

// 在 final Local 的公开处理器边界接入观察器，以实际 process 调用统计逐条行情。
class ObservedLocal final : public hbt::LocalProcessor {
public:
    // 【核心逻辑42】接管原生处理器，并以 sample_every 指定本地处理耗时的系统采样间隔。
    ObservedLocal(std::unique_ptr<Local> local, MarketObserver& observer, std::size_t sample_every,
                  ReplayEventSink* sink)
        : local_(std::move(local)), observer_(observer), sample_every_(sample_every), sink_(sink) {}

    // 【核心逻辑43】更新原生本地状态后通知观察器，并按配置测量本地事件处理耗时。
    void process(const hbt::Event& event) override {
        /** 输入是引擎按 local_ts 交付的一条行情，输出包括观察统计和可选纳秒耗时样本。
         * sample_every 为零时关闭计时，为正时从第零条开始每隔该数量取样。
         * 样本覆盖 Local::process 与观察器调用，交易所处理、调度和文件 IO 位于此范围之外。
         */
        const bool sample = sample_every_ > 0 && observer_.statistics().total % sample_every_ == 0;
        const auto start = sample ? Clock::now() : Clock::time_point{};
        if (sink_) {
            sink_->before_event(event, local_->depth());
        }
        local_->process(event);
        if (sink_) {
            sink_->after_event(event, local_->depth());
        }
        observer_.on_event(event, local_->depth());
        if (sample) {
            samples_ns.push_back(std::chrono::duration<double, std::nano>(Clock::now() - start).count());
        }
        if (hbt::event_is(event, hbt::EXCH_EVENT)) {
            // 双路由共用同一记录时，最终统计需减去这部分交集，避免把物理行数加倍。
            ++combined_local_rows;
        }
        last_event = event;
    }

    // 【非核心】转交原生Local确定事件的本地可见时刻。
    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return local_->event_seen_timestamp(event);
    }
    // 【非核心】将订单回报的消费交给原生Local执行。
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        return local_->process_recv_order(timestamp, order_id);
    }
    // 【非核心】读取原生Local下一条待接收订单回报的时刻。
    std::int64_t earliest_recv_order_timestamp() const noexcept override {
        return local_->earliest_recv_order_timestamp();
    }
    // 【非核心】读取原生Local下一条待发送请求的时刻。
    std::int64_t earliest_send_order_timestamp() const noexcept override {
        return local_->earliest_send_order_timestamp();
    }
    // 【非核心】将提交请求原样交给原生Local执行。
    void submit_order(hbt::OrderId order_id, hbt::Side side, double price, double quantity,
                      hbt::OrdType order_type, hbt::TimeInForce time_in_force, std::int64_t timestamp) override {
        local_->submit_order(order_id, side, price, quantity, order_type, time_in_force, timestamp);
    }
    // 【非核心】将改价改量请求原样交给原生Local执行。
    void modify(hbt::OrderId order_id, double price, double quantity, std::int64_t timestamp) override {
        local_->modify(order_id, price, quantity, timestamp);
    }
    // 【非核心】将撤单请求原样交给原生Local执行。
    void cancel(hbt::OrderId order_id, std::int64_t timestamp) override {
        local_->cancel(order_id, timestamp);
    }
    // 【非核心】调用原生Local移除已终结的订单记录。
    void clear_inactive_orders() override { local_->clear_inactive_orders(); }
    // 【非核心】读取原生账户的当前净仓位。
    double position() const noexcept override { return local_->position(); }
    // 【非核心】返回原生账户状态的只读引用。
    const hbt::StateValues& state_values() const noexcept override { return local_->state_values(); }
    // 【非核心】返回原生订单表的只读引用。
    const std::unordered_map<hbt::OrderId, hbt::Order<>>& orders() const noexcept override {
        return local_->orders();
    }
    // 【非核心】读取原生Local缓存的最近成交事件。
    const std::vector<hbt::Event>& last_trades() const noexcept override { return local_->last_trades(); }
    // 【非核心】清空原生Local缓存的最近成交事件。
    void clear_last_trades() noexcept override { local_->clear_last_trades(); }
    // 【非核心】读取最近行情的交易所与本地时间对。
    std::optional<std::pair<std::int64_t, std::int64_t>> feed_latency() const noexcept override {
        return local_->feed_latency();
    }
    // 【非核心】读取最近订单的请求、交易所处理及回报时刻。
    std::optional<std::tuple<std::int64_t, std::int64_t, std::int64_t>> order_latency() const noexcept override {
        return local_->order_latency();
    }
    // 【非核心】读取原生本地盘口的最优买价。
    double best_bid() const noexcept override { return local_->best_bid(); }
    // 【非核心】读取原生本地盘口的最优卖价。
    double best_ask() const noexcept override { return local_->best_ask(); }
    // 【非核心】读取最优买价对应的整数Tick档位。
    std::int64_t best_bid_tick() const noexcept override { return local_->best_bid_tick(); }
    // 【非核心】读取最优卖价对应的整数Tick档位。
    std::int64_t best_ask_tick() const noexcept override { return local_->best_ask_tick(); }
    // 【非核心】读取原生合约配置的价格步长。
    double tick_size() const noexcept override { return local_->tick_size(); }
    // 【非核心】读取原生合约配置的数量步长。
    double lot_size() const noexcept override { return local_->lot_size(); }
    // 【非核心】返回原生本地订单簿的只读引用。
    const Depth& depth() const noexcept { return local_->depth(); }

    std::uint64_t combined_local_rows = 0;
    std::vector<double> samples_ns;
    hbt::Event last_event{};

private:
    std::unique_ptr<Local> local_;
    MarketObserver& observer_;
    std::size_t sample_every_;
    ReplayEventSink* sink_;
};

// 在交易所处理器外围检查该侧时序并记录实际处理次数。
class CountedExchange final : public hbt::Processor {
public:
    // 【非核心】接管实际交易所处理器，保留其深度更新、排队和成交行为。
    explicit CountedExchange(std::unique_ptr<Exchange> exchange) : exchange_(std::move(exchange)) {}
    // 【非核心】按交易所时钟验证一条行情，再交给原生处理器应用。
    void process(const hbt::Event& event) override {
        /** 输入事件须按 exch_ts 非递减到达，输出为原生状态变化和侧内计数。
         * 时序错误直接抛出异常，用于检查跨分片后的交易所路由完整性。
         */
        if (count > 0 && event.exch_ts < last_timestamp_) {
            throw std::runtime_error("exchange event timestamps moved backwards");
        }
        exchange_->process(event);
        last_timestamp_ = event.exch_ts;
        ++count;
    }
    // 【非核心】读取原生交易所处理器确定的事件可见时刻。
    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return exchange_->event_seen_timestamp(event);
    }
    // 【非核心】将待处理订单请求交给原生交易所处理器。
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        return exchange_->process_recv_order(timestamp, order_id);
    }
    // 【非核心】读取原生交易所下一条待接收订单请求的时刻。
    std::int64_t earliest_recv_order_timestamp() const noexcept override {
        return exchange_->earliest_recv_order_timestamp();
    }
    // 【非核心】读取原生交易所下一条待发送订单回报的时刻。
    std::int64_t earliest_send_order_timestamp() const noexcept override {
        return exchange_->earliest_send_order_timestamp();
    }
    std::uint64_t count = 0;

private:
    std::unique_ptr<Exchange> exchange_;
    std::int64_t last_timestamp_ = 0;
};

// 【非核心】读取本进程截至当前时刻的驻留内存高水位，并统一换算成字节。
std::uint64_t peak_rss_bytes() {
    /** 输入来自操作系统 getrusage，返回累计 Peak RSS。
     * macOS 原值为字节，Linux 原值为 KiB，读取失败时抛出异常。
     */
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

// 【核心逻辑41】组装双侧原生处理器，完整消费共享行情缓存并汇总观察结果。
ReplayResult replay_reader(const ReplayConfig& config, std::shared_ptr<hbt::EventReaderCache> reader,
                           ReplayEventSink* sink) {
    /** config 指定步长、观察深度和采样频率，reader 必须提供非空且分侧有序的事件。
     * 返回 ReplayResult，包含本地分类计数、最终 Top-N、整体耗时和局部耗时样本。
     * Local 与 Exchange 共用缓存，Backtest 持有处理器，裸观测指针只在该会话内使用。
     */
    MarketObserver observer{config.depth_levels, config.snapshot_interval_ns};
    auto [exchange_bus, local_bus] = hbt::make_order_buses<Latency>(Latency{100'000, 50'000});
    auto native_local = std::make_unique<Local>(
        Depth{config.tick_size, config.lot_size},
        hbt::State<Asset, Fees>{Asset{1.0}, Fees{hbt::CommonFees{0.0, 0.0}}}, 0, std::move(local_bus));
    auto native_exchange = std::make_unique<Exchange>(
        Depth{config.tick_size, config.lot_size},
        hbt::State<Asset, Fees, Queue::OrderState>{Asset{1.0}, Fees{hbt::CommonFees{0.0, 0.0}}},
        Queue{}, std::move(exchange_bus));
    auto local = std::make_unique<ObservedLocal>(std::move(native_local), observer, config.latency_sample_every, sink);
    auto exchange = std::make_unique<CountedExchange>(std::move(native_exchange));
    auto* local_view = local.get();
    auto* exchange_view = exchange.get();
    hbt::Backtest backtest;
    backtest.add_asset(std::move(local), std::move(exchange), reader);

    const auto start = Clock::now();
    // close 自身进入原生调度循环，首次解压和后续分片加载也计入整体回放耗时。
    const auto status = backtest.close();
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    if (status != hbt::ElapseResult::EndOfData || observer.statistics().total == 0) {
        throw std::runtime_error("replay did not consume a nonempty stream to EndOfData");
    }
    observer.capture(local_view->last_event, local_view->depth());
    if (sink) {
        sink->finish(local_view->depth());
    }
    // M0 的观察流程应保持零订单、零仓位，这一检查隔离策略行为与回放验证。
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

// 【非核心】把有限浮点数写入 JSON，缺少有效行情指标时输出 null。
void write_number(std::ostream& output, double value) {
    /** output 是报告流，value 可包含空簿产生的 NaN。
     * 写入符合 JSON 语法的数值或 null，供报告中的 Mid、Spread 与采样指标复用。
     */
    if (std::isfinite(value)) {
        output << value;
    } else {
        output << "null";
    }
}

// 【非核心】将快照有效前缀导出为按最优价优先排列的价格数量二元组数组。
void write_levels(std::ostream& output, const std::array<PriceLevel, 10>& levels, std::size_t count) {
    /** levels 已按方向排好序，count 必须在数组容量以内。
     * 输出 JSON 数组，每个元素依次为实际价格和该价位挂单数量。
     */
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

// 【非核心】读取观察回放参数，并以配置所在项目解析行情清单与报告输出位置。
ReplayConfig load_config(const std::filesystem::path& filename) {
    const auto fields = detail::read_flat_config(filename, {
        "source", "date", "symbol", "tick_size", "lot_size", "depth_levels",
        "snapshot_interval_ns", "latency_sample_every", "output",
    });
    ReplayConfig config;
    config.source = fields.at("source");
    config.date = fields.at("date");
    config.symbol = fields.at("symbol");
    config.tick_size = detail::config_number<double>(fields, "tick_size");
    config.lot_size = detail::config_number<double>(fields, "lot_size");
    config.depth_levels = detail::config_number<std::size_t>(fields, "depth_levels");
    config.snapshot_interval_ns = detail::config_number<std::int64_t>(fields, "snapshot_interval_ns");
    config.latency_sample_every = detail::config_number<std::size_t>(fields, "latency_sample_every");
    config.output = fields.at("output");
    if (!std::isfinite(config.tick_size) || config.tick_size <= 0 ||
        !std::isfinite(config.lot_size) || config.lot_size <= 0 || config.symbol != "BTCUSDT") {
        throw std::runtime_error("expected BTCUSDT and positive finite tick_size/lot_size");
    }
    const auto root = std::filesystem::path{MM_PROJECT_ROOT};
    if (config.source != "synthetic") {
        config.source = (root / config.source).string();
    }
    if (config.output.empty()) {
        throw std::runtime_error("output must be configured");
    }
    config.output = root / config.output;
    return config;
}

// 【非核心】建立包含交易、深档挂单和删除买一事件的确定性回放夹具。
std::vector<hbt::Event> synthetic_events() {
    /** 无外部输入，返回同时具有 EXCH/LOCAL 标志的 25 条原生事件。
     * 初始四条沿用上游示例，扩展部分使原第十一档可在买一删除后进入前十档。
     * 用于检查观察深度、分类数量和双时钟回放接入是否正确。
     */
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

// 【核心逻辑75】为内存行情夹具建立双消费者缓存，再执行统一观察回放流程。
ReplayResult run_memory_replay(const ReplayConfig& config, std::vector<std::vector<hbt::Event>> chunks,
                               ReplayEventSink* sink) {
    /** 输入 chunks 按分片顺序提供原生事件，函数接管这些数组并返回 ReplayResult。
     * 两个消费者分别对应本地和交易所处理器，测试可据此验证跨分片释放语义。
     */
    return replay_reader(config, hbt::EventReaderCache::from_chunks(std::move(chunks), 2), sink);
}

// 【核心逻辑40】根据配置选择合成夹具或磁盘分片，并调用相同的原生回放接入层。
ReplayResult run_replay(const ReplayConfig& config, ReplayEventSink* sink) {
    /** 输入 config.source 为 synthetic 或可读取的转换清单，返回观察回放汇总。
     * 文件模式通过 event_files_from_manifest 和 EventReaderCache 逐片加载数据。
     */
    if (config.source == "synthetic") {
        return run_memory_replay(config, {synthetic_events()}, sink);
    }
    return replay_reader(config, hbt::event_reader_from_files(hbt::event_files_from_manifest(config.source)), sink);
}

// 【非核心】保存回放完整性、最终盘口与具有明确测量范围的性能指标。
void write_report(const ReplayConfig& config, const ReplayResult& result, const std::filesystem::path& output) {
    /** config 提供实验标签，result 必须来自已完整消费的回放，output 指定 JSON 文件。
     * 输出同时列出逻辑事件吞吐和双侧 dispatch 吞吐，便于解释物理记录与处理次数。
     * 局部耗时样本按 nearest-rank 求分位数，关闭采样时对应字段输出 null。
     */
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
