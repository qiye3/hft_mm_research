#pragma once

#include <cmath>
#include <deque>
#include <memory>
#include <vector>
#include "mm/trade_recorder.hpp"

namespace mm::detail {

using Asset = hbt::LinearAsset;
using Latency = hbt::ConstantLatency;
using Depth = hbt::HashMapMarketDepth;
using Fees = hbt::TradingValueFeeModel<hbt::CommonFees>;
using Queue = hbt::RiskAdverseQueueModel;
using Local = hbt::Local<Asset, Latency, Depth, Fees>;
using Exchange = hbt::PartialFillExchange<Asset, Latency, Queue, Depth, Fees>;
using ResponseQueue = std::deque<std::pair<hbt::Order<>, std::int64_t>>;

// 对上游 final Local 的公开接口做转发；只在回报边界记录，不接管订单状态。
class RecordingLocal final : public hbt::LocalProcessor {
public:
    RecordingLocal(std::unique_ptr<Local> local, std::shared_ptr<ResponseQueue> responses,
                    TradeRecorder& recorder)
        : local_(std::move(local)), responses_(std::move(responses)), recorder_(recorder) {}

    void process(const hbt::Event& event) override {
        local_->process(event);
        const double mid = (local_->best_bid() + local_->best_ask()) * 0.5;
        if (std::isfinite(mid)) {
            last_mid = mid;
        }
        ++event_count;
    }
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        std::vector<hbt::Order<>> received;
        for (auto& [response, receipt_timestamp] : *responses_) {
            if (receipt_timestamp != timestamp) {
                break;
            }
            received.push_back(response);
            // 上游拒绝回报可能携带原订单的旧 Partial/Filled 状态；它不是新成交。
            // Local 对状态盲目 apply_fill，因此仅规范化此类拒绝的计账字段。
            // 原始回报仍逐条导出；原生 Local 的 req==Rejected 状态分支保持不变。
            if (response.req == hbt::Status::Rejected &&
                (response.status == hbt::Status::PartiallyFilled || response.status == hbt::Status::Filled)) {
                response.status = hbt::Status::None;
                response.exec_qty = 0.0;
                recorder_.normalized_rejection();
            }
        }
        const bool received_requested = local_->process_recv_order(timestamp, order_id);
        for (const auto& response : received) {
            recorder_.response(response, timestamp, last_mid);
        }
        recorder_.reconcile(local_->state_values());
        return received_requested;
    }
    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return local_->event_seen_timestamp(event);
    }
    std::int64_t earliest_recv_order_timestamp() const noexcept override {
        return local_->earliest_recv_order_timestamp();
    }
    std::int64_t earliest_send_order_timestamp() const noexcept override {
        return local_->earliest_send_order_timestamp();
    }
    void submit_order(hbt::OrderId order_id, hbt::Side side, double price, double quantity,
                      hbt::OrdType order_type, hbt::TimeInForce tif, std::int64_t timestamp) override {
        local_->submit_order(order_id, side, price, quantity, order_type, tif, timestamp);
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

    std::uint64_t event_count = 0;
    double last_mid = std::numeric_limits<double>::quiet_NaN();

private:
    std::unique_ptr<Local> local_;
    std::shared_ptr<ResponseQueue> responses_;
    TradeRecorder& recorder_;
};

class CountingExchange final : public hbt::Processor {
public:
    explicit CountingExchange(std::unique_ptr<Exchange> exchange) : exchange_(std::move(exchange)) {}
    void process(const hbt::Event& event) override {
        exchange_->process(event);
        ++event_count;
    }
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        return exchange_->process_recv_order(timestamp, order_id);
    }
    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return exchange_->event_seen_timestamp(event);
    }
    std::int64_t earliest_recv_order_timestamp() const noexcept override {
        return exchange_->earliest_recv_order_timestamp();
    }
    std::int64_t earliest_send_order_timestamp() const noexcept override {
        return exchange_->earliest_send_order_timestamp();
    }
    std::uint64_t event_count = 0;

private:
    std::unique_ptr<Exchange> exchange_;
};

struct TradingEngine {
    hbt::Backtest backtest;
    RecordingLocal* local;
    CountingExchange* exchange;

    TradingEngine(const TradingConfig& config, TradeRecorder& recorder,
                  std::shared_ptr<hbt::EventReaderCache> reader) {
        hbt::OrderBus<> requests;
        hbt::OrderBus<> responses;
        const Latency latency{config.entry_latency_ns, config.response_latency_ns};
        auto native_local = std::make_unique<Local>(Depth{config.tick_size, config.lot_size},
            hbt::State<Asset, Fees>{Asset{1.0}, Fees{hbt::CommonFees{config.maker_fee, config.taker_fee}}},
            0, hbt::LocalToExch<Latency>{requests, responses, latency});
        auto native_exchange = std::make_unique<Exchange>(Depth{config.tick_size, config.lot_size},
            hbt::State<Asset, Fees, Queue::OrderState>{Asset{1.0}, Fees{hbt::CommonFees{config.maker_fee, config.taker_fee}}},
            Queue{}, hbt::ExchToLocal<Latency>{requests, responses, latency});
        auto recording_local = std::make_unique<RecordingLocal>(std::move(native_local), responses.shared(), recorder);
        auto counting_exchange = std::make_unique<CountingExchange>(std::move(native_exchange));
        local = recording_local.get();
        exchange = counting_exchange.get();
        backtest.add_asset(std::move(recording_local), std::move(counting_exchange), std::move(reader));
    }
};

} // namespace mm::detail
