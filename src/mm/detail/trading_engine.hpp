#pragma once

/**
 * M1 原生处理器接入层，组装共用订单总线的 Local 与 PartialFillExchange。
 * RecordingLocal 在回报被消费前保存原始载荷，在消费后把同批结果交给记录器核对。
 * 处理器所有权交给 Backtest，TradingEngine 保留会话内可用的观测指针。
 */

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

// 在原生 Local 更新状态的边界增加行情计数与逐条回报记录。
class RecordingLocal final : public hbt::LocalProcessor {
public:
    // 接管原生 Local，并关联它消费的共享响应队列和会话级记录器。
    RecordingLocal(std::unique_ptr<Local> local, std::shared_ptr<ResponseQueue> responses,
                    TradeRecorder& recorder)
        : local_(std::move(local)), responses_(std::move(responses)), recorder_(recorder) {}

    // 先让本地处理器应用一条行情，再保存当时可见的有效中间价。
    void process(const hbt::Event& event) override {
        /** 输入为调度器按 local_ts 交付的事件，输出为原生盘口更新和本地事件计数。
         * 空簿期间沿用最近有效 mid，后续账本可按当时最近已知价格盯市。
         */
        local_->process(event);
        const double mid = (local_->best_bid() + local_->best_ask()) * 0.5;
        if (std::isfinite(mid)) {
            last_mid = mid;
        }
        ++event_count;
    }
    // 在当前接收时刻保存全部原始回报，并在原生批次处理后进行账本核对。
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        /** 输入 timestamp 为队首响应的本地接收时刻，order_id 可指定等待的委托。
         * 返回原生 Local 是否收到了目标订单响应，供 Backtest 的等待语义使用。
         * 队列条目先复制到 received，才能在 Local 消费后逐笔导出原始载荷。
         */
        std::vector<hbt::Order<>> received;
        for (auto& [response, receipt_timestamp] : *responses_) {
            if (receipt_timestamp != timestamp) {
                break;
            }
            received.push_back(response);
            // 固定版本的 Local 在检查 req 前就按 Filled/Partial 状态调用 apply_fill。
            // 将拒绝载荷中的旧执行字段归零，并改为 None，原生 req=Rejected 分支仍可识别拒绝。
            // received 中的副本保留原字段，离线核对可以完整复现这次兼容处理。
            if (response.req == hbt::Status::Rejected &&
                (response.status == hbt::Status::PartiallyFilled || response.status == hbt::Status::Filled)) {
                response.status = hbt::Status::None;
                response.exec_qty = 0.0;
                recorder_.normalized_rejection();
            }
        }
        const bool received_requested = local_->process_recv_order(timestamp, order_id);
        // 同时刻可能有方向相反的多笔成交，逐条回放账本后再与批次结束的原生状态核对。
        for (const auto& response : received) {
            recorder_.response(response, timestamp, last_mid);
        }
        recorder_.reconcile(local_->state_values());
        return received_requested;
    }
    // 以下接口按 LocalProcessor 契约直接转发，时间选择、订单变更及只读取值由原生对象完成。
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
    // 行情延迟返回最近行情的 exchange/local 时间对，订单延迟返回请求、交易所及回报时间三元组。
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

// 在原生交易所处理器外围统计实际交付事件，供单日完整性检查使用。
class CountingExchange final : public hbt::Processor {
public:
    // 接管原生交易所处理器，后续所有行情和请求按原接口转发。
    explicit CountingExchange(std::unique_ptr<Exchange> exchange) : exchange_(std::move(exchange)) {}
    // 应用一条交易所可见行情，并增加该侧逻辑事件计数。
    void process(const hbt::Event& event) override {
        /** 输入为调度器按 exch_ts 交付的事件，输出为原生交易所状态更新。
         * 每次调用对应一次交易所侧实际处理，结束时与转换清单的逻辑事件数核对。
         */
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

// 单次策略回测的组装结果，两个观测指针的有效期与 backtest 持有的处理器一致。
struct TradingEngine {
    hbt::Backtest backtest;
    RecordingLocal* local;
    CountingExchange* exchange;

    // 把原生处理器、共享请求响应队列和行情缓存注册到同一回测会话。
    TradingEngine(const TradingConfig& config, TradeRecorder& recorder,
                  std::shared_ptr<hbt::EventReaderCache> reader) {
        /** config 提供步长、费率和固定延迟，recorder 必须比本会话存活更久。
         * reader 由 Local 和 Exchange 共同消费，输出为可通过 backtest 推进的会话。
         * 两个 OrderBus 的副本共享底层队列，RecordingLocal 通过公开 shared 接口观测响应。
         */
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
