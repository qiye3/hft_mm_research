#pragma once

/**
 * M1 原生处理器接入层，组装共用订单总线的 Local 与 PartialFillExchange。
 * RecordingLocal 在回报被消费前保存原始载荷，在消费后把同批结果交给记录器核对。
 * 处理器所有权交给 Backtest，TradingEngine 保留会话内可用的观测指针。
 */

#include <limits>
#include <deque>
#include <memory>
#include <stdexcept>
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
    // 【核心逻辑49】接管原生 Local，并关联它消费的共享响应队列和会话级记录器。
    RecordingLocal(std::unique_ptr<Local> local, std::shared_ptr<ResponseQueue> responses,
                    TradeRecorder& recorder)
        : local_(std::move(local)), responses_(std::move(responses)), recorder_(recorder) {}

    // 【核心逻辑51】先让本地处理器应用一条行情，再保存当时可见的中间价。
    void process(const hbt::Event& event) override {
        /** 输入为调度器按 local_ts 交付的事件，输出为原生盘口更新和本地事件计数。
         * 当前盘口无效时保留非有限值，后续成交和最终盯市必须明确失败。
         */
        local_->process(event);
        const double mid = (local_->best_bid() + local_->best_ask()) * 0.5;
        last_mid = mid;
        ++event_count;
    }
    // 【核心逻辑60】在当前接收时刻保存全部原始回报，并在原生批次处理后进行账本核对。
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        /** 输入 timestamp 为队首响应的本地接收时刻，order_id 可指定等待的委托。
         * 返回原生 Local 是否收到了目标订单响应，供 Backtest 的等待语义使用。
         * 队列条目先复制到 received，才能在 Local 消费后逐笔导出原始载荷。
         */
        std::vector<hbt::Order<>> received;
        for (const auto& [response, receipt_timestamp] : *responses_) {
            if (receipt_timestamp != timestamp) {
                break;
            }
            received.push_back(response);
            // 拒绝回报携带成交状态会触发固定引擎的重复记账，必须在消费前终止。
            if (response.req == hbt::Status::Rejected &&
                (response.status == hbt::Status::PartiallyFilled || response.status == hbt::Status::Filled)) {
                throw std::runtime_error("rejected response carries a fill status; pinned engine cannot process it safely");
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
    // 【非核心】转交原生Local确定事件的本地可见时刻。
    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return local_->event_seen_timestamp(event);
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
                      hbt::OrdType order_type, hbt::TimeInForce tif, std::int64_t timestamp) override {
        local_->submit_order(order_id, side, price, quantity, order_type, tif, timestamp);
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
    // 【非核心】接管原生交易所处理器，后续所有行情和请求按原接口转发。
    explicit CountingExchange(std::unique_ptr<Exchange> exchange) : exchange_(std::move(exchange)) {}
    // 【非核心】应用一条交易所可见行情，并增加该侧逻辑事件计数。
    void process(const hbt::Event& event) override {
        /** 输入为调度器按 exch_ts 交付的事件，输出为原生交易所状态更新。
         * 每次调用对应一次交易所侧实际处理，结束时与转换清单的逻辑事件数核对。
         */
        exchange_->process(event);
        ++event_count;
    }
    // 【非核心】将待处理订单请求交给原生交易所处理器。
    bool process_recv_order(std::int64_t timestamp, std::optional<hbt::OrderId> order_id) override {
        return exchange_->process_recv_order(timestamp, order_id);
    }
    // 【非核心】读取原生交易所处理器确定的事件可见时刻。
    std::optional<std::int64_t> event_seen_timestamp(const hbt::Event& event) const noexcept override {
        return exchange_->event_seen_timestamp(event);
    }
    // 【非核心】读取原生交易所下一条待接收订单请求的时刻。
    std::int64_t earliest_recv_order_timestamp() const noexcept override {
        return exchange_->earliest_recv_order_timestamp();
    }
    // 【非核心】读取原生交易所下一条待发送订单回报的时刻。
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

    // 【核心逻辑48】把原生处理器、共享请求响应队列和行情缓存注册到同一回测会话。
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
