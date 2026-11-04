#pragma once

/** M1 报价模块的公开入口，连接原生本地状态、固定报价规则与交易记录器。 */

#include <optional>
#include <hbt/hbt.hpp>
#include "mm/trade_recorder.hpp"

namespace mm {

// 买卖目标价格以 tick 档位表示，提交订单时再乘 tick_size 转成 USDT。
struct QuoteTicks {
    std::int64_t bid;
    std::int64_t ask;
};

// 输入有效 BBO 档位与正半宽，返回向外取整的双边目标档位。
QuoteTicks fixed_quotes(std::int64_t best_bid, std::int64_t best_ask, std::int64_t half_width);
// 从 config 内的平铺 YAML 建立已校验参数，并解析相对输入输出位置。
TradingConfig load_trading_config(const std::filesystem::path& filename);
// 对配置执行数量网格和调度约束检查，非法参数以异常报告给入口程序。
void validate_trading_config(const TradingConfig& config);

/**
 * 保存固定报价策略的决策状态，库存与订单生命周期从 hbt::Backtest 读取。
 * recorder 必须覆盖策略对象的存活期，价格锚与订单 ID 只服务于报价决策。
 */
class MarketMaker {
public:
    // 关联策略配置和记录器，并在建立决策状态前检查参数约束。
    MarketMaker(const TradingConfig& config, TradeRecorder& recorder);
    // 执行一次双边报价决策，stop 为真时请求撤销现有委托。
    void on_decision(hbt::Backtest& backtest, bool stop = false);
    // 返回买卖方向未终结订单的剩余 BTC 数量，包含新单与撤单在途。
    [[nodiscard]] std::pair<double, double> reservations(const hbt::Backtest& backtest) const;

private:
    // 使用原生订单视图决定单侧动作，已有风险订单终结后才补入替代单。
    void quote_side(hbt::Backtest& backtest, hbt::Side side, std::int64_t target_tick, bool stop);
    TradingConfig config_;
    TradeRecorder& recorder_;
    hbt::OrderId next_order_id_ = 1;
    std::optional<double> reference_tick_; // 最近一次更新目标时的中间价，可含半 tick。
    QuoteTicks targets_{};
};

// 使用指定共享分片缓存运行完整做市链路，供内存夹具和文件回测共用。
TradingSummary run_baseline(const TradingConfig& config,
                            std::shared_ptr<hbt::EventReaderCache> reader);
// 从 config.source 清单创建原生 reader，并返回已核对的交易汇总。
TradingSummary run_baseline(const TradingConfig& config);

} // namespace mm
