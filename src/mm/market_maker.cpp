/**
 * M1 做市主流程，将本地盘口转换成目标报价并驱动原生回测接口。
 * MarketMaker 读取框架库存与订单，TradeRecorder 核对回报后的交易账本。
 * 输入为 TradingConfig 和行情分片，输出为交易汇总及逐笔 CSV。
 */
#include "mm/market_maker.hpp"
#include "mm/detail/trading_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace mm {
namespace {

// 判断订单的剩余数量是否仍占用方向性库存预算。
bool reserves_inventory(const hbt::Order<>& order) {
    /** 输入为框架本地订单，返回它是否仍可能带来新增成交。
     * active 覆盖已接受和部分成交订单，新单尚未确认时由 req=New 识别。
     * 撤单在途的订单仍为 active，风险预算要持续保留到终结回报到达。
     */
    return order.active() || (order.status == hbt::Status::None && order.req == hbt::Status::New);
}

// 移除配置标量两端的空白字符，保留值内部的内容。
std::string strip(std::string value) {
    /** 输入为一行配置的键或值，输出为裁去首尾空白的字符串。
     * 全空白输入返回空串，供配置读取阶段跳过空行。
     */
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    return value.substr(start, value.find_last_not_of(" \t\r\n") - start + 1);
}

// 读取平铺 YAML 标量，并在字段重复时终止配置加载。
std::map<std::string, std::string> read_flat_config(const std::filesystem::path& filename) {
    /** 输入 filename 指向可读配置文件，输出键到原始标量字符串的映射。
     * 处理行尾注释和外层引号，数值转换留给 load_trading_config。
     * 无法读取、缺少分隔符或重复字段时抛出异常，调用方可定位配置错误。
     */
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("cannot open trading config: " + filename.string());
    }
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(input, line)) {
        line = strip(line.substr(0, line.find('#')));
        if (line.empty()) {
            continue;
        }
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            throw std::runtime_error("expected scalar key: value");
        }
        const auto key = strip(line.substr(0, separator));
        auto value = strip(line.substr(separator + 1));
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        if (!fields.emplace(key, value).second) {
            throw std::runtime_error("duplicate config key: " + key);
        }
    }
    return fields;
}

} // namespace

// 以中间价为中心，将固定半宽的双边报价向外取整到价格档位。
QuoteTicks fixed_quotes(std::int64_t best_bid, std::int64_t best_ask, std::int64_t half_width) {
    /** 输入 best_bid、best_ask 和 half_width 的单位均为 tick。
     * 盘口必须满足买价小于卖价，半宽必须为正，否则抛出 invalid_argument。
     * 输出买卖整数档位，买价向下取整、卖价向上取整，供后续限价委托使用。
     */
    if (best_bid >= best_ask || half_width <= 0) {
        throw std::invalid_argument("need noncrossed book and positive half width");
    }
    const double mid_tick = (static_cast<double>(best_bid) + static_cast<double>(best_ask)) * 0.5;
    return {static_cast<std::int64_t>(std::floor(mid_tick - static_cast<double>(half_width))),
            static_cast<std::int64_t>(std::ceil(mid_tick + static_cast<double>(half_width)))};
}

// 校验数量网格、风险上限和调度参数能否组成有效的做市实验。
void validate_trading_config(const TradingConfig& config) {
    /** 输入为已赋值的策略配置，正常返回表示通过当前阶段的参数约束。
     * 检查下单量与仓位上限均为整 lot，单笔数量不超过净仓位上限。
     * 同时检查更新模式、纳秒时长和文件位置，非法配置抛出 invalid_argument。
     */
    for (const double value : {config.tick_size, config.lot_size, config.order_quantity,
                               config.max_position, config.initial_capital}) {
        if (!std::isfinite(value) || value <= 0) {
            throw std::invalid_argument("size, capital and tick parameters must be positive finite");
        }
    }
    for (const double value : {config.order_quantity, config.max_position}) {
        if (std::abs(value / config.lot_size - std::round(value / config.lot_size)) > 1e-8) {
            throw std::invalid_argument("quantity must be an integer number of lots");
        }
    }
    if (config.order_quantity > config.max_position || config.half_width_ticks <= 0 ||
        config.decision_interval_ns <= 0 || config.price_threshold_ticks <= 0 ||
        config.entry_latency_ns < 0 || config.response_latency_ns < 0 ||
        !std::isfinite(config.maker_fee) || !std::isfinite(config.taker_fee) ||
        (config.update_mode != "time" && config.update_mode != "price") ||
        config.quote_stop_timestamp_ns <= 0 || config.source.empty() || config.output.empty()) {
        throw std::invalid_argument("invalid baseline trading configuration");
    }
}

// 将策略配置中的标量转换成有单位的参数，并确定输入输出位置。
TradingConfig load_trading_config(const std::filesystem::path& filename) {
    /** 输入 filename 位于项目的 config 目录，输出经过校验的 TradingConfig。
     * 缺省字段沿用结构体默认值，相对文件位置以配置目录的上一级解析。
     * 未知字段和数值转换错误会中止 CLI 启动，避免实验参数被静默忽略。
     */
    TradingConfig config;
    for (const auto& [key, value] : read_flat_config(filename)) {
        if (key == "name") {
            config.name = value;
        } else if (key == "date") {
            config.date = value;
        } else if (key == "source") {
            config.source = value;
        } else if (key == "output") {
            config.output = value;
        } else if (key == "update_mode") {
            config.update_mode = value;
        } else if (key == "tick_size") {
            config.tick_size = std::stod(value);
        } else if (key == "lot_size") {
            config.lot_size = std::stod(value);
        } else if (key == "order_quantity") {
            config.order_quantity = std::stod(value);
        } else if (key == "max_position") {
            config.max_position = std::stod(value);
        } else if (key == "initial_capital") {
            config.initial_capital = std::stod(value);
        } else if (key == "maker_fee") {
            config.maker_fee = std::stod(value);
        } else if (key == "taker_fee") {
            config.taker_fee = std::stod(value);
        } else if (key == "half_width_ticks") {
            config.half_width_ticks = std::stoll(value);
        } else if (key == "decision_interval_ns") {
            config.decision_interval_ns = std::stoll(value);
        } else if (key == "price_threshold_ticks") {
            config.price_threshold_ticks = std::stoll(value);
        } else if (key == "entry_latency_ns") {
            config.entry_latency_ns = std::stoll(value);
        } else if (key == "response_latency_ns") {
            config.response_latency_ns = std::stoll(value);
        } else if (key == "quote_stop_timestamp_ns") {
            config.quote_stop_timestamp_ns = std::stoll(value);
        } else {
            throw std::runtime_error("unknown trading field: " + key);
        }
    }
    const auto root = std::filesystem::absolute(filename).parent_path().parent_path();
    config.source = root / config.source;
    config.output = root / config.output;
    validate_trading_config(config);
    return config;
}

// 保存已验证的策略参数，并关联覆盖整个回测生命周期的记录器。
MarketMaker::MarketMaker(const TradingConfig& config, TradeRecorder& recorder)
    : config_(config), recorder_(recorder) { validate_trading_config(config); }

// 从框架订单中累计买卖两侧仍需预留的未成交数量。
std::pair<double, double> MarketMaker::reservations(const hbt::Backtest& backtest) const {
    /** 输入为当前原生回测状态，输出按买方、卖方排列的 BTC 数量对。
     * 使用 leaves_qty 处理部分成交后的剩余风险，供决策检查和权益记录共用。
     * 订单终结后才解除对应预留，因此取消请求本身不会扩大可下单预算。
     */
    double buys = 0.0;
    double sells = 0.0;
    for (const auto& [order_id, order] : backtest.orders(0)) {
        (void)order_id;
        if (!reserves_inventory(order)) {
            continue;
        }
        if (order.side == hbt::Side::Buy) {
            buys += order.leaves_qty;
        } else {
            sells += order.leaves_qty;
        }
    }
    return {buys, sells};
}

// 对单侧执行保留、撤单或新增委托，并把新增数量限制在可用库存预算内。
void MarketMaker::quote_side(hbt::Backtest& backtest, hbt::Side side, std::int64_t target_tick, bool stop) {
    /** 输入 side 指定方向，target_tick 是目标价格档位，stop 要求该侧退出。
     * 通过 backtest 提交原生请求并交给 recorder_ 记录，订单状态由框架推进。
     * 已有订单在途时本次直接返回，每侧同时最多保留一笔可能成交的订单。
     */
    const hbt::Order<>* existing = nullptr;
    for (const auto& [order_id, order] : backtest.orders(0)) {
        (void)order_id;
        if (order.side == side && reserves_inventory(order)) {
            if (existing != nullptr) {
                throw std::runtime_error("more than one working order on a side");
            }
            existing = &order;
        }
    }
    const double sign = hbt::side_sign(side);
    // 买侧预算为上限减当前仓位，卖侧预算为上限加当前仓位。
    // 减仓方向可跨越零仓位，最远到达相反方向的同一绝对上限。
    const double capacity = config_.max_position - sign * backtest.position(0);
    const auto available_lots = static_cast<std::int64_t>(std::floor(capacity / config_.lot_size + 1e-8));
    const auto order_lots = static_cast<std::int64_t>(std::llround(config_.order_quantity / config_.lot_size));
    const double quantity = static_cast<double>(std::max<std::int64_t>(0, std::min(order_lots, available_lots))) * config_.lot_size;
    if (existing != nullptr) {
        const bool replace = stop || quantity == 0.0 || existing->price_tick != target_tick;
        if (replace && existing->cancellable()) {
            const auto order_id = existing->order_id;
            backtest.cancel(0, order_id);
            recorder_.action("cancel_request", backtest.orders(0).at(order_id), backtest.current_timestamp());
        }
        // 当前分支同时覆盖保留原单和等待撤单确认，新目标在后续决策点补入。
        return;
    }
    if (stop || quantity == 0.0) {
        return;
    }
    const auto order_id = next_order_id_++;
    const double price = static_cast<double>(target_tick) * config_.tick_size;
    // 本地报价到达交易所前市场还可能变化，GTX 会使已经穿价的委托失效。
    if (side == hbt::Side::Buy) {
        backtest.submit_buy_order(0, order_id, price, quantity, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    } else {
        backtest.submit_sell_order(0, order_id, price, quantity, hbt::TimeInForce::GTX, hbt::OrdType::Limit, false);
    }
    recorder_.action("submit", backtest.orders(0).at(order_id), backtest.current_timestamp());
}

// 在一次决策时刻更新目标报价，再核对包含在途订单的双侧敞口。
void MarketMaker::on_decision(hbt::Backtest& backtest, bool stop) {
    /** 输入 backtest 提供本地可见盘口和实际仓位，stop 表示进入收尾阶段。
     * 输出体现为原生下单或撤单请求，结束时检查最坏单侧成交后的净仓位。
     * 空簿时触发退出，价格阈值模式以最近一次更新目标时的中间价为锚。
     */
    backtest.clear_inactive_orders(0);
    const auto best_bid = backtest.best_bid_tick(0);
    const auto best_ask = backtest.best_ask_tick(0);
    stop = stop || !std::isfinite(backtest.mid(0));
    if (!stop) {
        const double mid_tick = (static_cast<double>(best_bid) + static_cast<double>(best_ask)) * 0.5;
        if (!reference_tick_ || config_.update_mode == "time" ||
            std::abs(mid_tick - *reference_tick_) >= static_cast<double>(config_.price_threshold_ticks)) {
            targets_ = fixed_quotes(best_bid, best_ask, config_.half_width_ticks);
            reference_tick_ = mid_tick;
        }
    }
    // 即使价格阈值尚未触发，也要处理已成交侧的补单和库存方向限制。
    quote_side(backtest, hbt::Side::Buy, targets_.bid, stop);
    quote_side(backtest, hbt::Side::Sell, targets_.ask, stop);
    const auto [buys, sells] = reservations(backtest);
    if (backtest.position(0) + buys > config_.max_position + 1e-10 ||
        backtest.position(0) - sells < -config_.max_position - 1e-10) {
        throw std::runtime_error("working and pending orders exceed inventory budget");
    }
}

// 让原生双时钟引擎推进完整行情，在固定决策网格上执行做市并记录账本。
TradingSummary run_baseline(const TradingConfig& config, std::shared_ptr<hbt::EventReaderCache> reader) {
    /** 输入为有效策略配置及双方共用的行情缓存，缓存也可由测试夹具构造。
     * 输出为 TradingSummary，逐笔订单、成交和权益写入 config.output。
     * 每次 elapse 会处理目标时间前的全部市场事件和回报，策略随后才作决策。
     * 结束时撤单并排空回报，订单未终结或账本不符会抛出异常。
     */
    validate_trading_config(config);
    TradeRecorder recorder{config};
    detail::TradingEngine engine{config, recorder, std::move(reader)};
    auto& backtest = engine.backtest;
    MarketMaker maker{config, recorder};
    const auto start = std::chrono::steady_clock::now();
    auto status = backtest.elapse(0);
    while (status != hbt::ElapseResult::EndOfData) {
        const auto now = backtest.current_timestamp();
        maker.on_decision(backtest, now >= config.quote_stop_timestamp_ns);
        if (std::isfinite(engine.local->last_mid)) {
            const auto [buys, sells] = maker.reservations(backtest);
            recorder.checkpoint(now, engine.local->last_mid, backtest.state_values(0), buys, sells);
        }
        const auto next = (now / config.decision_interval_ns + 1) * config.decision_interval_ns;
        // 对齐绝对纳秒时间网格，避免随回报到达时刻累积决策时钟漂移。
        status = backtest.elapse(next - now);
    }
    // 数据结束后继续通过引擎处理撤单与回报，尾仓按最后本地有效中间价盯市。
    maker.on_decision(backtest, true);
    backtest.close();
    backtest.clear_inactive_orders(0);
    if (!backtest.orders(0).empty()) {
        throw std::runtime_error("unresolved orders at end of data; increase stop buffer");
    }
    recorder.checkpoint(backtest.current_timestamp(), engine.local->last_mid, backtest.state_values(0), 0, 0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    recorder.finish(backtest.state_values(0), engine.local->last_mid, engine.local->event_count,
                    engine.exchange->event_count, seconds);
    const auto result = recorder.summary();
    std::cout << config.name << " fills=" << result.fills << " pnl=" << result.net_pnl
              << " fees=" << result.fees << " max_inventory=" << result.max_abs_position
              << " seconds=" << seconds << '\n';
    return result;
}

// 从标准事件清单加载共享分片缓存，并启动与测试相同的策略执行链路。
TradingSummary run_baseline(const TradingConfig& config) {
    /** 输入 config.source 指向包含 shards 的转换清单，输出完整策略汇总。
     * 文件加载由原生 EventReaderCache 完成，真实数据与内存夹具共用上面的执行实现。
     */
    return run_baseline(config, hbt::event_reader_from_files(hbt::event_files_from_manifest(config.source.string())));
}

} // namespace mm
