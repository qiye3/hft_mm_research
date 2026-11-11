#pragma once

/** C++ 特征模块示例，使用前后 BBO 计算事件 OFI，供研究采样器调用。
 * 新增特征时可参考输入结构、纯计算函数与调用方负责区段管理的分工。
 */

namespace mm {

// 同一接收时刻可见的最优价格和挂单数量，价格数量必须来自有效 BBO。
struct BboState {
    double bid;
    double ask;
    double bid_quantity;
    double ask_quantity;
};

// 【核心逻辑175】根据一次有效盘口更新前后的最优价量，计算买方为正的 OFI，数量单位沿用源数据。
inline double calculate_bbo_ofi(const BboState& previous, const BboState& current) {
    /** 两个输入必须属于同一连续区段，且不能跨越快照初始化或无效盘口。
     * 返回逐事件 OFI，快照、缺口和窗口累计由调用方管理。
     */
    return (current.bid >= previous.bid ? current.bid_quantity : 0)
         - (current.bid <= previous.bid ? previous.bid_quantity : 0)
         - (current.ask <= previous.ask ? current.ask_quantity : 0)
         + (current.ask >= previous.ask ? previous.ask_quantity : 0);
}

} // namespace mm
