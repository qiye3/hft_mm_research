/** 校验曲线动态规划的终端成本、对称性和当前实际仓位。 */
#include "mm/strategies/curve_policy.hpp"
#include <stdexcept>

// 【非核心】固定解析场景校验当前目标，未来区间计划不能绕过换仓和终端成本。
int main() {
    mm::ReturnCurve curve{};
    mm::ReturnCurve risk{};
    risk.fill(.01);
    if (mm::curve_inventory_target(curve, risk, 1., 2., 0.) != 0.
        || mm::curve_inventory_target(curve, risk, 1., 2., 1.) != 0.) {
        throw std::runtime_error("zero curve must stay flat or reduce existing exposure");
    }
    curve[0] = 10.;
    if (mm::curve_inventory_target(curve, risk, .1, .1, 0.) != 2.) {
        throw std::runtime_error("positive near-term increment should enter long first");
    }
    curve[0] = -10.;
    if (mm::curve_inventory_target(curve, risk, .1, .1, 0.) != -2.) {
        throw std::runtime_error("negative near-term increment should enter short first");
    }
    return 0;
}
