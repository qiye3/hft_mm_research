"""从C++订单、成交和权益导出文件独立重建账本，供M1和模型回测共用。"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


# ==================== 逐笔订单与权益账本核对 ====================
# 【非核心】允许浮点累加误差，但对非有限值或超过容差的账本差异立即报错。
def close(actual: float, expected: float, name: str) -> None:
    """核对一个独立重算字段与 C++ 导出值。

    输入 actual 为文件或汇总中的值，expected 为 Python 重算值，name 标识检查位置。
    输出为正常返回或 ValueError，比较采用 1e-10 相对容差与 1e-7 绝对容差。
    用于吸收跨语言浮点表示与累加误差，同时保留可定位到字段的失败信息。
    """
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f"{name} 不一致：{actual} != {expected}")


# 【非核心】从订单流水重算请求和回报分类，建立供成交检查使用的原始委托量索引。
def audit_orders(directory: Path, summary: dict) -> dict[int, float]:
    """检查一组实验的 orders.csv 与汇总计数相符。

    输入
        directory 包含 C++ 导出的逐笔文件，summary 为同目录的 summary.json 内容。
    输出
        返回订单 ID 到提交数量的映射，重复 ID 或分类数量不符时抛出 ValueError。
    用途
        在成交核对前确定每个订单的原始数量，并区分拒绝请求与有经济变化的成交回报。
        拒绝回报携带成交状态时直接报错，避免再次计入历史执行量。
    """
    submitted = {}
    counts = Counter()
    with (directory / "orders.csv").open() as source:
        for row in csv.DictReader(source):
            kind = row["kind"]
            counts[kind] += 1
            order_id = int(row["order_id"])
            if kind == "submit":
                if order_id in submitted:
                    raise ValueError("订单 ID 重复")
                submitted[order_id] = float(row["quantity"])
            elif kind == "response":
                if row["request"] == "rejected":
                    counts["rejected"] += 1
                    if row["status"] in {"filled", "partial"}:
                        raise ValueError("拒绝回报携带成交状态，不能核算为有效回测")
                elif row["status"] in {"canceled", "expired"}:
                    counts[row["status"]] += 1
                elif row["status"] in {"filled", "partial"} and float(row["exec_quantity"]) > 0:
                    counts["fills"] += 1
    for summary_key, count_key in [
        ("submissions", "submit"),
        ("cancel_requests", "cancel_request"),
        ("cancellations", "canceled"),
        ("expired", "expired"),
        ("rejected", "rejected"),
        ("fills", "fills"),
    ]:
        if counts[count_key] != summary[summary_key]:
            raise ValueError(f"委托流水与汇总不符：{summary_key}")
    return submitted


# 【非核心】按逐笔成交重建平均成本账本，并检查累计成交量和每笔费用。
def audit_fills(directory: Path, summary: dict, submitted: dict[int, float]) -> dict:
    """独立重算交易现金流、净仓位、费用与已实现收益。

    输入
        directory 提供 fills.csv，summary 提供费率、最终标记价格和原生核对后的汇总。
        submitted 来自 audit_orders，指定每个有效订单可成交的原始数量。
    输出
        返回等长数组 times、prices、quantities、signs 和 mids，供后续 markout 使用。
        数量超出委托、字段差异或汇总成交数不符时抛出 ValueError。
    用途
        按新增执行量逐笔计账，验证方向抵消和部分成交都保留完整现金流。
        times 使用回报本地接收纳秒，signs 中买为 1、卖为 -1，金额单位为 USDT。
    """
    position = 0.0
    balance = 0.0
    fees = 0.0
    volume = 0.0
    turnover = 0.0
    average_cost = 0.0
    realized_gross = 0.0
    times = []
    prices = []
    quantities = []
    signs = []
    mids = []
    filled_by_order = Counter()
    with (directory / "fills.csv").open() as source:
        for row in csv.DictReader(source):
            quantity = float(row["quantity"])
            price = float(row["price"])
            sign = int(row["side"])
            order_id = int(row["order_id"])
            if order_id not in submitted or sign not in {-1, 1} or quantity <= 0:
                raise ValueError("成交缺少有效委托或数量/方向错误")
            filled_by_order[order_id] += quantity
            if filled_by_order[order_id] > submitted[order_id] + 1e-10:
                raise ValueError("累计成交量超过委托量")
            old_position = position
            position += sign * quantity
            if old_position == 0 or old_position * sign > 0:
                average_cost = (abs(old_position) * average_cost + price * quantity) / (abs(old_position) + quantity)
            else:
                # 先实现被对冲旧仓位的收益，剩余新方向仓位的成本改用本次成交价。
                realized_gross += min(abs(old_position), quantity) * (price - average_cost) * math.copysign(1, old_position)
                if abs(position) < 1e-12:
                    average_cost = 0.0
                elif old_position * position < 0:
                    average_cost = price
            if abs(position) < 1e-12:
                position = 0.0
            balance -= sign * quantity * price
            fee = price * quantity * (summary["maker_fee"] if row["maker"] == "1" else summary["taker_fee"])
            fees += fee
            volume += quantity
            turnover += price * quantity
            mid = float(row["mid"])
            for key, expected in [
                ("fee", fee), ("position", position), ("balance", balance),
                ("average_cost", average_cost), ("realized_gross", realized_gross),
                ("net_pnl", balance + position * mid - fees),
            ]:
                close(float(row[key]), expected, f"fill.{key}")
            times.append(int(row["local_timestamp_ns"]))
            prices.append(price)
            quantities.append(quantity)
            signs.append(sign)
            mids.append(mid)
    for key, expected in [
        ("position", position), ("balance", balance), ("fees", fees),
        ("volume", volume), ("turnover", turnover), ("average_cost", average_cost),
        ("realized_gross", realized_gross),
        ("net_pnl", balance + position * summary["final_mid"] - fees),
    ]:
        close(summary[key], expected, f"summary.{key}")
    if len(times) != summary["fills"]:
        raise ValueError("成交 CSV 数量不一致")
    return {
        "times": np.asarray(times, dtype=np.int64),
        "prices": np.asarray(prices),
        "quantities": np.asarray(quantities),
        "signs": np.asarray(signs),
        "mids": np.asarray(mids),
    }


# 【非核心】从权益观察点核对 PnL 分解、库存预留及累计金额回撤。
def audit_equity(directory: Path, summary: dict) -> tuple[np.ndarray, np.ndarray]:
    """验收逐点权益并取得后续 markout 所需的本地中间价序列。

    输入
        directory 包含 equity.csv，summary 给出初始资金、风险上限和最终风险指标。
    输出
        返回非递减的 int64 本地时间数组与对应 mid 数组，空序列或任一核对失败时抛异常。
    用途
        将现金流加库存市值减费用的表达与已实现加未实现收益分解交叉核对。
        回撤取已导出决策点与成交点的金额峰谷差，预留风险仅检查有相应字段的决策点。
    """
    times = []
    mids = []
    peak = summary["initial_capital"]
    max_drawdown = 0.0
    maximum_inventory = 0.0
    with (directory / "equity.csv").open() as source:
        for row in csv.DictReader(source):
            mid = float(row["mid"])
            position = float(row["position"])
            pnl = float(row["balance"]) + position * mid - float(row["fees"])
            equity = summary["initial_capital"] + pnl
            close(float(row["net_pnl"]), pnl, "equity.pnl")
            close(float(row["equity"]), equity, "equity.equity")
            close(float(row["realized_gross"]) + float(row["unrealized"]) - float(row["fees"]), pnl, "equity.decomposition")
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
            maximum_inventory = max(maximum_inventory, abs(position))
            if abs(position) > summary["max_position"] + 1e-10:
                raise ValueError("实际库存超过上限")
            if row["reserved_buy"]:
                # 成交点的预留字段留空，决策点的字符串 "0" 仍应进入实际预算检查。
                if position + float(row["reserved_buy"]) > summary["max_position"] + 1e-10:
                    raise ValueError("买方在途敞口超限")
                if position - float(row["reserved_sell"]) < -summary["max_position"] - 1e-10:
                    raise ValueError("卖方在途敞口超限")
            times.append(int(row["local_timestamp_ns"]))
            mids.append(mid)
    if not times or np.any(np.diff(times) < 0):
        raise ValueError("权益记录为空或时间逆序")
    close(summary["max_drawdown"], max_drawdown, "maximum drawdown")
    close(summary["max_abs_position"], maximum_inventory, "max inventory")
    return np.asarray(times, dtype=np.int64), np.asarray(mids)


# ==================== 完整日验收与成交质量分析 ====================
# 【核心逻辑199】核对完整日的一组实验，再按成交量加权计算本地回报后一秒的收益标记。
def analyze_run(directory: Path, manifest: dict) -> dict:
    """为一组已完成的 C++ 回测补充独立核对和短期成交质量统计。

    输入
        directory 包含 summary.json 和三份 CSV，manifest 为本日标准事件清单。
    输出
        返回加入 markout、观察样本数、末端剔除数和 csv_reconciled 的汇总字典。
        无可用一秒样本时，各 markout 指标为 None。
    用途
        先确认两侧完整消费同一日行情，再计算买卖方向统一的成交后标记。
        未来价格使用 t+1s 之前最后一个权益观察点，按成交 BTC 数量加权。
    """
    summary = json.loads((directory / "summary.json").read_text())
    if summary["date"] != manifest["date"]:
        raise ValueError("行情日期不符")
    if summary["local_events"] != manifest["logical_events"] or summary["exchange_events"] != manifest["logical_events"]:
        raise ValueError("没有消费同一完整日的双侧行情")
    submitted = audit_orders(directory, summary)
    fills = audit_fills(directory, summary, submitted)
    times, mids = audit_equity(directory, summary)
    valid = fills["times"] + 1_000_000_000 <= times[-1]
    if valid.any():
        # right 再减一得到目标时刻的 as-of 值，同时间戳取最后一条已记录观察。
        indices = np.searchsorted(times, fills["times"][valid] + 1_000_000_000, side="right") - 1
        sign = fills["signs"][valid]
        prices = fills["prices"][valid]
        future_mids = mids[indices]
        weights = fills["quantities"][valid]
        summary["mean_markout_1s_usdt_per_btc"] = float(np.average(sign * (future_mids - prices), weights=weights))
        summary["mean_markout_1s_bps"] = float(np.average(sign * (future_mids - prices) / prices * 10000, weights=weights))
        summary["mean_post_fill_mid_move_1s_usdt_per_btc"] = float(np.average(sign * (future_mids - fills["mids"][valid]), weights=weights))
    else:
        summary["mean_markout_1s_usdt_per_btc"] = None
        summary["mean_markout_1s_bps"] = None
        summary["mean_post_fill_mid_move_1s_usdt_per_btc"] = None
    summary["markout_observations"] = int(valid.sum())
    summary["markout_censored"] = int(len(valid) - valid.sum())
    summary["csv_reconciled"] = True
    return summary
