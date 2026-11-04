"""执行 M1 的 A/B/C 固定报价实验，并从逐笔文件独立检查交易结果。

run_experiments 将同一日行情交给 C++ baseline_mm 逐组运行，audit_orders、audit_fills
和 audit_equity 从导出 CSV 重建生命周期计数、平均成本账本与权益。
analyze_run 在核对后计算本地成交回报时刻起的一秒 markout，聚合 JSON 保存在实验输出目录，
write_report 生成中文报告。输入为策略配置、数据清单及实验 CSV，输出为核对后的七组机制实验结果。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
# 每组只覆盖对照因素，共用配置中的其余费率、延迟和数据日期。
CASES = {
    "A_width10": {"half_width_ticks": 10},
    "A_width20": {},
    "A_width50": {"half_width_ticks": 50},
    "B_loose": {"max_position": 0.05},
    "B_strict": {"max_position": 0.003},
    "C_time500ms": {"decision_interval_ns": 500_000_000},
    "C_price20ticks": {"update_mode": "price", "price_threshold_ticks": 20},
}


# ==================== 逐笔订单与权益账本核对 ====================
# 允许浮点累加误差，但对非有限值或超过容差的账本差异立即报错。
def close(actual: float, expected: float, name: str) -> None:
    """核对一个独立重算字段与 C++ 导出值。

    输入 actual 为文件或汇总中的值，expected 为 Python 重算值，name 标识检查位置。
    输出为正常返回或 ValueError，比较采用 1e-10 相对容差与 1e-7 绝对容差。
    用于吸收跨语言浮点表示与累加误差，同时保留可定位到字段的失败信息。
    """
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f"{name} 不一致：{actual} != {expected}")


# 从订单流水重算请求和回报分类，建立供成交检查使用的原始委托量索引。
def audit_orders(directory: Path, summary: dict) -> dict[int, float]:
    """检查一组实验的 orders.csv 与汇总计数相符。

    输入
        directory 包含 C++ 导出的逐笔文件，summary 为同目录的 summary.json 内容。
    输出
        返回订单 ID 到提交数量的映射，重复 ID 或分类数量不符时抛出 ValueError。
    用途
        在成交核对前确定每个订单的原始数量，并区分拒绝请求与有经济变化的成交回报。
        兼容分支保存的是原始载荷，因此旧 Partial/Filled 拒绝也可从流水中再次识别。
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
                    counts["normalized_rejections"] += int(row["status"] in {"filled", "partial"})
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
        ("normalized_rejections", "normalized_rejections"),
        ("fills", "fills"),
    ]:
        if counts[count_key] != summary[summary_key]:
            raise ValueError(f"委托流水与汇总不符：{summary_key}")
    return submitted


# 按逐笔成交重建平均成本账本，并检查累计成交量和每笔费用。
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


# 从权益观察点核对 PnL 分解、库存预留及累计金额回撤。
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
# 核对完整日的一组实验，再按成交量加权计算本地回报后一秒的收益标记。
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


# ==================== 实验编排与研究报告输出 ====================
# 逐组生成独立配置、运行同一 C++ 策略入口，并在每组成功后保存核对结果。
def run_experiments(output_root: Path) -> list[dict]:
    """串行执行 CASES 中定义的七组单日机制实验。

    输入
        output_root 为实验输出根目录，每个案例的子目录必须尚未存在。
        默认参数来自项目 config/baseline_mm.yaml，可执行文件来自 Release 构建目录。
    输出
        返回按 CASES 顺序排列的分析结果，并持续写入 output_root/analysis.json。
        每组生成的绝对位置配置保存在 output_root/configs 目录。
    用途
        统一数据、费率和延迟，仅按各案例覆盖指定因素。C++ 失败或 CSV 核对失败时停止，
        已完成组的配置、逐笔文件及进度结果可用于定位异常和复核。
    """
    base = yaml.safe_load((ROOT / "config/baseline_mm.yaml").read_text())
    manifest = json.loads((ROOT / base["source"]).read_text())
    configuration_directory = output_root / "configs"
    configuration_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for name, overrides in CASES.items():
        config = base | overrides | {
            "name": name,
            "source": str(ROOT / base["source"]),
            "output": str(output_root / name),
        }
        filename = configuration_directory / f"{name}.yaml"
        filename.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"运行 {name}", flush=True)
        subprocess.run([str(ROOT / "workspace/build/release/baseline_mm"), str(filename)], check=True, cwd=ROOT)
        result = analyze_run(Path(config["output"]), manifest)
        results.append(result)
        (output_root / "analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(f"核对通过 {name}: net_pnl={result['net_pnl']:.6f}", flush=True)
    return results


# 将固定七组的核对结果排成对照表，并用实测数值解释三个机制实验。
def write_report(results: list[dict], destination: Path) -> None:
    """保存可独立阅读的 M1 中文实验报告。

    输入
        results 应包含 CASES 的全部七个名称，当前报告叙述要求对照组具有有效 markout。
        destination 指定报告目录，目录可自动创建。
    输出
        仅写出 report.md，展示主要对照与测量口径。完整指标由调用方保存至实验输出目录。
    用途
        把报价宽度、库存上限和更新频率的变化与费用、回撤及成交后价格走势联系起来。
        本步骤使用已核对的结果，重新生成报告时可直接传入缓存分析字典。
    """
    destination.mkdir(parents=True, exist_ok=True)
    notes_link = Path(
        os.path.relpath(ROOT / "docs/implementation/m1_notes.md", destination.resolve())
    ).as_posix()
    lines = [
        "# 2026-11-03 — M1 固定报价做市机制实验",
        "",
        "全部实验使用 M0 同一份 2025-02-01 UTC BTCUSDT 完整日数据。原生 PartialFillExchange + RiskAdverseQueueModel，GTX 仅挂单，每笔 0.001 BTC，初始资金 10,000 USDT。",
        "",
        "假设 maker 费率 0.0002（2 bps）、taker 费率 0.0005（5 bps），请求/响应延迟 1 ms / 0.5 ms。费率及延迟是机制实验参数；历史合约规则未认证。PnL 为交易现金流、尾仓盯市减手续费，不含资金费率，不作期末强平。",
        "",
        "| 实验 | 成交回报数 | 撤单请求 | 净 PnL/USDT | 费用/USDT | 最大仓位/BTC | 回撤/USDT | 1s markout/USDT每BTC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        markout = result["mean_markout_1s_usdt_per_btc"]
        markout_text = "无可用样本" if markout is None else f"{markout:.3f}"
        cells = [
            result["name"],
            f"{result['fills']:,}",
            f"{result['cancel_requests']:,}",
            f"{result['net_pnl']:.3f}",
            f"{result['fees']:.3f}",
            f"{result['max_abs_position']:.3f}",
            f"{result['max_drawdown']:.3f}",
            markout_text,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    by_name = {result["name"]: result for result in results}
    base = by_name["A_width20"]
    narrow = by_name["A_width10"]
    wide = by_name["A_width50"]
    loose = by_name["B_loose"]
    strict = by_name["B_strict"]
    slow = by_name["C_time500ms"]
    threshold = by_name["C_price20ticks"]
    lines.extend([
        "",
        "## 三个实验的实际变化",
        "",
        f"A：半宽从 10 增至 50 ticks，成交从 {narrow['fills']:,} 降至 {wide['fills']:,}，费用从 {narrow['fees']:.2f} 降至 {wide['fees']:.2f} USDT，净 PnL 从 {narrow['net_pnl']:.2f} 变为 {wide['net_pnl']:.2f}。较宽报价降低成交机会和周转成本；其 1 秒 markout 仍为 {wide['mean_markout_1s_usdt_per_btc']:.2f} USDT/BTC。成交后有符号 Mid 变化均值为 {wide['mean_post_fill_mid_move_1s_usdt_per_btc']:.2f}，说明这些较深报价成交时仍伴随不利的价格移动。",
        "",
        f"B：宽松 0.05 BTC 与严格 0.003 BTC 上限下，实际最大库存分别为 {loose['max_abs_position']:.3f}/{strict['max_abs_position']:.3f} BTC，最终库存 {loose['position']:.3f}/{strict['position']:.3f} BTC，回撤 {loose['max_drawdown']:.2f}/{strict['max_drawdown']:.2f} USDT。严格限制使成交减少至 {strict['fills']:,}，费用减少至 {strict['fees']:.2f}；该日改善同时包含更低周转与更低敞口，不能解释为已经找到最优库存控制。",
        "",
        f"C：基线 100 ms 更新撤单 {base['cancel_requests']:,} 次；500 ms 为 {slow['cancel_requests']:,} 次、净 PnL {slow['net_pnl']:.2f}，成交与费用也同步下降。20-tick 阈值更新撤单 {threshold['cancel_requests']:,} 次，但成交 {threshold['fills']:,} 次，净 PnL {threshold['net_pnl']:.2f}，费用 {threshold['fees']:.2f}，该日未因撤单减少而改善收益。保留较久的报价同时改变排队优先级和价格陈旧程度，本实验没有将两者再作因果拆分。",
        "",
        f"20-tick 基线的账本分解为：{base['realized_gross']:.6f} 已实现毛收益 + {base['unrealized']:.6f} 尾仓浮盈亏 − {base['fees']:.6f} 手续费 = {base['net_pnl']:.6f} USDT。其 1 秒 markout 为 {base['mean_markout_1s_usdt_per_btc']:.3f} USDT/BTC，支持检查逆向选择这一后续方向；此处没有开发保护信号。",
        "",
        "所有默认委托都是一个 lot，因此真实日部分成交计数为零；部分成交与撤单竞态由合成多 lot 回归验证。一天结果用于理解机制，不据此判断稳定盈利能力。",
        "",
        "## 设计与指标",
        "",
        "A：半宽 10/20/50 ticks（1/2/5 USDT），其他参数相同。B：保留 width20 基线，将安全净仓上限从 0.01 BTC 改为宽松 0.05 或严格 0.003；没有库存 skew。C：比较基线 100 ms 定时、500 ms 定时，以及每 100 ms 检查参考中间价累计变化 20 ticks 才更新目标报价。阈值模式仍会补充已成交订单，并立即执行库存预算限制。",
        "",
        "同一侧最多一笔未终结委托；撤单或新单在途继续占用剩余数量预算，确认前不发替代单。所有报价均向外取整到 Tick。若交易所接收时变为可立即成交，GTX 使其失效而不吃单。23:59:59 停止新报价，排空撤单及成交回报；尾仓按最后本地中间价估值。",
        "",
        "成交量、现金、手续费、仓位与框架 StateValues 在每批真实回报后核对。Python 从 CSV 独立核对成交累计量不超过订单量、平均成本法已实现收益、逐点权益、费用、敞口上限和最大回撤；每组双侧都必须恰好处理 78,619,443 条逻辑事件。",
        "",
        "1 秒 markout 以成交回报本地接收时刻为起点，使用 t+1s 前最后一个权益观察点的本地中间价，买方取正号、卖方取负号，按成交数量加权；不是交易所时间 markout。样本末端不足 1 秒者剔除并记录。权益按决策点及每笔成交记录，回撤基于这些观察点，非逐行情 Tick 的连续盯市。",
        "",
        "## 如何理解结果",
        "",
        "净 PnL = 毛已实现收益 + 尾仓浮盈亏 − 手续费。不能只看买卖报价之间的价差；成交后的中间价走势、不能立刻撤掉的委托、累积库存和成本都会改变结果。具体组间变化以本表实测值为准。",
        "",
        "## 复现",
        "",
        "在项目根目录执行 `.venv/bin/python scripts/analyze_baseline.py --run-experiments`。每个实验子目录必须不存在，以避免覆盖一次实验；可用 --output-root 指定新目录。已有输出可用 `--analyze-existing workspace/results/m1/BTCUSDT_2025-02-01` 重新核对，完整参数见 config/baseline_mm.yaml 与脚本 CASES。C++ 入口为 src/apps/baseline_mm.cpp。",
        "",
        "原始逐笔文件保存在 workspace/results/m1/BTCUSDT_2025-02-01/<实验>/{orders.csv,fills.csv,equity.csv,summary.json}，不纳入 Git。完整聚合指标保存在实验输出根目录的 analysis.json，报告目录保存方法说明与结果解读。",
        "",
        f"上游旧拒绝回报的兼容处理、假设范围与单元测试见 [M1 实现笔记]({notes_link})。没有修改第三方源码，未开发 Alpha 或机器学习。",
    ])
    (destination / "report.md").write_text("\n".join(lines) + "\n")


# 根据命令行选择新运行或读取已有实验，最后生成同一格式的报告。
def main() -> None:
    """提供实验编排与离线复核的命令行入口。

    输入为互斥的 --run-experiments 或 --analyze-existing，以及输出和报告目录选项。
    输出为实验目录下的 analysis.json 和报告目录下的 report.md，失败异常向调用命令传播。
    仅分析已有实验时读取各组 CSV，新的回测由 run_experiments 分支启动。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-experiments", action="store_true")
    group.add_argument("--analyze-existing", type=Path)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs/reports/m1_2025-02-01")
    parser.add_argument("--output-root", type=Path, default=ROOT / "workspace/results/m1/BTCUSDT_2025-02-01")
    arguments = parser.parse_args()
    if arguments.run_experiments:
        results = run_experiments(arguments.output_root.resolve())
    else:
        base = yaml.safe_load((ROOT / "config/baseline_mm.yaml").read_text())
        manifest = json.loads((ROOT / base["source"]).read_text())
        results = [analyze_run(arguments.analyze_existing / name, manifest) for name in CASES]
        (arguments.analyze_existing / "analysis.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
    write_report(results, arguments.report_dir)


if __name__ == "__main__":
    main()
