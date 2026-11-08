/* 从已发布日统计派生诊断视图，不读取行情和全量因子序列。 */
(function () {
  'use strict';

  // 【非核心】仅有限日指标参与等权平均，空集合保留缺失。
  function mean(values) {
    const valid = values.filter(Number.isFinite);
    return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : null;
  }

  // 【非核心】样本标准差用于描述日IC波动，未做年化。
  function deviation(values) {
    const valid = values.filter(Number.isFinite);
    if (valid.length < 2) {
      return null;
    }
    const average = mean(valid);
    return Math.sqrt(valid.reduce((total, value) => total + (value - average) ** 2, 0) / (valid.length - 1));
  }

  // 【非核心】真实日历七日窗口至少五个有效日，跨缺失日期不缩短窗口。
  function rollingIC(daily) {
    return daily.map(day => {
      const end = Date.parse(day.date);
      const window = daily.filter(row => Date.parse(row.date) >= end - 6 * 86400000 && Date.parse(row.date) <= end);
      const valid = window.map(row => row.ic).filter(Number.isFinite);
      return {date: day.date, ic: day.ic, rolling: valid.length >= 5 ? mean(valid) : null};
    });
  }

  // 【非核心】日IC直方图只分箱有效日，常数序列合成一个明确区间。
  function icHistogram(daily) {
    const values = daily.map(row => row.ic).filter(Number.isFinite);
    if (!values.length) {
      return [];
    }
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const bins = minimum === maximum ? 1 : 6;
    const width = (maximum - minimum) / bins;
    return Array.from({length: bins}, (_, index) => {
      const start = minimum + width * index;
      const end = minimum + width * (index + 1);
      const count = values.filter(value => value >= start && (index === bins - 1 ? value <= end : value < end)).length;
      return {
        group: ((start + end) / 2).toFixed(2), count, valid_days: count,
        tip: `日IC区间 ${start.toFixed(4)} 至 ${end.toFixed(4)}\n有效日 ${count}\n占比 ${(100 * count / values.length).toFixed(1)}%`,
      };
    });
  }

  // 【非核心】月度指标按有效日等权，月初月末的不完整月份使用实际天数。
  function monthlyIC(daily) {
    const months = [...new Set(daily.map(row => row.date.slice(0, 7)))];
    return months.map(month => {
      const selected = daily.filter(row => row.date.startsWith(month));
      const values = selected.map(row => row.ic).filter(Number.isFinite);
      return {
        label: month.slice(2), ic: mean(values), rank_ic: mean(selected.map(row => row.rank_ic)),
        positive: values.length ? values.filter(value => value > 0).length / values.length : null,
        valid_days: values.length,
      };
    });
  }

  // 【非核心】预测周期热图固定月份和周期，均值的分母只包含有效日。
  function horizonHeatmap(result) {
    const months = [...new Set(result.daily_horizons.map(row => row.date.slice(0, 7)))];
    const horizons = result.horizons.map(row => row.horizon_ns);
    return months.map(month => ({
      label: month,
      cells: horizons.map(horizon => {
        const values = result.daily_horizons.filter(row => row.date.startsWith(month) && row.horizon_ns === horizon).map(row => row.ic).filter(Number.isFinite);
        return {label: horizon / 1e9 + 's', value: mean(values), days: values.length};
      }),
    }));
  }

  // 【非核心】完整端点共同存在的日期才计算高低组差，空组不补零。
  function dailySpread(result, groupCount) {
    return result.daily_ic.map(day => {
      const groups = result.daily_groups.filter(row => row.date === day.date);
      const low = groups.find(row => row.group === 1);
      const high = groups.find(row => row.group === groupCount);
      const available = low && high && Number.isFinite(low.mean_return_bps) && Number.isFinite(high.mean_return_bps);
      return {date: day.date, spread: available ? high.mean_return_bps - low.mean_return_bps : null};
    });
  }

  // 【非核心】按日期关联分布与当前采样的IC，未观测日期不参与散点。
  function qualityRows(result) {
    const distribution = new Map(result.distribution.map(row => [row.date, row]));
    return result.daily_ic.map(day => ({
      name: day.date, date: day.date, ic: day.ic,
      coverage: day.rows ? day.paired_rows / day.rows : null,
      missing_rate: distribution.get(day.date).missing_rate,
      zero_rate: distribution.get(day.date).zero_rate,
    }));
  }

  // 【非核心】稳定性聚焦日分布、日历滚动和月度方向，原始方向保留正负号。
  function stability(result, context) {
    const {Charts, panel} = context;
    const monthly = monthlyIC(result.daily_ic);
    return panel('日 IC 分布', '横轴为分箱中心 · 悬浮查看完整区间', Charts.bars(icHistogram(result.daily_ic), 'count', {unit: '日数'})) +
      panel('七日滚动 IC', '日历7日 · 至少5个有效日 · 不年化', Charts.lines(rollingIC(result.daily_ic), [{key: 'ic', label: '日IC'}, {key: 'rolling', label: '7日均值'}], {zero: true})) +
      panel('月度预测能力', '每个月内的有效日等权', Charts.lines(monthly, [{key: 'ic', label: 'IC'}, {key: 'rank_ic', label: 'Rank IC'}], {zero: true})) +
      panel('月度正 IC 日比例', 'IC严格大于0的天数 / 有效日', Charts.lines(monthly, [{key: 'positive', label: '正IC日比例'}], {percent: true}));
  }

  // 【非核心】同一训练冻结分组下比较周期与样本切分，热图只汇总已有日IC。
  function horizons(result, context) {
    const {Charts, panel} = context;
    const periods = result.horizons.map(row => ({...row, label: row.horizon_ns / 1e9 + 's'}));
    const splitLabels = {train: '训练', validation: '验证', test: '测试'};
    const splits = ['train', 'validation', 'test'].map(split => ({...result.split_summary.find(row => row.split === split), label: splitLabels[split]}));
    return panel('月份 × 预测周期', '日均IC · 对称配色 · 悬浮查看有效日数', Charts.heatmap(horizonHeatmap(result), periods.map(row => row.label))) +
      panel('训练 / 验证 / 测试', '当前周期及采样口径 · 每段日期等权', Charts.lines(splits, [{key: 'ic_mean', label: 'IC'}, {key: 'rank_ic_mean', label: 'Rank IC'}], {zero: true})) +
      panel('多周期预测能力', '横轴为离散周期', Charts.lines(periods, [{key: 'ic_mean', label: 'IC'}, {key: 'rank_ic_mean', label: 'Rank IC'}], {zero: true})) +
      panel('周期与高低组收益差', '未扣费的未来Mid变化 · 日期等权', Charts.lines(periods, [{key: 'top_bottom_bps', label: '高低组差'}], {zero: true, unit: 'bps'}));
  }

  // 【非核心】分组差分只使用相邻两组均有效的汇总值，不改变原阈值。
  function groups(result, context) {
    const {Charts, panel, groupRows} = context;
    const rows = groupRows(result);
    const total = rows.reduce((count, row) => count + row.count, 0);
    const shares = rows.map(row => ({...row, label: row.group, share: total ? row.count / total : null}));
    const adjacent = rows.slice(1).map((row, index) => ({
      label: rows[index].group + '→' + row.group,
      delta: Number.isFinite(row.bps) && Number.isFinite(rows[index].bps) ? row.bps - rows[index].bps : null,
    }));
    return panel('组别与收益方向', '原始未来Mid变化 · bps', Charts.bars(rows, 'bps', {unit: 'bps'})) +
      panel('组别样本占比', '各组样本 / 所有分组有效样本合计', Charts.lines(shares, [{key: 'share', label: '样本占比'}], {percent: true})) +
      panel('相邻组收益变化', '各组日均收益之差 · bps · 有效日期可能不同', Charts.lines(adjacent, [{key: 'delta', label: '相邻组差'}], {zero: true, unit: 'bps'})) +
      panel('逐日高低组差', '仅两端点组均存在的日期 · 毛收益', Charts.lines(dailySpread(result, rows.length), [{key: 'spread', label: '高低组差'}], {zero: true, unit: 'bps'}));
  }

  // 【非核心】质量散点展示同日关联，横轴的缺失率来自因子日分布表。
  function quality(result, context) {
    const {Charts, panel} = context;
    const rows = qualityRows(result);
    return panel('覆盖率与预测能力', '每点一天 · 横轴共同覆盖率，纵轴IC', Charts.scatter(rows, {x: 'coverage', y: 'ic', xLabel: '共同覆盖率', yLabel: 'IC', xPercent: true})) +
      panel('缺失率与预测能力', '每点一天 · 日因子缺失率口径', Charts.scatter(rows, {x: 'missing_rate', y: 'ic', xLabel: '缺失率', yLabel: 'IC', xPercent: true})) +
      panel('零值率与预测能力', '每点一天 · 有限因子值中的零值率', Charts.scatter(rows, {x: 'zero_rate', y: 'ic', xLabel: '零值率', yLabel: 'IC', xPercent: true})) +
      panel('日度质量变化', '覆盖率、缺失率、零值率各自保留原始分母', Charts.lines(rows, [{key: 'coverage', label: '共同覆盖'}, {key: 'missing_rate', label: '缺失率'}, {key: 'zero_rate', label: '零值率'}], {percent: true}));
  }

  // 【非核心】每个视角最多四张双列图，切换视角只派生紧凑日统计。
  function render(result, view, context) {
    const views = {stability: '稳定性', horizons: '预测周期', groups: '分组结构', quality: '数据质量'};
    const navigation = Object.entries(views).map(([key, label]) => `<button type="button" data-diagnostic-view="${key}" class="${view === key ? 'active' : ''}">${label}</button>`).join('');
    const renderers = {stability, horizons, groups, quality};
    return `<div class="fb-diagnostic-nav">${navigation}</div><div class="board-panels">${renderers[view](result, context)}</div>`;
  }

  // 【非核心】因子散点按当前选择与真实覆盖比较，未按日IC相关冒充因子相关。
  function compare(results, context) {
    const rows = results.map(result => {
      const expected = result.daily_ic.reduce((count, row) => count + row.rows, 0);
      return {
        name: result.id, ic: result.summary.ic_mean,
        volatility: deviation(result.daily_ic.map(row => row.ic)),
        positive: result.summary.positive_ic_day_ratio,
        coverage: expected ? result.summary.paired_rows / expected : null,
      };
    });
    return context.panel('预测能力与日间波动', '每点一个因子 · IC日波动为样本标准差', context.Charts.scatter(rows, {x: 'volatility', y: 'ic', xLabel: 'IC日波动', yLabel: '日均IC'})) +
      context.panel('预测能力与覆盖率', '每点一个因子 · 同范围、同周期、同采样', context.Charts.scatter(rows, {x: 'coverage', y: 'ic', xLabel: '共同覆盖率', yLabel: '日均IC', xPercent: true}));
  }
  window.HFTProductionDiagnostics = {render, compare};
})();
