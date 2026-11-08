/* 服务器视图只展示已发布评估，沿用资料库外壳和浏览器笔记。 */
(function () {
  'use strict';
  const Data = window.HFTData;
  const Charts = window.HFTProductionCharts;
  const state = {diagnosticView: 'stability', scope: 'test', horizon: 1000, sampling: 'all', costs: false, fee: 5, direction: 1};
  let manifest;
  let refresh;
  let renderVersion = 0;
  const scopeLabels = {train: '训练期', validation: '验证期', test: '测试期'};

  // 【非核心】来自结果文件的名称和定义统一转义。
  function escapeText(value) {
    return String(value).replace(/[&<>"]/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[character]));
  }

  // 【非核心】统计空值保留缺失展示，不能当作零。
  function number(value, digits = 4) {
    return Number.isFinite(value) ? value.toFixed(digits) : '—';
  }

  // 【非核心】图卡复用原有紧凑双列布局。
  function panel(title, description, chart) {
    return `<section class="board-panel"><h3>${escapeText(title)}</h3><p class="panel-description">${escapeText(description)}</p>${chart}</section>`;
  }

  // 【非核心】切分、周期、采样均来自真实评测选项，成本为独立展示情景。
  function controls() {
    const labels = {train: '训练期', validation: '验证期', test: '测试期'};
    document.getElementById('controls').innerHTML = `
      <label>范围<select data-result-control="scope">${manifest.scopes.map(scope => `<option value="${scope}" ${scope === state.scope ? 'selected' : ''}>${labels[scope]}</option>`).join('')}</select></label>
      <label>预测周期<select data-result-control="horizon">${manifest.horizons_ms.map(horizon => `<option value="${horizon}" ${horizon === state.horizon ? 'selected' : ''}>${horizon / 1000}s</option>`).join('')}</select></label>
      <label>采样<select data-result-control="sampling"><option value="all" ${state.sampling === 'all' ? 'selected' : ''}>全部整秒</option><option value="nonoverlap" ${state.sampling === 'nonoverlap' ? 'selected' : ''}>非重叠标签</option></select></label>
      <label>方向<select data-result-control="direction"><option value="1" ${state.direction === 1 ? 'selected' : ''}>高多低空</option><option value="-1" ${state.direction === -1 ? 'selected' : ''}>高空低多</option></select></label>
      <label><input type="checkbox" data-result-control="costs" ${state.costs ? 'checked' : ''}>扣除交易成本</label>
      <label>单边费率 bps<input type="number" min="0" max="1000" step="0.1" data-result-control="fee" value="${state.fee}"></label>
      <span>列表固定为测试期1s · 训练期冻结分组</span>`;
  }

  // 【非核心】读取现有日统计，不触发主研究任务。
  function evaluation(id) {
    return Data.request('evaluations/' + id, {scope: state.scope, horizon: state.horizon, sampling: state.sampling});
  }

  // 【非核心】双侧等权高低组收益扣往返费率，不将重叠标签累加为净值。
  function netReturn(result) {
    const spread = result.summary.top_bottom_bps;
    return Number.isFinite(spread) ? state.direction * spread / 2 - (state.costs ? 2 * state.fee : 0) : null;
  }

  // 【非核心】组别按现有阈值展示，空组保留位置；正负零组不伪装分位组。
  function groupRows(result) {
    const count = result.boundaries.mode === 'sign' ? 3 : result.boundaries.cutoffs.length + 1;
    return Array.from({length: count}, (_, index) => {
      const group = result.groups.find(row => row.group === index + 1);
      const label = result.boundaries.mode === 'sign' ? ['负值', '零值', '正值'][index] : 'G' + (index + 1);
      return {group: label, bps: group ? group.bps : null, count: group ? group.samples : 0, valid_days: group ? group.valid_days : 0};
    });
  }

  // 【非核心】主评估统计采用日等权，样本量来自实际关联记录。
  function overview(result) {
    const summary = result.summary;
    const expected = result.daily_ic.reduce((total, day) => total + day.rows, 0);
    const values = [
      ['日均 IC', number(summary.ic_mean), `有效日 ${summary.valid_ic_days}`],
      ['日均 Rank IC', number(summary.rank_ic_mean), `正IC日 ${number(summary.positive_ic_day_ratio * 100, 1)}%`],
      ['共同覆盖率', number(100 * summary.paired_rows / expected, 2) + '%', summary.paired_rows.toLocaleString() + ' 个样本'],
      ['双侧等权' + (state.costs ? '扣费后' : '扣费前'), number(netReturn(result)) + ' bps', '0.5×方向×高低组差−往返手续费'],
    ];
    const horizons = horizonRows(result);
    const period = result.daily_ic;
    return `<div class="compact-overview"><div class="fb-context"><span class="fb-period">${period[0].date} — ${period.at(-1).date}</span><span>${scopeLabels[state.scope]}</span><span>${state.horizon / 1000}s预测</span><span>${state.sampling === 'all' ? '全部整秒' : '非重叠采样'}</span></div><div class="metric-grid">${values.map(([title, value, description], index) => {
      const metric = [summary.ic_mean, summary.rank_ic_mean, null, netReturn(result)][index];
      const tone = Number.isFinite(metric) && metric < 0 ? 'negative' : 'positive';
      return `<div class="metric-card" data-tone="${tone}"><span>${title}</span><strong>${value}</strong><small>${description}</small></div>`;
    }).join('')}</div><div data-overview-notes></div><div class="board-panels">
      ${panel('每日预测相关性', '同一批共同有效样本 · 点击图例切换系列', Charts.lines(result.daily_ic, [{key: 'ic', label: 'IC'}, {key: 'rank_ic', label: 'Rank IC'}], {zero: true}))}
      ${panel('分组未来价格变化', `${result.boundaries.mode === 'sign' ? '负 / 零 / 正' : '训练期冻结五组'} · 日期等权`, Charts.bars(groupRows(result), 'bps', {unit: 'bps'}))}
      ${panel('预测周期与信号强弱', '各周期日均相关性 · 横轴为离散周期', Charts.lines(horizons, [{key: 'ic_mean', label: 'IC'}, {key: 'rank_ic_mean', label: 'Rank IC'}], {zero: true}))}
      ${panel('每日共同覆盖率', '因子与标签共同有效 / 当前采样网格', Charts.lines(period.map(row => ({date: row.date, coverage: row.rows ? row.paired_rows / row.rows : null})), [{key: 'coverage', label: '共同覆盖率'}], {percent: true}))}
      </div></div>`;
  }

  // 【非核心】预测周期转换为秒，界面不暴露难读的纳秒整数。
  function horizonRows(result) {
    return result.horizons.map(row => ({...row, label: row.horizon_ns / 1e9 + 's'}));
  }

  // 【非核心】表格仅插入已保存列，清晰区分日分布统计和全期分布。
  function table(rows, columns) {
    const integerFields = new Set(['valid_rows', 'paired_rows', 'count', 'valid_days']);
    const signedFields = new Set(['ic_mean', 'rank_ic_mean', 'top_bottom_bps', 'net', 'bps']);
    return `<div class="table-scroll"><table><thead><tr>${columns.map(([, label]) => `<th>${escapeText(label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${columns.map(([key]) => {
      const value = row[key];
      let text = value ?? '—';
      let tone = value == null ? 'fb-value-missing' : '';
      if (typeof value === 'number') {
        text = integerFields.has(key) ? value.toLocaleString('en-US') : Charts.format(value, key.endsWith('_rate'));
        if (signedFields.has(key)) {
          tone = value < 0 ? 'fb-value-negative' : 'fb-value-positive';
        }
      }
      return `<td class="${tone} ${key === 'factor' ? 'fb-factor-cell' : ''}" title="${escapeText(text)}">${escapeText(text)}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>`;
  }

  // 【非核心】各标签对应真实现有结果，未计算的日内曲线明确说明。
  function detail(result, tab) {
    if (tab === 'overview') {
      return overview(result);
    }
    if (tab === 'diagnostics') {
      return window.HFTProductionDiagnostics.render(result, state.diagnosticView, {Charts, panel, groupRows});
    }
    if (tab === 'groups') {
      return `<p class="board-note">${result.boundaries.mode === 'sign' ? '负值、零值、正值分组' : '训练期冻结五分组'} · 日期等权未来Mid变化。手续费情景见总览的双侧等权收益。</p><div class="board-panels">` +
        panel('原始分组收益', '正负颜色表示未来价格变化方向', Charts.bars(groupRows(result), 'bps', {unit: 'bps'})) +
        panel('分组样本分布', '保留空组位置 · 样本数为所选范围合计', Charts.bars(groupRows(result), 'count', {unit: '样本'})) + '</div>' +
        table(groupRows(result), [['group', '组别'], ['bps', '日均收益 bps'], ['count', '总样本'], ['valid_days', '有效日']]);
    }
    if (tab === 'distribution') {
      return '<p class="board-note">分位数按每日有限因子值计算。悬停查看当日精确值，点击图例开关曲线。</p><div class="board-panels">' +
        panel('每日因子分位数', 'P5 / 中位数 / P95', Charts.lines(result.distribution, [{key: 'q05', label: 'P5'}, {key: 'median', label: 'P50'}, {key: 'q95', label: 'P95'}])) +
        panel('缺失值与零值', '各自沿用上游日分布统计口径', Charts.lines(result.distribution, [{key: 'missing_rate', label: '缺失率'}, {key: 'zero_rate', label: '零值率'}], {percent: true})) + '</div>' +
        table(result.distribution, [['date', '日期'], ['valid_rows', '有效样本'], ['missing_rate', '缺失率'], ['zero_rate', '零值率'], ['q01', 'P1'], ['q05', 'P5'], ['median', 'P50'], ['q95', 'P95'], ['q99', 'P99']]);
    }
    if (tab === 'stability') {
      return '<div class="board-panels">' +
        panel('每日 IC 与 Rank IC', '可在评估设置中切换非重叠采样', Charts.lines(result.daily_ic, [{key: 'ic', label: 'IC'}, {key: 'rank_ic', label: 'Rank IC'}], {zero: true})) +
        panel('多周期高低组收益差', '原始收益 · 日期等权', Charts.lines(horizonRows(result), [{key: 'top_bottom_bps', label: '高低组差'}], {zero: true, unit: 'bps'})) + '</div>' +
        table(horizonRows(result), [['label', '预测周期'], ['ic_mean', '日均IC'], ['rank_ic_mean', '日均Rank IC'], ['paired_rows', '共同样本'], ['top_bottom_bps', '高低组差 bps']]);
    }
    if (tab === 'timeseries') {
      return '<p class="board-note">这里展示已有评测保存的日级因子统计。</p><div class="board-panels">' +
        panel('每日因子均值', '原始因子单位', Charts.lines(result.distribution, [{key: 'mean', label: '均值'}], {zero: true})) +
        panel('每日因子波动', '有限值标准差 · 与均值独立坐标', Charts.lines(result.distribution, [{key: 'std', label: '标准差'}])) + '</div>';
    }
    return `<section class="board-panel"><h3>评估口径</h3><p>因子每秒输出，标签单位bps。训练结束2026-05-31，验证结束2026-08-31，测试截至2026-10-21。分界清除300s并检查标签结束时间。列表固定展示测试期1s、全部整秒指标。</p><p>分组按训练期因子分布冻结。重复分位边界使用正负零分组，空组保持缺失。总览高低组差沿用上游完整分组日口径。</p><p>成本为每边${number(state.fee, 2)}bps的手续费情景，双侧等权收益扣除${state.costs ? number(2 * state.fee, 2) : '0'}bps往返费用。此处为条件收益，不是订单成交后的策略PnL。</p><pre>${escapeText(JSON.stringify({definition_version: result.definition_version, config: result.config, definition: result.definition, boundaries: result.boundaries}, null, 2))}</pre></section>`;
  }

  // 【非核心】异步响应只更新当前详情，防止切换条目后覆盖新页。
  async function renderInto(element, id, tab) {
    const version = ++renderVersion;
    element.textContent = '读取服务器评测…';
    try {
      const result = await evaluation(id);
      if (version !== renderVersion) {
        return;
      }
      element.innerHTML = detail(result, tab);
      const notes = document.querySelector('#notes-staging .research-notes');
      const slot = element.querySelector('[data-overview-notes]');
      if (slot && notes) {
        slot.append(notes);
      } else {
        document.getElementById('notes-staging').hidden = false;
      }
    } catch (error) {
      element.textContent = error.message;
    }
  }

  // 【非核心】对比现有指标，不用日IC相关冒充因子值相关性。
  async function renderComparison(element, ids) {
    const version = ++renderVersion;
    element.textContent = '读取因子对比…';
    try {
      const results = [];
      for (const id of ids) {
        results.push(await evaluation(id));
      }
      if (version !== renderVersion) {
        return;
      }
      element.innerHTML = '<div class="board-panels">' +
        panel('因子预测能力对比', '日均IC · 相同范围、周期和采样口径', comparisonBars(results, 'ic_mean')) +
        panel('排序预测能力对比', '日均Rank IC · 完整因子ID可悬停查看', comparisonBars(results, 'rank_ic_mean')) +
        window.HFTProductionDiagnostics.compare(results, {Charts, panel}) + '</div>' +
        table(results.map(result => ({...result.summary, net: netReturn(result)})), [['factor', '因子'], ['ic_mean', 'IC'], ['rank_ic_mean', 'Rank IC'], ['top_bottom_bps', '高低组差bps'], ['net', '双侧等权情景bps'], ['paired_rows', '共同样本']]);
    } catch (error) {
      element.textContent = error.message;
    }
  }

  // 【非核心】对比条共用对称尺度，完整ID保留在悬停提示中。
  function comparisonBars(results, field) {
    const values = results.map(result => result.summary[field]).filter(Number.isFinite);
    const extent = Math.max(0.001, ...values.map(Math.abs));
    return '<div class="fb-compare-bars">' + results.map(result => {
      const value = result.summary[field];
      const width = Number.isFinite(value) ? Math.abs(value) / extent * 48 : 0;
      const left = value < 0 ? 50 - width : 50;
      const color = value < 0 ? '#bd6874' : '#14847c';
      return `<div class="fb-compare-row"><span title="${escapeText(result.id)}">${escapeText(result.id)}</span><span class="fb-compare-track"><i style="left:${left}%;width:${width}%;background:${color}"></i></span><b style="color:${color}">${number(value)}</b></div>`;
    }).join('') + '</div>';
  }

  // 【非核心】设置只作用于看板，不写研究配置。
  function initialize(config, factors, onChange) {
    document.body.classList.add('production-board');
    Charts.attachInteractions();
    manifest = config;
    refresh = onChange;
    controls();
    let previousWidth = document.getElementById('detail').clientWidth;
    let resizeTimer;
    new ResizeObserver(entries => {
      const width = Math.round(entries[0].contentRect.width);
      if (Math.abs(width - previousWidth) < 2) {
        return;
      }
      previousWidth = width;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(refresh, 180);
    }).observe(document.getElementById('detail'));
    document.getElementById('detail').addEventListener('click', event => {
      const button = event.target.closest('[data-diagnostic-view]');
      if (button) {
        state.diagnosticView = button.dataset.diagnosticView;
        refresh();
      }
    });
    document.getElementById('controls').addEventListener('change', event => {
      const key = event.target.dataset.resultControl;
      if (!key || !event.target.checkValidity()) {
        return;
      }
      if (key === 'costs') {
        state[key] = event.target.checked;
      } else if (['horizon', 'fee', 'direction'].includes(key)) {
        state[key] = Number(event.target.value);
      } else {
        state[key] = event.target.value;
      }
      controls();
      refresh();
    });
  }
  window.HFTEvaluation = {initialize, renderInto, renderComparison};
})();
