/* 独立研究看板：读取项目结果，十二组和成本情景属于看板扩展。 */
(function () {
  'use strict';
  const Data = window.HFTData;
  const Charts = window.HFTCharts;
  const ResearchCharts = window.HFTResearchCharts;
  const state = {
    manifest: null, factors: [], selected: '', compared: new Set(), tab: 'overview',
    scope: 'full', mode: 'daily', horizon: 1000, costs: false,
    fee: 5, slippage: 0, spreadCost: 0, direction: 1,
    seriesDate: '2026-07-01', seriesStart: '12:00:00', seriesDuration: 100,
    showPast: true, correlationMethod: 'pearson', query: '',
  };
  const cache = new Map();
  let renderVersion = 0;
  let content = null;
  let onChange = null;
  // 【非核心】界面文本转义，数值缺失显示横线而不是零。
  function escapeText(value) {
    return String(value).replace(/[&<>"]/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
    })[character]);
  }

  // 【非核心】统一数值显示精度，零值保留为有效结果。
  function number(value, digits = 3) {
    return Number.isFinite(value) ? value.toFixed(digits) : '—';
  }

  // 【非核心】图卡明确写出样本范围、单位和汇总口径。
  function panel(title, description, chart, wide = false) {
    return `<section class="board-panel ${wide ? 'span-two' : ''}"><h3>${title}</h3><p class="panel-description">${description}</p>${chart}</section>`;
  }

  // 【非核心】成本是固定往返bps情景，开关不会改变IC或原始Mid标签。
  function costBps() {
    return state.costs ? 2 * state.fee + 2 * state.slippage + state.spreadCost : 0;
  }

  // 【非核心】低半组做空、高半组做多，方向由用户显式指定，不用全期收益自动选方向。
  function directionalGroups(groups) {
    return groups.map((group, index) => {
      const side = (index <= 5 ? -1 : 1) * state.direction;
      const result = Number.isFinite(group.bps) ? side * group.bps - costBps() : null;
      return {...group, bps: result, side: side > 0 ? '多' : '空'};
    });
  }

  // 【非核心】设置项只控制看板范围和成本假设，不修改研究配置。
  function renderControls() {
    const scopeOptions = Object.entries(state.manifest.scopes).map(([key, scope]) =>
      `<option value="${key}" ${state.scope === key ? 'selected' : ''}>${scope[2]}</option>`).join('');
    document.getElementById('controls').innerHTML = `
      <label>范围<select id="scope">${scopeOptions}</select></label>
      <label>分组<select id="group-mode"><option value="daily" ${state.mode === 'daily' ? 'selected' : ''}>逐日回顾分位</option><option value="fixed" ${state.mode === 'fixed' ? 'selected' : ''}>开发期固定阈值</option></select></label>
      <label>预测周期<select id="horizon">${[100, 500, 1000, 5000].map(period => `<option value="${period}" ${state.horizon === period ? 'selected' : ''}>${period}ms</option>`).join('')}</select></label>
      <label>方向<select id="direction"><option value="1" ${state.direction === 1 ? 'selected' : ''}>高多低空</option><option value="-1" ${state.direction === -1 ? 'selected' : ''}>高空低多</option></select></label>
      <label class="cost-switch"><input id="cost-toggle" type="checkbox" ${state.costs ? 'checked' : ''}>扣除交易成本</label>
      <label>单边费率 bps<input id="fee" type="number" min="0" max="1000" step="0.1" value="${state.fee}"></label>
      <label>单边滑点 bps<input id="slippage" type="number" min="0" max="1000" step="0.1" value="${state.slippage}"></label>
      <label>往返价差 bps<input id="spread-cost" type="number" min="0" max="1000" step="0.1" value="${state.spreadCost}"></label>
      <span class="cost-total">扣减 ${number(costBps(), 2)} bps / 往返</span>`;
  }

  // 【非核心】快照版本、范围、分组与周期共同作为视图缓存键。
  async function getEvaluation(name) {
    const key = [state.manifest.run_id, name, state.scope, state.mode, state.horizon].join('|');
    if (!cache.has(key)) {
      const response = await Data.request('evaluations/' + name, {
        scope: state.scope, mode: state.mode, horizon: state.horizon,
      });
      cache.set(key, response);
    }
    return cache.get(key);
  }

  // 【非核心】十二组样本表中的平均数量按所选日数计算，空组也计入分母。
  function groupTable(groups) {
    return `<div class="table-scroll"><table class="sample-table"><thead><tr><th>组别</th>${groups.map(group => `<th>${group.group}</th>`).join('')}</tr></thead><tbody>
      <tr><th>平均每日样本</th>${groups.map(group => `<td>${number(group.average_count, 0)}</td>`).join('')}</tr>
      <tr><th>有效日</th>${groups.map(group => `<td>${group.valid_days}</td>`).join('')}</tr>
      <tr><th>原始收益 bps</th>${groups.map(group => `<td>${number(group.bps, 4)}</td>`).join('')}</tr>
      <tr><th>方向化${state.costs ? '扣费后' : '扣费前'}</th>${directionalGroups(groups).map(group => `<td>${group.side} ${number(group.bps, 4)}</td>`).join('')}</tr>
    </tbody></table></div>`;
  }

  // 【非核心】等权双侧每一单位总名义资金收益，区别于未归一化G10−G1价差。
  function twoSidedReturn(evaluation) {
    return Number.isFinite(evaluation.spread) ? state.direction * evaluation.spread / 2 - costBps() : null;
  }

  // 【非核心】显示原始IC和共同覆盖，避免把因子覆盖误称关联后覆盖。
  function overview(evaluation) {
    const coverage = evaluation.coverage;
    const interval = evaluation.ic_ci;
    const cards = [
      ['日均 IC', number(evaluation.ic, 4), interval ? `95%区间 [${number(interval[0], 4)}, ${number(interval[1], 4)}]` : '区间不可计算'],
      ['日均 Rank IC', number(evaluation.rank_ic, 4), `ICIR ${number(evaluation.icir, 2)} · 未年化`],
      ['共同覆盖率', number(100 * coverage.paired / coverage.expected, 2) + '%', `${coverage.paired.toLocaleString()} / ${coverage.expected.toLocaleString()} 网格`],
      [`双侧等权${state.costs ? '扣费后' : '扣费前'}`, number(twoSidedReturn(evaluation), 4) + ' bps', '0.5×方向×(G10−G1)−往返成本'],
    ];
    return `<div class="compact-overview"><div class="metric-grid">${cards.map(([title, value, description]) => `<div class="metric-card"><span>${title}</span><strong>${value}</strong><small>${description}</small></div>`).join('')}</div>
      <div data-overview-notes></div><div class="board-panels">
      ${panel('逐日IC', 'Pearson · 每日共同有效样本；7日线仅展示平滑', Charts.icTimeline(evaluation.daily_ic))}
      ${panel('十二组原始Mid变化', `${state.horizon}ms · bps · 日期等权；两尾重叠`, ResearchCharts.bars(evaluation.groups, 'bps', '十二组原始Mid变化'))}
      ${panel('十二组方向化收益', `${state.costs ? '扣费后' : '扣费前'} · bps · 情景估算`, ResearchCharts.bars(directionalGroups(evaluation.groups), 'bps', '方向化成本情景'))}
      ${panel('多周期预测关系', '日均 Pearson IC · 不随成本开关变化', Charts.lineChart(evaluation.horizons.map(row => ({label: row.h + 'ms', ic: row.ic, rank: row.rank_ic})), ['ic', 'rank'], {zero: true, decimal: 3}))}
      </div><p class="overview-scope">${evaluation.dates[0]}—${evaluation.dates.at(-1)} · ${evaluation.scope_title} · ${state.horizon}ms · 条件收益bps；完整分组表见“12组收益”</p></div>`;
  }

  // 【非核心】原始组收益、成本情景和组内IC分开展示，避免扣成本时方向混淆。
  function groupsPage(evaluation) {
    const rows = evaluation.daily_group_ic.map(day => {
      const result = {day: day.day};
      day.groups.forEach((group, index) => {result['g' + index] = group.ic;});
      return result;
    });
    return `<div class="board-panels">
      ${panel('十二组未来Mid变化', '左尾1% + 十分位 + 右尾1% · 原始收益不扣费', ResearchCharts.bars(evaluation.groups, 'bps', '十二组原始收益'))}
      ${panel(`十二组方向化${state.costs ? '扣费后' : '扣费前'}收益`, `低半组${state.direction === 1 ? '空' : '多'}，高半组${state.direction === 1 ? '多' : '空'} · 扣减${number(costBps(), 2)}bps`, ResearchCharts.bars(directionalGroups(evaluation.groups), 'bps', '方向化净收益情景'))}
      ${panel('各分位组内IC', '因子取值被分组限制后的Pearson · 与全体IC分开解读', ResearchCharts.bars(evaluation.groups, 'ic', '组内IC'))}
      ${panel('各分位组逐日IC', '同色对应同组 · 缺失日断线', Charts.lineChart(rows, evaluation.groups.map((group, index) => 'g' + index), {palette: ResearchCharts.colors, zero: true, decimal: 3}))}
      </div><div class="group-legend">${evaluation.groups.map((group, index) => `<span><i style="background:${ResearchCharts.colors[index]}"></i>${group.group}</span>`).join('')}</div>
      ${groupTable(evaluation.groups)}${methodFootnote(evaluation)}`;
  }

  // 【非核心】损耗使用互斥类别，四项之和回到理论网格总数。
  function distributionPage(evaluation) {
    const coverage = evaluation.coverage;
    const rows = [['文件缺网格', coverage.missing_grid], ['因子无效', coverage.factor_invalid],
      ['因子有效但标签无效', coverage.label_invalid_only], ['共同有效', coverage.paired]];
    return `<div class="coverage-grid">${rows.map(([title, count]) => `<div><strong>${count.toLocaleString()}</strong><span>${title} · ${number(count / coverage.expected * 100, 3)}%</span></div>`).join('')}</div>
      <p class="board-note">因子覆盖 ${number(100 * coverage.factor_valid / coverage.expected, 2)}% · 标签覆盖 ${number(100 * coverage.label_valid / coverage.expected, 2)}% · 有效标签零变化 ${number(100 * coverage.zero_labels / coverage.label_valid, 2)}%。标签无效原因当前没有逐条来源字段，因此不进一步猜分原因。</p>
      <div class="board-panels">
      ${panel('因子分布', '选定范围有限值 · 全期统一36桶边界', Charts.histogram(evaluation.hist))}
      ${panel('逐日分位带', '线性插值 P10/P25/P50/P75/P90', Charts.percentiles(evaluation.pct))}
      ${panel('逐日密度', '统一边界 · 每日归一化', Charts.heatmap(evaluation.densities, evaluation.hist.map(row => row.x)), true)}
      </div>`;
  }

  // 【非核心】通过真实UTC时间查询局部数据，不跨缺失点插值。
  function seriesControls(evaluation) {
    if (!evaluation.dates.includes(state.seriesDate)) {
      state.seriesDate = evaluation.dates[0];
    }
    return `<form id="series-form" class="series-form">
      <label>UTC日期<select id="series-date">${evaluation.dates.map(date => `<option ${date === state.seriesDate ? 'selected' : ''}>${date}</option>`).join('')}</select></label>
      <label>UTC开始时间<input id="series-start" type="time" step="1" value="${state.seriesStart}" required></label>
      <label>时长<select id="series-duration">${[5, 10, 24, 100, 300, 600].map(duration => `<option value="${duration}" ${duration === state.seriesDuration ? 'selected' : ''}>${duration}s</option>`).join('')}</select></label>
      <label><input id="series-show-past" type="checkbox" ${state.showPast ? 'checked' : ''}>显示过去1s收益</label>
      <button type="submit">读取时段</button></form><div id="series-chart" class="board-panel">读取时序…</div>`;
  }

  // 【非核心】异步时序只更新当前页面，快速切换不覆盖新页面。
  async function loadSeries(version) {
    const seconds = state.seriesStart.split(':').map(Number);
    const start = seconds[0] * 3600 + seconds[1] * 60 + (seconds[2] || 0);
    const payload = await Data.request('series/' + state.selected, {
      date: state.seriesDate, start, duration: state.seriesDuration, horizon: state.horizon,
    });
    if (version !== renderVersion) {
      return;
    }
    const ages = payload.series.map(row => row.age_ms).filter(Number.isFinite).sort((left, right) => left - right);
    const age = ages.length ? ages[Math.floor((ages.length - 1) * .95)] : null;
    document.getElementById('series-chart').innerHTML = `<h3>因子 / 未来${state.horizon}ms / 过去1s</h3><p class="panel-description">${state.seriesDate} ${state.seriesStart} UTC · ${payload.series.length}个网格 · 盘口年龄P95 ${number(age, 1)}ms · 缺失点断线</p>${ResearchCharts.series(payload.series, state.showPast)}`;
  }

  // 【非核心】置信区间描述日指标的不确定性，不把行数当独立样本数。
  function stabilityPage(evaluation) {
    const spreadRows = evaluation.daily_spreads.map(row => ({day: row.day, spread: row.spread}));
    return `<p class="board-note">IC区间使用固定种子的2000次连续三日移动块bootstrap。日内重叠标签保留，不用独立行数做显著性推断。当前切分均为回溯演练。</p>
      <div class="board-panels">
      ${panel('日IC和日Rank IC', '原始方向 · 每日等权', Charts.lineChart(evaluation.daily_ic, ['ic', 'rank_ic'], {zero: true, decimal: 3}))}
      ${panel('逐日G10−G1毛价差', `bps · 当前分组模式；95%均值区间 ${evaluation.spread_ci ? evaluation.spread_ci.map(value => number(value, 4)).join(' 至 ') : '不可计算'}`, Charts.lineChart(spreadRows, ['spread'], {zero: true, decimal: 3}))}
      ${panel('等待之后还剩多少信息', '固定h=1s，目标 m(t+δ+1s)/m(t+δ)−1；δ为100ms整数倍', Charts.lineChart(evaluation.delays, ['ic'], {zero: true, decimal: 3}), true)}
      </div><div class="table-scroll"><table><thead><tr><th>等待δ</th><th>日均IC</th><th>共同样本</th></tr></thead><tbody>${evaluation.delays.map(row => `<tr><td>${row.label}</td><td>${number(row.ic, 4)}</td><td>${row.count.toLocaleString()}</td></tr>`).join('')}</tbody></table></div>`;
  }

  // 【非核心】成本假设和回溯分组范围随图表一起展示。
  function methodFootnote(evaluation) {
    return `<p class="board-note">${evaluation.dates[0]}—${evaluation.dates.at(-1)} UTC · ${evaluation.scope_title} · ${state.mode === 'daily' ? '当日因子分位阈值，回顾分析' : '07-01—07-18因子分布冻结阈值'}。G0.1≤P1，G9.9>P99，与端点组重叠。成本只作用于方向化收益，不改变IC和原始Mid变化。重叠标签条件收益不累加为净值。</p>`;
  }

  // 【非核心】比较页使用相同范围，明确全期Pearson与日均Spearman的区别。
  async function comparePage(version) {
    const factors = state.factors.filter(factor => state.compared.has(factor.id));
    if (factors.length < 2) {
      content.innerHTML = '<div class="board-empty">请在左侧勾选至少两个因子。</div>';
      return;
    }
    const [matrix, ...evaluations] = await Promise.all([
      Data.request('compare/correlation', {ids: factors.map(factor => factor.id).join(','), scope: state.scope, method: state.correlationMethod}),
      ...factors.map(factor => getEvaluation(factor.id)),
    ]);
    if (version !== renderVersion) {
      return;
    }
    const matrixFactors = factors.map(factor => ({id: factor.id}));
    const chart = Charts.matrix(matrixFactors, (left, right) => matrix.values[matrixFactors.indexOf(left)][matrixFactors.indexOf(right)]);
    content.innerHTML = `<label class="inline-control">相关性<select id="correlation-method"><option value="pearson" ${state.correlationMethod === 'pearson' ? 'selected' : ''}>全范围Pearson</option><option value="spearman" ${state.correlationMethod === 'spearman' ? 'selected' : ''}>逐日Spearman等权均值</option></select></label>
      ${panel('因子相关性', `${matrix.dates[0]}—${matrix.dates.at(-1)} · ${matrix.mask} · ${matrix.count.toLocaleString()}行`, chart)}
      <div class="table-scroll"><table><thead><tr><th>因子</th><th>IC</th><th>Rank IC</th><th>共同覆盖</th><th>G10−G1毛价差</th><th>双侧等权${state.costs ? '扣费后' : '扣费前'}bps</th></tr></thead><tbody>${evaluations.map(evaluation => `<tr><td>${evaluation.id}</td><td>${number(evaluation.ic, 4)}</td><td>${number(evaluation.rank_ic, 4)}</td><td>${number(100 * evaluation.coverage.paired / evaluation.coverage.expected, 2)}%</td><td>${number(evaluation.spread, 4)}</td><td>${number(twoSidedReturn(evaluation), 4)}</td></tr>`).join('')}</tbody></table></div>
      <h3 class="section-title">十二组原始收益横向检查</h3><div class="board-panels">${evaluations.map(evaluation => panel(evaluation.id, `${state.horizon}ms · 日期等权 · bps`, ResearchCharts.bars(evaluation.groups, 'bps', evaluation.id))).join('')}</div>`;
  }

  // 【非核心】已实现与后续研究分开列出，不用占位数值伪装评估结果。
  function methodology(evaluation) {
    const metadata = evaluation.metadata;
    return `<section class="board-panel method-panel"><h3>当前计算身份</h3>
      <dl><dt>因子版本</dt><dd>${state.manifest.factor_version}</dd><dt>Features版本</dt><dd>${state.manifest.feature_version}</dd><dt>标签版本</dt><dd>${state.manifest.label_version}</dd><dt>窗口</dt><dd>10/50点，min_samples=8/40；跨日连续，null不重置窗口</dd><dt>结果</dt><dd>${state.manifest.run_id}</dd><dt>代码</dt><dd>${state.manifest.code_commit.slice(0, 12)} · ${state.manifest.code_state}</dd></dl>
      <h3>成本情景</h3><p>每组代表独立单位名义仓位的假设往返交易。低半组做空、高半组做多，可反向。方向化收益=方向×原始组收益−(2×单边手续费+2×单边滑点+往返价差)。双侧等权收益=0.5×方向×(G10−G1)−往返成本。方向由用户选择，不按全期结果自动拟合。</p>
      <p>默认单边手续费5bps来自现有基线taker参数，仅为假设；滑点和往返价差默认0，可自行设置。固定bps成本不模拟真实买卖报价、Maker排队或成交概率，不能称作策略费用后PnL。</p>
      <h3>十二组与时间切分</h3><p>G1—G10为十分位，G0.1为最低1%，G9.9为最高1%。重复值不强拆，空组显示缺失。阈值使用因子自身的有限值，不以标签是否有效选样本。固定模式只用07-01—07-18的因子值确定阈值。</p>
      <p>30天都已用于探索。开发/验证/测试切分是回溯演练，不是未触碰的样本外验证。标签维持主项目原有单日边界和质量有效规则。</p>
      <h3>后续仍需独立研究实现</h3><p>真实滚动样本外、市场状态诊断、成对有效相关矩阵、样本外增量模型和真实订单执行结果，尚未作为本次看板结果提供。现有页面只展示已经计算的统计。</p>
      <details><summary>完整因子与数据定义</summary><pre>${escapeText(JSON.stringify(metadata, null, 2))}</pre></details></section>`;
  }

  // 【非核心】评估只挂载在原资料库详情内部，不拥有资料库组织状态。
  async function renderInto(element, factorId, tab) {
    const version = ++renderVersion;
    content = element;
    state.selected = factorId;
    state.tab = tab;
    content.innerHTML = '<div class="board-empty">读取研究结果…</div>';
    try {
      if (tab === 'compare') {
        await comparePage(version);
        return;
      }
      const evaluation = await getEvaluation(factorId);
      if (version !== renderVersion) {
        return;
      }
      const pages = {overview, groups: groupsPage, distribution: distributionPage,
        timeseries: seriesControls, stability: stabilityPage, method: methodology};
      content.innerHTML = pages[tab](evaluation);
      const notes = document.querySelector('#notes-staging .research-notes');
      const notesSlot = content.querySelector('[data-overview-notes]');
      if (notesSlot) {
        notesSlot.append(notes);
      } else {
        document.getElementById('notes-staging').hidden = false;
      }
      if (tab === 'timeseries') {
        await loadSeries(version);
      }
    } catch (error) {
      if (version === renderVersion) {
        content.innerHTML = `<div class="board-error">${escapeText(error.message)}</div>`;
      }
    }
  }

  // 【非核心】对比集合直接来自原资料库，不另建第二套勾选状态。
  function renderComparison(element, factorIds) {
    state.compared = new Set(factorIds);
    return renderInto(element, factorIds[0], 'compare');
  }

  // 【非核心】事件入口统一校验可编辑成本，拒绝负值或非数字输入。
  function changeControl(event) {
    const target = event.target;
    const costFields = {fee: 'fee', slippage: 'slippage', 'spread-cost': 'spreadCost'};
    if (target.id in costFields) {
      if (!target.checkValidity() || !Number.isFinite(target.valueAsNumber)) {
        target.reportValidity();
        return;
      }
      state[costFields[target.id]] = target.valueAsNumber;
    } else if (target.id === 'scope') {
      state.scope = target.value;
    } else if (target.id === 'group-mode') {
      state.mode = target.value;
    } else if (target.id === 'horizon') {
      state.horizon = Number(target.value);
    } else if (target.id === 'direction') {
      state.direction = Number(target.value);
    } else if (target.id === 'cost-toggle') {
      state.costs = target.checked;
    } else {
      return;
    }
    renderControls();
    onChange();
  }

  // 【非核心】只监听增强评估区域的表单，不干预收藏、分类或原详情事件。
  function attachEvents() {
    document.getElementById('controls').addEventListener('change', changeControl);
    document.addEventListener('change', event => {
      if (event.target.id === 'correlation-method') {
        state.correlationMethod = event.target.value;
        onChange();
      }
    });
    document.addEventListener('submit', event => {
      if (event.target.id !== 'series-form') {
        return;
      }
      event.preventDefault();
      state.seriesDate = document.getElementById('series-date').value;
      state.seriesStart = document.getElementById('series-start').value;
      state.seriesDuration = Number(document.getElementById('series-duration').value);
      state.showPast = document.getElementById('series-show-past').checked;
      loadSeries(renderVersion).catch(error => {
        document.getElementById('series-chart').textContent = error.message;
      });
    });
  }

  // 【非核心】由原资料库初始化，增强区不替换外层导航和三栏布局。
  function initialize(manifest, factors, refresh) {
    state.manifest = manifest;
    state.factors = factors;
    state.fee = manifest.cost_defaults.fee_per_side_bps;
    state.seriesDate = manifest.dates[0];
    onChange = refresh;
    renderControls();
    attachEvents();
  }

  window.HFTEvaluation = {initialize, renderInto, renderComparison};
})();
