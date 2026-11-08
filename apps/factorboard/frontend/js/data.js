/* 资料库连接项目只读结果，状态与笔记单独保存到当前浏览器。 */
(function (global) {
  'use strict';
  const annotationKey = 'hft_fb_library_annotations_v1';
  const Data = {all: [], factors: [], features: [], gens: [], ready: false, mode: 'historical_files', dates: []};
  const saved = localStorage.getItem(annotationKey);
  const annotations = saved ? JSON.parse(saved) : {statuses: {}, notes: {}, displayNames: {}};

  // 【非核心】只读API错误明确抛出，不以空结果替代失败。
  async function request(path, parameters = {}) {
    const response = await fetch('/api/v1/' + path.replace(/^\//, '') + '?' + new URLSearchParams(parameters));
    if (!response.ok) {
      const error = new Error('读取失败：HTTP ' + response.status + '，' + path);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  // 【非核心】因子ID始终是主键，中文显示名只作为可空资料库字段。
  function model(factor, evaluation) {
    let category = 'trade';
    if (factor.id.startsWith('book_')) {
      category = 'book';
    } else if (factor.id.startsWith('cross_')) {
      category = 'cross';
    }
    let inputs = Object.keys(evaluation.metadata.definition.inputs);
    if (category === 'cross') {
      inputs = ['trade_imbalance_1s', 'imbalance_5'];
    }
    return {
      id: factor.id, name: factor.id, display_name: annotations.displayNames[factor.id] ?? factor.display_name ?? '',
      gen: 'Gen001', category, desc: '10/50点均值差，最少8/40点，跨日连续计算',
      baseName: factor.id, baseFeature: inputs.join(' − '), rawFields: inputs,
      operation: 'mean_diff', window: '10 / 50', post: 'raw',
      formula: `mean(${inputs.join(' − ')}, 10) − mean(${inputs.join(' − ')}, 50)`,
      meanIC: evaluation.ic, rank: evaluation.rank_ic, spread: evaluation.spread,
      icir: evaluation.icir, positive: evaluation.positive,
      valid: evaluation.coverage.paired / evaluation.coverage.expected, zero: factor.zero_rate,
      n: evaluation.dates.length, status: annotations.statuses[factor.id] ?? global.HFTResearchStates.defaultName(),
      notes: annotations.notes[factor.id] ?? '', tags: [category, 'mean_diff', 'Gen001'], kind: '已计算',
    };
  }

  // 【非核心】资料库状态独立保存，不改动因子Parquet和研究配置。
  function persistAnnotations() {
    localStorage.setItem(annotationKey, JSON.stringify(annotations));
  }

  // 【非核心】恢复原状态筛选所需的即时内存和浏览器持久化状态。
  async function putStatus(id, status) {
    annotations.statuses[id] = status;
    Data.find(id).status = status;
    persistAnnotations();
  }

  // 【非核心】笔记只写本机浏览器，保存失败直接向界面报告。
  async function putNote(id, note) {
    annotations.notes[id] = note;
    Data.find(id).notes = note;
    persistAnnotations();
  }

  // 【非核心】中文短名可以留空，保存名称不改变任何因子标识和路径。
  async function putDisplayName(id, displayName) {
    annotations.displayNames[id] = displayName.trim();
    Data.find(id).display_name = displayName.trim();
    persistAnnotations();
  }

  // 【非核心】使用已有结果发布记录填充原任务页，不伪造后台任务执行。
  async function listJobs() {
    return {items: [{id: Data.config.run_id, factor_id: '看板诊断快照', status: 'completed',
      progress: 1, created_at: Data.config.created_at_utc}]};
  }

  // 【非核心】登记保持入口，当前资料库只收录已经发布的项目因子。
  async function addFactor() {
    throw new Error('先在研究项目生成因子，再运行看板prepare；当前不创建占位因子');
  }

  // 【非核心】全期1s指标供原资料库排序，详情使用独立的可切换范围。
  async function reload() {
    const [manifest, registry] = await Promise.all([request('config'), request('factors')]);
    const evaluations = await Promise.all(registry.items.map(factor => request('evaluations/' + factor.id)));
    Data.all.splice(0, Data.all.length, ...registry.items.map((factor, index) => model(factor, evaluations[index])));
    Data.factors = Data.all;
    Data.features = [];
    Data.gens = [{id: 'Gen001', title: '买卖压力短长窗口差'}];
    Data.config = manifest;
    Data.dates = manifest.dates;
    document.querySelector('.demo-mark').textContent = manifest.dates.length + ' 天研究数据';
  }

  Object.assign(Data, {
    request, reload, putStatus, putNote, putDisplayName, listJobs, addFactor,
    find: id => Data.all.find(factor => factor.id === id),
    categoryLabel: category => ({book: '盘口', trade: '主动成交', cross: '交互', ofi: '事件OFI', returns: '价格路径'})[category] ?? category,
    getStatusMap: () => annotations.statuses,
    getNotesMap: () => annotations.notes,
  });
  global.HFTData = Data;
  reload().then(() => {
    Data.ready = true;
    window.dispatchEvent(new CustomEvent('hft-data-ready'));
  }).catch(error => {
    document.querySelector('.demo-mark').textContent = error.message;
  });
})(window);
