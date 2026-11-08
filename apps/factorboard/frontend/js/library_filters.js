/* 指标条件仅筛选现有资料库，不重新计算因子。 */
(function (global) {
  'use strict';
  const metrics = [
    ['meanIC', 'IC 1s', 1], ['rank', 'Rank IC', 1], ['spread', '高低组毛价差 bps', 1],
    ['icir', 'ICIR', 1], ['positive', '正IC日 %', 100], ['valid', '共同有效率 %', 100], ['zero', '零值率 %', 100],
  ];
  const operations = [['ge', '≥'], ['le', '≤'], ['gt', '>'], ['lt', '<'], ['abs_ge', '|值| ≥'], ['abs_le', '|值| ≤']];
  const defaultKey = 'hft_fb_default_metric_filters_v1';
  const savedDefault = localStorage.getItem(defaultKey);
  let conditions = savedDefault ? JSON.parse(savedDefault) : [];
  const presetKey = 'hft_fb_metric_filter_presets_v1';
  const savedPresets = localStorage.getItem(presetKey);
  const presets = savedPresets ? JSON.parse(savedPresets) : [];
  let activePresetId = '';
  let editingPresetId = '';
  let dialog;
  let refresh;

  // 【非核心】活动指标缺失时不通过筛选，百分比阈值使用界面百分数单位。
  function matches(factor) {
    return conditions.every(condition => {
      const metric = metrics.find(item => item[0] === condition.metric);
      const value = factor[condition.metric];
      if (!Number.isFinite(value)) {
        return false;
      }
      const scaled = value * metric[2];
      const threshold = condition.threshold;
      switch (condition.operation) {
        case 'ge': return scaled >= threshold;
        case 'le': return scaled <= threshold;
        case 'gt': return scaled > threshold;
        case 'lt': return scaled < threshold;
        case 'abs_ge': return Math.abs(scaled) >= threshold;
        case 'abs_le': return Math.abs(scaled) <= threshold;
      }
    });
  }

  // 【非核心】条件编辑行使用原生数值约束，不把空值或无效输入当作零。
  function addCondition(condition = {metric: 'meanIC', operation: 'ge', threshold: 0}) {
    const row = document.createElement('div');
    row.className = 'metric-filter-row';
    row.innerHTML = '<select aria-label="筛选指标"></select><select aria-label="比较条件"></select><input type="number" step="any" required aria-label="阈值"><button type="button" aria-label="移除条件">×</button>';
    const selects = row.querySelectorAll('select');
    for (const [index, options] of [[0, metrics], [1, operations]]) {
      options.forEach(([value, label]) => selects[index].add(new Option(label, value)));
    }
    selects[0].value = condition.metric;
    selects[1].value = condition.operation;
    row.querySelector('input').value = condition.threshold;
    row.querySelector('button').addEventListener('click', () => row.remove());
    dialog.querySelector('.metric-filter-rows').append(row);
  }

  // 【非核心】所有条件按AND组合，取消编辑不影响已应用条件。
  function open() {
    dialog.querySelector('.metric-filter-rows').replaceChildren();
    conditions.forEach(addCondition);
    if (!conditions.length) {
      addCondition();
    }
    renderPresets(activePresetId);
    dialog.showModal();
  }

  // 【非核心】只读取通过原生表单验证的编辑条件，供应用和保存默认共用。
  function editedConditions() {
    return [...dialog.querySelectorAll('.metric-filter-row')].map(row => {
      const selects = row.querySelectorAll('select');
      return {metric: selects[0].value, operation: selects[1].value,
        threshold: row.querySelector('input').valueAsNumber};
    });
  }

  // 【非核心】默认条件独立保存在当前浏览器，清除本次筛选不删除默认。
  function restoreDefault() {
    const saved = localStorage.getItem(defaultKey);
    conditions = saved ? JSON.parse(saved) : [];
    activePresetId = '';
    dialog.close();
    refresh();
  }

  // 【非核心】复制条件防止编辑草稿直接改动已保存方案。
  function copyConditions(source) {
    return source.map(condition => ({...condition}));
  }

  // 【非核心】列表快速切换只应用已保存方案，不重算原始指标。
  function applyPreset(id) {
    const preset = presets.find(item => item.id === id);
    conditions = copyConditions(preset.conditions);
    activePresetId = id;
    dialog.close();
    refresh();
  }

  // 【非核心】把每套方案的名称、条件和应用按钮直接列出，使用原生DOM安全显示名称。
  function renderPresets(id = '') {
    editingPresetId = id;
    const list = dialog.querySelector('[data-preset-list]');
    list.replaceChildren();
    for (const preset of presets) {
      const row = document.createElement('div');
      row.className = 'saved-filter-row';
      row.dataset.presetId = preset.id;
      const description = document.createElement('div');
      const title = document.createElement('strong');
      title.textContent = preset.name;
      description.append(title);
      if (preset.id === activePresetId) {
        const badge = document.createElement('span');
        badge.className = 'active-filter-badge';
        badge.textContent = '正在使用';
        description.append(badge);
      }
      const details = document.createElement('small');
      details.textContent = preset.conditions.map(condition => {
        const metric = metrics.find(item => item[0] === condition.metric);
        const operation = operations.find(item => item[0] === condition.operation);
        return metric[1] + ' ' + operation[1] + ' ' + condition.threshold;
      }).join('，且 ');
      if (!preset.conditions.length) {
        details.textContent = '全部因子';
      }
      description.append(details);
      row.append(description);
      const actions = document.createElement('div');
      actions.className = 'saved-filter-actions';
      for (const [label, action] of [
        ['应用', () => applyPreset(preset.id)],
        ['编辑', () => loadPresetEditor(preset.id)],
        ['删除', () => deletePreset(preset.id)],
      ]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = label;
        button.setAttribute('aria-label', label + '方案 ' + preset.name);
        button.addEventListener('click', action);
        actions.append(button);
      }
      row.append(actions);
      list.append(row);
    }
    if (!presets.length) {
      const empty = document.createElement('p');
      empty.className = 'saved-filter-empty';
      empty.textContent = '还没有方案。编辑下方条件，填写方案名称后保存。';
      list.append(empty);
    }
    const preset = presets.find(item => item.id === id);
    const nameInput = dialog.querySelector('[name=preset-name]');
    nameInput.value = preset ? preset.name : '';
    nameInput.setCustomValidity('');
    dialog.querySelector('[data-editor-title]').textContent = preset ? '编辑方案 · ' + preset.name : '新建方案或临时筛选';
    dialog.querySelector('[data-save-preset]').textContent = preset ? '保存修改并应用' : '保存为新方案并应用';
    const active = presets.find(item => item.id === activePresetId);
    dialog.querySelector('[data-active-filter]').textContent = active ? '当前正在使用方案「' + active.name + '」' : '当前使用临时／默认条件，共 ' + conditions.length + ' 条';
  }

  // 【非核心】编辑已有方案只载入草稿，点击应用前不改变列表筛选。
  function loadPresetEditor(id) {
    renderPresets(id);
    dialog.querySelector('.metric-filter-rows').replaceChildren();
    const preset = presets.find(item => item.id === id);
    const source = preset ? preset.conditions : conditions;
    source.forEach(addCondition);
    if (!source.length) {
      addCondition();
    }
    dialog.querySelector('[data-filter-editor]').scrollIntoView({block: 'nearest'});
  }

  // 【非核心】同名检查集中在保存边界，选择已有方案时可更新条件或改名。
  function savePreset() {
    const form = dialog.querySelector('form');
    if (!form.reportValidity()) {
      return;
    }
    const input = form.elements.namedItem('preset-name');
    const name = input.value.trim();
    const id = editingPresetId;
    if (!name || presets.some(preset => preset.id !== id && preset.name === name)) {
      input.setCustomValidity(name ? '方案名称已存在' : '请输入方案名称');
      input.reportValidity();
      return;
    }
    const edited = editedConditions();
    let preset;
    if (id) {
      preset = presets.find(item => item.id === id);
      preset.name = name;
      preset.conditions = edited;
    } else {
      preset = {id: crypto.randomUUID(), name, conditions: edited};
      presets.push(preset);
    }
    localStorage.setItem(presetKey, JSON.stringify(presets));
    applyPreset(preset.id);
  }

  // 【非核心】删除方案不影响另存的默认条件；若删除当前方案则清除本次筛选。
  function deletePreset(id) {
    presets.splice(presets.findIndex(preset => preset.id === id), 1);
    localStorage.setItem(presetKey, JSON.stringify(presets));
    if (activePresetId === id) {
      activePresetId = '';
      conditions = [];
    }
    dialog.close();
    refresh();
  }

  // 【非核心】筛选弹窗集中处理边界校验，只向列表提供确定的条件集合。
  function initialize(onApply) {
    refresh = onApply;
    dialog = document.createElement('dialog');
    dialog.className = 'metric-filter-dialog';
    dialog.setAttribute('aria-label', '指标筛选');
    dialog.innerHTML = '<form><h3>指标筛选</h3><p>条件同时满足才显示；使用列表1s指标，缺失值不通过筛选。</p><section class="saved-filter-presets"><div class="saved-filter-heading"><h4>已保存方案</h4><button type="button" data-new-preset>＋ 新建方案</button></div><p data-active-filter class="active-filter-description"></p><div data-preset-list></div></section><fieldset class="filter-condition-editor" data-filter-editor><legend data-editor-title>新建方案或临时筛选</legend><div class="metric-filter-rows"></div><button type="button" data-add-condition>＋ 添加条件</button><div class="save-filter-row"><label>方案名称<input name="preset-name" maxlength="60" placeholder="例如：强IC且覆盖率高" aria-label="筛选方案名称"></label><button type="button" data-save-preset>保存为新方案并应用</button></div></fieldset><div class="filter-dialog-footer"><button type="button" data-save-default>保存为默认</button><button type="button" data-restore-default>恢复默认</button><button type="button" data-clear-filter>清除筛选</button><button type="button" data-cancel-filter>取消</button><button type="submit">应用下方条件</button></div></form>';
    document.body.append(dialog);
    dialog.querySelector('[data-new-preset]').addEventListener('click', () => loadPresetEditor(''));
    dialog.querySelector('[name=preset-name]').addEventListener('input', event => event.target.setCustomValidity(''));
    dialog.querySelector('[data-save-preset]').addEventListener('click', savePreset);
    dialog.querySelector('[data-add-condition]').addEventListener('click', () => addCondition());
    dialog.querySelector('[data-cancel-filter]').addEventListener('click', () => dialog.close());
    dialog.querySelector('[data-save-default]').addEventListener('click', () => {
      if (!dialog.querySelector('form').reportValidity()) {
        return;
      }
      conditions = editedConditions();
      activePresetId = '';
      localStorage.setItem(defaultKey, JSON.stringify(conditions));
      dialog.close();
      refresh();
    });
    dialog.querySelector('[data-restore-default]').addEventListener('click', restoreDefault);
    dialog.querySelector('[data-clear-filter]').addEventListener('click', () => {
      conditions = [];
      activePresetId = '';
      dialog.close();
      refresh();
    });
    dialog.querySelector('form').addEventListener('submit', event => {
      event.preventDefault();
      conditions = editedConditions();
      const preset = presets.find(item => item.id === editingPresetId);
      activePresetId = '';
      if (preset && JSON.stringify(preset.conditions) === JSON.stringify(conditions)) {
        activePresetId = preset.id;
      }
      dialog.close();
      refresh();
    });
  }
  global.HFTLibraryFilters = {initialize, open, matches, clear: () => { conditions = []; activePresetId = ''; }, presets: () => presets, applyPreset, activePreset: () => activePresetId, count: () => conditions.length};
})(window);
