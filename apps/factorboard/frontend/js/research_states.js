/* 研究状态定义与颜色；分类成员、因子数据和研究计算不受影响。 */
(function (global) {
  'use strict';
  const key = 'hft_fb_research_states_v1';
  const saved = localStorage.getItem(key);
  const definitions = saved ? JSON.parse(saved) : [
    {id: 'candidate', name: '候选', color: '#8095a7'},
    {id: 'watch', name: '观察', color: '#d4a05b'},
    {id: 'selected', name: '已入选', color: '#3d9d84'},
    {id: 'rejected', name: '已淘汰', color: '#b98282'},
  ];
  // 移出某个研究状态时使用明确的未设置状态，保留既有状态名称与成员。
  if (!definitions.some(definition => definition.id === 'unassigned')) {
    definitions.push({id: 'unassigned', name: '未设置', color: null});
  }
  let dialog;
  let editing;
  let callbacks;

  // 【非核心】同一状态定义统一控制列表背景、状态单元格与右键菜单。
  function find(name) {
    return definitions.find(definition => definition.name === name);
  }

  // 【非核心】状态编辑不新建因子，也不把状态当作自定义分类。
  function open(id = '') {
    editing = id ? definitions.find(definition => definition.id === id) : null;
    const form = dialog.querySelector('form');
    form.reset();
    form.elements.namedItem('state-name').value = editing ? editing.name : '';
    form.elements.namedItem('state-name').setCustomValidity('');
    form.elements.namedItem('state-color').value = editing?.color ?? '#6ea8a0';
    form.elements.namedItem('use-state-color').checked = editing ? Boolean(editing.color) : true;
    dialog.querySelector('h3').textContent = editing ? '编辑研究状态' : '新增研究状态';
    dialog.showModal();
    form.elements.namedItem('state-name').focus();
  }

  // 【非核心】改名时同步原状态成员，防止刷新后出现找不到定义的旧状态名。
  async function save(event) {
    event.preventDefault();
    const form = dialog.querySelector('form');
    const input = form.elements.namedItem('state-name');
    const name = input.value.trim();
    if (!name || definitions.some(definition => definition !== editing && definition.name === name)) {
      input.setCustomValidity(name ? '状态名称已存在' : '请输入状态名称');
      input.reportValidity();
      return;
    }
    const color = form.elements.namedItem('use-state-color').checked ? form.elements.namedItem('state-color').value : null;
    if (editing) {
      const previousName = editing.name;
      const members = global.HFTData.all.filter(factor => factor.status === previousName);
      await Promise.all(members.map(factor => global.HFTData.putStatus(factor.id, name)));
      editing.name = name;
      editing.color = color;
    } else {
      definitions.push({id: crypto.randomUUID(), name, color});
    }
    localStorage.setItem(key, JSON.stringify(definitions));
    dialog.close();
    callbacks.refresh();
    callbacks.notify('研究状态与颜色已保存到本机');
  }

  // 【非核心】新增和编辑入口只注册一次，资料库重绘不会重复绑定。
  function initialize(handlers) {
    callbacks = handlers;
    dialog = document.createElement('dialog');
    dialog.className = 'research-state-dialog collection-dialog';
    dialog.setAttribute('aria-label', '研究状态设置');
    dialog.innerHTML = '<form><h3>研究状态</h3><label>状态名称<input name="state-name" maxlength="40" required autocomplete="off"></label><label class="collection-color-setting"><span><input name="use-state-color" type="checkbox" checked>启用颜色</span><input type="color" name="state-color" aria-label="状态颜色" value="#6ea8a0"></label><div><button type="button" data-cancel-state>取消</button><button type="submit">保存状态</button></div></form>';
    document.body.append(dialog);
    dialog.querySelector('[data-cancel-state]').addEventListener('click', () => dialog.close());
    dialog.querySelector('[name=state-name]').addEventListener('input', event => event.target.setCustomValidity(''));
    dialog.querySelector('form').addEventListener('submit', event => save(event).catch(error => callbacks.notify(error.message)));
    document.addEventListener('click', event => {
      if (event.target.closest('[data-new-state]')) {
        open();
      }
      const edit = event.target.closest('[data-edit-state]');
      if (edit) {
        open(edit.dataset.editState);
      }
    });
  }
  global.HFTResearchStates = {initialize, find, all: () => definitions,
    unassignedName: () => definitions.find(definition => definition.id === 'unassigned').name,
    defaultName: () => definitions.find(definition => definition.id === 'candidate').name};
})(window);
