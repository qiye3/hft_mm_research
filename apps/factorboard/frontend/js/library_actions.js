/* 资料库快捷分类与详情折叠；只修改当前浏览器的组织信息。 */
(function (global) {
  'use strict';
  const collectionKey = 'hft_fb_custom_collections_v1';
  const collapsedKey = 'hft_fb_detail_collapsed';
  const stored = localStorage.getItem(collectionKey);
  const collections = stored ? JSON.parse(stored) : [];
  let handlers;
  let contextFactor;
  let contextFactors = [];
  let editingCollection = null;
  let menu;
  let dialog;

  // 【非核心】自定义分类与数据来源、Gen互相独立，同一因子可加入多个分类。
  function contains(collectionId, factorId) {
    return collections.find(collection => collection.id === collectionId).factorIds.includes(factorId);
  }

  // 【非核心】仅持久化分类成员，不修改因子文件、状态定义或研究配置。
  function persistCollections() {
    localStorage.setItem(collectionKey, JSON.stringify(collections));
  }

  // 【非核心】关闭右键菜单；键盘关闭时把焦点交还原条目。
  function closeMenu(restoreFocus = false) {
    menu.hidden = true;
    if (restoreFocus) {
      const row = document.querySelector(`[data-open="${contextFactor}"]`);
      if (row) {
        row.focus();
      }
    }
  }

  // 【非核心】用原生按钮保留键盘操作和当前分类勾选状态。
  function menuButton(label, action, collectionId = '') {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.dataset.contextAction = action;
    if (collectionId) {
      button.dataset.collectionId = collectionId;
      button.setAttribute('role', 'menuitemcheckbox');
      button.setAttribute('aria-checked', String(contextFactors.every(id => contains(collectionId, id))));
    } else {
      button.setAttribute('role', 'menuitem');
    }
    return button;
  }

  // 【非核心】右键操作目标独立于当前详情选择，不误改另一个因子。
  function openMenu(event, row) {
    event.preventDefault();
    contextFactor = row.dataset.open;
    const selected = handlers.selection();
    contextFactors = selected.includes(contextFactor) ? selected : [contextFactor];
    menu.replaceChildren();
    const heading = document.createElement('div');
    heading.className = 'context-factor-title';
    heading.textContent = contextFactors.length > 1 ? `已选 ${contextFactors.length} 个因子` : contextFactor;
    menu.append(heading);
    menu.append(menuButton(contextFactors.every(handlers.isFavorite) ? '取消收藏' : '加入收藏', 'favorite'));
    const compared = contextFactors.filter(handlers.isCompared).length;
    const removeComparison = compared === contextFactors.length;
    let compareLabel = '加入因子对比组';
    if (removeComparison) {
      compareLabel = '移出因子对比组';
    } else if (compared) {
      compareLabel = '补齐加入因子对比组';
    }
    menu.append(menuButton(compareLabel, removeComparison ? 'remove-compare' : 'add-compare'));
    const stateSection = document.createElement('div');
    stateSection.className = 'context-section';
    stateSection.textContent = '研究状态（互斥）';
    menu.append(stateSection);
    for (const definition of global.HFTResearchStates.all().filter(item => item.id !== 'unassigned')) {
      const assigned = contextFactors.every(id => global.HFTData.find(id).status === definition.name);
      const button = menuButton((assigned ? '移出' : '设为') + definition.name, assigned ? 'remove-status' : 'set-status');
      button.dataset.statusName = definition.name;
      button.setAttribute('role', 'menuitemradio');
      button.setAttribute('aria-checked', String(assigned));
      if (definition.color) {
        button.style.borderLeft = '3px solid ' + definition.color;
      }
      menu.append(button);
    }
    const section = document.createElement('div');
    section.className = 'context-section';
    section.textContent = '自定义分类（可多选）';
    menu.append(section);
    for (const collection of collections) {
      const count = contextFactors.filter(id => contains(collection.id, id)).length;
      const remove = count === contextFactors.length;
      let prefix = '加入分类：';
      if (remove) {
        prefix = '移出分类：';
      } else if (count) {
        prefix = '补齐加入分类：';
      }
      const button = menuButton(prefix + collection.name, remove ? 'remove-collection' : 'add-collection', collection.id);
      let checked = 'mixed';
      if (remove) {
        checked = 'true';
      } else if (count === 0) {
        checked = 'false';
      }
      button.setAttribute('aria-checked', checked);
      menu.append(button);
    }
    menu.append(menuButton('新建分类并添加…', 'new-collection'));
    menu.hidden = false;
    const rowBounds = row.getBoundingClientRect();
    const left = event.clientX || rowBounds.left;
    const top = event.clientY || rowBounds.top;
    const bounds = menu.getBoundingClientRect();
    menu.style.left = Math.max(8, Math.min(left, innerWidth - bounds.width - 8)) + 'px';
    menu.style.top = Math.max(8, Math.min(top, innerHeight - bounds.height - 8)) + 'px';
    menu.querySelector('button').focus();
  }

  // 【非核心】新建分类使用本地对话框，名称去空格且不允许重复。
  function openCollectionDialog(factorIds = [], collectionId = '') {
    closeMenu();
    editingCollection = collectionId ? collections.find(item => item.id === collectionId) : null;
    contextFactors = factorIds;
    const form = dialog.querySelector('form');
    form.reset();
    const input = form.elements.namedItem('collection-name');
    input.setCustomValidity('');
    input.value = editingCollection ? editingCollection.name : '';
    dialog.querySelector('h3').textContent = editingCollection ? '编辑自定义分类' : '新建分类';
    dialog.querySelector('p').textContent = factorIds.length ? `保存后添加 ${factorIds.length} 个因子` : '分类仅管理成员，不改变研究状态、信息来源或Gen。';
    dialog.showModal();
    input.focus();
  }

  // 【非核心】分类只保存名称和批量成员，不再承担颜色设置。
  function createCollection(event) {
    event.preventDefault();
    const form = dialog.querySelector('form');
    const input = form.elements.namedItem('collection-name');
    const name = input.value.trim();
    if (!name || collections.some(collection => collection !== editingCollection && collection.name === name)) {
      input.setCustomValidity(name ? '分类名称已存在' : '请输入分类名称');
      input.reportValidity();
      return;
    }
    if (editingCollection) {
      editingCollection.name = name;
      delete editingCollection.color;
    } else {
      collections.push({id: crypto.randomUUID(), name, factorIds: [...contextFactors]});
    }
    persistCollections();
    dialog.close();
    handlers.refresh();
    handlers.notify('自定义分类已保存到本机浏览器');
  }

  // 【非核心】菜单操作只改变右键因子的组织信息，切换分类成员时可直接移出。
  async function applyMenuAction(event) {
    const button = event.target.closest('[data-context-action]');
    if (!button) {
      return;
    }
    const action = button.dataset.contextAction;
    closeMenu();
    if (action === 'new-collection') {
      openCollectionDialog(contextFactors);
      return;
    }
    if (action === 'add-compare' || action === 'remove-compare') {
      handlers.setComparison(contextFactors, action === 'add-compare');
      handlers.notify(`已更新对比组：${contextFactors.length}个因子`);
      return;
    }
    if (action === 'favorite') {
      handlers.setFavorites(contextFactors, !contextFactors.every(handlers.isFavorite));
      return;
    }
    if (['add-collection', 'remove-collection'].includes(action)) {
      const collection = collections.find(item => item.id === button.dataset.collectionId);
      const remove = action === 'remove-collection';
      const members = new Set(collection.factorIds);
      contextFactors.forEach(id => {
        if (remove) {
          members.delete(id);
        } else {
          members.add(id);
        }
      });
      collection.factorIds = [...members];
      persistCollections();
    } else {
      const status = action === 'remove-status' ? global.HFTResearchStates.unassignedName() : button.dataset.statusName;
      await Promise.all(contextFactors.map(id => global.HFTData.putStatus(id, status)));
    }
    handlers.refresh();
    handlers.notify('已更新右键条目的分类');
  }

  // 【非核心】折叠只隐藏说明和摘要，保留因子ID与页签，避免丢失导航。
  function toggleDetail() {
    const collapsed = localStorage.getItem(collapsedKey) !== 'true';
    localStorage.setItem(collapsedKey, String(collapsed));
    const header = document.querySelector('.factor-detail-header');
    header.classList.toggle('factor-header-collapsed', collapsed);
    const button = header.querySelector('[data-toggle-detail]');
    button.textContent = collapsed ? '展开详情 ▾' : '收起详情 ▴';
    button.setAttribute('aria-expanded', String(!collapsed));
  }

  // 【非核心】右键菜单、分类对话框和折叠监听各初始化一次，资料库重绘不重复绑定。
  function initialize(callbacks) {
    handlers = callbacks;
    menu = document.createElement('div');
    menu.className = 'factor-context-menu';
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', '因子快捷分类');
    menu.hidden = true;
    document.body.append(menu);
    dialog = document.createElement('dialog');
    dialog.className = 'collection-dialog';
    dialog.setAttribute('aria-label', '新建自定义分类');
    dialog.innerHTML = '<form><h3>新建分类</h3><p></p><label>分类名称<input name="collection-name" required maxlength="80" autocomplete="off"></label><div><button type="button" data-cancel-collection>取消</button><button type="submit">创建</button></div></form>';
    document.body.append(dialog);
    dialog.querySelector('form').addEventListener('submit', createCollection);
    dialog.querySelector('input').addEventListener('input', event => event.target.setCustomValidity(''));
    dialog.querySelector('[data-cancel-collection]').addEventListener('click', () => dialog.close());
    menu.addEventListener('click', event => applyMenuAction(event).catch(error => handlers.notify(error.message)));
    document.getElementById('records').addEventListener('contextmenu', event => {
      const row = event.target.closest('[data-open]');
      if (row) {
        openMenu(event, row);
      }
    });
    document.addEventListener('click', event => {
      const edit = event.target.closest('[data-edit-collection]');
      if (edit) {
        openCollectionDialog([], edit.dataset.editCollection);
      }
      const batch = event.target.closest('[data-bulk-collections]');
      if (batch) {
        const selected = handlers.selection();
        const bounds = batch.getBoundingClientRect();
        openMenu({preventDefault() {}, clientX: bounds.left, clientY: bounds.bottom},
          {dataset: {open: selected[0]}, getBoundingClientRect: () => bounds});
      }
      if (event.target.closest('[data-toggle-detail]')) {
        toggleDetail();
      }
      if (event.target.closest('[data-new-collection]')) {
        openCollectionDialog();
      }
    });
    document.addEventListener('pointerdown', event => {
      if (!menu.contains(event.target)) {
        closeMenu();
      }
    });
    menu.addEventListener('keydown', event => {
      if (event.key === 'Escape') {
        event.preventDefault();
        closeMenu(true);
      } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        const buttons = [...menu.querySelectorAll('button')];
        const index = buttons.indexOf(document.activeElement);
        const next = index + (event.key === 'ArrowDown' ? 1 : -1);
        buttons[(next + buttons.length) % buttons.length].focus();
      }
    });
    window.addEventListener('resize', () => closeMenu());
    document.addEventListener('scroll', event => {
      if (!menu.contains(event.target)) {
        closeMenu();
      }
    }, true);
  }

  global.HFTLibraryActions = {
    initialize, contains,
    forceCollapsed: () => localStorage.setItem(collapsedKey, 'true'),
    memberships: factorId => collections.filter(collection => collection.factorIds.includes(factorId)),
    collections: () => collections,
    collapsed: () => localStorage.getItem(collapsedKey) === 'true',
  };
})(window);
