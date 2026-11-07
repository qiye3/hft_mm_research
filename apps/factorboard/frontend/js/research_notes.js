/* 长笔记在独立对话框编辑，列表和总览仅显示摘要。 */
(function (global) {
  'use strict';
  let dialog;
  let factorId;
  let handlers;

  // 【非核心】弹窗绑定打开时的因子ID，长笔记保留原始换行和全文。
  function open(id) {
    const factor = global.HFTData.find(id);
    factorId = id;
    const form = dialog.querySelector('form');
    dialog.querySelector('.note-factor-id').textContent = id;
    form.elements.namedItem('note').value = factor.notes;
    form.elements.namedItem('display-name').value = factor.display_name;
    const statuses = form.elements.namedItem('status');
    statuses.replaceChildren();
    global.HFTResearchStates.all().forEach(definition => statuses.add(new Option(definition.name, definition.name)));
    statuses.value = factor.status;
    dialog.querySelector('.note-formula').textContent = factor.formula;
    dialog.showModal();
    form.elements.namedItem('note').focus();
  }

  // 【非核心】笔记、可选短名和单一研究状态一次明确保存，不在取消时写入。
  async function save(event) {
    event.preventDefault();
    const fields = dialog.querySelector('form').elements;
    await Promise.all([
      global.HFTData.putNote(factorId, fields.namedItem('note').value),
      global.HFTData.putDisplayName(factorId, fields.namedItem('display-name').value),
      global.HFTData.putStatus(factorId, fields.namedItem('status').value),
    ]);
    dialog.close();
    handlers.refresh();
    handlers.notify('完整笔记已保存到本机浏览器');
  }

  // 【非核心】总览摘要只负责打开弹窗，编辑器不参与图表尺寸计算。
  function initialize(callbacks) {
    handlers = callbacks;
    dialog = document.createElement('dialog');
    dialog.className = 'research-note-dialog';
    dialog.setAttribute('aria-label', '编辑研究笔记');
    dialog.innerHTML = '<form><div class="note-dialog-head"><h3>研究笔记</h3><button type="button" data-cancel-note aria-label="关闭笔记">×</button></div><p class="note-factor-id"></p><div class="note-dialog-fields"><label>研究状态<select name="status"></select></label><label>中文短名（可选）<input name="display-name"></label></div><label class="long-note-label">详细笔记<textarea name="note" placeholder="记录研究结论、异常日期、构造思路和后续实验…"></textarea></label><details><summary>因子定义</summary><code class="note-formula"></code></details><div class="note-dialog-footer"><span>保存到当前浏览器</span><button type="button" data-cancel-note>取消</button><button type="submit">保存笔记</button></div></form>';
    document.body.append(dialog);
    dialog.querySelectorAll('[data-cancel-note]').forEach(button => button.addEventListener('click', () => dialog.close()));
    dialog.querySelector('form').addEventListener('submit', event => save(event).catch(error => handlers.notify(error.message)));
    document.addEventListener('click', event => {
      const summary = event.target.closest('[data-open-note]');
      if (summary) {
        open(summary.dataset.openNote);
      }
    });
  }
  global.HFTResearchNotes = {initialize};
})(window);
