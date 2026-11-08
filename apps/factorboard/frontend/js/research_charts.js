/* 十二组与连续时间查询专用图表，缺失值始终断线。 */
(function (global) {
  'use strict';
  const colors = ['#263e76', '#3f65a4', '#577fb8', '#739ac5', '#92b4d0', '#b3cdd8',
    '#e5cbb6', '#d9aa91', '#cc8b78', '#b96c68', '#a44f5b', '#78374e'];

  // 【非核心】将文本安全插入SVG的标题和轴标签。
  function escapeText(value) {
    return String(value).replace(/[&<>"]/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'
    })[character]);
  }

  // 【非核心】十二组固定位置保留空组，避免移位后误读端点。
  function bars(rows, field, title) {
    const valid = rows.map(row => row[field]).filter(Number.isFinite);
    if (!valid.length) {
      return '<div class="board-empty">此范围没有有效分组样本</div>';
    }
    const width = 1020;
    const height = 300;
    const left = 62;
    const right = 18;
    const top = 28;
    const bottom = 42;
    const extent = Math.max(.001, ...valid.map(Math.abs)) * 1.3;
    const center = (height - bottom + top) / 2;
    const scale = (height - bottom - top) / (2 * extent);
    const slot = (width - left - right) / rows.length;
    let marks = '';
    for (let tick = -2; tick <= 2; tick += 1) {
      const value = tick * extent / 2;
      const vertical = center - value * scale;
      marks += `<line x1="${left}" x2="${width - right}" y1="${vertical}" y2="${vertical}" stroke="#e5ebef"/>`;
      marks += `<text x="${left - 8}" y="${vertical + 4}" text-anchor="end" class="tick">${value.toFixed(3)}</text>`;
    }
    rows.forEach((row, index) => {
      const horizontal = left + (index + .5) * slot;
      const value = row[field];
      if (Number.isFinite(value)) {
        const vertical = center - value * scale;
        const barTop = Math.min(vertical, center);
        const barHeight = Math.max(1, Math.abs(vertical - center));
        marks += `<rect x="${horizontal - slot * .33}" y="${barTop}" width="${slot * .66}" height="${barHeight}" rx="2" fill="${colors[index]}"><title>${escapeText(row.group)} · ${value.toFixed(5)} · 样本${row.count} · ${row.valid_days}日</title></rect>`;
        marks += `<text x="${horizontal}" y="${vertical + (value >= 0 ? -8 : 17)}" text-anchor="middle" class="tick">${value.toFixed(3)}</text>`;
      }
      marks += `<text x="${horizontal}" y="${height - 12}" text-anchor="middle" class="tick">${escapeText(row.group)}</text>`;
    });
    return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeText(title)}">${marks}</svg>`;
  }

  // 【非核心】两条不同量纲曲线各自缩放，横轴严格保留真实时间和缺失位置。
  function series(rows, showPast) {
    if (!rows.length) {
      return '<div class="board-empty">所选时段没有网格数据</div>';
    }
    const width = 1020;
    const height = 330;
    const left = 62;
    const right = 70;
    const top = 30;
    const bottom = 40;
    const first = rows[0].ts;
    const span = Math.max(1, rows.at(-1).ts - first);
    const factorValues = rows.map(row => row.factor).filter(Number.isFinite);
    const returnValues = rows.flatMap(row => showPast ? [row.label, row.past] : [row.label]).filter(Number.isFinite);
    const factorExtent = Math.max(.01, ...factorValues.map(Math.abs)) * 1.15;
    const returnExtent = Math.max(.01, ...returnValues.map(Math.abs)) * 1.15;
    const center = (height - bottom + top) / 2;
    const horizontal = timestamp => left + (width - left - right) * (timestamp - first) / span;
    let marks = '';
    for (let index = 0; index <= 4; index += 1) {
      const ratio = 1 - index / 2;
      const vertical = top + (height - top - bottom) * index / 4;
      marks += `<line x1="${left}" x2="${width - right}" y1="${vertical}" y2="${vertical}" stroke="#e5ebef"/>`;
      marks += `<text x="${left - 7}" y="${vertical + 4}" text-anchor="end" class="tick">${(factorExtent * ratio).toFixed(3)}</text>`;
      marks += `<text x="${width - right + 7}" y="${vertical + 4}" class="tick">${(returnExtent * ratio).toFixed(3)}</text>`;
    }
    const fields = [['factor', factorExtent, '#087d75'], ['label', returnExtent, '#c46a65']];
    if (showPast) {
      fields.push(['past', returnExtent, '#8b98ad']);
    }
    fields.forEach(([field, extent, color]) => {
      let connected = false;
      const segments = rows.map(row => {
        if (!Number.isFinite(row[field])) {
          connected = false;
          return '';
        }
        const command = connected ? 'L' : 'M';
        connected = true;
        const vertical = center - row[field] / extent * (height - top - bottom) / 2;
        return command + horizontal(row.ts).toFixed(2) + ',' + vertical.toFixed(2);
      });
      marks += `<path d="${segments.join(' ')}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
    });
    for (let index = 0; index <= 5; index += 1) {
      const timestamp = first + span * index / 5;
      const label = new Date(timestamp / 1e6).toISOString().slice(11, 23);
      marks += `<text x="${horizontal(timestamp)}" y="${height - 12}" text-anchor="middle" class="tick">${label}</text>`;
    }
    marks += `<text x="${left}" y="15" class="tick">Factor</text><text x="${width - right}" y="15" text-anchor="end" class="tick">未来收益 / 历史收益 · bps</text>`;
    return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="真实UTC时间因子与收益时序">${marks}</svg>`;
  }

  global.HFTResearchCharts = {bars, series, colors};
})(window);
