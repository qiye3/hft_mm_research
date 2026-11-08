/* 生产评测图表。使用原生SVG，坐标按容器宽度绘制，空值保持断线。 */
(function () {
  'use strict';
  const colors = ['#14847c', '#6574c7', '#c28146', '#af637b'];

  // 【非核心】结果名称和提示文本统一转义后进入HTML或SVG。
  function escapeText(value) {
    return String(value).replace(/[&<>"]/g, character => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[character]));
  }

  // 【非核心】小量级保留有效数字，比例按百分数展示。
  function format(value, percent = false) {
    if (!Number.isFinite(value)) {
      return '—';
    }
    if (percent) {
      return (value * 100).toFixed(2) + '%';
    }
    if (value !== 0 && Math.abs(value) < 0.001) {
      return value.toExponential(2);
    }
    return value.toLocaleString('en-US', {maximumFractionDigits: 4});
  }

  // 【非核心】实际容器宽度决定坐标字体大小，避免大画布缩小后文字无法辨认。
  function dimensions(wide) {
    const container = document.getElementById('enhanced-evaluation');
    const available = container.clientWidth;
    const columns = !wide && available >= 620 ? 2 : 1;
    const width = Math.max(260, Math.floor((available - (columns - 1) * 12) / columns) - 26);
    const height = window.innerHeight < 820 ? 108 : 156;
    return {width, height, left: 54, right: 12, top: 12, bottom: 26};
  }

  // 【非核心】轴范围覆盖全部有限值，常数序列仍保留可读的上下空间。
  function valueRange(values, zero) {
    if (zero) {
      const extent = Math.max(0.001, ...values.map(Math.abs)) * 1.16;
      return [-extent, extent];
    }
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const margin = Math.max((maximum - minimum) * 0.12, Math.abs(maximum) * 0.02, 0.00001);
    return [minimum - margin, maximum + margin];
  }

  // 【非核心】坐标采用简短刻度，精确值保留在悬浮提示中。
  function axisLabel(value, percent) {
    if (percent) {
      return (value * 100).toFixed(1) + '%';
    }
    if (Math.abs(value) >= 10000) {
      return (value / 1000000).toFixed(1) + 'M';
    }
    return String(Number(value.toPrecision(3)));
  }

  // 【非核心】每图保留四条辅助线，零轴单独突出。
  function axes(size, range, labels, percent) {
    const plotHeight = size.height - size.top - size.bottom;
    let marks = '';
    for (let index = 0; index <= 4; index += 1) {
      const vertical = size.top + plotHeight * index / 4;
      const value = range[1] - (range[1] - range[0]) * index / 4;
      marks += `<line x1="${size.left}" x2="${size.width - size.right}" y1="${vertical}" y2="${vertical}" class="fb-grid"/><text x="${size.left - 8}" y="${vertical + 3}" text-anchor="end" class="fb-axis">${escapeText(axisLabel(value, percent))}</text>`;
    }
    if (range[0] < 0 && range[1] > 0) {
      const vertical = size.top + range[1] / (range[1] - range[0]) * plotHeight;
      marks += `<line x1="${size.left}" x2="${size.width - size.right}" y1="${vertical}" y2="${vertical}" class="fb-zero"/>`;
    }
    labels.forEach(label => {
      marks += `<text x="${label.position}" y="${size.height - 8}" text-anchor="${label.anchor || 'middle'}" class="fb-axis">${escapeText(label.text)}</text>`;
    });
    return marks;
  }

  // 【非核心】图例可独立开关系列，键盘也可聚焦逐点提示。
  function wrap(size, marks, series, unit) {
    const legend = series.map((field, index) => `<button type="button" data-chart-toggle="${index}" aria-pressed="true"><i style="background:${colors[index]}"></i>${escapeText(field.label)}</button>`).join('');
    return `<div class="fb-chart"><div class="fb-chart-legend">${legend}<span>${escapeText(unit || '')}</span></div><svg viewBox="0 0 ${size.width} ${size.height}" role="img" aria-label="${escapeText(series.map(field => field.label).join(' / '))}">${marks}</svg><div class="fb-chart-tooltip" role="status" hidden></div></div>`;
  }

  // 【非核心】同一横轴绘制真实日期或离散预测周期，缺失点直接断开路径。
  function lines(rows, series, options = {}) {
    const values = rows.flatMap(row => series.map(field => row[field.key])).filter(Number.isFinite);
    if (!values.length) {
      return '<div class="fb-empty">当前范围无有效观测</div>';
    }
    const size = dimensions(options.wide);
    const range = valueRange(values, options.zero);
    if (options.percent) {
      range[0] = Math.max(0, range[0]);
      range[1] = Math.min(1, range[1]);
    }
    const plotWidth = size.width - size.left - size.right;
    const plotHeight = size.height - size.top - size.bottom;
    const timeValues = rows.map((row, index) => row.date ? Date.parse(row.date) : index);
    const timeSpan = timeValues.at(-1) - timeValues[0] || 1;
    const horizontal = index => size.left + (timeValues[index] - timeValues[0]) / timeSpan * plotWidth;
    const vertical = value => size.top + (range[1] - value) / (range[1] - range[0]) * plotHeight;
    const tickCount = rows.length <= 10 ? rows.length : Math.max(2, Math.floor(plotWidth / 80));
    const indices = [...new Set(Array.from({length: tickCount}, (_, index) => Math.round(index * (rows.length - 1) / Math.max(1, tickCount - 1))))];
    const labels = indices.map(index => ({position: horizontal(index), text: rows[index].date ? rows[index].date.slice(5) : rows[index].label}));
    let marks = axes(size, range, labels, options.percent);
    series.forEach((field, seriesIndex) => {
      let connected = false;
      const segments = rows.map((row, index) => {
        if (!Number.isFinite(row[field.key])) {
          connected = false;
          return '';
        }
        const command = connected ? 'L' : 'M';
        connected = true;
        return `${command}${horizontal(index).toFixed(2)},${vertical(row[field.key]).toFixed(2)}`;
      });
      marks += `<g data-chart-series="${seriesIndex}"><path d="${segments.join(' ')}" stroke="${colors[seriesIndex]}" stroke-width="1.8" fill="none" stroke-linejoin="round"/>`;
      if (rows.length <= 10) {
        marks += rows.map((row, index) => Number.isFinite(row[field.key]) ? `<circle cx="${horizontal(index)}" cy="${vertical(row[field.key])}" r="3" fill="${colors[seriesIndex]}"/>` : '').join('');
      }
      marks += '</g>';
    });
    rows.forEach((row, index) => {
      const previous = index === 0 ? size.left : (horizontal(index - 1) + horizontal(index)) / 2;
      const next = index === rows.length - 1 ? size.width - size.right : (horizontal(index) + horizontal(index + 1)) / 2;
      const details = series.map(field => `${field.label}  ${format(row[field.key], options.percent)}`).join('\n');
      marks += `<rect x="${previous}" y="${size.top}" width="${Math.max(1, next - previous)}" height="${plotHeight}" class="fb-hit" tabindex="${index === 0 ? 0 : -1}" data-chart-tip="${escapeText((row.date || row.label) + '\n' + details)}"/><line x1="${horizontal(index)}" x2="${horizontal(index)}" y1="${size.top}" y2="${size.height - size.bottom}" class="fb-crosshair"/>`;
    });
    return wrap(size, marks, series, options.unit);
  }

  // 【非核心】收益颜色表达正负，缺失组保留原位置和缺失提示。
  function bars(rows, field, options = {}) {
    const values = rows.map(row => row[field]).filter(Number.isFinite);
    if (!values.length) {
      return '<div class="fb-empty">当前范围缺少有效分组</div>';
    }
    const size = dimensions(options.wide);
    const range = valueRange(values, true);
    if (['样本', '日数'].includes(options.unit)) {
      range[0] = 0;
    }
    const plotWidth = size.width - size.left - size.right;
    const plotHeight = size.height - size.top - size.bottom;
    const step = plotWidth / rows.length;
    const center = size.top + range[1] / (range[1] - range[0]) * plotHeight;
    const scale = plotHeight / (range[1] - range[0]);
    const labels = rows.map((row, index) => ({position: size.left + (index + 0.5) * step, text: row.group}));
    let marks = axes(size, range, labels, false);
    rows.forEach((row, index) => {
      const horizontal = size.left + (index + 0.5) * step;
      const value = row[field];
      if (Number.isFinite(value)) {
        const vertical = center - value * scale;
        marks += `<rect x="${horizontal - step * 0.29}" y="${Math.min(center, vertical)}" width="${step * 0.58}" height="${Math.max(1, Math.abs(vertical - center))}" rx="3" fill="${value >= 0 ? '#14847c' : '#bd6874'}"/><text x="${horizontal}" y="${vertical + (value >= 0 ? -6 : 14)}" text-anchor="middle" class="fb-bar-label">${escapeText(options.unit === '样本' ? new Intl.NumberFormat('en', {notation: 'compact', maximumFractionDigits: 1}).format(value) : format(value))}</text>`;
      } else {
        marks += `<text x="${horizontal}" y="${center - 5}" text-anchor="middle" class="fb-axis">—</text>`;
      }
      const tip = row.tip || `${row.group}\n${options.unit || 'bps'}  ${format(value)}\n样本  ${row.count.toLocaleString()}\n有效日  ${row.valid_days}`;
      marks += `<rect x="${horizontal - step / 2}" y="${size.top}" width="${step}" height="${plotHeight}" class="fb-hit" tabindex="${index === 0 ? 0 : -1}" data-chart-tip="${escapeText(tip)}"/>`;
    });
    return wrap(size, marks, [], options.unit || 'bps');
  }

  // 【非核心】散点保留每个日期或因子的独立观测，悬浮展示精确数值。
  function scatter(rows, options) {
    const valid = rows.filter(row => Number.isFinite(row[options.x]) && Number.isFinite(row[options.y]));
    if (!valid.length) {
      return '<div class="fb-empty">当前范围没有可配对观测</div>';
    }
    const size = dimensions(false);
    const xRange = valueRange(valid.map(row => row[options.x]), false);
    const yRange = valueRange(valid.map(row => row[options.y]), false);
    if (options.xPercent) {
      xRange[0] = Math.max(0, xRange[0]);
      xRange[1] = Math.min(1, xRange[1]);
    }
    if (options.yPercent) {
      yRange[0] = Math.max(0, yRange[0]);
      yRange[1] = Math.min(1, yRange[1]);
    }
    const plotWidth = size.width - size.left - size.right;
    const plotHeight = size.height - size.top - size.bottom;
    const labels = Array.from({length: 3}, (_, index) => ({
      position: size.left + index * plotWidth / 2,
      text: axisLabel(xRange[0] + index * (xRange[1] - xRange[0]) / 2, options.xPercent),
    }));
    let marks = axes(size, yRange, labels, options.yPercent);
    valid.forEach((row, index) => {
      const horizontal = size.left + (row[options.x] - xRange[0]) / (xRange[1] - xRange[0]) * plotWidth;
      const vertical = size.top + (yRange[1] - row[options.y]) / (yRange[1] - yRange[0]) * plotHeight;
      const tooltip = `${row.name}\n${options.xLabel}  ${format(row[options.x], options.xPercent)}\n${options.yLabel}  ${format(row[options.y], options.yPercent)}`;
      marks += `<circle cx="${horizontal}" cy="${vertical}" r="4" fill="#14847c" fill-opacity=".55" class="fb-scatter-point" tabindex="${index === 0 ? 0 : -1}" data-chart-tip="${escapeText(tooltip)}"/>`;
    });
    return wrap(size, marks, [], `${options.xLabel} → / ↑ ${options.yLabel}`);
  }

  // 【非核心】热图显示已计算日指标的月度等权均值，缺失单元保持空白。
  function heatmap(rows, columns) {
    const values = rows.flatMap(row => row.cells.map(cell => cell.value)).filter(Number.isFinite);
    const extent = Math.max(0.001, ...values.map(Math.abs));
    const header = columns.map(column => `<th>${escapeText(column)}</th>`).join('');
    const body = rows.map(row => `<tr><th>${escapeText(row.label)}</th>${row.cells.map(cell => {
      const valid = Number.isFinite(cell.value);
      const alpha = valid ? 0.06 + Math.abs(cell.value) / extent * 0.65 : 0;
      const color = cell.value < 0 ? `rgba(189,104,116,${alpha})` : `rgba(20,132,124,${alpha})`;
      const tooltip = `${row.label} · ${cell.label} · 日均IC ${format(cell.value)} · 有效日 ${cell.days}`;
      return `<td style="background:${color}" title="${escapeText(tooltip)}">${valid ? cell.value.toFixed(3) : '—'}</td>`;
    }).join('')}</tr>`).join('');
    return `<div class="fb-heatmap"><table><thead><tr><th>月份</th>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  // 【非核心】提示固定在图内，触控或键盘聚焦也可读取完整数值。
  function showTooltip(target, clientX, clientY) {
    const chart = target.closest('.fb-chart');
    const tooltip = chart.querySelector('.fb-chart-tooltip');
    tooltip.textContent = target.dataset.chartTip;
    tooltip.hidden = false;
    const bounds = chart.getBoundingClientRect();
    const left = Math.min(clientX - bounds.left + 12, bounds.width - tooltip.offsetWidth - 6);
    const top = Math.min(clientY - bounds.top + 12, bounds.height - tooltip.offsetHeight - 4);
    tooltip.style.left = Math.max(0, left) + 'px';
    tooltip.style.top = Math.max(0, top) + 'px';
  }

  // 【非核心】委托事件随详情替换继续生效，不在每次渲染重复绑定。
  function attachInteractions() {
    document.addEventListener('pointermove', event => {
      const target = event.target.closest('[data-chart-tip]');
      if (target) {
        showTooltip(target, event.clientX, event.clientY);
      }
    });
    document.addEventListener('pointerout', event => {
      const target = event.target.closest('[data-chart-tip]');
      if (target) {
        target.closest('.fb-chart').querySelector('.fb-chart-tooltip').hidden = true;
      }
    });
    document.addEventListener('focusin', event => {
      const target = event.target.closest('[data-chart-tip]');
      if (target) {
        const bounds = target.getBoundingClientRect();
        showTooltip(target, bounds.x, bounds.y);
      }
    });
    document.addEventListener('focusout', event => {
      const target = event.target.closest('[data-chart-tip]');
      if (target) {
        target.closest('.fb-chart').querySelector('.fb-chart-tooltip').hidden = true;
      }
    });
    document.addEventListener('keydown', event => {
      const target = event.target.closest('[data-chart-tip]');
      if (!target || !['ArrowLeft', 'ArrowRight'].includes(event.key)) {
        return;
      }
      event.preventDefault();
      const points = [...target.closest('svg').querySelectorAll('[data-chart-tip]')];
      const step = event.key === 'ArrowRight' ? 1 : -1;
      const next = points[points.indexOf(target) + step];
      if (next) {
        target.setAttribute('tabindex', '-1');
        next.setAttribute('tabindex', '0');
        next.focus();
      }
    });
    document.addEventListener('click', event => {
      const button = event.target.closest('[data-chart-toggle]');
      if (!button) {
        return;
      }
      const visible = button.getAttribute('aria-pressed') !== 'true';
      button.setAttribute('aria-pressed', String(visible));
      button.closest('.fb-chart').querySelector(`[data-chart-series="${button.dataset.chartToggle}"]`).style.display = visible ? '' : 'none';
    });
  }
  window.HFTProductionCharts = {lines, bars, scatter, heatmap, format, attachInteractions};
})();
