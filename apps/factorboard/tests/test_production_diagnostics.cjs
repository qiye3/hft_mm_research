/* 直接验证看板派生统计的窗口、空值和分组边界。运行 node tests/test_production_diagnostics.cjs。 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const sandbox = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../frontend/js/production_diagnostics.js'), 'utf8'), sandbox);
const diagnostics = sandbox.window.HFTProductionDiagnostics;
let captured = [];
// 【非核心】捕获图表输入，直接核对派生值，避免把截图当统计验收。
function capture(kind, rows) {
  captured.push({kind, rows});
  return '';
}
const context = {
  Charts: {
    lines: rows => capture('lines', rows), bars: rows => capture('bars', rows),
    heatmap: rows => capture('heatmap', rows), scatter: rows => capture('scatter', rows),
  },
  panel: () => '',
  groupRows: () => [
    {group: 'G1', bps: 1, count: 10},
    {group: 'G2', bps: null, count: 0},
    {group: 'G3', bps: 3, count: 20},
  ],
};
const daily = Array.from({length: 8}, (_, index) => ({
  date: `2026-01-0${index + 1}`, ic: index / 10, rank_ic: index / 20,
  rows: 100, paired_rows: 95,
}));
const result = {
  id: 'test_factor', daily_ic: daily,
  horizons: [{horizon_ns: 1e9}],
  daily_horizons: daily.map(row => ({...row, horizon_ns: 1e9})),
  split_summary: ['train', 'validation', 'test'].map(split => ({split, ic_mean: .1, rank_ic_mean: .2})),
  daily_groups: [{date: '2026-01-01', group: 1, mean_return_bps: 1}, {date: '2026-01-01', group: 3, mean_return_bps: 3}],
  distribution: daily.map(row => ({date: row.date, missing_rate: .02, zero_rate: .1})),
  summary: {ic_mean: .35, positive_ic_day_ratio: .875, paired_rows: 760},
};
diagnostics.render(result, 'stability', context);
assert.equal(captured[0].rows.reduce((count, row) => count + row.count, 0), 8);
assert.equal(captured[1].rows[3].rolling, null);
assert.ok(Math.abs(captured[1].rows[7].rolling - .4) < 1e-12);
assert.equal(captured[3].rows[0].positive, 7 / 8);
captured = [];
diagnostics.render({...result, daily_ic: [...daily.slice(0, 5), {...daily[5], date: '2026-01-20'}]}, 'stability', context);
assert.equal(captured[1].rows.at(-1).rolling, null);
captured = [];
diagnostics.render(result, 'horizons', context);
assert.ok(Math.abs(captured[0].rows[0].cells[0].value - .35) < 1e-12);
assert.equal(captured[0].rows[0].cells[0].days, 8);
captured = [];
diagnostics.render(result, 'groups', context);
assert.equal(captured[2].rows[0].delta, null);
assert.equal(captured[3].rows[0].spread, 2);
assert.equal(captured[3].rows[1].spread, null);
captured = [];
diagnostics.render(result, 'quality', context);
assert.equal(captured[0].rows[0].coverage, .95);
captured = [];
diagnostics.compare([result], context);
assert.ok(Math.abs(captured[0].rows[0].volatility - Math.sqrt(.06)) < 1e-12);
assert.equal(captured[1].rows[0].coverage, .95);
