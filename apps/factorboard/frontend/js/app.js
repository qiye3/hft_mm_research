/* FactorBoard research workbench. Numerical values are served by the evaluation API. */
(function(){
'use strict';
const D=window.HFTData, C=window.HFTCharts;
const folders=document.getElementById('folders'), records=document.getElementById('records'), detail=document.getElementById('detail'), shell=document.getElementById('workspace'), nav=document.getElementById('workspace-nav');
const listSize=35;
const columns=[
 {key:'name',label:'因子名称',width:235},
 {key:'display_name',label:'中文短名（可选）',width:125},
 {key:'gen',label:'Gen',width:66},
 {key:'category',label:'分类',width:80},
 {key:'ic',label:'IC 1s',width:70},
 {key:'rank',label:'Rank IC',width:78},
 {key:'spread',label:'高低组毛价差',width:79},
 {key:'icir',label:'ICIR',width:65},
 {key:'positive',label:'正 IC 日',width:76},
 {key:'valid',label:'共同有效率',width:72},
 {key:'zero',label:'零值率',width:72},
 {key:'status',label:'状态',width:72}
];
const columnByKey=new Map(columns.map(c=>[c.key,c]));
const defaultColumns=['name','ic','spread','status'];
const numericColumns=new Set(['ic','rank','spread','icir','positive','valid','zero']);
const prefKey='hft_fb_frontend_v4_pref';
function sortValue(f,key){switch(key){
 case 'name':return f.id;
 case 'display_name':return f.display_name;
 case 'gen':return f.gen;
 case 'category':return D.categoryLabel(f.category);
 case 'status':return statusOf(f);
 case 'ic':return f.meanIC;
 case 'rank':return f.rank;
 case 'spread':return f.spread;
 case 'icir':return f.icir;
 case 'positive':return f.positive;
 case 'valid':return f.valid;
 case 'zero':return f.zero;
 default:return f.id;
}}
function columnText(f,key){switch(key){
 case 'name':return `<div class="record-name" title="${esc(f.id)}">${esc(f.id)}</div>`;
 case 'display_name':return `<span class="record-cell label">${esc(f.display_name)}</span>`;
 case 'gen':return `<span class="record-cell label">${esc(f.gen)}</span>`;
 case 'category':return `<span class="record-cell label">${esc(D.categoryLabel(f.category))}</span>`;
 case 'ic':return `<span class="record-cell number ${f.meanIC>=0?'pos':'neg'}">${sign(f.meanIC,3)}</span>`;
 case 'rank':return `<span class="record-cell number ${f.rank>=0?'pos':'neg'}">${sign(f.rank,3)}</span>`;
 case 'spread':return `<span class="record-cell number ${f.spread>=0?'pos':'neg'}">${sign(f.spread,3)}</span>`;
 case 'icir':return `<span class="record-cell number">${val(f.icir,2)}</span>`;
 case 'positive':return `<span class="record-cell number">${(f.positive==null?'—':(f.positive*100).toFixed(0)+'%')}</span>`;
 case 'valid':return `<span class="record-cell number">${(f.valid==null?'—':(f.valid*100).toFixed(1)+'%')}</span>`;
 case 'zero':return `<span class="record-cell number">${(f.zero==null?'—':(f.zero*100).toFixed(1)+'%')}</span>`;
 case 'status':return `<span class="record-status centered-status" style="--status-color:${window.HFTResearchStates.find(statusOf(f)).color ?? '#a0aeb5'}"><i class="status-dot"></i>${esc(statusOf(f))}</span>`;
 default:return '';
}}
// 【非核心】名称列使用固定紧凑宽度；完整ID可悬停查看，不随剩余空间拉宽。
function columnLayout() {
 const widths = state.columns.filter(key => key !== 'name').map(key => columnByKey.get(key).width + 'px');
 const minimum = state.columns.reduce((sum, key) => sum + columnByKey.get(key).width, 0) + 19 + state.columns.length * 8 + 24;
 return {grid: ['19px', '235px', ...widths].join(' '), min: minimum};
}
function renderWorkspaceTabs(){nav.innerHTML=`<button type="button" data-workspace="library" class="workspace-tab ${state.view==='library'?'active':''}">资料库</button><button type="button" data-workspace="compare" class="workspace-tab ${state.view==='compare'?'active':''}">因子对比 <span class="workspace-count">${state.compare.length}</span></button><button type="button" data-workspace="gens" class="workspace-tab ${state.view==='gens'?'active':''}">Gen 批次</button><button type="button" data-workspace="jobs" class="workspace-tab ${state.view==='jobs'?'active':''}">评估任务</button>${state.view==='library'?`<div class="workspace-pane-controls"><button class="pane-toggle ${state.foldersOpen?'on':''}" type="button" data-toggle-pane="folders" aria-pressed="${state.foldersOpen}" title="展开或收起左侧分类栏"><span>▤</span> 分类 ${state.foldersOpen?'‹':'›'}</button><button class="pane-toggle ${state.recordsOpen?'on':''}" type="button" data-toggle-pane="records" aria-pressed="${state.recordsOpen}" title="展开或收起中间因子列表"><span>▦</span> 列表 ${state.recordsOpen?'‹':'›'}</button></div>`:''}`;}
function renderColumnSettings(){return `<div class="column-picker"><button type="button" id="column-toggle" class="small-column-btn" aria-expanded="${state.columnPicker}" aria-controls="column-menu">列设置 ▥</button>${state.columnPicker?`<div id="column-menu" class="column-menu" role="group" aria-label="显示列"><div class="column-menu-title">列表字段</div>${columns.map(c=>`<label class="column-option"><input data-column="${c.key}" type="checkbox" ${state.columns.includes(c.key)?'checked':''} ${c.key==='name'?'disabled':''}><span>${esc(c.label)}</span></label>`).join('')}<div class="column-menu-foot"><button type="button" id="column-reset">恢复默认</button></div></div>`:''}</div>`;}
function tableHeading(){return `<div class="table-heading" style="--record-grid:${columnLayout().grid}"><input type="checkbox" data-select-all aria-label="全选当前筛选结果">${state.columns.map(k=>`<button class="column-heading ${k==='name'?'name-h':'numeric-h'}" type="button" data-sort="${k}" title="按${esc(columnByKey.get(k).label)}排序">${esc(columnByKey.get(k).label)}<i class="sort-caret">${state.sort===k?(state.desc?'↓':'↑'):''}</i></button>`).join('')}</div>`;}

const state={view:'library',collection:'factors',kind:'all',query:'',sort:'name',desc:false,page:1,columns:defaultColumns.slice(),columnPicker:false,panelOpen:window.innerWidth>740,foldersOpen:true,recordsOpen:true,
 selected:'book_obi_l5_mean_10',tab:'overview',horizon:1000,distDate:-1,tsLength:240,
 compare:['book_obi_l5_mean_10','trade_aggr_imb_100ms_mean_10'],favorites:new Set(),showPast:false,batchSelection:new Set()};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sign=(x,d=4)=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':(x>=0?'+':'')+Number(x).toFixed(d);
const val=(x,d=3)=>x===null||x===undefined||!Number.isFinite(Number(x))?'—':Number(x).toFixed(d);
const typeName=f=>f.gen==='FEATURES'?'Feature':'Factor';
function statusOf(f){return D.getStatusMap()[f.id]||f.status;}
function noteOf(f){return D.getNotesMap()[f.id]||'';}
// 【非核心】原资料库偏好使用原存储键，收藏和面板开关不随评估模块替换。
function loadPref() {
 const saved = localStorage.getItem(prefKey);
 if (!saved) {
  return;
 }
 const preferences = JSON.parse(saved);
 state.compare = preferences.compare.filter(id => D.find(id));
 state.favorites = new Set(preferences.favorites);
 state.columns = preferences.columns;
 state.sort = preferences.sort;
 state.desc = preferences.desc;
 state.foldersOpen = preferences.foldersOpen;
 state.recordsOpen = preferences.recordsOpen;
}

// 【非核心】资料库偏好保存失败明确报告，不静默吞掉浏览器存储错误。
function savePref() {
 localStorage.setItem(prefKey, JSON.stringify({
  compare: state.compare, favorites: [...state.favorites], columns: state.columns,
  sort: state.sort, desc: state.desc, foldersOpen: state.foldersOpen, recordsOpen: state.recordsOpen
 }));
}
let toastTimer;
function toast(text){const el=document.getElementById('message');el.textContent=text;el.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('show'),2000)}
function countFor(key){return D.all.filter(f=>matchCollection(f,key)).length}
function matchCollection(f,key){if(key==='all')return true;if(key==='factors')return f.gen!=='FEATURES';if(key==='features')return f.gen==='FEATURES';if(key==='favorites')return state.favorites.has(f.id);if(key==='comparison')return state.compare.includes(f.id);if(key.startsWith('status:'))return statusOf(f)===window.HFTResearchStates.all().find(definition=>definition.id===key.slice(7)).name;if(key.startsWith('custom:'))return window.HFTLibraryActions.contains(key.slice(7),f.id);if(key.startsWith('gen:'))return f.gen===key.slice(4);if(key.startsWith('cat:'))return f.category===key.slice(4);return true;}
const folder=(key,title,icon,num)=>`<button class="folder-btn ${state.view==='library'&&state.collection===key?'active':''}" data-folder="${esc(key)}"><span class="folder-icon">${icon}</span><span class="folder-name">${esc(title)}</span><span class="folder-count">${num}</span></button>`;
function renderFolders(){
 const counts={};for(const f of D.all){counts[f.gen]=(counts[f.gen]||0)+1;counts[f.category]=(counts[f.category]||0)+1;}
 folders.innerHTML=`<div class="folders-top">研究资料库 <span style="color:#a4afb7;font:10px var(--mono)">BTC</span></div>
 <div class="folder-head">资料库</div>
 ${folder('all','全部条目','▤',D.all.length)}
 ${folder('factors','Factors','ƒ',D.factors.length)}
 ${folder('features','Features','◇',D.features.length)}
 ${folder('favorites','收藏','☆',state.favorites.size)}
 ${folder('comparison','对比组','⇄',state.compare.length)}

 <hr class="folder-divider">
 <div class="folder-head custom-collection-heading">自定义分类<button data-new-collection type="button" title="新建分类">＋</button></div>
 ${window.HFTLibraryActions.collections().map(collection => `<div class="custom-folder-row">${folder('custom:'+collection.id, collection.name, '▸', D.all.filter(factor => collection.factorIds.includes(factor.id)).length)}<button data-edit-collection="${collection.id}" title="编辑分类" aria-label="编辑分类 ${esc(collection.name)}">⚙</button></div>`).join('')}
 <hr class="folder-divider">
 <div class="folder-head custom-collection-heading">研究状态<button data-new-state type="button" title="新增研究状态">＋</button></div>
 ${window.HFTResearchStates.all().map(definition => `<div class="custom-folder-row">${folder('status:'+definition.id, definition.name, `<i class="state-color-indicator" style="background:${definition.color ?? '#a0aeb5'}"></i>`, D.all.filter(factor=>statusOf(factor)===definition.name).length)}<button data-edit-state="${definition.id}" title="编辑研究状态 ${esc(definition.name)}">⚙</button></div>`).join('')}
 <hr class="folder-divider">
 <div class="folder-head">Gen 批次</div>
 ${D.gens.map(g=>folder('gen:'+g.id,`${g.id} · ${g.title}`,'▸',counts[g.id]||0)).join('')}
 <hr class="folder-divider">
 <div class="folder-head">信息来源</div>
 ${[['book','盘口'],['trade','主动成交'],['cross','交互'],['ofi','订单流'],['returns','价格路径']].map(([k,t])=>folder('cat:'+k,t,'·',counts[k]||0)).join('')}
 <div class="folder-footer">FactorBoard · BTCUSDT</div>`;
}
function filterList(){let arr=D.all.filter(f=>matchCollection(f,state.collection)&&(state.kind==='all'||(state.kind==='features'?f.gen==='FEATURES':f.gen!=='FEATURES')));arr=arr.filter(window.HFTLibraryFilters.matches);if(state.query.trim()){let q=state.query.trim().toLowerCase();arr=arr.filter(f=>[f.id,f.display_name,f.gen,f.desc,f.operation,f.category,...f.tags].join(' ').toLowerCase().includes(q));}
 arr.sort((a,b)=>{const av=sortValue(a,state.sort),bv=sortValue(b,state.sort);const v=numericColumns.has(state.sort)?(av==null?1:bv==null?-1:av-bv):String(av).localeCompare(String(bv),'zh-Hans');return (state.desc?-v:v)||a.id.localeCompare(b.id);});return arr;}
function activeName(){if(state.collection==='all')return '全部因子';if(state.collection==='factors')return 'Factors';if(state.collection==='features')return 'Features';if(state.collection==='favorites')return '收藏';if(state.collection==='comparison')return '对比组';if(state.collection.startsWith('status:'))return window.HFTResearchStates.all().find(definition=>definition.id===state.collection.slice(7)).name;if(state.collection.startsWith('custom:'))return window.HFTLibraryActions.collections().find(collection => collection.id===state.collection.slice(7)).name;if(state.collection.startsWith('gen:'))return state.collection.slice(4);if(state.collection.startsWith('cat:'))return D.categoryLabel(state.collection.slice(4));return '因子';}
// 【非核心】搜索与标题并排，指标条件和批量操作集成到原列表工具栏。
function renderRecords() {
 const matches = filterList();
 records.innerHTML = `<div class="records-head">
  <div class="records-headingline"><div class="records-title"><h2 id="records-title">${esc(activeName())}</h2><span id="records-count">${matches.length}项</span></div><div class="searchbox"><input id="search" type="search" autocomplete="off" placeholder="搜索ID / 名称 / Gen" value="${esc(state.query)}" aria-label="搜索因子"></div></div>
  <div class="record-toolbar compact-record-tools"><div class="segmented"><button data-kind="all" class="${state.kind==='all'?'active':''}">全部</button><button data-kind="factors" class="${state.kind==='factors'?'active':''}">Factor</button><button data-kind="features" class="${state.kind==='features'?'active':''}">Feature</button></div><button type="button" class="small-column-btn" data-open-filters>筛选${window.HFTLibraryFilters.count() ? ' ('+window.HFTLibraryFilters.count()+')' : ''}</button>${renderColumnSettings()}<select class="filter-preset-quick" data-filter-preset aria-label="快速应用筛选方案"><option value="">筛选方案…</option>${window.HFTLibraryFilters.presets().map(preset => `<option value="${preset.id}" ${window.HFTLibraryFilters.activePreset()===preset.id?'selected':''}>${esc(preset.name)}</option>`).join('')}</select><button class="small-column-btn" type="button" data-action="add-factor">＋ 登记</button></div>
  <div class="bulk-toolbar"><span id="batch-count"></span><button type="button" class="small-column-btn" data-bulk-collections ${state.batchSelection.size?'':'disabled'}>批量分类</button><button type="button" class="small-column-btn" data-clear-batch ${state.batchSelection.size?'':'disabled'}>清空多选</button></div>
 </div><div class="records-table-scroll"><div class="records-table-surface" style="--record-min:${columnLayout().min}px">${tableHeading()}<div class="record-list" id="record-list"></div></div></div><div class="record-footer" id="record-footer"></div>`;
 updateRows(false);
}

// 【非核心】行背景只由研究状态控制，自定义分类不再拥有颜色。
function statusAppearance(factor) {
 const definition = window.HFTResearchStates.find(statusOf(factor));
 return definition.color ? `--state-bg:${definition.color}18;--state-edge:${definition.color};` : '';
}

// 【非核心】多选只管理批量分类，与已有四因子对比选择完全独立。
function updateBatchToolbar() {
 const selected = state.batchSelection.size;
 const counter = document.getElementById('batch-count');
 if (!counter) {
  return;
 }
 counter.textContent = `多选 ${selected} 项`;
 document.querySelector('[data-bulk-collections]').disabled = selected === 0;
 document.querySelector('[data-clear-batch]').disabled = selected === 0;
 const visible = filterList();
 const count = visible.filter(factor => state.batchSelection.has(factor.id)).length;
 const selectAll = document.querySelector('[data-select-all]');
 selectAll.checked = visible.length > 0 && count === visible.length;
 selectAll.indeterminate = count > 0 && count < visible.length;
}

function updateRows(preserveScroll){const arr=filterList(),n=Math.max(1,Math.ceil(arr.length/listSize));state.page=Math.max(1,Math.min(state.page,n));const first=(state.page-1)*listSize,items=arr.slice(first,first+listSize);const box=document.getElementById('record-list');if(!box)return;let oldScroll=box.scrollTop;
 box.innerHTML = items.map(factor => {
  const appearance = statusAppearance(factor);
  return `<div class="record-row ${state.selected===factor.id?'selected':''} ${appearance?'has-state-color':''} ${state.batchSelection.has(factor.id)?'batch-picked':''}" data-open="${esc(factor.id)}" tabindex="0" role="button" aria-label="选择因子 ${esc(factor.id)}" style="--record-grid:${columnLayout().grid};${appearance}">
   <input type="checkbox" data-bulk-select="${esc(factor.id)}" title="多选以批量分类" aria-label="多选 ${esc(factor.id)}" ${state.batchSelection.has(factor.id)?'checked':''}>
   ${state.columns.map(key => `<div class="record-value ${key==='name'?'name-value':''}">${columnText(factor,key)}</div>`).join('')}</div>`;
 }).join('') || '<div class="placeholder" style="padding:30px 10px">没有匹配的因子</div>';
 updateBatchToolbar();
 if(preserveScroll)box.scrollTop=oldScroll;else box.scrollTop=0;
 const footer=document.getElementById('record-footer');footer.innerHTML=`<span>${arr.length?first+1:0}—${Math.min(arr.length,first+listSize)} / ${arr.length}</span><span class="pager"><button data-page-step="-1" ${state.page<=1?'disabled':''}>上一页</button><span>${state.page} / ${n}</span><button data-page-step="1" ${state.page>=n?'disabled':''}>下一页</button></span>`;
 const headerCount=document.getElementById('records-count');if(headerCount)headerCount.textContent=arr.length+' 项';}
function tags(f){return `<span class="tag blue">${esc(f.gen)}</span><span class="tag">${esc(D.categoryLabel(f.category))}</span><span class="tag">${esc(f.operation)}</span><span class="tag green">${esc(f.post)}</span>`}
// 【非核心】顶部摘要可独立折叠，切换因子时沿用浏览器中的折叠选择。
function header(f){return `<div class="detail-header factor-detail-header ${window.HFTLibraryActions.collapsed()?'factor-header-collapsed':''}"><div class="detail-path"><strong>${esc(typeName(f))}</strong><span> / </span>${esc(f.gen)} <span> / </span> ${esc(D.categoryLabel(f.category))}</div>
 <div class="factor-top"><div class="factor-headline"><h1>${esc(f.id)}</h1><div class="factor-desc">${tags(f)}<span class="meta-sep">·</span><span>${esc(f.window)}</span></div></div>
 <div class="factor-actions"><button type="button" class="small-btn detail-collapse-toggle" data-toggle-detail aria-expanded="${!window.HFTLibraryActions.collapsed()}">${window.HFTLibraryActions.collapsed()?'展开详情 ▾':'收起详情 ▴'}</button><button class="small-btn mobile-back" data-action="mobile-back">← 列表</button><button class="small-btn ${state.favorites.has(f.id)?'toggled':''}" data-action="favorite" data-id="${esc(f.id)}">${state.favorites.has(f.id)?'★':'☆'}</button><button class="small-btn" data-action="run-eval" data-id="${esc(f.id)}">▷ 评估</button><button class="small-btn ${state.compare.includes(f.id)?'toggled':''}" data-action="add-compare" data-id="${esc(f.id)}">${state.compare.includes(f.id)?'✓ 对比':'＋ 对比'}</button></div></div>
 <div class="tiny-stats"><span>IC 1s <b class="${f.meanIC>=0?'pos':'neg'}">${sign(f.meanIC,4)}</b></span><i></i><span>逐日正向 <b>${f.positive==null?'—':Math.round(f.positive*100)+'%'}</b></span><i></i><span>高低组毛价差 <b class="${f.spread>=0?'pos':'neg'}">${sign(f.spread,3)} bp</b></span><i></i><span>共同有效率 <b>${(f.valid==null?'—':(f.valid*100).toFixed(1)+'%')}</b></span></div>
 <div class="tabs">${[['overview','评估总览'],['distribution','数据与分布'],['timeseries','时序'],['stability','IC与衰减'],['groups','分组收益'],...(D.config.mode === 'production_results' ? [['diagnostics','多维诊断']] : []),['method','口径与版本']].map(([k,t])=>`<button class="${state.tab===k?'active':''}" data-tab="${k}">${t}</button>`).join('')}</div></div>`;}
// 【非核心】保留原资料库详情与笔记，只将数值评估交给增强视图。
// 【非核心】笔记、可选显示名和状态归属资料库，保存到浏览器。
function researchNotes(factor) {
 const note = noteOf(factor).replace(/\s+/g, ' ').trim();
 return `<section class="research-notes note-summary-row"><button type="button" data-open-note="${esc(factor.id)}" title="打开独立笔记编辑器"><strong>研究笔记</strong><span>${esc(note || '暂无笔记，点击编写')}</span><em>编辑详情 ↗</em></button></section>`;
}

// 【非核心】评估区域由增强模块挂载，外层组织功能保留。
function renderDetail() {
 if (state.view === 'compare') {
  renderCompare();
  return;
 }
 if (state.view === 'jobs') {
  renderJobs();
  return;
 }
 if (state.view === 'gens') {
  renderGens();
  return;
 }
 const factor = D.find(state.selected);
 if (!factor) {
  detail.innerHTML = '<div class="placeholder">没有选中因子</div>';
  return;
 }
 detail.innerHTML = header(factor) + '<div class="view-scroll" id="view-scroll"><div id="enhanced-evaluation"></div><div id="notes-staging" hidden>' + researchNotes(factor) + '</div></div>';
 window.HFTEvaluation.renderInto(document.getElementById('enhanced-evaluation'), factor.id, state.tab);
}

// 【非核心】对比选择和弹窗沿用原组织系统，评估图接入十二组与成本设置。
function renderCompare() {
 const factors = state.compare.map(D.find).filter(Boolean);
 detail.innerHTML = `<div class="detail-header compare-header"><div class="detail-path"><strong>Research</strong> / Compare</div><div class="compare-head"><h2>因子对比</h2><div class="compare-head-actions"><span>${factors.length} 项</span><button class="small-btn compare-add" data-action="pick-compare">＋ 选择因子</button></div></div><div class="compare-list">${factors.map(factor => `<span class="compare-chip">${esc(factor.id)} <button aria-label="移除 ${esc(factor.id)}" data-remove="${esc(factor.id)}">×</button></span>`).join('')}</div></div><div class="view-scroll"><div id="enhanced-evaluation"></div></div>`;
 window.HFTEvaluation.renderComparison(document.getElementById('enhanced-evaluation'), state.compare);
}

function renderGens(){detail.innerHTML=`<div class="detail-header"><div class="detail-path"><strong>Research</strong> / Generations</div><div class="factor-top"><div class="factor-headline"><h1 style="font-family:var(--ui)">Gen 批次</h1><div class="factor-desc">母式与基础变量</div></div><button class="small-btn mobile-back" data-action="mobile-back">← 列表</button></div></div><div class="view-scroll"><div class="batch-list">${D.gens.map(g=>`<button class="batch-row" data-batch="${g.id}"><strong>${g.id}</strong><span>${esc(g.title)}</span><small>${D.factors.filter(f=>f.gen===g.id).length} 项 →</small></button>`).join('')}</div></div>`;}
function setHash(route){let h=route==='home'?'#/library':route==='compare'?'#/compare':route==='gens'?'#/gens':route==='jobs'?'#/jobs':'#/factor/'+encodeURIComponent(state.selected);if(location.hash!==h)location.hash=h;else renderAll();}
// 【非核心】品牌链接回全部因子列表，清空本次筛选但不删除用户保存的默认方案。
function showHome() {
 window.HFTLibraryFilters.clear();
 state.collection = 'all';
 state.query = '';
 state.kind = 'all';
 state.page = 1;
 state.panelOpen = false;
 state.recordsOpen = true;
 setHash('home');
}
function changeCollection(key){state.collection=key;state.view='library';state.page=1;state.query='';state.kind='all';let matches=filterList();if(matches.length)state.selected=matches[0].id;state.tab='overview';setHash('library');}
function openFactor(id){state.selected=id;state.view='library';state.tab='overview';state.panelOpen=true;setHash('library');}
function selectFactor(id){if(state.view!=='library'){openFactor(id);return;}const changed=state.selected!==id;state.selected=id;if(changed)state.tab='overview';document.querySelectorAll('.record-row').forEach(row=>row.classList.toggle('selected',row.dataset.open===id));if(state.panelOpen)renderDetail();history.replaceState(null,'','#/factor/'+encodeURIComponent(id));}
// 【非核心】双击同一因子切换右侧看板，双击不同因子打开对应详情。
function toggleRecordPanel(id, previous) {
 const shouldOpen = !state.panelOpen || previous !== id;
 state.selected = id;
 state.view = 'library';
 state.tab = 'overview';
 state.panelOpen = shouldOpen;
 if (shouldOpen) {
  window.HFTLibraryActions.forceCollapsed();
 }
 renderAll();
}
function syncLayout(){const wide=['compare','gens','jobs'].includes(state.view);shell.classList.toggle('view-compare',state.view==='compare');shell.classList.toggle('view-gens',state.view==='gens'||state.view==='jobs');shell.classList.toggle('folders-collapsed',state.view==='library'&&!state.foldersOpen);shell.classList.toggle('records-collapsed',state.view==='library'&&!state.recordsOpen);shell.classList.toggle('detail-open',wide||state.panelOpen);
 const mobile=window.innerWidth<=740;let parts=[];
 if(wide){parts=['minmax(0,1fr)'];}
 else if(mobile&&state.panelOpen){parts=['minmax(0,1fr)'];}
 else{if(state.foldersOpen)parts.push(mobile?'140px':'var(--fb-folder-width)');if(state.recordsOpen)parts.push(state.panelOpen&&!mobile?'minmax(280px,var(--fb-record-width))':'minmax(0,1fr)');if(state.panelOpen&&!mobile)parts.push('minmax(0,1fr)');if(!parts.length)parts=['minmax(0,1fr)'];}
 document.querySelector('.appgrid').style.gridTemplateColumns=parts.join(' ');
}
function togglePane(pane){if(state.view!=='library')return;const mobile=window.innerWidth<=740;const wasDetailOpen=state.panelOpen;
 if(pane==='folders'){state.foldersOpen=!state.foldersOpen;if(mobile&&state.foldersOpen){state.panelOpen=false;state.recordsOpen=true;}}
 if(pane==='records'){state.recordsOpen=!state.recordsOpen;if(!state.recordsOpen)state.panelOpen=true;else if(mobile)state.panelOpen=false;}
 if(!state.recordsOpen&&!state.panelOpen)state.panelOpen=true;
 savePref();renderWorkspaceTabs();if(!wasDetailOpen&&state.panelOpen)renderDetail();syncLayout();}
function renderAll(){if(state.view==='library'&&!state.query){const arr=filterList();const idx=arr.findIndex(f=>f.id===state.selected);if(idx>=0)state.page=Math.floor(idx/listSize)+1;}
 state.columnPicker=false;renderWorkspaceTabs();if(state.view==='library'){renderFolders();renderRecords();}else{folders.replaceChildren();records.replaceChildren();}renderDetail();syncLayout();}
function parseHash(){let s=decodeURIComponent(location.hash.replace(/^#\/?/,''));if(s==='library'){state.view='library';state.collection='all';state.query='';state.kind='all';state.panelOpen=false;state.recordsOpen=true;}else if(s==='compare'){state.view='compare';}else if(s==='gens'){state.view='gens';}else if(s==='jobs'){state.view='jobs';}else if(s.startsWith('factor/')){let id=s.slice(7);if(D.all.some(f=>f.id===id))state.selected=id;state.view='library';}else{state.view='library';}renderAll();}
// 【非核心】对比组没有数量上限，与通用多选集合分别保存。
function setComparison(ids, enabled) {
 const members = new Set(state.compare);
 ids.forEach(id => {
  if (enabled) {
   members.add(id);
  } else {
   members.delete(id);
  }
 });
 state.compare = [...members];
 savePref();
 renderWorkspaceTabs();
 if (state.view === 'library') {
  renderFolders();
  updateRows(true);
 }
 if (state.view === 'compare' || (state.view === 'library' && state.panelOpen)) {
  renderDetail();
 }
}

// 【非核心】详情按钮继续使用同一个对比组，不再限制四个因子。
function updateCompare(id, enabled) {
 setComparison([id], enabled);
 return true;
}

function toggleFavorite(id){if(state.favorites.has(id))state.favorites.delete(id);else state.favorites.add(id);savePref();renderFolders();if(state.collection==='favorites')updateRows(true);if(state.panelOpen)renderDetail();}
function toggleColumnsPanel(){state.columnPicker=!state.columnPicker;const current=document.getElementById('column-toggle');const picker=current?.closest('.column-picker');if(picker)picker.outerHTML=renderColumnSettings();}
function setColumns(next){state.columns=['name',...columns.map(c=>c.key).filter(k=>k!=='name'&&next.includes(k))];if(!state.columns.includes(state.sort)){state.sort='name';state.desc=false;}state.columnPicker=true;savePref();renderRecords();}
function changeSort(key){if(!columnByKey.has(key))return;if(state.sort===key)state.desc=!state.desc;else{state.sort=key;state.desc=numericColumns.has(key);}savePref();const table=document.querySelector('.table-heading');if(table)table.outerHTML=tableHeading();updateRows(true);}
const pickerDialog=document.getElementById('factor-picker');
const picker={draft:new Set(),query:'',type:'all',gen:'all'};
const factorPickerLimit=70;
function pickerMatch(f){if(picker.type==='factors'&&f.gen==='FEATURES')return false;if(picker.type==='features'&&f.gen!=='FEATURES')return false;if(picker.gen!=='all'&&f.gen!==picker.gen)return false;const q=picker.query.trim().toLowerCase();return !q||[f.id,f.display_name,f.gen,f.category,D.categoryLabel(f.category),f.operation,f.desc,...f.tags].join(' ').toLowerCase().includes(q);}
function pickerCounts(){const badge=pickerDialog.querySelector('#picker-counter');if(badge)badge.textContent=`${picker.draft.size} 项`;}
function renderPickerSelection(){const area=pickerDialog.querySelector('#picker-selected');area.innerHTML=[...picker.draft].map(id=>`<span class="picker-chip">${esc(id)}<button data-picker-remove="${esc(id)}" type="button" title="移除 ${esc(id)}">×</button></span>`).join('')||'<span class="picker-muted">尚未选择</span>';pickerCounts();}
function renderPickerResults(){const result=pickerDialog.querySelector('#picker-results');const matches=D.all.filter(pickerMatch);const pinned=matches.filter(f=>picker.draft.has(f.id));const others=matches.filter(f=>!picker.draft.has(f.id));const shown=[...pinned,...others].slice(0,factorPickerLimit);result.innerHTML=shown.map(f=>{const checked=picker.draft.has(f.id);return `<button type="button" class="picker-result ${checked?'picked':''}" data-picker-id="${esc(f.id)}" aria-pressed="${checked}"><span class="picker-check">${checked?'✓':'＋'}</span><span class="picker-result-main"><strong>${esc(f.id)}</strong><small>${esc(f.gen)} · ${esc(D.categoryLabel(f.category))}</small></span><span class="picker-result-metric ${f.meanIC>=0?'pos':'neg'}">${sign(f.meanIC,3)}</span></button>`;}).join('')||'<div class="picker-empty">没有匹配的因子</div>';pickerDialog.querySelector('#picker-result-count').textContent=`${matches.length} 项${matches.length>factorPickerLimit?` · 显示前 ${factorPickerLimit} 项`:''}`;renderPickerSelection();}
function openFactorPicker(){picker.draft=new Set(state.compare);picker.query='';picker.type='all';picker.gen='all';pickerDialog.querySelector('#picker-search').value='';pickerDialog.querySelector('#picker-type').value='all';const genSelect=pickerDialog.querySelector('#picker-gen');genSelect.innerHTML='<option value="all">全部 Gen</option>'+D.gens.map(g=>`<option value="${esc(g.id)}">${esc(g.id)}</option>`).join('')+'<option value="FEATURES">Features</option>';genSelect.value='all';renderPickerResults();if(!pickerDialog.open)pickerDialog.showModal();pickerDialog.querySelector('#picker-search').focus();}
function closeFactorPicker(){if(pickerDialog.open)pickerDialog.close();}
// 【非核心】对比选择器保留全部勾选，不截断为四项。
function pickerToggle(id) {
 if (picker.draft.has(id)) {
  picker.draft.delete(id);
 } else {
  picker.draft.add(id);
 }
 renderPickerResults();
}
function pickerApply(){state.compare=[...picker.draft];savePref();closeFactorPicker();renderWorkspaceTabs();if(state.view==='library')renderFolders();if(state.view==='compare')renderDetail();else if(state.view==='library')updateRows(true);}
pickerDialog.addEventListener('click',e=>{if(e.target===pickerDialog){closeFactorPicker();return;}const result=e.target.closest('[data-picker-id]');if(result){pickerToggle(result.dataset.pickerId);return;}const rm=e.target.closest('[data-picker-remove]');if(rm){pickerToggle(rm.dataset.pickerRemove);return;}if(e.target.closest('#picker-close')||e.target.closest('#picker-cancel')){closeFactorPicker();return;}if(e.target.closest('#picker-apply')){pickerApply();return;}});
pickerDialog.querySelector('#picker-search').addEventListener('input',e=>{picker.query=e.target.value;renderPickerResults();});
pickerDialog.querySelector('#picker-type').addEventListener('change',e=>{picker.type=e.target.value;renderPickerResults();});
pickerDialog.querySelector('#picker-gen').addEventListener('change',e=>{picker.gen=e.target.value;renderPickerResults();});

let precedingSelectionForRow=null;
document.addEventListener('click',e=>{const sidebarFactor=e.target.closest('[data-sidebar-factor]');if(sidebarFactor){openFactor(sidebarFactor.dataset.sidebarFactor);return;}const sidebarRemove=e.target.closest('[data-sidebar-remove]');if(sidebarRemove){updateCompare(sidebarRemove.dataset.sidebarRemove,false);return;}if(e.target.closest('[data-home]')){showHome();return;}const workspaceBtn=e.target.closest('[data-workspace]');if(workspaceBtn){setHash(workspaceBtn.dataset.workspace);return;}const paneBtn=e.target.closest('[data-toggle-pane]');if(paneBtn){togglePane(paneBtn.dataset.togglePane);return;}const folderBtn=e.target.closest('[data-folder]');if(folderBtn){changeCollection(folderBtn.dataset.folder);return;}const routeBtn=e.target.closest('[data-page]');if(routeBtn){setHash(routeBtn.dataset.page);return;}
 const kind=e.target.closest('[data-kind]');if(kind){state.kind=kind.dataset.kind;state.page=1;renderRecords();return;}
 const sort=e.target.closest('[data-sort]');if(sort){changeSort(sort.dataset.sort);return;}
 const step=e.target.closest('[data-page-step]');if(step){state.page+=Number(step.dataset.pageStep);updateRows(false);return;}
 const bulkCheckbox = e.target.closest('[data-bulk-select]');
 if (bulkCheckbox) {
  if (bulkCheckbox.checked) {
   state.batchSelection.add(bulkCheckbox.dataset.bulkSelect);
  } else {
   state.batchSelection.delete(bulkCheckbox.dataset.bulkSelect);
  }
  updateRows(true);
  return;
 }
 if (e.target.matches('[data-select-all]')) {
  for (const factor of filterList()) {
   if (e.target.checked) {
    state.batchSelection.add(factor.id);
   } else {
    state.batchSelection.delete(factor.id);
   }
  }
  updateRows(true);
  return;
 }
 if (e.target.closest('[data-clear-batch]')) {
  state.batchSelection.clear();
  updateRows(true);
  return;
 }
 if (e.target.closest('[data-open-filters]')) {
  window.HFTLibraryFilters.open();
  return;
 }
 const row=e.target.closest('[data-open]');if(row){const id=row.dataset.open;if(e.detail===2){toggleRecordPanel(id,precedingSelectionForRow);precedingSelectionForRow=null;}else{precedingSelectionForRow=state.selected;selectFactor(id);}return;}
 if(e.target.id==='column-toggle'){toggleColumnsPanel();return;}
 if(e.target.id==='column-reset'){setColumns(defaultColumns);return;}
 const tab=e.target.closest('[data-tab]');if(tab){state.tab=tab.dataset.tab;renderDetail();return;}
 const action=e.target.closest('[data-action]');if(action){let id=action.dataset.id||state.selected;switch(action.dataset.action){case 'run-eval':runEvaluation([id]);return;case 'evaluate-compare':runEvaluation(state.compare);return;case 'add-factor':openRegister();return;case 'refresh-jobs':renderJobs();return;case 'add-compare':updateCompare(id,!state.compare.includes(id));return;case 'favorite':toggleFavorite(id);return;case 'mobile-back':state.panelOpen=false;state.recordsOpen=true;renderWorkspaceTabs();syncLayout();return;case 'pick-compare':openFactorPicker();return;}}
 const horizon=e.target.closest('[data-horizon]');if(horizon){state.horizon=Number(horizon.dataset.horizon);renderDetail();return;}
 const len=e.target.closest('[data-len]');if(len){state.tsLength=Number(len.dataset.len);renderDetail();return;}
 const rem=e.target.closest('[data-remove]');if(rem){updateCompare(rem.dataset.remove,false);return;}
 const batch=e.target.closest('[data-batch]');if(batch){changeCollection('gen:'+batch.dataset.batch);return;}

});
document.addEventListener('input',e=>{if(e.target.id==='search'){state.query=e.target.value;state.page=1;updateRows(false);}});
document.addEventListener('change',e=>{if(e.target.matches('[data-filter-preset]')){if(e.target.value)window.HFTLibraryFilters.applyPreset(e.target.value);return;}if(e.target.matches('[data-column]')){const col=e.target.dataset.column;const selected=new Set(state.columns);if(e.target.checked)selected.add(col);else selected.delete(col);setColumns([...selected]);}
 else if(e.target.id==='dist-date'){state.distDate=Number(e.target.value);renderDetail();}
 else if(e.target.id==='show-past'){state.showPast=e.target.checked;renderDetail();}});
document.addEventListener('keydown',e=>{const row=e.target.closest('[data-open]');if(row&&(e.key==='Enter'||e.key===' ')){e.preventDefault();if(e.key==='Enter'&&state.selected===row.dataset.open){state.panelOpen=!state.panelOpen;renderAll();}else openFactor(row.dataset.open);}});
document.addEventListener('click',e=>{if(state.columnPicker&&!e.target.closest('.column-picker')){state.columnPicker=false;const picker=document.querySelector('.column-picker');if(picker)picker.outerHTML=renderColumnSettings();}});
window.addEventListener('hashchange',parseHash);window.addEventListener('resize',syncLayout);

// 【非核心】看板不调用主项目计算，明确提示使用既有完整结果。
function runEvaluation() {
 toast(D.config.mode === 'production_results' ? '已完成评测每分钟自动收录，刷新页面更新列表' : '当前展示已生成结果；更新数据请运行看板prepare入口');
}
let jobsCache=[];let jobsLoading=false;
function renderJobs(){
 if(!jobsLoading){jobsLoading=true;D.listJobs().then(x=>{jobsCache=x.items;jobsLoading=false;if(state.view==='jobs')displayJobs();}).catch(e=>{jobsLoading=false;toast(e.message);});}
 displayJobs();
}
function displayJobs(){
 const rows=jobsCache.slice(0,70);
 detail.innerHTML=`<div class="detail-header"><div class="detail-path"><strong>Research</strong> / Jobs</div><div class="factor-top"><div class="factor-headline"><h1 style="font-family:var(--ui)">Evaluation 任务</h1><div class="factor-desc">看板结果发布记录 · 不运行主项目任务</div></div><button class="small-btn" data-action="refresh-jobs">刷新</button></div></div>
 <div class="view-scroll"><div class="jobs-page"><table class="compare-table"><thead><tr><th>任务 / 因子</th><th>状态</th><th>进度</th><th>创建时间</th></tr></thead><tbody>${rows.map(j=>`<tr><td><strong>${esc(j.factor_id)}</strong><small>${esc(j.id)}</small></td><td>${esc(j.status)}${j.error?`<small style="color:#b65b50">${esc(j.error)}</small>`:''}</td><td>${Math.round(j.progress*100)}%</td><td>${esc(j.created_at.slice(0,19))}</td></tr>`).join('')||'<tr><td colspan="4">暂无计算任务。当前使用已发布的研究结果。</td></tr>'}</tbody></table></div></div>`;
}
function openRegister(){
 const dialog=document.getElementById('register-factor');
 if(dialog&&!dialog.open)dialog.showModal();
}
const registerForm=document.getElementById('register-factor');
if(registerForm){
 registerForm.querySelector('[data-register-close]').addEventListener('click',()=>registerForm.close());
 registerForm.querySelector('form').addEventListener('submit',async e=>{
  e.preventDefault();const form=new FormData(e.target);
  const payload={id:form.get('id').trim(),gen:form.get('gen').trim(),category:form.get('category'),desc:form.get('desc'),base_name:form.get('id').trim(),base_columns:form.get('base_columns').split(',').map(x=>x.trim()).filter(Boolean),base_op:form.get('base_op'),operation:form.get('operation'),short_window:form.get('short_window')?Number(form.get('short_window')):null,long_window:form.get('long_window')?Number(form.get('long_window')):null,post:'raw',raw_fields:[],status:'候选',unit:'ratio'};
  try{await D.addFactor(payload);registerForm.close();state.collection='all';state.kind='all';renderAll();toast('因子已登记');}catch(err){toast(err.message);}
 });
}
function initialize(){
 window.HFTResearchNotes.initialize({refresh:renderAll, notify:toast});
 window.HFTResearchStates.initialize({refresh:renderAll, notify:toast});
 folders.addEventListener('dblclick',event=>{if(event.target.closest('.folders-top'))togglePane('folders');});
 window.HFTLibraryActions.initialize({
  refresh: renderAll, notify: toast,
  isFavorite: id => state.favorites.has(id),
  selection: () => [...state.batchSelection],
  isCompared: id => state.compare.includes(id),
  setComparison,
  setFavorites: (ids, enabled) => {
   ids.forEach(id => {
    if (enabled) {
     state.favorites.add(id);
    } else {
     state.favorites.delete(id);
    }
   });
   savePref();
   renderAll();
  }
 });
 window.HFTLibraryFilters.initialize(() => { state.page=1; renderRecords(); });
 window.HFTEvaluation.initialize(D.config, D.all, renderDetail);
 state.selected=D.find(state.selected)?.id||D.all[0]?.id||'';
 state.compare=state.compare.filter(id=>D.find(id));
 if(state.compare.length<2)state.compare=D.factors.slice(0,2).map(f=>f.id);
 loadPref();if(!location.hash){history.replaceState(null,'','#/factor/'+encodeURIComponent(state.selected));}parseHash();
 if(D.loadError)toast(D.loadError);
}
if(D.ready)initialize();else window.addEventListener('hft-data-ready',initialize,{once:true});
})();
