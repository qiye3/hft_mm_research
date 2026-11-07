/* Lightweight SVG charts for server-computed historical evaluations. */
(function(global){'use strict';
const W=760,H=262,L=51,R=18,T=15,B=39;const ink='#60738a',grid='#e8edf1',pos='#087d75',neg='#ca7164',blue='#4a78bb',purple='#8b75ba';
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const n=v=>Number.isFinite(v)?v:0;
const val=(v,dec=3)=>n(v).toFixed(dec);
function tag(inner,height=H){return `<svg viewBox="0 0 ${W} ${height}" role="img" aria-label="历史数据评估图表" preserveAspectRatio="xMidYMid meet" class="chart-svg">${inner}</svg>`;}
function path(pts){return pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(2)+' '+p[1].toFixed(2)).join(' ')}
function ticksY(min,max,formatter=(v)=>v.toFixed(2)){let out='';for(let i=0;i<5;i++){let y=T+(H-T-B)*i/4,v=max-(max-min)*i/4;out+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="${grid}"/><text x="${L-11}" y="${y+4}" class="tick" text-anchor="end">${esc(formatter(v))}</text>`}return out;}
function ticksX(labels,count=5){let out='';for(let i=0;i<count;i++){let idx=Math.round((labels.length-1)*i/(count-1)),x=L+(W-L-R)*idx/Math.max(1,labels.length-1);out+=`<text x="${x}" y="${H-11}" text-anchor="middle" class="tick">${esc(labels[idx])}</text>`;}return out;}
function axes(min,max,labels,format){return ticksY(min,max,format)+ticksX(labels)+`<line x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" stroke="#a8b5c1"/>`;}
function lineChart(data,keys,opt={}){
 if(!data?.length)return `<div class="viz-empty">暂无该统计</div>`;
 const valid=data.flatMap(d=>keys.map(k=>d[k]).filter(v=>v!==null&&Number.isFinite(Number(v))).map(Number));
 if(!valid.length)return `<div class="viz-empty">该预测期限缺少有效标签</div>`;
 const lo=Math.min(...valid), hi=Math.max(...valid),span=hi-lo||Math.max(.01,Math.abs(hi)*.1),pad=span*.2;
 const ymin=opt.zero?Math.min(0,lo-pad):lo-pad, ymax=opt.zero?Math.max(0,hi+pad):hi+pad;
 const y=v=>T+(H-T-B)*(ymax-v)/(ymax-ymin),x=i=>L+(W-L-R)*i/Math.max(1,data.length-1);
 let s=axes(ymin,ymax,data.map(d=>d.label??d.day??''),opt.yformat||((v)=>v.toFixed(opt.decimal??2)));
 const palette=opt.palette||[pos,blue,purple,neg];
 keys.forEach((k,j)=>{
  let run=[];const pieces=[];
  data.forEach((d,i)=>{const v=d[k];if(v!==null&&v!==undefined&&Number.isFinite(Number(v)))run.push([x(i),y(Number(v))]);else if(run.length){pieces.push(run);run=[];}});
  if(run.length)pieces.push(run);
  for(const part of pieces){if(part.length<2)continue;s+=`<path d="${path(part)}" fill="none" stroke="${palette[j%palette.length]}" stroke-width="${opt.width||2.4}" stroke-linejoin="round" stroke-linecap="round"/>`;}
 });return tag(s);
}
function histogram(bins,opt={}){
 if(!bins?.length)return `<div class="viz-empty">暂无分布统计</div>`;
 const values=bins.map(d=>Number(d.v)||0),max=Math.max(.001,...values)*1.18;
 const xmin=bins[0].x,xmax=bins[bins.length-1].x||xmin+1;
 const x=i=>L+(W-L-R)*i/bins.length,bw=(W-L-R)/bins.length,y=v=>T+(H-T-B)*(max-v)/max;
 let s=ticksY(0,max,v=>(v*100).toFixed(1)+'%');
 for(let j=0;j<=5;j++){let value=xmin+(xmax-xmin)*j/5, xx=L+(W-L-R)*j/5;s+=`<text x="${xx}" y="${H-11}" text-anchor="middle" class="tick">${Number(value).toPrecision(3)}</text>`;}
 const points=[];
 bins.forEach((b,i)=>{const v=Number(b.v)||0,xx=x(i)+1,hh=Math.max(0,(H-B)-y(v));s+=`<rect x="${xx}" y="${y(v)}" width="${Math.max(1,bw-2)}" height="${hh}" rx="1.5" fill="${opt.color||blue}" fill-opacity=".74"><title>值 ${val(b.x,3)} · 占比 ${(v*100).toFixed(2)}%</title></rect>`;points.push([xx+bw/2,y(v)]);});
 s+=`<path d="${path(points)}" stroke="${opt.color||blue}" stroke-width="2" fill="none"/>`;
 if(xmin<=0&&xmax>=0){const z=L+(W-L-R)*(0-xmin)/(xmax-xmin||1);s+=`<line x1="${z}" y1="${T}" x2="${z}" y2="${H-B}" stroke="#b59b7a" stroke-dasharray="4 4"/>`;}
 return tag(s);
}
function percentiles(items){
 const rows=(items||[]).filter(d=>d.p10!==null&&d.p90!==null);
 if(!rows.length)return `<div class="viz-empty">暂无逐日分位数</div>`;
 let vals=rows.flatMap(d=>[d.p10,d.p90]),mn=Math.min(...vals),mx=Math.max(...vals),pad=(mx-mn||.1)*.15;mn-=pad;mx+=pad;
 const x=i=>L+(W-L-R)*i/Math.max(1,rows.length-1),y=v=>T+(H-T-B)*(mx-v)/(mx-mn);
 let s=axes(mn,mx,rows.map(d=>d.day.slice(5)),v=>v.toFixed(3));
 const area=(lo,hi,color,op)=>{const all=[...rows.map((d,i)=>[x(i),y(d[hi])]),...rows.slice().reverse().map((d,j)=>[x(rows.length-1-j),y(d[lo])])];return `<polygon points="${all.map(p=>p.join(',')).join(' ')}" fill="${color}" fill-opacity="${op}"/>`;};
 s+=area('p10','p90',blue,.12)+area('p25','p75',blue,.30);
 s+=`<path d="${path(rows.map((d,i)=>[x(i),y(d.p50)]))}" fill="none" stroke="${pos}" stroke-width="2.7"/>`;
 return tag(s);
}
function heatmap(rows,bins){
 if(!rows?.length||!rows[0]?.length)return `<div class="viz-empty">暂无时序密度统计</div>`;
 const x0=74,y0=27,cw=(W-x0-20)/rows.length,ch=(H-y0-40)/rows[0].length;
 let s='';for(let i=0;i<rows.length;i++)for(let j=0;j<rows[i].length;j++){
  const v=rows[i][j]??0,a=(.07+Math.min(1,v)*.90).toFixed(2),x=x0+i*cw,y=y0+(rows[i].length-1-j)*ch;
  s+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(1,cw+.25).toFixed(2)}" height="${Math.max(1,ch+.25).toFixed(2)}" fill="${pos}" fill-opacity="${a}"><title>日序 ${i+1} · Bin ${j+1}: ${(v*100).toFixed(2)}%</title></rect>`;
 }
 for(let i=0;i<5;i++){let xx=x0+i*(W-x0-20)/4;s+=`<text x="${xx}" y="${H-11}" class="tick" text-anchor="middle">D${1+Math.round(i*(rows.length-1)/4)}</text>`;}
 if(bins?.length){for(let j=0;j<5;j++){const i=Math.min(bins.length-1,Math.round((bins.length-1)*(1-j/4)));s+=`<text x="${x0-8}" y="${y0+12+j*(H-y0-40)/4}" class="tick" text-anchor="end">${Number(bins[i]).toPrecision(3)}</text>`;}}
 s+=`<text x="${W-19}" y="13" class="small-label" text-anchor="end">每日归一化分桶占比</text>`;return tag(s);
}
function icTimeline(days){
 const data=(days||[]).filter(d=>d.ic!==null&&Number.isFinite(Number(d.ic)));
 if(!data.length)return `<div class="viz-empty">暂无有效 IC</div>`;
 const scale=Math.max(.015,...data.map(d=>Math.abs(d.ic)))*1.18,x=i=>L+(W-L-R)*i/data.length,y=v=>T+(H-T-B)*(scale-v)/(2*scale),width=(W-L-R)/data.length;
 let s=ticksY(-scale,scale,v=>v.toFixed(3))+ticksX(data.map(d=>d.day.slice(5)),Math.min(6,data.length));
 s+=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}" stroke="#9aa8b5" stroke-width="1.3"/>`;
 data.forEach((d,i)=>{let yy=y(d.ic),base=y(0),top=Math.min(yy,base),hh=Math.max(1,Math.abs(yy-base));s+=`<rect x="${x(i)+2}" y="${top}" width="${Math.max(2,width-3)}" height="${hh}" fill="${d.ic>=0?pos:neg}" fill-opacity=".78"><title>${esc(d.day)} ${esc(d.split||'')} IC ${val(d.ic,5)}</title></rect>`;});
 let avg=data.map((d,i)=>data.slice(Math.max(0,i-6),i+1).reduce((s,x)=>s+x.ic,0)/(i-Math.max(0,i-6)+1));
 s+=`<path d="${path(avg.map((v,i)=>[x(i)+width/2,y(v)]))}" fill="none" stroke="#283c59" stroke-width="2.7"/>`;
 return tag(s);
}
function groups(rows){
 if(!rows?.length)return `<div class="viz-empty">暂无分组收益</div>`;
 const values=rows.map(d=>d.bps).filter(v=>v!==null&&Number.isFinite(Number(v))).map(Number);
 if(!values.length)return `<div class="viz-empty">该期限无有效收益标签</div>`;
 const min=Math.min(0,...values)-Math.max(.01,(Math.max(...values)-Math.min(...values))*.1),max=Math.max(0,...values)+Math.max(.01,(Math.max(...values)-Math.min(...values))*.1);
 const x=i=>L+(W-L-R)*(i+.5)/rows.length,y=v=>T+(H-T-B)*(max-v)/(max-min);let s=ticksY(min,max,v=>v.toFixed(3));
 rows.forEach((d,i)=>{const xx=x(i)-(W-L-R)/rows.length*.32;if(d.bps!=null){const yy=y(Number(d.bps)),top=Math.min(y(0),yy),hh=Math.max(1,Math.abs(yy-y(0)));s+=`<rect x="${xx}" y="${top}" width="${Math.min(80,(W-L-R)/rows.length*.64)}" height="${hh}" rx="3" fill="${d.bps>=0?pos:neg}" fill-opacity=".85"><title>${esc(d.group)}: ${Number(d.bps).toFixed(5)} bps, N=${d.count}</title></rect><text x="${x(i)}" y="${top-7}" text-anchor="middle" class="value-label">${d.bps>0?'+':''}${Number(d.bps).toFixed(3)}</text>`;}s+=`<text x="${x(i)}" y="${H-11}" text-anchor="middle" class="tick">${esc(d.group)}</text>`;});
 s+=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}" stroke="#a5b5bf"/>`;
 return tag(s);
}
function dualSeries(data){let x=i=>L+(W-L-R)*i/(data.length-1),m=1.3,y=v=>T+(H-T-B)*(m-v)/(2*m);let s=ticksY(-m,m,v=>v.toFixed(1))+ticksX(data.map(d=>(d.t*0.1).toFixed(0)+'s'),6);s+=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}" stroke="#b4c0c9" stroke-dasharray="4 4"/>`;
 const series=[['factor',pos],['label',neg],['past','#a5b2c2']];series.forEach(([k,color],j)=>{s+=`<path d="${path(data.map((d,i)=>[x(i),y(d[k])]))}" stroke="${color}" stroke-width="${j===0?2.4:1.9}" opacity="${j===2?.62:.9}" fill="none"/>`;});return tag(s);}
function multiHorizon(factors,getMetrics){const hor=[100,500,1000,5000],data=hor.map((h,i)=>({label:['100ms','500ms','1s','5s'][i],...Object.fromEntries(factors.map((f,j)=>['s'+j,getMetrics(f).hor[i].ic]))}));return lineChart(data,factors.map((f,i)=>'s'+i),{palette:[pos,blue,purple,neg],zero:true,decimal:3});}
function matrix(factors,corr){const count=factors.length,width=160, x0=210,y0=13, sz=Math.min(101,(W-x0-20)/count),height=y0+count*sz+37;let s='';for(let i=0;i<count;i++){s+=`<text x="${x0-10}" y="${y0+sz*(i+.58)}" class="matrix-label" text-anchor="end">${esc(factors[i].id.slice(0,23))}</text>`;for(let j=0;j<count;j++){const v=corr(factors[i],factors[j]);const a=v==null?0:Math.abs(v),color=v==null?'#8896a4':v>=0?pos:neg;s+=`<rect x="${x0+j*sz}" y="${y0+i*sz}" width="${sz-4}" height="${sz-4}" rx="4" fill="${color}" fill-opacity="${(.1+.76*a).toFixed(2)}"><title>${esc(factors[i].id)} / ${esc(factors[j].id)}: ${v==null?'暂无共同样本':v.toFixed(3)}</title></rect><text x="${x0+j*sz+(sz-4)/2}" y="${y0+i*sz+(sz-4)/2+4}" text-anchor="middle" fill="${a>.5?'#fff':'#284356'}" font-size="12" font-weight="700">${v==null?'—':v.toFixed(2)}</text>`;}}
 return tag(s,height);}
function miniLine(values,color=pos){let W0=98,H0=31,min=Math.min(...values),max=Math.max(...values),rng=max-min||1,p=values.map((v,i)=>[i*(W0-2)/(values.length-1)+1,H0-3-(v-min)/rng*(H0-8)]);return `<svg class="spark" viewBox="0 0 ${W0} ${H0}" aria-hidden="true"><path d="${path(p)}" stroke="${color}" stroke-width="2" fill="none" stroke-linecap="round"/></svg>`;}
global.HFTCharts={histogram,percentiles,heatmap,icTimeline,groups,dualSeries,multiHorizon,matrix,miniLine,lineChart};
})(window);
