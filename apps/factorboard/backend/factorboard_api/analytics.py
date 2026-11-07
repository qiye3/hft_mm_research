"""Offline evaluation on registered features/factors. No synthetic metrics, no use of test labels for bin selection.

A file contains one UTC date of 100-ms sampled rows with timestamp_ns, features, and ideally
future y_mid_{h}ms_bps columns. If only mid_price exists, targets are derived using exact
100-ms time spacing; never interpolate across gaps or segment changes.
"""
from __future__ import annotations
import json, re, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from .config import DATA_DIR

HORIZONS=(100,500,1000,5000)
TIMESTAMPS=('timestamp_ns','ts_local_ns','local_ts_ns','sample_ts_local_ns')
LABEL_CANDIDATES={
 100:('y_mid_100ms_bps','target_100ms_bps','label_100ms_bps'),
 500:('y_mid_500ms_bps','target_500ms_bps','label_500ms_bps'),
 1000:('y_mid_1s_bps','target_1s_bps','label_1s_bps','future_return_1s_bps','y_1s_bps'),
 5000:('y_mid_5s_bps','y_mid_5000ms_bps','target_5s_bps','label_5s_bps'),
}


def discover():
    """The configured directory must point at one homogeneous daily dataset version."""
    paths=[]
    if DATA_DIR.exists():
        paths=[p for p in DATA_DIR.rglob('*') if p.is_file() and p.suffix.lower() in ('.parquet','.csv') and not p.name.startswith('.')]
    out={}
    for p in sorted(paths):
        dates=re.findall(r'20\d{2}-[01]\d-[0-3]\d', str(p.relative_to(DATA_DIR)))
        if dates:
            date=dates[-1]
            if date in out:
                raise ValueError(f'同一日期存在多个文件 ({date})；FACTORBOARD_DATA_DIR 应指向单个数据集版本目录')
            out[date]=p
    return out


def sample_dates():return sorted(discover().keys())


def read_daily(path:Path, columns:list[str]|None=None):
    """Projection: only requested columns reach memory; Polars is preferred for Parquet."""
    if path.suffix.lower()=='.csv':
        if columns:
            h=pd.read_csv(path,nrows=0).columns
            return pd.read_csv(path,usecols=[c for c in columns if c in h])
        return pd.read_csv(path)
    try:
        import polars as pl
    except ImportError:
        try:
            return pd.read_parquet(path,columns=columns)
        except ImportError as exc:
            raise RuntimeError('Parquet 读取需安装 polars（或 pyarrow）；参照 requirements-research.txt') from exc
    lazy=pl.scan_parquet(str(path))
    available=set(lazy.collect_schema().names())
    wanted=[c for c in columns if c in available] if columns else list(available)
    if not wanted:return pd.DataFrame()
    # Avoid polars->pandas requiring pyarrow in lightweight deployments.
    return pd.DataFrame(lazy.select(wanted).collect(engine='streaming').to_dict(as_series=False))


def _group_segments(data,category):
    ts=data['timestamp_ns'].to_numpy(dtype=np.int64)
    reset=np.r_[True,np.diff(ts)!=100_000_000]
    segs=['book_segment_id'] if category=='book' else ['trade_segment_id'] if category=='trade' else ['book_segment_id','trade_segment_id']
    for field in segs:
        if field in data:
            old=data[field].to_numpy()
            reset[1:]|= old[1:]!=old[:-1]
    # Rows with invalid input break window continuity rather than becoming implicit zeros.
    flags=['book_feature_valid','book_valid'] if category=='book' else ['trade_feature_valid','trade_valid'] if category=='trade' else ['book_feature_valid','trade_feature_valid','book_valid','trade_valid']
    for field in flags:
        if field in data:reset|=~data[field].fillna(False).to_numpy(dtype=bool)
    return np.cumsum(reset)


def factor_series(data,definition):
    """Compute from pre-existing 100ms sampled features, not from next-period labels."""
    columns=definition['base_columns']
    missing=[c for c in columns if c not in data]
    if missing: raise ValueError(f"{definition['id']}: 缺少输入字段 {', '.join(missing)}")
    op=definition['base_op']
    vals=[pd.to_numeric(data[c],errors='coerce') for c in columns]
    if op=='identity':base=vals[0]
    elif op=='difference':base=vals[0]-vals[1]
    elif op=='ratio':
        den=vals[1].replace(0,np.nan);base=vals[0]/den
    else:raise ValueError(f'不支持的里层运算 {op}')
    base=base.replace([np.inf,-np.inf],np.nan).astype('float64')
    quality_flags=['book_feature_valid','book_valid'] if definition['category']=='book' else ['trade_feature_valid','trade_valid'] if definition['category']=='trade' else ['book_feature_valid','trade_feature_valid','book_valid','trade_valid']
    for flag in quality_flags:
        if flag in data:base=base.where(data[flag].fillna(False).to_numpy(dtype=bool))
    segment=_group_segments(data,definition['category'])
    base=base.where(np.isfinite(base))
    valid=base.notna()
    # Every invalid row creates a new segment so rolling never skips unknown data.
    segment=np.cumsum(np.r_[True, (np.diff(segment)!=0) | (~valid.to_numpy()[1:]) | (~valid.to_numpy()[:-1])])
    outer=definition['operation']; w=definition.get('short_window'); long=definition.get('long_window')
    if outer=='identity':out=base
    else:
        if not w or w<1:raise ValueError('滚动窗口必须为正整数')
        short=base.groupby(segment,sort=False).transform(lambda s:s.rolling(int(w),min_periods=int(w)).mean())
        if outer=='mean':out=short
        elif outer=='mean_diff':
            if not long or long<=w:raise ValueError('long_window 必须大于 short_window')
            long_value=base.groupby(segment,sort=False).transform(lambda s:s.rolling(int(long),min_periods=int(long)).mean())
            out=short-long_value
        else:raise ValueError(f'不支持的外层运算 {outer}')
    return out.where(np.isfinite(out)).astype('float64')


def labels(data,h):
    for key in LABEL_CANDIDATES[h]:
        if key in data:
            vals=pd.to_numeric(data[key],errors='coerce').to_numpy(dtype=np.float64)
            for flag in ('label_valid','target_valid', f'label_{h}ms_valid'):
                if flag in data: vals=np.where(data[flag].fillna(False).to_numpy(dtype=bool),vals,np.nan)
            return np.where(np.isfinite(vals), vals, np.nan),key
    if 'mid_price' not in data:return np.full(len(data),np.nan),None
    step=h//100; ts=data['timestamp_ns'].to_numpy(dtype=np.int64)
    mid=pd.to_numeric(data['mid_price'],errors='coerce').to_numpy(dtype=float)
    fut=np.r_[mid[step:],np.full(step,np.nan)]
    future_ts=np.r_[ts[step:],np.full(step,-1,dtype=np.int64)]
    good=(future_ts-ts==h*1_000_000)&np.isfinite(mid)&(mid>0)&np.isfinite(fut)
    for col in ('book_segment_id','book_valid'):
        if col in data:
            if col=='book_segment_id':
                a=data[col].to_numpy(); future_seg=np.r_[a[step:],np.full(step,None,dtype=object)]
                good&=(a==future_seg)
            else:good&=data[col].fillna(False).to_numpy(dtype=bool)
    y=np.where(good,(fut/mid-1)*10000,np.nan)
    return y,'derived_mid_price'


def daily_eval_frame(path,definition,horizons=HORIZONS):
    keys=['timestamp_ns',*TIMESTAMPS,'mid_price','return_1s_bps','book_segment_id','trade_segment_id',
          'book_feature_valid','trade_feature_valid','book_valid','trade_valid','label_valid','target_valid',*[f'label_{h}ms_valid' for h in horizons],*definition['base_columns']]
    for h in horizons:keys.extend(LABEL_CANDIDATES[h])
    df=read_daily(path,list(dict.fromkeys(keys)))
    if len(df)==0:raise ValueError(f'{path.name}: 文件没有可用列')
    time_col=next((k for k in TIMESTAMPS if k in df),None)
    if not time_col:raise ValueError(f'{path.name}: 缺少 timestamp_ns/ts_local_ns')
    df['timestamp_ns']=pd.to_numeric(df[time_col],errors='coerce').astype('int64')
    # Check monotonic, no silently re-order raw rows or ignore duplicates.
    if df['timestamp_ns'].duplicated().any():raise ValueError(f'{path.name}: 采样时间戳重复，请先处理数据质量')
    if not df['timestamp_ns'].is_monotonic_increasing:raise ValueError(f'{path.name}: 时间戳未严格递增')
    factor=factor_series(df,definition)
    outs={h:labels(df,h) for h in horizons}
    return df['timestamp_ns'].to_numpy(dtype=np.int64),factor.to_numpy(dtype=float),outs,(pd.to_numeric(df['return_1s_bps'],errors='coerce').to_numpy(dtype=float) if 'return_1s_bps' in df else np.full(len(df),np.nan))


def date_split(dates):
    # All selections are chronological. For a single date, report in-sample exploration only.
    n=len(dates)
    if n==0:raise ValueError('未找到可评估的日频数据')
    if n==1:return {dates[0]:'train'}
    if n==2:return {dates[0]:'train',dates[1]:'test'}
    if n<5:return {d:('train' if i<n-1 else 'test') for i,d in enumerate(dates)}
    ntrain=max(2,int(n*.6));nval=max(1,int(n*.2));
    if ntrain+nval>=n:nval=n-ntrain-1
    return {d:('train' if i<ntrain else 'valid' if i<ntrain+nval else 'test') for i,d in enumerate(dates)}


def finite_float(x):
    try:
        v=float(x)
        return round(v,8) if math.isfinite(v) else None
    except(TypeError,ValueError):return None

def pearson(x,y):
    if len(x)<6 or np.std(x)<1e-14 or np.std(y)<1e-14:return None
    return finite_float(np.corrcoef(x,y)[0,1])

def daily_ic(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    x,y=x[mask],y[mask]
    p=pearson(x,y)
    # Spearman on evenly spaced sample; for >200k rows use a fixed-stride subsample
    # and explicitly document that rank IC is sampled, not full-population rank IC.
    stride=max(1,math.ceil(len(x)/200_000))
    xs=pd.Series(x[::stride]).rank().to_numpy();ys=pd.Series(y[::stride]).rank().to_numpy()
    return p,pearson(xs,ys),int(len(x)),stride


def _quantiles(x):
    return np.percentile(x,[10,25,50,75,90]).tolist() if len(x) else [None]*5


def evaluation(definition,paths,progress=None):
    if not paths:raise ValueError('没有日期文件，请将 FACTORBOARD_DATA_DIR 指向已有研究表目录')
    dates=sorted(paths)
    split=date_split(dates)
    # Pass 1. Train-only bin thresholds, with bounded deterministic sample per day.
    sample=[]
    for i,date in enumerate(dates):
        if split[date]!='train':continue
        ts,f,ys,past=daily_eval_frame(paths[date],definition,[1000])
        idx=np.where(np.isfinite(f)&np.isfinite(ys[1000][0]))[0]
        if len(idx):sample.append(f[idx[::max(1,math.ceil(len(idx)/100_000))]])
        if progress:progress(min(.22,(i+1)/len(dates)*.2))
    if not sample:raise ValueError('训练日期没有合法的因子/1s标签配对样本')
    train_values=np.concatenate(sample)
    train_thresholds=np.quantile(train_values,[.2,.4,.6,.8]).tolist()
    unique_ties=len(set(round(x,10) for x in train_thresholds))<4
    method='sign' if unique_ties and np.min(train_values)<0<np.max(train_values) else 'quintile'
    if method=='sign':group_ids=['Neg','Zero','Pos']
    else:group_ids=['G1','G2','G3','G4','G5']
    qlow,qhigh=np.quantile(train_values,[.005,.995])
    if qhigh-qlow<1e-10:qlow,qhigh=float(np.min(train_values))-.5,float(np.max(train_values))+.5
    edges=np.linspace(qlow,qhigh,37)
    hist_counts=np.zeros(36,dtype=np.float64)
    density=[]; pct=[]; days_rows=[]; daily_by_h=[]; groups_by_h=defaultdict(lambda:defaultdict(list))
    sample_series=[]; total_valid=0;total_rows=0;total_zeros=0
    support={h:False for h in HORIZONS}
    label_origins={str(h):set() for h in HORIZONS}
    for i,date in enumerate(dates):
        ts,f,targets,past=daily_eval_frame(paths[date],definition)
        total_rows+=len(f);mask=np.isfinite(f);total_valid+=int(mask.sum());total_zeros+=int(np.count_nonzero(f[mask]==0))
        valid=f[mask]
        p10,p25,p50,p75,p90=_quantiles(valid)
        pct.append({'day':date,'p10':finite_float(p10),'p25':finite_float(p25),'p50':finite_float(p50),'p75':finite_float(p75),'p90':finite_float(p90),'zero':finite_float(np.mean(valid==0)) if len(valid) else None,'valid':finite_float(mask.mean()),'split':split[date]})
        counts=np.histogram(valid,bins=edges)[0].astype(float)
        density.append((counts/counts.sum()).tolist() if counts.sum()>0 else [0]*36)
        if split[date]=='test' or (len(dates)==1 and split[date]=='train'):
            hist_counts+=counts
        if method=='sign':groups=np.where(f<0,0,np.where(f>0,2,1))
        else:groups=np.searchsorted(train_thresholds,f,'right')
        for horizon in HORIZONS:
            y,origin=targets[horizon];support[horizon]=bool(support[horizon] or (origin is not None and np.isfinite(y).any()))
            if origin:label_origins[str(horizon)].add(origin)
            aligned=mask&np.isfinite(y)
            ic,rankic,count,stride=daily_ic(f[aligned],y[aligned])
            row={'day':date,'h':horizon,'ic':ic,'rankic':rankic,'count':count,'rank_stride':stride,'split':split[date]}
            daily_by_h.append(row)
            if horizon==1000:days_rows.append(row)
            if split[date]=='test' or (len(dates)==1 and split[date]=='train'):
                for gi,g in enumerate(group_ids):
                    gmask=aligned&(groups==gi)
                    n=int(gmask.sum());value=finite_float(np.mean(y[gmask])) if n else None
                    groups_by_h[horizon][g].append({'day':date,'bps':value,'n':n})
        if not sample_series and split[date]=='test':
            y=targets[1000][0]
            idx=np.where(mask&np.isfinite(y))[0]
            sample_series=[{'t':j,'ts':int(ts[k]),'factor':finite_float(f[k]),'label':finite_float(y[k]),'past':finite_float(past[k])} for j,k in enumerate(idx[:2400])]
        if progress:progress(.22+.72*(i+1)/len(dates))
    if not sample_series:
        ts,f,tars,past=daily_eval_frame(paths[dates[-1]],definition)
        valid=np.where(np.isfinite(f)&np.isfinite(tars[1000][0]))[0]
        sample_series=[{'t':j,'ts':int(ts[k]),'factor':finite_float(f[k]),'label':finite_float(tars[1000][0][k]),'past':finite_float(past[k])} for j,k in enumerate(valid[:2400])]
    hor=[];grouped={}
    for h in HORIZONS:
        rows=[r for r in daily_by_h if r['h']==h and (r['split']=='test' or len(dates)==1)]
        ics=[r['ic'] for r in rows if r['ic'] is not None]
        group_rows=[]
        for g in group_ids:
            allr=groups_by_h[h][g];values=[x['bps'] for x in allr if x['bps'] is not None]
            group_rows.append({'group':g,'bps':finite_float(np.mean(values)) if values else None,'count':sum(x['n'] for x in allr)})
        grouped[str(h)]=group_rows
        hi=group_rows[-1]['bps'];lo=group_rows[0]['bps']
        spread=finite_float(hi-lo) if hi is not None and lo is not None else None
        hor.append({'h':h,'ic':finite_float(np.mean(ics)) if ics else None,'bps':spread,'n_days':len(ics),'label_available':support[h]})
    testdays=[r for r in days_rows if r['split']=='test' or len(dates)==1]
    ics=[r['ic'] for r in testdays if r['ic'] is not None]
    rankics=[r['rankic'] for r in testdays if r['rankic'] is not None]
    icmean=float(np.mean(ics)) if ics else None
    icstd=float(np.std(ics,ddof=1)) if len(ics)>1 else None
    norm=hist_counts.sum() or 1
    hist=[{'x':finite_float((edges[i]+edges[i+1])/2),'v':finite_float(hist_counts[i]/norm)} for i in range(len(edges)-1)]
    return {
      'factor_id':definition['id'],'data_kind':'historical_computation','schema_version':1,'dates':dates,'split':split,
      'factor_definition':{k:definition.get(k) for k in ('id','gen','base_columns','base_op','operation','short_window','long_window','post','unit')},
      'data_manifest':[{'date':d,'file':str(p.relative_to(DATA_DIR)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for d,p in sorted(paths.items())],
      'clock_spec':'dataset.timestamp_ns (100ms grid), no future feature values',
      'scope':'test' if len(dates)>1 else 'train_exploration','group_method':method,
      'group_thresholds':train_thresholds,'hist':hist,'pct':pct,'densities':density,
      'days':days_rows,'daily_by_horizon':daily_by_h,'hor':hor,'groups':grouped['1000'], 'groups_by_horizon':grouped,
      'series':sample_series,'series_step_ms':100,'train_threshold_sample':int(len(train_values)), 'label_origins':{k:sorted(v) for k,v in label_origins.items()},
      'summary':{
         'meanIC':finite_float(icmean),'rank':finite_float(np.mean(rankics)) if rankics else None,
         'icir':finite_float(icmean/icstd) if icmean is not None and icstd and icstd>1e-12 else None,
         'positive':finite_float(np.mean(np.array(ics)>0)) if ics else None,
         'spread':next((x['bps'] for x in hor if x['h']==1000),None),
         'valid':finite_float(total_valid/total_rows) if total_rows else None,
         'zero':finite_float(total_zeros/total_valid) if total_valid else None,
         'n':len(ics),'row_count':int(total_rows),
         'label_status':{str(h):support[h] for h in HORIZONS},
         'sample':{'train':sum(v=='train' for v in split.values()),'valid':sum(v=='valid' for v in split.values()),'test':sum(v=='test' for v in split.values())}
      }
    }


def correlation_matrix(defs, paths, max_days=3, max_sample_per_day=60_000):
    """Pairwise Pearson on common valid timestamps. Never estimates correlation from displayed plot lines."""
    dates=sorted(paths)[-max_days:]
    corr=[]
    for d in dates:
        frames=[]
        # Read multiple features once/day using a projection; no longer than 3 days by default.
        cols=set(['timestamp_ns',*TIMESTAMPS,'book_segment_id','trade_segment_id','book_feature_valid','trade_feature_valid','book_valid','trade_valid'])
        for defn in defs:cols.update(defn['base_columns'])
        df=read_daily(paths[d],list(cols))
        tcol=next((k for k in TIMESTAMPS if k in df),None)
        if tcol is None:raise ValueError('找不到采样时间字段')
        df['timestamp_ns']=pd.to_numeric(df[tcol],errors='coerce').astype('int64')
        for definition in defs:frames.append(factor_series(df,definition))
        frame=pd.concat(frames,axis=1)
        frame.columns=[x['id'] for x in defs]
        idx=np.arange(0,len(frame),max(1,math.ceil(len(frame)/max_sample_per_day)))
        corr.append(frame.iloc[idx])
    combined=pd.concat(corr,ignore_index=True) if corr else pd.DataFrame()
    c=combined.corr(method='pearson',min_periods=20)
    return {'ids':[d['id'] for d in defs], 'values':[[finite_float(c.loc[a['id'],b['id']]) for b in defs] for a in defs], 'days':dates,'sample_size':len(combined)}
