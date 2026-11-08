"""FactorBoard API with static v4 user-approved frontend."""
from __future__ import annotations
import os,json,uuid,hashlib,re,sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from . import db
from .config import FRONTEND, READ_ONLY, EVALUATIONS, DATA_DIR
from .analytics import sample_dates, discover, daily_eval_frame, correlation_matrix
from .dispatch import dispatch

@asynccontextmanager
async def lifespan(app):
    db.init_db()
    dispatch()
    yield

app=FastAPI(title='BTCUSDT FactorBoard',version='1.0.0',docs_url='/api/docs',openapi_url='/api/openapi.json',lifespan=lifespan)

@app.middleware('http')
async def token_auth(request:Request, call_next):
    token=os.environ.get('FACTORBOARD_API_KEY','')
    if token and request.url.path.startswith('/api/v1/'):
        from hmac import compare_digest
        if not compare_digest(request.headers.get('X-API-Key',''),token):
            return JSONResponse(status_code=401,content={'detail':'API Key 无效；请通过客户端连接设置填写。'})
    return await call_next(request)


def write_ok():
    if READ_ONLY:raise HTTPException(403,'服务器为只读模式')

class FactorCreate(BaseModel):
    id:str=Field(pattern=r'^[a-z][a-z0-9_]{2,140}$')
    gen:str=Field(min_length=1,max_length=35)
    category:str=Field(pattern=r'^[a-z_]{2,30}$')
    desc:str=''
    base_name:str=''
    base_columns:list[str]=Field(min_length=1,max_length=2)
    base_op:str='identity'
    operation:str='identity'
    short_window:int|None=None
    long_window:int|None=None
    post:str='raw'
    raw_fields:list[str]=Field(default_factory=list)
    status:str='候选'
    unit:str='ratio'
    @field_validator('base_columns')
    @classmethod
    def columns_valid(cls,values):
        if any(not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]*',x) for x in values):raise ValueError('输入列名只能是合法字段标识符')
        return values
    @model_validator(mode='after')
    def coherent(self):
        if self.base_op not in ('identity','difference','ratio'):raise ValueError('不支持的里层计算')
        if self.base_op=='identity' and len(self.base_columns)!=1:raise ValueError('identity 需要1个输入列')
        if self.base_op in ('difference','ratio') and len(self.base_columns)!=2:raise ValueError('difference/ratio 需要2列')
        if self.operation not in ('identity','mean','mean_diff'):raise ValueError('不支持的外层算子')
        if self.operation!='identity' and (not self.short_window or not 1<=self.short_window<=100000):raise ValueError('short_window 必须为正整数')
        if self.operation=='mean_diff' and (not self.long_window or self.long_window<=self.short_window or self.long_window>100000):raise ValueError('long_window 必须大于 short_window')
        if self.post!='raw':raise ValueError('首版仅支持 raw 后处理')
        return self

class FactorUpdate(BaseModel):
    status:str|None=None
    notes:str|None=None
    desc:str|None=None

class GenCreate(BaseModel):
    id:str=Field(pattern=r'^Gen\d{3,}$')
    title:str=Field(min_length=1,max_length=120)
    family:str=''
    hypothesis:str=''
    status:str='已设计'

class EvalJobInput(BaseModel):
    factor_ids:list[str]=Field(min_length=1,max_length=60)
    dates:list[str]|None=None

@app.get('/api/v1/health')
def health():return {'ok':True,'service':'factorboard','version':'1.0.0'}

@app.get('/api/v1/config')
def config():
    return {'name':'BTCUSDT FactorBoard','dataset_dates':sample_dates(),'mode':('synthetic_fixture' if (DATA_DIR/'synthetic_manifest.json').exists() else 'historical_files') if sample_dates() else 'no_data','horizons_ms':[100,500,1000,5000], 'sample_ms':100,'read_only':READ_ONLY}

@app.get('/api/v1/factors')
def factors(limit:int=Query(default=5000,ge=1,le=10000)):
    items=db.list_factors()
    return {'items':items[:limit],'total':len(items)}

@app.post('/api/v1/factors',status_code=201)
def factor_create(body:FactorCreate):
    write_ok()
    x=body.model_dump()
    x['base_name']=x['base_name'] or x['id']
    try:db.insert_factor(x)
    except sqlite3.IntegrityError:raise HTTPException(409,'因子名已存在')
    return db.get_factor(x['id'])

@app.get('/api/v1/factors/{factor_id}')
def factor_detail(factor_id:str):
    f=db.get_factor(factor_id)
    if not f:raise HTTPException(404,'因子不存在')
    return f

@app.patch('/api/v1/factors/{factor_id}')
def factor_update(factor_id:str,body:FactorUpdate):
    write_ok()
    if not db.get_factor(factor_id):raise HTTPException(404,'因子不存在')
    db.edit_factor(factor_id,**body.model_dump(exclude_unset=True))
    return db.get_factor(factor_id)

@app.get('/api/v1/gens')
def gens():return {'items':db.list_gens()}

@app.post('/api/v1/gens',status_code=201)
def gen_create(body:GenCreate):
    write_ok()
    try:
        with db.connect() as conn: conn.execute('INSERT INTO gens(id,title,family,hypothesis,status) VALUES(?,?,?,?,?)',tuple(body.model_dump()[k] for k in ('id','title','family','hypothesis','status')))
    except sqlite3.IntegrityError:raise HTTPException(409,'Gen 已存在')
    return body

@app.get('/api/v1/datasets')
def datasets():
    try:
        files=discover()
    except Exception as exc:raise HTTPException(422,str(exc))
    return {'dates':sorted(files),'files':[{'date':date,'file':str(p.name),'bytes':p.stat().st_size} for date,p in sorted(files.items())]}

@app.post('/api/v1/jobs/evaluate',status_code=202)
def submit_evaluations(body:EvalJobInput):
    write_ok()
    found={x['id'] for x in db.list_factors()}
    unknown=[x for x in body.factor_ids if x not in found]
    if unknown:raise HTTPException(404,f'未知因子：{unknown}')
    try:dates=sample_dates()
    except Exception as exc:raise HTTPException(422,str(exc))
    if not dates:raise HTTPException(422,'未找到研究数据。请配置 FACTORBOARD_DATA_DIR 指向日频 Parquet / CSV。')
    if body.dates:
        invalid=set(body.dates)-set(dates)
        if invalid:raise HTTPException(422,f'不存在的日期：{sorted(invalid)}')
    jobs=[]
    for fid in dict.fromkeys(body.factor_ids):
        job_id=uuid.uuid4().hex[:16]
        db.create_job(job_id,fid,{'dates':body.dates or []})
        jobs.append(db.get_job(job_id))
    dispatch()
    return {'jobs':jobs}

@app.get('/api/v1/jobs')
def jobs(limit:int=Query(default=100,ge=1,le=1000)):
    dispatch()
    return {'items':db.list_jobs(limit)}

@app.get('/api/v1/jobs/{job_id}')
def job_detail(job_id:str):
    j=db.get_job(job_id)
    if not j:raise HTTPException(404,'任务不存在')
    return j

@app.get('/api/v1/evaluations/{factor_id}')
def factor_evaluation(factor_id:str):
    rec=db.evaluation_record(factor_id)
    if not rec:raise HTTPException(404,'尚未评估')
    path=Path(rec['result_path'])
    if not path.is_file():raise HTTPException(410,'评估文件已丢失，需要重新计算')
    return JSONResponse(content=json.loads(path.read_text(encoding='utf-8')))

@app.get('/api/v1/evaluations/{factor_id}/series')
def factor_series_detail(factor_id:str,date:str,limit:int=Query(2400,ge=10,le=6000)):
    from .analytics import labels
    definition=db.get_factor(factor_id)
    if not definition:raise HTTPException(404,'因子不存在')
    try:
        files=discover()
        if date not in files:raise HTTPException(404,'日期不存在')
        ts,x,y,past=daily_eval_frame(files[date],definition,[1000])
    except HTTPException:raise
    except Exception as exc:raise HTTPException(422,str(exc))
    import numpy as np
    idx=np.where(np.isfinite(x)&np.isfinite(y[1000][0]))[0][:limit]
    return {'factor_id':factor_id,'date':date,'series':[{'t':int(i),'ts':int(ts[k]),'factor':float(x[k]),'label':float(y[1000][0][k]),'past':float(past[k]) if np.isfinite(past[k]) else None} for i,k in enumerate(idx)]}

@app.get('/api/v1/compare/correlation')
def compare_correlation(ids:str):
    names=[i for i in ids.split(',') if i]
    if len(names)<2 or len(names)>4 or len(set(names))!=len(names):raise HTTPException(422,'比较数量必须是2—4个不同因子')
    defs=[db.get_factor(n) for n in names]
    if any(x is None for x in defs):raise HTTPException(404,'包含未知因子')
    try: files=discover()
    except Exception as exc:raise HTTPException(422,str(exc))
    if not files:raise HTTPException(422,'无可用研究数据')
    key=hashlib.sha256(('|'.join(names)+'|'+','.join(sorted(files))).encode()).hexdigest()[:20]
    path=EVALUATIONS/f'corr_{key}.json'
    if path.is_file():return json.loads(path.read_text())
    try:result=correlation_matrix(defs,files)
    except Exception as exc:raise HTTPException(422,str(exc))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,allow_nan=False),encoding='utf8')
    return result

@app.get('/')
def home():return FileResponse(FRONTEND/'index.html')
app.mount('/css',StaticFiles(directory=str(FRONTEND/'css')),name='css')
app.mount('/js',StaticFiles(directory=str(FRONTEND/'js')),name='js')
