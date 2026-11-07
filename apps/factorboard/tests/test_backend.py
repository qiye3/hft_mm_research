import os,sys, tempfile, json
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
TMP=Path(tempfile.mkdtemp(prefix='factorboard_test_'))
os.environ['FACTORBOARD_STATE_DIR']=str(TMP/'state')
os.environ['FACTORBOARD_DATA_DIR']=str(TMP/'daily')
sys.path.insert(0,str(HERE/'backend'))
sys.path.insert(0,str(HERE/'scripts'))
from make_demo_dataset import make
make(TMP/'daily',days=7,points=450)
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from factorboard_api import db
from factorboard_api.analytics import evaluation,discover,labels,factor_series,correlation_matrix,date_split
from factorboard_api.main import app
from factorboard_api.worker import run


def test_00_registry_and_data():
 db.init_db()
 assert len(db.list_factors())==28
 assert len(db.list_gens())==2
 assert len(discover())==7
 x=db.get_factor('book_obi_l5_mean_diff_10_50')
 assert x['base_columns']==['imbalance_5']
 assert x['operation']=='mean_diff' and x['short_window']==10 and x['long_window']==50


def test_01_rolling_continuity_and_label_edges():
 n=120
 data=pd.DataFrame({'timestamp_ns':np.arange(n)*100_000_000,'book_segment_id':[1]*70+[2]*50,'book_feature_valid':[True]*n,'imbalance_5':np.arange(n,dtype=float),'mid_price':100+np.arange(n,dtype=float)*.01})
 defs=db.get_factor('book_obi_l5_mean_10')
 a=factor_series(data,defs)
 assert np.isnan(a.iloc[8])
 assert a.iloc[9]==4.5
 assert np.isnan(a.iloc[70])
 assert a.iloc[79]==74.5
 b=factor_series(data,db.get_factor('book_obi_l5_mean_diff_10_50'))
 assert np.isnan(b.iloc[48]); assert abs(b.iloc[49] - (44.5-24.5)) <1e-9
 assert np.isnan(b.iloc[100]); assert abs(b.iloc[119] - (114.5-94.5)) <1e-9
 y,origin=labels(data,1000)
 assert origin=='derived_mid_price'
 assert np.isnan(y[69]) and np.isnan(y[-1])
 assert np.isfinite(y[10])
 # Data gaps are not implicitly forward-filled.
 data.loc[53:,'timestamp_ns']+=100_000_000
 y2,_=labels(data,100)
 assert np.isnan(y2[52])


def test_02_evaluation_and_splits():
 files=discover()
 result=evaluation(db.get_factor('book_obi_l5_mean_10'),files)
 assert result['summary']['n']==2
 assert result['scope']=='test'
 assert len(result['pct'])==7 and len(result['days'])==7
 assert all(len(x)==36 for x in result['densities'])
 assert result['summary']['label_status']['1000']
 assert result['groups'] and result['group_thresholds']
 assert sum(x['count'] for x in result['groups'])>0
 assert len(result['series'])>0
 assert all(x['split']=='test' for x in result['days'][-2:])


def test_03_api_no_fabricated_metrics_and_persistence(monkeypatch):
 db.init_db()
 with TestClient(app) as client:
  x=client.get('/api/v1/factors').json()
  assert x['total']==28
  f=next(f for f in x['items'] if f['id']=='book_obi_l5_mean_10')
  assert f['summary'] is None  # unless a worker actually completed
  assert client.get('/api/v1/evaluations/book_obi_l5_mean_10').status_code==404
  assert client.get('/api/v1/config').json()['mode']=='synthetic_fixture'
  assert client.get('/api/v1/health').json()['ok']
  updated=client.patch('/api/v1/factors/book_obi_l5_mean_10',json={'notes':'研究记录：验证时钟口径。','status':'观察'})
  assert updated.status_code==200 and updated.json()['notes'].startswith('研究记录')
  assert client.get('/api/v1/factors/book_obi_l5_mean_10').json()['status']=='观察'
  # Avoid worker auto-spawn; validate queue and synchronous worker execution.
  import factorboard_api.main as api_module
  monkeypatch.setattr(api_module,'dispatch',lambda:[])
  result=client.post('/api/v1/jobs/evaluate',json={'factor_ids':['book_obi_l5_mean_10']})
  assert result.status_code==202
  job=result.json()['jobs'][0]
  assert job['status']=='queued'
  import factorboard_api.worker as wk
  import factorboard_api.dispatch as dispatcher
  monkeypatch.setattr(dispatcher,'dispatch',lambda:[])
  worker_result=run(job['id'])
  assert worker_result['factor_id']=='book_obi_l5_mean_10'
  assert client.get('/api/v1/jobs/'+job['id']).json()['status']=='completed'
  ev=client.get('/api/v1/evaluations/book_obi_l5_mean_10')
  assert ev.status_code==200 and ev.json()['scope']=='test'
  after=client.get('/api/v1/factors').json()['items']
  item=next(f for f in after if f['id']=='book_obi_l5_mean_10')
  assert item['summary'] is not None and item['summary']['meanIC'] is not None
  corr=client.get('/api/v1/compare/correlation',params={'ids':'book_obi_l5_mean_10,book_obi_l1_mean_10'})
  assert corr.status_code==200
  assert corr.json()['values'][0][0]==1


def test_04_input_validation_and_security():
 with TestClient(app) as client:
  assert client.post('/api/v1/factors',json={'id':'x; rm -rf /','gen':'Gen002','category':'book','base_columns':['imbalance_5']}).status_code==422
  assert client.post('/api/v1/jobs/evaluate',json={'factor_ids':['unknown_factor']}).status_code==404
  assert client.get('/api/v1/compare/correlation',params={'ids':'foo'}).status_code==422
  assert client.get('/').status_code==200
  assert client.get('/js/app.js').status_code==200
