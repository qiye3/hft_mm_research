"""Persistent evaluation worker, launched by the API as a separate OS process."""
from __future__ import annotations
import sys, json, os
from pathlib import Path
from . import db
from .analytics import evaluation, discover
from .config import EVALUATIONS

def run(job_id):
    job=db.get_job(job_id)
    if not job:raise ValueError('找不到任务')
    def progress(x):db.update_job(job_id,progress=round(float(x),4))
    try:
        db.update_job(job_id,status='running',progress=0.01)
        factor=db.get_factor(job['factor_id'])
        if not factor:raise ValueError('因子不存在')
        files=discover()
        chosen=job['config'].get('dates')
        if chosen:
            files={d:p for d,p in files.items() if d in chosen}
        result=evaluation(factor,files,progress=progress)
        result['job_id']=job_id
        EVALUATIONS.mkdir(parents=True,exist_ok=True)
        path=EVALUATIONS/f'{factor["id"]}.json'
        tmp=EVALUATIONS/f'{factor["id"]}.{job_id}.tmp'
        tmp.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
        tmp.replace(path)
        db.put_evaluation(factor['id'],job_id,path,result['summary'])
        db.update_job(job_id,status='completed',progress=1,result_path=str(path))
        return result
    except Exception as exc:
        db.update_job(job_id,status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        from .dispatch import dispatch
        dispatch()

if __name__=='__main__':
    run(sys.argv[1])
