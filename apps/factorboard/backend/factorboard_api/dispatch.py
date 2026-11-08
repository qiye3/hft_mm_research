"""SQLite queue dispatcher; API remains responsive while CPU calculations run in worker processes."""
import os,sys,subprocess
from . import db
from .config import ROOT, STATE_DIR, MAX_JOBS


def dispatch():
    # Reserve worker slots in a write transaction. Multiple requests cannot claim same job.
    with db.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        count=conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
        rows=[r['id'] for r in conn.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT ?",(max(0,MAX_JOBS-count),))]
        for job_id in rows:
            conn.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?",(db.now(),job_id))
    logs=STATE_DIR/'logs';logs.mkdir(exist_ok=True,parents=True)
    for job_id in rows:
        env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'backend')+os.pathsep+env.get('PYTHONPATH','')
        try:
            with open(logs/f'{job_id}.log','ab') as stream:
                subprocess.Popen([sys.executable,'-m','factorboard_api.worker',job_id],cwd=str(ROOT),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        except Exception as exc:
            db.update_job(job_id,status='failed',error=f'worker spawn failed: {exc}')
    return rows
