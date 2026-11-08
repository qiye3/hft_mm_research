"""SQLite registry. Database is small; large arrays and chart results live outside it."""
import sqlite3, json, datetime, hashlib
from contextlib import contextmanager
from .config import DATABASE, DATA_DIR, ensure_dirs
from .seed import GENS, seed_factor_definitions

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

@contextmanager
def connect():
    ensure_dirs()
    db=sqlite3.connect(DATABASE, timeout=30)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA busy_timeout=30000')
    db.execute('PRAGMA journal_mode=WAL')
    try:
        yield db
        db.commit()
    finally: db.close()

FACTOR_COLUMNS = ('id','gen','category','desc','base_name','base_columns','base_op','operation','short_window','long_window','post','raw_fields','status','unit')

def init_db():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS gens(id TEXT PRIMARY KEY,title TEXT NOT NULL,family TEXT,hypothesis TEXT,status TEXT);
        CREATE TABLE IF NOT EXISTS app_state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS factors(
           id TEXT PRIMARY KEY, gen TEXT NOT NULL,category TEXT NOT NULL,desc TEXT,
           base_name TEXT NOT NULL,base_columns TEXT NOT NULL,base_op TEXT NOT NULL,
           operation TEXT NOT NULL,short_window INTEGER,long_window INTEGER,post TEXT NOT NULL,
           raw_fields TEXT NOT NULL,status TEXT NOT NULL,unit TEXT,
           notes TEXT NOT NULL DEFAULT '',updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs(
           id TEXT PRIMARY KEY,kind TEXT NOT NULL,factor_id TEXT,config TEXT NOT NULL,
           status TEXT NOT NULL,progress REAL NOT NULL DEFAULT 0,
           error TEXT,result_path TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluations(
           factor_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
           result_path TEXT NOT NULL,summary TEXT NOT NULL,updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS jobs_latest ON jobs(created_at DESC);
        ''')
        for g in GENS:
            db.execute('INSERT OR IGNORE INTO gens(id,title,family,hypothesis,status) VALUES(?,?,?,?,?)', tuple(g[k] for k in ('id','title','family','hypothesis','status')))
        for f in seed_factor_definitions():
            x=dict(f);x['base_columns']=json.dumps(x['base_columns']);x['raw_fields']=json.dumps(x['raw_fields']);x['updated_at']=now()
            fields=list(FACTOR_COLUMNS)+['updated_at']
            db.execute('INSERT OR IGNORE INTO factors('+','.join(fields)+') VALUES('+','.join('?' for _ in fields)+')', [x.get(k) for k in fields])
        # Never present cached statistics for a different data source or a changed dataset.
        source_parts=[str(DATA_DIR)]
        if DATA_DIR.exists():
            source_parts.extend(f'{p.relative_to(DATA_DIR)}:{p.stat().st_size}:{p.stat().st_mtime_ns}'
              for p in sorted(DATA_DIR.rglob('*')) if p.is_file() and p.suffix.lower() in ('.parquet','.csv'))
        signature=hashlib.sha256('|'.join(source_parts).encode()).hexdigest()
        previous=db.execute("SELECT value FROM app_state WHERE key='dataset_signature'").fetchone()
        if previous and previous['value']!=signature:
            db.execute('DELETE FROM evaluations')
        db.execute("INSERT INTO app_state(key,value) VALUES('dataset_signature',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(signature,))
        # A process killed in-flight cannot claim a completed evaluation.
        db.execute("UPDATE jobs SET status='interrupted',updated_at=? WHERE status='running'",(now(),))

def row_factor(r):
    if not r:return None
    d=dict(r)
    for col in ('base_columns','raw_fields'):d[col]=json.loads(d[col])
    return d

def list_factors():
    with connect() as db:
        vals=[row_factor(r) for r in db.execute('SELECT * FROM factors ORDER BY gen,id')]
        sums={r['factor_id']:json.loads(r['summary']) for r in db.execute('SELECT factor_id,summary FROM evaluations')}
        for x in vals:
            x['summary']=sums.get(x['id'])
        return vals

def get_factor(id):
    with connect() as db:return row_factor(db.execute('SELECT * FROM factors WHERE id=?',(id,)).fetchone())

def insert_factor(f):
    obj=dict(f);obj['base_columns']=json.dumps(obj['base_columns']);obj['raw_fields']=json.dumps(obj.get('raw_fields',[]));obj['updated_at']=now()
    fields=list(FACTOR_COLUMNS)+['updated_at']
    with connect() as db:
        if obj['gen']!='FEATURES':
            db.execute('INSERT OR IGNORE INTO gens(id,title,family,hypothesis,status) VALUES(?,?,?,?,?)', (obj['gen'],obj['gen'],'','','已设计'))
        db.execute('INSERT INTO factors('+','.join(fields)+') VALUES('+','.join('?' for _ in fields)+')', [obj.get(k) for k in fields])

def edit_factor(id,**updates):
    allowed={'status','notes','desc'}
    args={k:v for k,v in updates.items() if k in allowed and v is not None}
    if not args:return
    args['updated_at']=now()
    with connect() as db: db.execute('UPDATE factors SET '+','.join(k+'=?' for k in args)+' WHERE id=?',[*args.values(),id])

def list_gens():
    with connect() as db:return [dict(r) for r in db.execute('SELECT * FROM gens ORDER BY id')]

def get_job(id):
    with connect() as db:
        r=db.execute('SELECT * FROM jobs WHERE id=?',(id,)).fetchone()
        if not r:return None
        x=dict(r);x['config']=json.loads(x['config']);return x

def list_jobs(limit=100):
    with connect() as db:
        a=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,))]
        for x in a:x['config']=json.loads(x['config'])
        return a

def create_job(id,factor_id,config):
    t=now()
    with connect() as db:
        db.execute('INSERT INTO jobs(id,kind,factor_id,config,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',(id,'evaluation',factor_id,json.dumps(config), 'queued',t,t))

def update_job(id,**k):
    allowed={'status','progress','error','result_path'}
    args={a:b for a,b in k.items() if a in allowed}
    args['updated_at']=now()
    with connect() as db: db.execute('UPDATE jobs SET '+','.join(i+'=?' for i in args)+' WHERE id=?',[*args.values(),id])

def put_evaluation(factor_id,job_id,result_path,summary):
    with connect() as db:
        db.execute('INSERT INTO evaluations(factor_id,job_id,result_path,summary,updated_at) VALUES(?,?,?,?,?) ON CONFLICT(factor_id) DO UPDATE SET job_id=excluded.job_id,result_path=excluded.result_path,summary=excluded.summary,updated_at=excluded.updated_at',(factor_id,job_id,str(result_path),json.dumps(summary,allow_nan=False),now()))

def evaluation_record(factor_id):
    with connect() as db:
        r=db.execute('SELECT * FROM evaluations WHERE factor_id=?',(factor_id,)).fetchone()
        return dict(r) if r else None
