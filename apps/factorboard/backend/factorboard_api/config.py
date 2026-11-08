"""Local server configuration; keep source data outside the repository by default."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv('FACTORBOARD_DATA_DIR', str(ROOT / 'workspace/data/market_groups_v2'))).expanduser().resolve()
STATE_DIR = Path(os.getenv('FACTORBOARD_STATE_DIR', str(ROOT / 'workspace'))).expanduser().resolve()
DATABASE = STATE_DIR / 'factorboard.sqlite3'
EVALUATIONS = STATE_DIR / 'evaluations'
FRONTEND = ROOT / 'frontend'
MAX_JOBS = max(1, int(os.getenv('FACTORBOARD_MAX_JOBS', '2')))
READ_ONLY = os.getenv('FACTORBOARD_READ_ONLY', '').lower() in ('1', 'true', 'yes')


def ensure_dirs():
    for p in (STATE_DIR, EVALUATIONS):
        p.mkdir(parents=True, exist_ok=True)
