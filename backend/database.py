import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = os.environ.get("DB_PATH", "data.db")
if not os.path.isabs(DB_PATH):
    # Resolve relative DB_PATH against repo root (parent of backend/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(BASE_DIR, DB_PATH)

def get_db():
    # WAL + busy_timeout reduce "database is locked" under UI polling and background writers
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    try:
        conn.execute("PRAGMA busy_timeout=8000")
    except Exception:
        pass
    return conn

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

