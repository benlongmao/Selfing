"""
Thin entrypoint: configure paths, optional .env, then run uvicorn on backend.app:app.

All HTTP routes, static /ui, lifespan, and background loops live in backend/app.py.
Use manage_services.sh or `uvicorn backend.app:app` directly if you prefer.
"""
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PYTHONPATH", PROJECT_ROOT)

# Optional: load .env from repo root (does not override variables already set in the shell)
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass

# Default DB only when still unset (after .env)
if not os.environ.get("DB_PATH"):
    os.environ["DB_PATH"] = os.path.join(PROJECT_ROOT, "data.db")

# Apply DEEPSEEK_FORCE_IPV4 etc. before any HTTP client imports
try:
    from backend.net_prefs import apply_net_prefs

    apply_net_prefs()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import uvicorn

host = os.environ.get("BACKEND_HOST", "0.0.0.0")
port = int(os.environ.get("BACKEND_PORT", "8080"))
log_level = os.environ.get("UVICORN_LOG_LEVEL", "info").lower()

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host=host, port=port, log_level=log_level)
