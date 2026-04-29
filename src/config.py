import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data paths
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = DATA_DIR / "eval"

# Subdirectories
OPINIONS_RAW_DIR = RAW_DIR / "opinions"
UPLOADS_RAW_DIR = RAW_DIR / "uploads"
TEXT_DIR = PROCESSED_DIR / "text"
METADATA_DIR = PROCESSED_DIR / "metadata"

# CourtListener API
CL_BASE_URL = "https://www.courtlistener.com/api/rest/v4"
CL_API_TOKEN = os.environ.get("COURTLISTENER_TOKEN", "")

# Default fetch settings
CL_DEFAULT_PAGE_SIZE = 20
CL_MAX_PAGES = 5  # safety cap per fetch run
CL_MAX_RETRIES = 3
CL_REQUEST_TIMEOUT = 30  # seconds

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "10"))
LLM_TIMEOUT = 60

# HTTP API settings (consumed by app/api.py + app/cli.py 'serve')
API_HOST = os.environ.get("PARACITE_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("PARACITE_API_PORT", "8000"))

# Ensure directories exist
for d in [OPINIONS_RAW_DIR, UPLOADS_RAW_DIR, TEXT_DIR, METADATA_DIR, INDEX_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
