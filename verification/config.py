import os
from pathlib import Path

from dotenv import load_dotenv

MODEL_ID = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
MAX_QUERY_RESULTS = 5
SEARCH_BACKEND = "auto"
SOURCE_PRIORITY = ["sochfactcheck.com", "afp.com", "dawn.com"]
NO_EVIDENCE_CONFIDENCE = 0.3
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


def _find_env_file() -> Path | None:
    for directory in (Path.cwd(), Path(__file__).resolve().parent.parent):
        env_path = directory / ".env"
        if env_path.exists():
            return env_path
    return None


_env_path = _find_env_file()
if _env_path:
    load_dotenv(_env_path)


def get_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")
