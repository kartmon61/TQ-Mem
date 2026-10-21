"""Load `.env` from the project root once, at process start. No dependency (python-dotenv not required).

Rules:
- Only KEY=VALUE lines; `#` comments and blank lines ignored; surrounding quotes stripped.
- Existing process environment wins (a value already exported is never overwritten).
- If the file supplies AWS_ACCESS_KEY_ID, AWS_PROFILE is dropped so boto3 uses the static keys (not a stale profile / session token).
Credentials are therefore read once from the file by the credential chain and never appear in a command line.
"""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_loaded = False

def load_env(path: str | None = None) -> dict:
    global _loaded
    if _loaded and path is None:
        return {}
    p = path or os.path.join(ROOT, ".env")
    loaded = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v; loaded[k] = v
        if "AWS_ACCESS_KEY_ID" in loaded or "AWS_ACCESS_KEY_ID" in os.environ:
            os.environ.pop("AWS_PROFILE", None); os.environ.pop("AWS_DEFAULT_PROFILE", None)
    # mem0 phones home to PostHog by default; behind the corporate SSL interception that only produces "[PostHog] error uploading" noise
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    _loaded = True
    return loaded
