from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub heavy AI/ML deps that aren't installed in the test environment.
# These stubs must be injected into sys.modules BEFORE any project module
# that imports them is loaded — conftest.py runs first.
# ---------------------------------------------------------------------------

def _stub(name: str) -> MagicMock:
    mod = MagicMock(name=name)
    mod.__spec__ = None
    return mod


for _mod in [
    # AI / LLM clients
    "dspy",
    "dspy.predict",
    "dspy.teleprompt",
    "anthropic",
    "openai",
    # Scheduler / server (not needed for unit/integration tests)
    "apscheduler",
    "apscheduler.schedulers",
    "apscheduler.schedulers.background",
    "uvicorn",
    "fastapi",
    "fastapi.responses",
    "fastapi.staticfiles",
    "fastapi.templating",
    "starlette",
    "starlette.requests",
    "starlette.websockets",
    # Feed / news parsing
    "feedparser",
    "newsapi",
    "newsapi.newsapi_client",
    # financedata shared library (installed on Pi, absent in CI)
    "financedata",
    "financedata.live",
    "financedata.fx",
    "financedata.cache",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = _stub(_mod)
