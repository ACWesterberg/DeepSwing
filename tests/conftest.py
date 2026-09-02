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
    "starlette.middleware",
    "starlette.middleware.base",
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


# ---------------------------------------------------------------------------
# Process-global scan state must not leak between tests. The PASS memo is keyed
# (track, ticker), so one test's cached PASS would silently answer the next
# test's decision for the same ticker.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_scan_caches():
    from src.agent import news_analyzer
    from src.scheduler import scan_loop

    scan_loop._pass_memo.clear()
    news_analyzer._SUMMARY_CACHE.clear()
    yield
    scan_loop._pass_memo.clear()
    news_analyzer._SUMMARY_CACHE.clear()
