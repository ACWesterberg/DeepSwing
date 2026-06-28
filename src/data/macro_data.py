from __future__ import annotations

import os

from config.settings import settings

os.environ.setdefault("FRED_API_KEY", settings.fred_api_key or "")

from financedata import get_macro_context  # noqa: F401, E402
