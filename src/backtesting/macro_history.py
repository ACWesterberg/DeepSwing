from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from src.backtesting.history import asof_close, load_series_history

logger = logging.getLogger(__name__)

_SYMBOLS = {
    "vix": "^VIX",
    "spx": "^GSPC",
    "omx": "^OMX",
    "tnx": "^TNX",
    "usdsek": "SEK=X",
}


class MacroHistory:
    """Point-in-time macro context reconstructed from index/FX daily histories.
    All values are as of the previous close, so no same-day look-ahead."""

    def __init__(self, start: date, end: date):
        self._series: dict[str, Optional[pd.DataFrame]] = {
            key: load_series_history(symbol, start, end)
            for key, symbol in _SYMBOLS.items()
        }

    def vix(self, day: date) -> Optional[float]:
        return asof_close(self._series["vix"], day, strictly_before=True)

    def context(self, market: str, day: date) -> str:
        lines = ["Macro snapshot (as of previous close):"]

        vix = self.vix(day)
        if vix is not None:
            lines.append(f"- VIX: {vix:.1f} ({_vix_label(vix)})")

        index_key, index_name = ("omx", "OMXS30") if market == "nordic" else ("spx", "S&P 500")
        index_line = self._index_line(index_key, index_name, day)
        if index_line:
            lines.append(index_line)
        if market == "nordic":
            spx_line = self._index_line("spx", "S&P 500", day)
            if spx_line:
                lines.append(spx_line)

        tnx = asof_close(self._series["tnx"], day, strictly_before=True)
        if tnx is not None:
            lines.append(f"- US 10Y yield: {tnx / 10:.2f}%")

        usdsek = asof_close(self._series["usdsek"], day, strictly_before=True)
        if usdsek is not None:
            lines.append(f"- USD/SEK: {usdsek:.2f}")

        if len(lines) == 1:
            return "No macro data available."
        return "\n".join(lines)

    def _index_line(self, key: str, name: str, day: date) -> Optional[str]:
        df = self._series.get(key)
        if df is None or df.empty:
            return None
        sliced = df[df.index.date < day]
        if len(sliced) < 51:
            return None
        closes = sliced["Close"]
        level = float(closes.iloc[-1])
        chg_20 = (level / float(closes.iloc[-21]) - 1.0) * 100
        ma_50 = float(closes.iloc[-50:].mean())
        rel = "above" if level > ma_50 else "below"
        return f"- {name}: {level:,.0f} | 20-session change: {chg_20:+.1f}% | {rel} 50-day MA"


def _vix_label(vix: float) -> str:
    if vix < 15:
        return "calm"
    if vix < 25:
        return "normal"
    if vix < 35:
        return "elevated"
    return "extreme"
