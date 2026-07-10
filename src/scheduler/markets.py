from __future__ import annotations

from typing import Literal

MarketType = Literal["nordic", "eu", "us"]
SCAN_MARKETS: tuple[MarketType, ...] = ("nordic", "eu", "us")
