from __future__ import annotations

from datetime import datetime

from src.scheduler.market_hours import is_exchange_open, is_market_open

# 2026 DST mismatch window: US springs forward Mar 8, EU not until Mar 29.
# During those weeks NYSE trades 14:30–21:00 CET, not the usual 15:30–22:00.
# All datetimes below are naive == UTC (market_hours localizes them as UTC).


class TestUSMarketHoursDST:
    def test_us_open_first_hour_during_dst_mismatch(self):
        # Mon 2026-03-16 13:45 UTC = 09:45 ET (open) = 14:45 CET
        # A fixed-CET window (15:15 open) would wrongly say closed.
        dt = datetime(2026, 3, 16, 13, 45)
        assert is_market_open("us", dt) is True

    def test_us_closed_after_et_close_during_dst_mismatch(self):
        # Mon 2026-03-16 20:30 UTC = 16:30 ET (closed) = 21:30 CET
        # A fixed-CET window (22:15 close) would wrongly say open.
        dt = datetime(2026, 3, 16, 20, 30)
        assert is_market_open("us", dt) is False

    def test_us_open_normal_aligned_period(self):
        # Mon 2026-06-15 14:00 UTC = 10:00 ET = 16:00 CEST — open under both
        dt = datetime(2026, 6, 15, 14, 0)
        assert is_market_open("us", dt) is True

    def test_us_closed_on_weekend(self):
        dt = datetime(2026, 6, 13, 14, 0)  # Saturday
        assert is_market_open("us", dt) is False

    def test_us_exchange_open_uses_et(self):
        # 13:45 UTC on 2026-03-16 = 09:45 ET → inside official 09:30–16:00
        dt = datetime(2026, 3, 16, 13, 45)
        assert is_exchange_open("us", dt) is True


class TestNordicMarketHours:
    def test_nordic_open_mid_session(self):
        dt = datetime(2026, 3, 16, 9, 0)  # 10:00 CET Monday
        assert is_market_open("nordic", dt) is True

    def test_nordic_closed_evening(self):
        dt = datetime(2026, 3, 16, 17, 30)  # 18:30 CET
        assert is_market_open("nordic", dt) is False

    def test_nordic_closed_on_weekend(self):
        dt = datetime(2026, 3, 14, 9, 0)  # Saturday
        assert is_market_open("nordic", dt) is False


class TestEuMarketHours:
    def test_eu_open_mid_session(self):
        dt = datetime(2026, 3, 16, 9, 0)  # 10:00 CET Monday
        assert is_market_open("eu", dt) is True

    def test_eu_closed_evening(self):
        dt = datetime(2026, 3, 16, 17, 30)  # 18:30 CET
        assert is_market_open("eu", dt) is False


class TestDailyEntryClock:
    def test_us_session_date_follows_new_york(self):
        from datetime import date, datetime
        from src.scheduler.market_hours import session_date
        # 02:00 UTC on the 15th is still the 14th in New York
        assert session_date("us", datetime(2026, 10, 15, 2, 0)) == date(2026, 10, 14)
        assert session_date("nordic", datetime(2026, 10, 15, 2, 0)) == date(2026, 10, 15)

    def test_exchange_open(self):
        from datetime import time
        from src.scheduler.market_hours import exchange_open
        assert exchange_open("us") == ("America/New_York", time(9, 30))
        assert exchange_open("nordic") == ("Europe/Stockholm", time(9, 0))

    def test_forming_bar_is_dropped(self, monkeypatch):
        from datetime import date
        import pandas as pd
        import src.scheduler.scan_loop as sl
        monkeypatch.setattr(sl, "session_date", lambda market: date(2026, 10, 15))
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]},
                          index=pd.to_datetime(["2026-10-13", "2026-10-14", "2026-10-15"]))
        assert list(sl._completed_bars(df, "nordic")["Close"]) == [1.0, 2.0]
