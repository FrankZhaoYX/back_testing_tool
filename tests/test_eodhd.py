import pandas as pd
import pytest

from backtest_tool import eodhd


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


ROWS = [
    {"date": "2024-01-03", "open": 2, "high": 2, "low": 2, "close": 2, "adjusted_close": 2, "volume": 20},
    {"date": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "adjusted_close": 1, "volume": 10},
]


def test_ticker_suffix():
    assert eodhd.to_eodhd_ticker("SPY") == "SPY.US"
    assert eodhd.to_eodhd_ticker("VOD.LSE") == "VOD.LSE"


def test_fetch_eod_builds_request_and_sorts():
    session = FakeSession([FakeResponse(ROWS)])
    df = eodhd.fetch_eod("SPY", "2024-01-01", "2024-01-31", api_key="k", session=session)
    url, params = session.calls[0]
    assert url.endswith("/eod/SPY.US")
    assert params["api_token"] == "k" and params["from"] == "2024-01-01" and params["to"] == "2024-01-31"
    assert list(df.columns) == eodhd.CSV_COLUMNS
    assert df["date"].is_monotonic_increasing and df["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_fetch_eod_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(eodhd.time, "sleep", lambda s: None)
    session = FakeSession([FakeResponse([], status=429), FakeResponse(ROWS)])
    assert len(eodhd.fetch_eod("SPY", api_key="k", session=session)) == 2


def test_fetch_eod_empty_raises():
    with pytest.raises(ValueError):
        eodhd.fetch_eod("NOPE", api_key="k", session=FakeSession([FakeResponse([])]))


def test_missing_key(monkeypatch):
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        eodhd.get_api_key()
