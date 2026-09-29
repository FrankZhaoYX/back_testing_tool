"""Download daily OHLCV bars from the EODHD end-of-day API and cache them as CSV."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

EOD_URL = "https://eodhd.com/api/eod/{ticker}"
CSV_COLUMNS = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]


def get_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("EODHD_API_KEY")
    if not key:
        raise RuntimeError("EODHD API key missing: pass --api-key or set EODHD_API_KEY")
    return key


def to_eodhd_ticker(symbol: str, exchange: str = "US") -> str:
    """'SPY' -> 'SPY.US'; symbols that already carry an exchange suffix are kept."""
    return symbol if "." in symbol else f"{symbol}.{exchange}"


def fetch_eod(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    exchange: str = "US",
    session: requests.Session | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily bars for one symbol. Returns a frame with CSV_COLUMNS, sorted by date."""
    params = {"api_token": get_api_key(api_key), "fmt": "json", "period": "d", "order": "a"}
    if start:
        params["from"] = start
    if end:
        params["to"] = end
    http = session or requests.Session()
    url = EOD_URL.format(ticker=to_eodhd_ticker(symbol, exchange))

    for attempt in range(retries):
        resp = http.get(url, params=params, timeout=30)
        if resp.status_code == 429 and attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))
            continue
        resp.raise_for_status()
        break
    rows = resp.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"EODHD returned no data for {symbol}: {rows!r}")

    df = pd.DataFrame(rows)
    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"EODHD response for {symbol} is missing columns {sorted(missing)}")
    df = df[CSV_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def download_universe(
    symbols: list[str],
    csv_dir: str | Path,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    exchange: str = "US",
) -> dict[str, Path]:
    """Download every symbol to ``<csv_dir>/<SYMBOL>.csv``; returns the written paths."""
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    key = get_api_key(api_key)
    written = {}
    with requests.Session() as session:
        for symbol in symbols:
            df = fetch_eod(symbol, start, end, api_key=key, exchange=exchange, session=session)
            path = csv_dir / f"{symbol.upper()}.csv"
            df.to_csv(path, index=False, date_format="%Y-%m-%d")
            written[symbol.upper()] = path
            print(f"{symbol}: {len(df)} bars {df['date'].iloc[0]:%Y-%m-%d} -> {df['date'].iloc[-1]:%Y-%m-%d}")
    return written
