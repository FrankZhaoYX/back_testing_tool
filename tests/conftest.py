import numpy as np
import pandas as pd
import pytest

from backtest_tool import DEFAULT_UNIVERSE
from backtest_tool.qlib_dump import dump_to_qlib


def make_bars(symbol: str, start="2018-01-02", end="2023-12-29", seed=0, split_on=None) -> pd.DataFrame:
    """Random-walk OHLCV in EODHD's CSV shape. `split_on` injects a 2:1 split on that date."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    adj = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, len(dates))))
    raw_close = adj.copy()
    if split_on is not None:
        raw_close[dates < pd.Timestamp(split_on)] *= 2  # pre-split raw prices were double
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": raw_close * 0.999,
        "high": raw_close * 1.01,
        "low": raw_close * 0.99,
        "close": raw_close,
        "adjusted_close": adj,
        "volume": rng.integers(1_000_000, 5_000_000, len(dates)),
    })
    return df


@pytest.fixture(scope="session")
def qlib_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    csv_dir = root / "csv"
    csv_dir.mkdir()
    for i, sym in enumerate(DEFAULT_UNIVERSE):
        split = "2021-06-01" if sym == "TQQQ" else None
        make_bars(sym, seed=i, split_on=split).to_csv(csv_dir / f"{sym}.csv", index=False)
    out = root / "qlib"
    dump_to_qlib(csv_dir, out)
    from backtest_tool.engine import init_qlib

    init_qlib(out)
    return out
