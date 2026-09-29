"""Signal generators that turn Qlib market data into target portfolio weights.

Every strategy returns a DataFrame (index: trade date, columns: symbols) of
target weights **decided with data up to that date's close**. Only the dates
on which the portfolio should be rebalanced are kept; the engine executes each
row on the following trading day, so no look-ahead bias is introduced.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
from qlib.data import D


def load_fields(symbols: list[str], fields: dict[str, str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Evaluate Qlib expressions; returns {name: DataFrame[date x symbol]}."""
    raw = D.features(symbols, list(fields.values()), start_time=start, end_time=end, freq="day")
    raw.columns = list(fields.keys())
    return {name: raw[name].unstack(level="instrument").sort_index() for name in fields}


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last available trading date of each period ('W', 'M', 'Q'), plus the first date."""
    period = {"W": "W", "M": "M", "Q": "Q"}[freq.upper()]
    s = pd.Series(index, index=index)
    last = s.groupby(index.to_period(period)).max()
    return pd.DatetimeIndex([index[0]]).union(pd.DatetimeIndex(last.values))


def keep_changes(weights: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """Keep the first row and each row whose weights differ from the previous row."""
    diff = weights.fillna(0).diff().abs().max(axis=1)
    diff.iloc[0] = 1.0
    return weights[diff > tol]


def buy_and_hold(symbols, start, end, **_) -> pd.DataFrame:
    """Equal-weight on the first day, then let positions drift."""
    close = load_fields(symbols, {"close": "$close"}, start, end)["close"]
    first = close.dropna(how="all").index[0]
    return pd.DataFrame(1.0 / len(symbols), index=[first], columns=symbols)


def equal_weight(symbols, start, end, rebalance: str = "M", **_) -> pd.DataFrame:
    """Equal weight across all symbols, rebalanced at the end of each period."""
    close = load_fields(symbols, {"close": "$close"}, start, end)["close"].dropna(how="all")
    avail = close.notna()
    weights = avail.div(avail.sum(axis=1), axis=0)
    return weights.loc[_rebalance_dates(weights.index, rebalance)]


def sma_trend(symbols, start, end, window: int = 200, **_) -> pd.DataFrame:
    """Hold each symbol (1/N of capital) only while its close is above its N-day SMA."""
    data = load_fields(symbols, {"close": "$close", "sma": f"Mean($close, {window})"}, start, end)
    close, sma = data["close"], data["sma"]
    # Require a full window of history before the SMA is trusted.
    enough = close.notna().cumsum() >= window
    on = (close > sma) & enough
    weights = on.astype(float) / len(symbols)
    weights = weights[enough.any(axis=1)]
    return keep_changes(weights)


def momentum_rotation(symbols, start, end, lookback: int = 63, top_k: int = 1, rebalance: str = "M", **_) -> pd.DataFrame:
    """Each period hold the top-k symbols by trailing return (only if that return is positive)."""
    data = load_fields(symbols, {"mom": f"$close / Ref($close, {lookback}) - 1"}, start, end)
    mom = data["mom"].dropna(how="all")
    rank = mom.rank(axis=1, ascending=False, method="first")
    picked = (rank <= top_k) & (mom > 0)
    weights = picked.astype(float) / top_k
    return weights.loc[_rebalance_dates(weights.index, rebalance)]


STRATEGIES: dict[str, Callable[..., pd.DataFrame]] = {
    "buy_and_hold": buy_and_hold,
    "equal_weight": equal_weight,
    "sma_trend": sma_trend,
    "momentum": momentum_rotation,
}
