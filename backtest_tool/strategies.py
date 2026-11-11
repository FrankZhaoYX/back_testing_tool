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


def _base_weights(symbols: list[str], weights: dict[str, float]) -> pd.Series:
    unknown = set(weights) - set(symbols)
    if unknown:
        raise KeyError(f"weights given for symbols not in the universe: {sorted(unknown)}")
    base = pd.Series(weights, dtype=float).reindex(symbols).fillna(0.0)
    if base.sum() > 1 + 1e-9 or (base < 0).any():
        raise ValueError("base weights must be non-negative and sum to at most 1")
    return base


def fixed_weight(symbols, start, end, weights: dict[str, float], rebalance: str = "Q", **_) -> pd.DataFrame:
    """Strategic allocation: rebalance back to fixed ``weights`` at the end of each period."""
    base = _base_weights(symbols, weights)
    close = load_fields(symbols, {"close": "$close"}, start, end)["close"].dropna(how="all")
    dates = _rebalance_dates(close.index, rebalance)
    return pd.DataFrame([base.to_numpy()] * len(dates), index=dates, columns=symbols)


def alpha_tilt(
    symbols,
    start,
    end,
    weights: dict[str, float],
    rebalance: str = "M",
    mom_windows: tuple[int, ...] = (21, 63, 126, 252),
    tilt: float = 0.5,
    trend_window: int | None = 200,
    trend_cut: float = 1.0,
    defensive: str | None = None,
    inv_vol_window: int | None = None,
    **_,
) -> pd.DataFrame:
    """Strategic weights tilted by alpha signals, decided at each period end.

    1. Momentum tilt: score = average of trailing returns over ``mom_windows``;
       the cross-sectional z-score (clipped to +-1.5) scales each base weight by
       ``1 + tilt * z / 1.5``, i.e. within ``[1 - tilt, 1 + tilt]``. The risky
       sleeve is then renormalised so the tilt only moves money *between* assets.
    2. Optional risk scaling: multiply by inverse ``inv_vol_window``-day volatility
       before renormalising (lower-vol assets get more capital).
    3. Trend filter: an asset below its ``trend_window``-day SMA loses ``trend_cut``
       of its weight, which moves to ``defensive`` (or stays in cash if None).

    Assets with a zero base weight, and the defensive asset, are never tilted.
    """
    base = _base_weights(symbols, weights)
    risky = [s for s in symbols if base[s] > 0 and s != defensive]
    fields = {"close": "$close"}
    fields.update({f"mom{w}": f"$close / Ref($close, {w}) - 1" for w in mom_windows})
    if trend_window:
        fields["sma"] = f"Mean($close, {trend_window})"
    if inv_vol_window:
        fields["vol"] = f"Std($close / Ref($close, 1) - 1, {inv_vol_window})"
    data = load_fields(symbols, fields, start, end)
    close = data["close"].dropna(how="all")
    dates = _rebalance_dates(close.index, rebalance)

    score = sum(data[f"mom{w}"] for w in mom_windows) / len(mom_windows)
    score = score.loc[dates, risky]
    z = score.sub(score.mean(axis=1), axis=0).div(score.std(axis=1, ddof=0).replace(0, 1), axis=0)
    mult = 1 + tilt * z.clip(-1.5, 1.5).fillna(0) / 1.5
    if inv_vol_window:
        mult = mult / data["vol"].loc[dates, risky]
    risky_budget = base[risky].sum()
    tilted = mult.mul(base[risky], axis=1)
    tilted = tilted.div(tilted.sum(axis=1), axis=0) * risky_budget

    out = pd.DataFrame(0.0, index=dates, columns=symbols)
    out[risky] = tilted
    if defensive:
        out[defensive] = base[defensive]
    if trend_window:
        below = (close.loc[dates, risky] < data["sma"].loc[dates, risky]).astype(float)
        freed = out[risky] * below * trend_cut
        out[risky] = out[risky] - freed
        if defensive:
            out[defensive] = out[defensive] + freed.sum(axis=1)
    return out.dropna(how="any")


def take_profit(
    symbols,
    start,
    end,
    weights: dict[str, float],
    take_profit: float = 0.2,
    reentry_drop: float = 0.1,
    max_wait: int = 126,
    defensive: str | None = None,
    rebalance: str = "Q",
    **_,
) -> pd.DataFrame:
    """Fixed weights with a per-asset take-profit rule.

    An asset is sold once its close is ``take_profit`` above its entry price; the
    weight moves to ``defensive`` (or cash). It is bought back when the price falls
    ``reentry_drop`` below the sale price, or after ``max_wait`` trading days.
    Rows are emitted on every state change and at each ``rebalance`` period end.
    """
    base = _base_weights(symbols, weights)
    risky = [s for s in symbols if base[s] > 0 and s != defensive]
    close = load_fields(symbols, {"close": "$close"}, start, end)["close"].dropna(how="all")
    held = pd.DataFrame(True, index=close.index, columns=risky)
    for sym in risky:
        prices = close[sym].to_numpy()
        in_position, ref, waited = True, None, 0
        for i, price in enumerate(prices):
            if price != price:  # not listed yet
                held.iat[i, held.columns.get_loc(sym)] = False
                continue
            if ref is None:
                ref = price
            if in_position and price >= ref * (1 + take_profit):
                in_position, ref, waited = False, price, 0
            elif not in_position:
                waited += 1
                if price <= ref * (1 - reentry_drop) or waited >= max_wait:
                    in_position, ref = True, price
            held.iat[i, held.columns.get_loc(sym)] = in_position
    out = pd.DataFrame(0.0, index=close.index, columns=symbols)
    out[risky] = held.astype(float).mul(base[risky], axis=1)
    if defensive:
        out[defensive] = base[defensive] + (base[risky].sum() - out[risky].sum(axis=1))
    changed = keep_changes(out).index
    return out.loc[changed.union(_rebalance_dates(close.index, rebalance))]


STRATEGIES: dict[str, Callable[..., pd.DataFrame]] = {
    "buy_and_hold": buy_and_hold,
    "equal_weight": equal_weight,
    "sma_trend": sma_trend,
    "momentum": momentum_rotation,
    "fixed_weight": fixed_weight,
    "alpha_tilt": alpha_tilt,
    "take_profit": take_profit,
}
