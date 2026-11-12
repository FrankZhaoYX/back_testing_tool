"""Entry-timing research for accumulating shares of one ETF.

Every variant receives the *same* contributions (the price of one share on the
first trading day of each month) and never sells, so the only thing that differs
is *when* the money is put to work. Final wealth is therefore directly comparable.

A signal is a boolean Series decided at the close of day ``t``; the purchase is
filled at the close of day ``t + 1`` (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

Signal = Callable[[pd.DataFrame], pd.Series]


def load_bars(csv_path: str | Path) -> pd.DataFrame:
    """Raw OHLC plus the cash dividend per share implied by the adjusted close."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    factor = df["adjusted_close"] / df["close"]
    dividend = df["close"].shift(1) * (1 - factor.shift(1) / factor)
    df["dividend"] = dividend.where(dividend > 0.05, 0.0).fillna(0.0)
    return df


# ---------------------------------------------------------------- signals


def monthly(df: pd.DataFrame) -> pd.Series:
    """Baseline: last trading day of each month, so the buy fills on the first day of the next."""
    idx = df.index
    last = idx.to_series().groupby(idx.to_period("M")).max()
    return pd.Series(idx.isin(last.values), index=idx)


def down_streak(n: int) -> Signal:
    """Close lower than the previous close for ``n`` days in a row."""
    return lambda df: (df["close"].diff() < 0).rolling(n).sum() == n


def below_ma(window: int, pct: float = 0.0) -> Signal:
    """Close at least ``pct`` below its ``window``-day moving average."""
    return lambda df: df["close"] < df["close"].rolling(window).mean() * (1 - pct)


def above_ma(window: int) -> Signal:
    return lambda df: df["close"] > df["close"].rolling(window).mean()


def drawdown(pct: float, window: int = 252) -> Signal:
    """Close at least ``pct`` below the highest close of the last ``window`` days."""
    return lambda df: df["close"] <= df["close"].rolling(window).max() * (1 - pct)


def rsi_below(period: int, level: float) -> Signal:
    def signal(df: pd.DataFrame) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        return rsi < level

    return signal


def bollinger_low(window: int = 20, width: float = 2.0) -> Signal:
    def signal(df: pd.DataFrame) -> pd.Series:
        ma, sd = df["close"].rolling(window).mean(), df["close"].rolling(window).std()
        return df["close"] < ma - width * sd

    return signal


def day_drop(pct: float) -> Signal:
    """Single-day fall of at least ``pct``."""
    return lambda df: df["close"].pct_change() <= -pct


def all_of(*signals: Signal) -> Signal:
    return lambda df: pd.concat([s(df) for s in signals], axis=1).all(axis=1)


def any_of(*signals: Signal) -> Signal:
    return lambda df: pd.concat([s(df) for s in signals], axis=1).any(axis=1)


# ---------------------------------------------------------------- simulation


@dataclass
class EntryResult:
    trades: pd.DataFrame  # date, price, shares
    equity: pd.DataFrame  # daily shares, cash, value
    stats: dict


def simulate(
    df: pd.DataFrame,
    signal: pd.Series,
    start: str,
    end: str,
    max_shares: int = 2,
    month_end_sweep: bool = False,
    slippage_bps: float = 5.0,
    cash_rate: float = 0.0,
    advance_signal: pd.Series | None = None,
    advance_shares: int = 0,
) -> EntryResult:
    """Accumulate shares: each signal buys up to ``max_shares`` whole shares with the cash on hand.

    Cash arrives on the first trading day of each month (one share's worth at that
    day's close) and from dividends. With ``month_end_sweep`` any cash still idle on
    the last trading day of a month is invested regardless of the signal.
    ``cash_rate`` is the annual interest earned on idle cash.

    ``advance_signal`` may spend future contributions early: when it fires, the cash
    balance is allowed to go negative by up to ``advance_shares`` shares' worth, which
    later contributions repay before normal buying resumes.
    """
    fire = signal.reindex(df.index).fillna(False).astype(bool).shift(1, fill_value=False)  # fill at t+1
    if advance_signal is None:
        advance = pd.Series(False, index=df.index)
    else:
        advance = advance_signal.reindex(df.index).fillna(False).astype(bool).shift(1, fill_value=False)
    window = df.loc[start:end]
    idx = window.index
    periods = idx.to_period("M")
    first_of_month = ~pd.Series(periods, index=idx).duplicated()
    last_of_month = ~pd.Series(periods, index=idx).duplicated(keep="last")
    slip = 1 + slippage_bps / 10_000

    shares, cash, contributed = 0, 0.0, 0.0
    trades, daily, flows = [], [], []
    for day in idx:
        price = float(window.at[day, "close"])
        cash *= 1 + cash_rate / 252
        cash += shares * float(window.at[day, "dividend"])
        if first_of_month[day]:
            cash += price
            contributed += price
            flows.append((day, price))
        cost = price * slip
        n = 0
        if fire[day]:
            n = min(max_shares, max(int(cash // cost), 0))
        if advance[day]:
            n = max(n, min(max_shares, max(int((cash + advance_shares * cost) // cost), 0)))
        if month_end_sweep and last_of_month[day] and day != idx[-1]:
            n = max(n, max(int(cash // cost), 0))
        if n > 0:
            shares += n
            cash -= n * cost
            trades.append({"date": day, "price": price, "shares": n})
        daily.append({"shares": shares, "cash": cash, "value": shares * price + cash})

    equity = pd.DataFrame(daily, index=idx)
    trade_df = pd.DataFrame(trades, columns=["date", "price", "shares"])
    final_value = float(equity["value"].iloc[-1])
    bought = int(trade_df["shares"].sum())
    invested = float((trade_df["price"] * trade_df["shares"]).sum())
    # what a "buy a little every day" investor paid in the same calendar months
    month_avg = window["close"].groupby(periods).transform("mean")
    vs_month = (trade_df["price"].to_numpy() / month_avg.reindex(trade_df["date"]).to_numpy() - 1) if bought else []
    stats = {
        "signal_days": int(fire.loc[idx].sum()),
        "buy_days": len(trade_df),
        "shares": bought,
        "contributed": round(contributed, 2),
        "avg_cost": round(invested / bought, 2) if bought else np.nan,
        "price_vs_month_avg": round(float(np.average(vs_month, weights=trade_df["shares"])), 4) if bought else np.nan,
        "avg_idle_cash_pct": round(float((equity["cash"] / equity["value"].clip(lower=1e-9)).mean()), 4),
        "min_cash": round(float(equity["cash"].min()), 2),
        "final_cash": round(float(equity["cash"].iloc[-1]), 2),
        "final_value": round(final_value, 2),
        "profit": round(final_value - contributed, 2),
        "irr": round(_irr(flows, idx[-1], final_value), 4),
    }
    return EntryResult(trade_df, equity, stats)


def _irr(flows: list[tuple[pd.Timestamp, float]], end: pd.Timestamp, final_value: float) -> float:
    """Annualised money-weighted return of the contributions (bisection on NPV)."""
    years = np.array([(end - day).days / 365.25 for day, _ in flows])
    amounts = np.array([amount for _, amount in flows])
    lo, hi = -0.99, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (amounts * (1 + mid) ** years).sum() > final_value:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def forward_returns(df: pd.DataFrame, signal: pd.Series, start: str, end: str, horizons=(21, 63, 252)) -> dict:
    """Event study: return from the fill (close of t+1) to ``h`` days later, vs. every day in the window."""
    close = df["close"]
    fire = signal.reindex(df.index).fillna(False).astype(bool).shift(1, fill_value=False).loc[start:end]
    out = {}
    for h in horizons:
        fwd = (close.shift(-h) / close - 1).loc[start:end].dropna()
        hit = fwd[fire.reindex(fwd.index)]
        out[f"fwd{h}_mean"] = round(float(hit.mean()), 4) if len(hit) else np.nan
        out[f"fwd{h}_win"] = round(float((hit > 0).mean()), 4) if len(hit) else np.nan
        out[f"fwd{h}_all_days"] = round(float(fwd.mean()), 4)
    return out
