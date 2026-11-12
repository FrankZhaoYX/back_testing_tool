"""Dollar-cost averaging: drip a lump sum into one asset in equal monthly instalments."""

from __future__ import annotations

import pandas as pd
from qlib.data import D

from .engine import BacktestResult, _clamp_end
from .metrics import summarize


def dca_backtest(
    symbol: str,
    start: str,
    end: str,
    months: int = 12,
    park: str | None = None,
    benchmark: str | None = None,
    capital: float = 100_000,
    cost_bps: float = 5.0,
) -> BacktestResult:
    """Invest ``capital / months`` into ``symbol`` at the close of the first trading day of each month.

    Money not yet invested sits in ``park`` (e.g. a short-bond ETF) or in cash at 0%
    when ``park`` is None. Fractional shares are allowed; ``cost_bps`` is charged per side.
    The report has the same columns the journal reads from a Qlib backtest.
    """
    end = _clamp_end(end)
    names = [s for s in dict.fromkeys([symbol, park, benchmark]) if s]
    close = D.features(names, ["$close"], start_time=start, end_time=end, freq="day")["$close"]
    close = close.unstack(level="instrument").sort_index().ffill().astype("float64")
    cost = cost_bps / 10_000
    instalment = capital / months
    buy_days = set(close.index.to_series().groupby(close.index.to_period("M")).min().iloc[:months])

    shares = park_shares = cash = 0.0
    rows = []
    prev_account = capital
    for i, day in enumerate(close.index):
        price = close.at[day, symbol]
        park_price = close.at[day, park] if park else None
        traded = fees = 0.0
        if i == 0:
            cash = capital
            if park:  # park everything, the first instalment is bought out of it below
                fees += cash * cost
                park_shares, traded, cash = cash * (1 - cost) / park_price, cash, 0.0
        if day in buy_days:
            if park:
                amount = min(instalment, park_shares * park_price)
                if day == max(buy_days):
                    amount = park_shares * park_price  # last instalment: move whatever is left
                park_shares -= amount / park_price
                fees += amount * cost
                traded += amount
                amount *= 1 - cost
            else:
                amount = cash if day == max(buy_days) else min(instalment, cash)
                cash -= amount
            fees += amount * cost
            shares += amount * (1 - cost) / price
            traded += amount
        value = shares * price + (park_shares * park_price if park else 0.0)
        account = value + cash
        rows.append(
            {
                "account": account,
                "return": (account + fees) / prev_account - 1,
                "cost": fees / prev_account,
                "turnover": traded / prev_account,
                "value": value,
                "cash": cash,
                "bench": 0.0,
            }
        )
        prev_account = account
    report = pd.DataFrame(rows, index=close.index)
    bench = benchmark or symbol
    report["bench"] = close[bench].pct_change().fillna(0.0)
    weights = pd.DataFrame({symbol: [1.0 / months]}, index=[close.index[0]])
    result = BacktestResult(name=f"dca_{months}m", report=report, weights=weights)
    result.stats = summarize(result.net_returns, report["turnover"])
    result.bench_stats = summarize(report["bench"])
    return result
