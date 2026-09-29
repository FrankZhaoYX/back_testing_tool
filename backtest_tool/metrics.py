"""Performance statistics computed from a daily net-return series (compounded)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def summarize(returns: pd.Series, turnover: pd.Series | None = None) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {}
    years = len(r) / TRADING_DAYS
    total = float((1 + r).prod() - 1)
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    downside = r[r < 0].std(ddof=1) * np.sqrt(TRADING_DAYS) if (r < 0).sum() > 1 else np.nan
    mdd = max_drawdown(r)
    stats = {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if vol > 0 else np.nan,
        "sortino": float(r.mean() * TRADING_DAYS / downside) if downside and downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "win_rate": float((r > 0).mean()),
    }
    if turnover is not None:
        stats["ann_turnover"] = float(turnover.mean() * TRADING_DAYS)
    return stats
