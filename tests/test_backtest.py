import numpy as np
import pandas as pd
import pytest

from backtest_tool import DEFAULT_UNIVERSE
from backtest_tool.engine import run_backtest
from backtest_tool.metrics import max_drawdown, summarize
from backtest_tool.strategies import STRATEGIES, keep_changes, sma_trend

START, END = "2020-01-02", "2022-12-30"


def test_metrics_basic():
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert summarize(r)["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)


def test_keep_changes():
    w = pd.DataFrame({"A": [1, 1, 0, 0], "B": [0, 0, 1, 1]}, index=pd.bdate_range("2024-01-01", periods=4))
    assert list(keep_changes(w).index) == [w.index[0], w.index[2]]


def test_sma_trend_weights_never_exceed_one(qlib_dir):
    w = sma_trend(DEFAULT_UNIVERSE, "2018-01-02", END)
    assert (w.sum(axis=1) <= 1 + 1e-9).all() and (w >= 0).all().all()


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_run(qlib_dir, name):
    res = run_backtest(name, DEFAULT_UNIVERSE, START, END)
    assert res.report.index[0] == pd.Timestamp(START)
    assert np.isfinite(res.stats["total_return"])
    assert res.report["value"].iloc[0] > 0  # invested from day one (warm-up signal carried over)


def test_buy_and_hold_tracks_constituents(qlib_dir):
    """A single-asset buy & hold should match that asset's return minus costs."""
    res = run_backtest("buy_and_hold", ["SPY"], START, END, cost_bps=0, risk_degree=1.0)
    bench = (1 + res.report["bench"]).prod() - 1
    strat = (1 + res.net_returns).prod() - 1
    # bench includes the first day's move, which the portfolio (bought at that close) misses
    first = res.report["bench"].iloc[0]
    assert strat == pytest.approx((1 + bench) / (1 + first) - 1, abs=0.01)


def test_cash_sleeve_is_respected(qlib_dir):
    """Partial targets (e.g. 2 of 4 ETFs on = 50%) must leave the rest in cash;
    qlib would otherwise renormalise the weights to 100% invested."""
    res = run_backtest("sma_trend", DEFAULT_UNIVERSE, START, END, window=50, cost_bps=0, risk_degree=1.0)
    invested = res.report["value"] / res.report["account"]
    days = invested.index
    # each signal row dated t is executed on the next trading day
    for signal_day, row in res.weights.fillna(0).iterrows():
        pos = days.searchsorted(signal_day, side="right")
        if pos < len(days):
            assert invested.iloc[pos] == pytest.approx(row.sum(), abs=0.02)


def test_unknown_strategy(qlib_dir):
    with pytest.raises(KeyError):
        run_backtest("nope", DEFAULT_UNIVERSE, START, END)
