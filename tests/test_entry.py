import numpy as np
import pandas as pd
import pytest

from backtest_tool import entry as en


@pytest.fixture
def bars():
    idx = pd.bdate_range("2020-01-01", "2022-12-30")
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx))))
    return pd.DataFrame({"close": close, "dividend": 0.0}, index=idx)


START, END = "2021-01-01", "2022-12-30"


def test_signals_match_definitions(bars):
    close = bars["close"]
    streak = en.down_streak(3)(bars)
    day = streak[streak].index[0]
    pos = bars.index.get_loc(day)
    assert (np.diff(close.iloc[pos - 3 : pos + 1]) < 0).all()
    below = en.below_ma(21)(bars)
    assert below.iloc[-1] == (close.iloc[-1] < close.iloc[-21:].mean())
    dd = en.drawdown(0.05)(bars)
    assert dd.iloc[-1] == (close.iloc[-1] <= close.iloc[-252:].max() * 0.95)
    both = en.all_of(en.down_streak(3), en.below_ma(21))(bars)
    assert (both == (streak & below)).all()


def test_baseline_buys_one_share_every_month(bars):
    res = en.simulate(bars, en.monthly(bars), START, END, slippage_bps=0)
    assert res.stats["buy_days"] == 24 and res.stats["shares"] == 24
    assert res.stats["final_cash"] == pytest.approx(0, abs=1e-6)
    assert res.stats["final_value"] == pytest.approx(24 * bars["close"].iloc[-1])
    # bought on the first trading day of each month at that day's close
    first_days = bars.loc[START:END].index.to_series().groupby(bars.loc[START:END].index.to_period("M")).min()
    assert list(res.trades["date"]) == list(first_days)


def test_signal_fills_next_day_and_respects_cash(bars):
    signal = pd.Series(False, index=bars.index)
    fire_day = bars.loc["2021-06-10":].index[0]
    signal[fire_day] = True
    res = en.simulate(bars, signal, START, END, slippage_bps=0)
    assert len(res.trades) == 1
    assert res.trades["date"].iloc[0] == bars.index[bars.index.get_loc(fire_day) + 1]  # no look-ahead
    assert res.trades["shares"].iloc[0] == 2  # capped at max_shares although 6 months of cash were waiting
    assert res.stats["final_cash"] > 0


def test_month_end_sweep_invests_idle_cash(bars):
    never = pd.Series(False, index=bars.index)
    assert en.simulate(bars, never, START, END).stats["shares"] == 0
    swept = en.simulate(bars, never, START, END, month_end_sweep=True, slippage_bps=0)
    assert swept.stats["shares"] >= 20  # roughly one share a month, bought at month end


def test_advance_borrows_from_future_months_and_repays(bars):
    always = pd.Series(True, index=bars.index)
    base = en.simulate(bars, en.monthly(bars), START, END, slippage_bps=0)
    ahead = en.simulate(bars, en.monthly(bars), START, END, slippage_bps=0, advance_signal=always, advance_shares=3)
    assert ahead.stats["min_cash"] < 0
    assert ahead.stats["min_cash"] >= -3 * bars.loc[START:END, "close"].max()
    assert ahead.stats["contributed"] == base.stats["contributed"]  # same money, only earlier
    assert ahead.equity["shares"].iloc[5] > base.equity["shares"].iloc[5]


def test_dividends_add_cash():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    bars = pd.DataFrame({"close": 100.0, "dividend": 0.0}, index=idx)
    bars.loc[idx[30], "dividend"] = 5.0
    res = en.simulate(bars, en.monthly(bars), "2021-01-01", "2021-03-31", slippage_bps=0)
    held = res.equity["shares"].loc[idx[29]]
    assert res.stats["final_value"] == pytest.approx(res.stats["contributed"] + 5.0 * held)


def test_irr_of_flat_price_is_zero():
    idx = pd.bdate_range("2021-01-01", "2022-12-30")
    bars = pd.DataFrame({"close": 100.0, "dividend": 0.0}, index=idx)
    res = en.simulate(bars, en.monthly(bars), "2021-01-01", "2022-12-30", slippage_bps=0)
    assert res.stats["irr"] == pytest.approx(0, abs=1e-3)


def test_forward_returns(bars):
    out = en.forward_returns(bars, pd.Series(True, index=bars.index), START, END, horizons=(21,))
    assert out["fwd21_mean"] == pytest.approx(out["fwd21_all_days"], abs=1e-3)
