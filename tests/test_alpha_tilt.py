import csv

import pandas as pd
import pytest

from backtest_tool import DEFAULT_UNIVERSE
from backtest_tool.engine import run_backtest
from backtest_tool.journal import COLUMNS, log_run
from backtest_tool.strategies import alpha_tilt, fixed_weight

START, END = "2020-01-02", "2022-12-30"
BASE = {"SPY": 0.4, "TQQQ": 0.2, "SOXX": 0.2}  # SMH is the defensive asset


def test_fixed_weight_rows_equal_base(qlib_dir):
    w = fixed_weight(DEFAULT_UNIVERSE, "2019-01-02", END, weights=BASE, rebalance="Q")
    assert (w["SPY"] == 0.4).all() and (w["SMH"] == 0).all()
    assert len(w) >= 16  # one row per quarter (plus the first day)


def test_unknown_symbol_in_weights(qlib_dir):
    with pytest.raises(KeyError):
        fixed_weight(DEFAULT_UNIVERSE, "2019-01-02", END, weights={"NOPE": 1.0})


def test_momentum_tilt_preserves_budget_and_bounds(qlib_dir):
    w = alpha_tilt(DEFAULT_UNIVERSE, "2019-01-02", END, weights=BASE, tilt=0.5, trend_window=None)
    assert w[["SPY", "TQQQ", "SOXX"]].sum(axis=1).round(9).eq(0.8).all()  # money only moves between assets
    assert (w["SMH"] == 0).all() and (w >= 0).all().all()
    assert not w["SPY"].round(6).eq(0.4).all()  # the tilt actually changes weights


def test_zero_tilt_without_trend_equals_base(qlib_dir):
    w = alpha_tilt(DEFAULT_UNIVERSE, "2019-01-02", END, weights=BASE, tilt=0.0, trend_window=None)
    assert w["SPY"].round(9).eq(0.4).all() and w["TQQQ"].round(9).eq(0.2).all()


def test_trend_filter_moves_weight_to_defensive(qlib_dir):
    kwargs = dict(weights=BASE, tilt=0.0, trend_window=100, defensive="SMH")
    full = alpha_tilt(DEFAULT_UNIVERSE, "2019-01-02", END, trend_cut=1.0, **kwargs)
    half = alpha_tilt(DEFAULT_UNIVERSE, "2019-01-02", END, trend_cut=0.5, **kwargs)
    assert full.sum(axis=1).round(9).eq(0.8).all()  # freed weight lands in the defensive asset
    assert (full["SMH"] > 0).any()
    assert ((full["SMH"] - 2 * half["SMH"]).abs() < 1e-9).all()
    cash = alpha_tilt(DEFAULT_UNIVERSE, "2019-01-02", END, weights=BASE, tilt=0.0, trend_window=100)
    assert (cash.sum(axis=1) < 0.8 - 1e-9).any() and (cash["SMH"] == 0).all()


def test_backtest_and_journal(qlib_dir, tmp_path):
    path = tmp_path / "journal.csv"
    for i in range(2):
        res = run_backtest(
            "alpha_tilt", DEFAULT_UNIVERSE, START, END, capital=50_000,
            weights=BASE, tilt=0.5, trend_window=100, trend_cut=0.5, defensive="SMH",
        )  # fmt: skip
        row = log_run(res, experiment=f"测试 {i}", change="改动", verdict="结论", weights=BASE,
                      capital=50_000, cost_bps=5, benchmark="SPY", path=path)  # fmt: skip
        assert row["run_id"] == i + 1
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == COLUMNS and len(rows) == 2
    assert rows[1]["experiment"] == "测试 1" and float(rows[1]["final_value"]) > 0
    assert float(rows[0]["total_cost"]) > 0
    assert pd.Timestamp(rows[0]["start"]) == pd.Timestamp(START)


def test_full_weight_in_one_asset_does_not_crash(qlib_dir):
    """All risky assets below trend -> 100% defensive; float round-off must not break qlib."""
    res = run_backtest("fixed_weight", DEFAULT_UNIVERSE, START, END, weights={"SPY": 1.0000000000000002})
    assert res.report["value"].iloc[-1] > 0


def test_take_profit_sells_and_rebuys(qlib_dir):
    from backtest_tool.strategies import take_profit

    w = take_profit(DEFAULT_UNIVERSE, "2019-01-02", END, weights=BASE, take_profit=0.1,
                    reentry_drop=0.05, max_wait=60, defensive="SMH")  # fmt: skip
    assert w.sum(axis=1).round(9).eq(0.8).all()  # sold weight is parked in the defensive asset
    risky = w[["SPY", "TQQQ", "SOXX"]].round(9)
    assert ((risky == 0) | (risky == pd.Series(BASE))).all().all()  # each asset is fully in or fully out
    steps = risky.diff().dropna()
    assert (steps > 0).any().any() and (steps < 0).any().any()  # something was sold and bought back


def test_dca_invests_gradually_and_ends_fully_invested(qlib_dir):
    from backtest_tool.dca import dca_backtest

    kwargs = dict(start=START, end=END, capital=50_000, cost_bps=0)
    lump = dca_backtest("SPY", months=1, **kwargs)
    dca = dca_backtest("SPY", months=12, **kwargs)
    assert lump.report["cash"].iloc[0] == pytest.approx(0)  # one instalment == lump sum
    assert dca.report["value"].iloc[0] == pytest.approx(50_000 / 12)
    assert dca.report["cash"].iloc[0] == pytest.approx(50_000 * 11 / 12)
    assert dca.report["cash"].iloc[-1] == pytest.approx(0)
    assert (dca.report["cash"].diff().dropna() <= 1e-9).all()  # cash only ever goes down
    # with no costs, daily returns compound exactly to the final account value
    assert (1 + dca.net_returns).prod() * 50_000 == pytest.approx(dca.report["account"].iloc[-1], rel=1e-5)

    parked = dca_backtest("SPY", months=12, park="SMH", **kwargs)
    assert parked.report["cash"].abs().max() == pytest.approx(0)  # idle money sits in the parking asset
    assert parked.report["account"].iloc[0] == pytest.approx(50_000)
