import numpy as np
import pandas as pd
import pytest

from backtest_tool.correlation import analyze, beta_matrix, daily_returns, last_years, render_html


@pytest.fixture
def prices():
    """Y = 1.5 * X + noise, Z independent: known rho and beta."""
    rng = np.random.default_rng(1)
    n = 2600
    x = rng.normal(0.0005, 0.01, n)
    y = 1.5 * x + rng.normal(0, 0.005, n)
    z = rng.normal(0.0003, 0.012, n)
    idx = pd.bdate_range("2015-01-02", periods=n + 1)
    rets = pd.DataFrame({"SPY": x, "QQQ": y, "SMH": z}, index=idx[1:])
    start = pd.DataFrame({"SPY": [100.0], "QQQ": [100.0], "SMH": [100.0]}, index=idx[:1])
    return pd.concat([start, 100 * (1 + rets).cumprod()])


def test_returns_formula(prices):
    r = daily_returns(prices)
    p = prices["SPY"]
    assert r["SPY"].iloc[0] == pytest.approx((p.iloc[1] - p.iloc[0]) / p.iloc[0])


def test_rho_and_beta_match_definitions(prices):
    r = daily_returns(prices)
    x, y = r["SPY"], r["QQQ"]
    rho = np.cov(x, y)[0, 1] / (x.std() * y.std())
    beta = np.cov(x, y)[0, 1] / x.var()
    res = analyze(prices)
    assert res["corr"].loc["SPY", "QQQ"] == pytest.approx(rho)
    assert res["beta"].loc["QQQ", "SPY"] == pytest.approx(beta)
    assert beta == pytest.approx(1.5, abs=0.05)
    assert abs(res["corr"].loc["SPY", "SMH"]) < 0.1
    assert np.allclose(np.diag(beta_matrix(r)), 1.0)


def test_last_years_keeps_leading_day(prices):
    trimmed = last_years(prices, 5)
    span = trimmed.index[-1] - trimmed.index[1]
    assert span <= pd.Timedelta(days=round(5 * 365.25))
    assert trimmed.index[-1] - trimmed.index[0] >= pd.Timedelta(days=round(5 * 365.25))


def test_report_renders(prices, tmp_path):
    html = render_html(analyze(prices))
    assert "相关系数矩阵" in html and "data:image/png;base64," in html
