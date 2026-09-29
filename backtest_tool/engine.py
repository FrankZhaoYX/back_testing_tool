"""Run target-weight strategies through Qlib's backtest engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import qlib
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest import executor as qlib_executor
from qlib.constant import REG_US
from qlib.contrib.strategy.order_generator import OrderGenWInteract
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
from qlib.data import D
from qlib.log import set_global_logger_level

from . import DEFAULT_BENCHMARK
from .metrics import summarize
from .strategies import STRATEGIES

_INITIALIZED_URI: str | None = None


def init_qlib(qlib_dir: str | Path) -> None:
    """Initialise Qlib once per process for the given provider directory."""
    global _INITIALIZED_URI
    uri = str(Path(qlib_dir).expanduser().resolve())
    if _INITIALIZED_URI != uri:
        qlib.init(
            provider_uri=uri,
            region=REG_US,
            kernels=1,
            expression_cache=None,
            dataset_cache=None,
            logging_level=logging.WARNING,
        )
        set_global_logger_level(logging.ERROR)
        _INITIALIZED_URI = uri


class TargetWeightStrategy(WeightStrategyBase):
    """Rebalance to the weights in ``signal`` (Series indexed by datetime, instrument).

    Qlib hands the strategy the signal of the *previous* trading day, so a row
    dated ``t`` is executed at ``t+1``. Days with no signal row are no-trade days.

    Qlib renormalises target weights to sum to 1, so any uninvested remainder
    (``1 - sum(weights)``) is kept in cash by scaling the step's risk degree.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._invested = 1.0

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        weights = {inst: float(w) for inst, w in score.dropna().items() if w > 0}
        self._invested = min(sum(weights.values()), 1.0)
        return weights

    def get_risk_degree(self, trade_step=None):
        return self.risk_degree * self._invested


@dataclass
class BacktestResult:
    name: str
    report: pd.DataFrame  # qlib daily report: return, cost, bench, turnover, account, ...
    weights: pd.DataFrame  # target weights on rebalance dates
    stats: dict = field(default_factory=dict)
    bench_stats: dict = field(default_factory=dict)

    @property
    def net_returns(self) -> pd.Series:
        return self.report["return"] - self.report["cost"]

    @property
    def equity(self) -> pd.Series:
        return self.report["account"]


def run_backtest(
    strategy: str,
    symbols: list[str],
    start: str,
    end: str,
    benchmark: str = DEFAULT_BENCHMARK,
    capital: float = 100_000,
    cost_bps: float = 5.0,
    deal_price: str = "close",
    risk_degree: float = 0.99,
    **strategy_kwargs,
) -> BacktestResult:
    """Backtest a named strategy from ``STRATEGIES``. Qlib must already be initialised."""
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    end = _clamp_end(end)
    # Load a bit of history before `start` so indicators are warm on day one.
    warmup_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    weights = STRATEGIES[strategy](symbols, warmup_start, end, **strategy_kwargs)
    weights = _clip_to_start(weights, start)
    if weights.empty:
        raise ValueError(f"{strategy} produced no signals between {start} and {end}")

    signal = weights.stack()
    signal.index.names = ["datetime", "instrument"]
    # OrderGenWInteract sizes orders with the execution-day deal price.
    strat = TargetWeightStrategy(signal=signal, risk_degree=risk_degree, order_generator_cls_or_obj=OrderGenWInteract)
    executor = qlib_executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    cost = cost_bps / 10_000
    portfolio, _ = qlib_backtest(
        start_time=start,
        end_time=end,
        strategy=strat,
        executor=executor,
        benchmark=benchmark,
        account=capital,
        exchange_kwargs={
            "freq": "day",
            "codes": sorted(set(symbols) | {benchmark}),
            "deal_price": deal_price,
            "limit_threshold": None,
            "open_cost": cost,
            "close_cost": cost,
            "min_cost": 0,
            "trade_unit": None,
        },
    )
    report, _positions = portfolio["1day"]
    result = BacktestResult(name=strategy, report=report, weights=weights)
    result.stats = summarize(result.net_returns, report.get("turnover"))
    result.bench_stats = summarize(report["bench"])
    return result


def _clamp_end(end: str) -> str:
    """Qlib needs the trading day *after* the last step, so stop one day before the data ends."""
    calendar = D.calendar(freq="day")
    if pd.Timestamp(end) >= calendar[-1]:
        return calendar[-2].strftime("%Y-%m-%d")
    return end


def _clip_to_start(weights: pd.DataFrame, start: str) -> pd.DataFrame:
    """Drop warm-up rows, carrying the last pre-start allocation so it is bought on day one.

    The carried row is dated on the last trading day before ``start`` because the
    strategy trades each day on the previous day's signal.
    """
    start_ts = pd.Timestamp(start)
    before, after = weights[weights.index < start_ts], weights[weights.index >= start_ts]
    if before.empty:
        return after
    prior_days = D.calendar(start_time=before.index[-1], end_time=start_ts, freq="day")
    prior_days = [d for d in prior_days if d < start_ts]
    carry = before.iloc[[-1]].copy()
    carry.index = [prior_days[-1]]
    return pd.concat([carry, after])
