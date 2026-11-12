"""Append-only CSV journal of every backtest run, so strategy changes can be reviewed later."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing qlib just to write a CSV
    from .engine import BacktestResult

DEFAULT_JOURNAL = "journal/strategy_journal.csv"
COLUMNS = [
    "run_id", "logged_at_utc", "experiment", "change", "rationale",
    "strategy", "weights", "params", "start", "end", "capital", "cost_bps", "benchmark",
    "final_value", "total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
    "win_rate", "ann_turnover", "total_cost",
    "bench_total_return", "bench_cagr", "bench_ann_vol", "bench_sharpe", "bench_max_drawdown",
    "excess_cagr", "verdict",
]  # fmt: skip


def _next_run_id(path: Path) -> int:
    if not path.exists():
        return 1
    with path.open(encoding="utf-8-sig", newline="") as fh:
        ids = [int(row["run_id"]) for row in csv.DictReader(fh) if row.get("run_id")]
    return max(ids, default=0) + 1


def log_run(
    result: BacktestResult,
    *,
    experiment: str,
    change: str = "",
    rationale: str = "",
    verdict: str = "",
    weights: dict | None = None,
    params: dict | None = None,
    start: str = "",
    end: str = "",
    capital: float = 0.0,
    cost_bps: float = 0.0,
    benchmark: str = "",
    path: str | Path = DEFAULT_JOURNAL,
) -> dict:
    """Append one row describing ``result`` to the journal and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report, stats, bench = result.report, result.stats, result.bench_stats
    # qlib's `cost` column is a daily cost rate relative to the previous day's account value
    prev_account = report["account"].shift(1).fillna(capital)
    row = {
        "run_id": _next_run_id(path),
        "logged_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": experiment,
        "change": change,
        "rationale": rationale,
        "strategy": result.name,
        "weights": json.dumps(weights or {}, ensure_ascii=False),
        "params": json.dumps(params or {}, ensure_ascii=False),
        "start": f"{report.index[0]:%Y-%m-%d}",
        "end": f"{report.index[-1]:%Y-%m-%d}",
        "capital": capital,
        "cost_bps": cost_bps,
        "benchmark": benchmark,
        "final_value": round(float(report["account"].iloc[-1]), 2),
        "total_cost": round(float((report["cost"] * prev_account).sum()), 2),
        "excess_cagr": round(stats["cagr"] - bench["cagr"], 4),
        "verdict": verdict,
    }
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "ann_turnover"):
        row[key] = round(float(stats[key]), 4)
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"):
        row[f"bench_{key}"] = round(float(bench[key]), 4)

    new_file = not path.exists()
    # utf-8-sig so Excel opens the Chinese text correctly
    with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


def append_rows(path: str | Path, columns: list[str], rows: list[dict]) -> None:
    """Append ``rows`` to a CSV journal, numbering them with a continuing ``run_id`` and a timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    next_id = _next_run_id(path)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    new_file = not path.exists()
    with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "logged_at_utc", *columns])
        if new_file:
            writer.writeheader()
        for offset, row in enumerate(rows):
            writer.writerow({"run_id": next_id + offset, "logged_at_utc": stamp, **row})
