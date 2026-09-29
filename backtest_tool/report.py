"""Persist backtest results: per-strategy daily reports, a summary table and an equity chart."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine import BacktestResult

PCT_COLS = ["total_return", "cagr", "ann_vol", "max_drawdown", "win_rate"]


def summary_table(results: list[BacktestResult], benchmark: str) -> pd.DataFrame:
    rows = {r.name: r.stats for r in results}
    if results:
        rows[f"benchmark ({benchmark})"] = results[0].bench_stats
    return pd.DataFrame(rows).T


def format_summary(table: pd.DataFrame) -> str:
    shown = table.copy()
    for col in shown.columns:
        fmt = "{:.2%}" if col in PCT_COLS else "{:.2f}"
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return shown.to_string()


def save_results(results: list[BacktestResult], benchmark: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = summary_table(results, benchmark)
    table.to_csv(out / "summary.csv")
    for r in results:
        r.report.to_csv(out / f"{r.name}_daily.csv")
        r.weights.to_csv(out / f"{r.name}_rebalances.csv")
    _plot_equity(results, benchmark, out / "equity.png")
    return out


def _plot_equity(results: list[BacktestResult], benchmark: str, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # plotting is optional
        return
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    curves = {r.name: (1 + r.net_returns).cumprod() for r in results}
    if results:
        curves[f"{benchmark} (bench)"] = (1 + results[0].report["bench"]).cumprod()
    for name, eq in curves.items():
        style = {"color": "0.35", "linestyle": "--"} if name.endswith("(bench)") else {}
        ax_eq.plot(eq.index, eq.values, label=name, linewidth=1.3, **style)
        ax_dd.plot(eq.index, (eq / eq.cummax() - 1).values, linewidth=1, **style)
    ax_eq.set_yscale("log")
    ax_eq.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax_eq.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_eq.set_ylabel("Growth of $1 (log)")
    ax_eq.legend(frameon=False)
    ax_eq.grid(alpha=0.3)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_dd.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
