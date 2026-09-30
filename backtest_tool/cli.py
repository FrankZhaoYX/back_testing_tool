"""Command line entry point: ``python -m backtest_tool <fetch|dump|backtest|run>``."""

from __future__ import annotations

import argparse
import warnings
from datetime import date

from . import DEFAULT_BENCHMARK, DEFAULT_UNIVERSE
from .strategies import STRATEGIES

DEFAULT_CSV_DIR = "data/csv"
DEFAULT_QLIB_DIR = "data/qlib"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backtest_tool", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p, fetch=False):
        p.add_argument("--csv-dir", default=DEFAULT_CSV_DIR)
        p.add_argument("--qlib-dir", default=DEFAULT_QLIB_DIR)
        if fetch:
            p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
            p.add_argument("--data-start", default="2008-01-01", help="first date to download")
            p.add_argument("--api-key", help="EODHD API key (default: $EODHD_API_KEY)")

    def bt_args(p):
        p.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
        p.add_argument("--strategy", nargs="+", default=["all"], choices=["all", *STRATEGIES])
        p.add_argument("--start", default="2012-01-01")
        p.add_argument("--end", default=date.today().isoformat())
        p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
        p.add_argument("--capital", type=float, default=100_000)
        p.add_argument("--cost-bps", type=float, default=5.0, help="cost per side in basis points")
        p.add_argument("--deal-price", default="close", choices=["close", "open"])
        p.add_argument("--sma-window", type=int, default=200)
        p.add_argument("--mom-lookback", type=int, default=63)
        p.add_argument("--top-k", type=int, default=1)
        p.add_argument("--rebalance", default="M", choices=["W", "M", "Q"])
        p.add_argument("--out", default="results")

    data_args(sub.add_parser("fetch", help="download daily bars from EODHD"), fetch=True)
    data_args(sub.add_parser("dump", help="convert CSVs to Qlib binary format"))
    bt = sub.add_parser("backtest", help="run strategies on the Qlib data")
    bt.add_argument("--qlib-dir", default=DEFAULT_QLIB_DIR)
    bt_args(bt)
    run = sub.add_parser("run", help="fetch + dump + backtest in one go")
    data_args(run)
    run.add_argument("--data-start", default="2008-01-01")
    run.add_argument("--api-key")
    bt_args(run)
    return parser


def cmd_fetch(args) -> None:
    from .eodhd import download_universe

    download_universe(args.symbols, args.csv_dir, start=args.data_start, api_key=args.api_key)


def cmd_dump(args) -> None:
    from .qlib_dump import dump_to_qlib

    symbols = dump_to_qlib(args.csv_dir, args.qlib_dir)
    print(f"wrote Qlib data for {', '.join(symbols)} -> {args.qlib_dir}")


def cmd_backtest(args) -> None:
    from .engine import init_qlib, run_backtest
    from .report import format_summary, save_results, summary_table

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    init_qlib(args.qlib_dir)
    names = list(STRATEGIES) if "all" in args.strategy else args.strategy
    params = {
        "window": args.sma_window,
        "lookback": args.mom_lookback,
        "top_k": args.top_k,
        "rebalance": args.rebalance,
    }
    results = [
        run_backtest(
            name,
            args.symbols,
            args.start,
            args.end,
            benchmark=args.benchmark,
            capital=args.capital,
            cost_bps=args.cost_bps,
            deal_price=args.deal_price,
            **params,
        )
        for name in names
    ]
    print()
    print(format_summary(summary_table(results, args.benchmark)))
    out = save_results(results, args.benchmark, args.out)
    print(f"\nsaved daily reports, rebalances, summary.csv and equity.png -> {out}/")


def main(argv=None) -> None:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    from .eodhd import EODHDError

    try:
        _dispatch(args)
    except EODHDError as exc:
        raise SystemExit(f"error: {exc}") from None


def _dispatch(args) -> None:
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "dump":
        cmd_dump(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "run":
        from .eodhd import download_universe

        download_universe(
            sorted(set(args.symbols) | {args.benchmark}), args.csv_dir, start=args.data_start, api_key=args.api_key
        )
        cmd_dump(args)
        cmd_backtest(args)
