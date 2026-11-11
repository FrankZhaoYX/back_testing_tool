"""Canadian-listed ETF portfolio: strategic weights + alpha signals, logged to the journal.

Run from the repo root (after `fetch` + `dump` for the tickers below):

    python -m backtest_tool fetch --symbols XUS.TO XQQ.TO XIC.TO XEG.TO XEF.TO XEC.TO CGL.TO XBB.TO CLF.TO VFV.TO --data-start 2013-01-01
    python -m backtest_tool dump
    python -m experiments.ca_portfolio            # appends every run to journal/strategy_journal.csv

Each experiment changes one thing relative to an earlier one; `change`, `rationale`
and `verdict` are written to the journal next to the measured results.
"""

from __future__ import annotations

import argparse
import warnings

from backtest_tool.engine import init_qlib, run_backtest
from backtest_tool.journal import DEFAULT_JOURNAL, log_run

UNIVERSE = ["XUS", "XQQ", "XIC", "XEG", "XEF", "XEC", "CGL", "XBB", "CLF", "VFV"]
BENCHMARK = "XUS"
CAPITAL = 50_000
COST_BPS = 5.0  # slippage per side; the ETFs are commission-free at most Canadian brokers
RECOMMENDED = "E08"
CHART_LABELS = {"E01": "original 4 x 25%", "E02": "core static", "E06": "core + trend filter", "E08": "core + momentum + half trend"}
WINDOWS = {"3y": ("2023-11-10", "2026-11-10"), "10y": ("2016-11-10", "2026-11-10")}

ORIGINAL = {"VFV": 0.25, "XQQ": 0.25, "XEG": 0.25, "CGL": 0.25}
CORE = {"XUS": 0.35, "XQQ": 0.10, "XIC": 0.20, "XEG": 0.05, "XEF": 0.15, "XEC": 0.05, "CGL": 0.10}
CORE_BOND = {"XUS": 0.30, "XQQ": 0.10, "XIC": 0.18, "XEG": 0.04, "XEF": 0.13, "XEC": 0.05, "CGL": 0.10, "XBB": 0.10}

EXPERIMENTS = [
    dict(
        id="E01", name="原方案-四只等权",
        change="基线：VFV(=VOO 加元版)/XQQ/XEG/CGL 各 25%，季度再平衡",
        rationale="原方案未给比例，按等权作为对照组",
        strategy="fixed_weight", weights=ORIGINAL, params={"rebalance": "Q"},
    ),
    dict(
        id="E02", name="新战略配置-无债券",
        change="换成 7 只 ETF：XUS35 XQQ10 XIC20 XEG5 XEF15 XEC5 CGL10，季度再平衡",
        rationale="按新思路：美国大盘为核心、科技降到 10%、加拿大以 XIC 为主只留 5% XEG、加入国际(XEF+XEC)、黄金 10%",
        strategy="fixed_weight", weights=CORE, params={"rebalance": "Q"},
    ),
    dict(
        id="E03", name="新战略配置-含10%债券",
        change="E02 各股票仓位等比例下调，腾出 10% 给 XBB",
        rationale="检验可选的债券仓位能否降低回撤",
        strategy="fixed_weight", weights=CORE_BOND, params={"rebalance": "Q"},
    ),
    dict(
        id="E04", name="E02+动量倾斜(0.5)",
        change="每月按 1/3/6/12 月平均动量的横截面 z 值，把各 ETF 权重在基准的 ±50% 内倾斜",
        rationale="第一个 alpha 信号：强者多配、弱者少配，总仓位不变",
        strategy="alpha_tilt", weights=CORE, params={"tilt": 0.5, "trend_window": None},
    ),
    dict(
        id="E05", name="E02+动量倾斜(1.0)",
        change="倾斜幅度从 ±50% 加到 ±100%",
        rationale="E04 信号方向对但幅度太小，测试加强后的效果",
        strategy="alpha_tilt", weights=CORE, params={"tilt": 1.0, "trend_window": None},
    ),
    dict(
        id="E06", name="E02+趋势过滤(全减)",
        change="每月检查：ETF 收盘低于 200 日均线则该仓位全部转入 CLF 短债；不做动量倾斜",
        rationale="第二个 alpha 信号：趋势过滤，目标是控制大回撤",
        strategy="alpha_tilt", weights=CORE, params={"tilt": 0.0, "trend_window": 200, "trend_cut": 1.0, "defensive": "CLF"},
    ),
    dict(
        id="E07", name="E02+动量(0.5)+趋势(全减)",
        change="E04 与 E06 两个信号叠加",
        rationale="看两个信号能否互补",
        strategy="alpha_tilt", weights=CORE, params={"tilt": 0.5, "trend_window": 200, "trend_cut": 1.0, "defensive": "CLF"},
    ),
    dict(
        id="E08", name="E02+动量(0.5)+趋势(减半)",
        change="跌破 200 日线只减一半仓位（而非全减）",
        rationale="E06/E07 回撤小但踏空成本高，减半是折中",
        strategy="alpha_tilt", weights=CORE, params={"tilt": 0.5, "trend_window": 200, "trend_cut": 0.5, "defensive": "CLF"},
    ),
    dict(
        id="E09", name="E02+波动率倒数加权",
        change="不用动量，改为按 63 日波动率倒数缩放权重（低波动多配）",
        rationale="检验风险平价式的配比是否优于固定比例",
        strategy="alpha_tilt", weights=CORE, params={"tilt": 0.0, "trend_window": None, "inv_vol_window": 63},
    ),
    dict(
        id="E10", name="E02+动量(1.0)+趋势(减半)",
        change="E08 的动量倾斜加到 ±100%",
        rationale="E05 显示强倾斜长期更好，与减半趋势过滤组合作为候选最终方案",
        strategy="alpha_tilt", weights=CORE, params={"tilt": 1.0, "trend_window": 200, "trend_cut": 0.5, "defensive": "CLF"},
    ),
    dict(
        id="E11", name="E02+固定止盈20%",
        change="每只 ETF 较买入价涨 20% 即全部卖出转入 CLF；较卖出价回落 10% 或等满 126 个交易日后买回",
        rationale="检验宽基 ETF 是否需要固定止盈规则",
        strategy="take_profit", weights=CORE,
        params={"take_profit": 0.2, "reentry_drop": 0.1, "max_wait": 126, "defensive": "CLF"},
    ),
    dict(
        id="E12", name="E02+固定止盈40%",
        change="止盈线从 20% 放宽到 40%，其余同 E11",
        rationale="看更宽的止盈线是否改变 E11 的结论",
        strategy="take_profit", weights=CORE,
        params={"take_profit": 0.4, "reentry_drop": 0.1, "max_wait": 126, "defensive": "CLF"},
    ),
]

# Conclusions, written after reviewing the measured results of each run (see the journal).
VERDICTS: dict[tuple[str, str], str] = {
    ("E01", "3y"): "对照组。收益最高(年化23.4%)，但靠的是 25% 黄金 + 25% 能源的集中押注，这三年两者恰好都大涨",
    ("E01", "10y"): "十年年化 15.7% 也最高，但最大回撤 -30.7% 是所有方案里最深的；只有 4 只且无国际分散",
    ("E02", "3y"): "年化 22.7%，比原方案低 0.7 个百分点，但波动更低(12.4% vs 13.3%)、夏普更高(1.72 vs 1.65)。作为后续信号的基准配置",
    ("E02", "10y"): "十年年化 13.7%，回撤 -28.2%。分散后收益略低于原方案和纯美股(XUS 15.5%)，回撤与 XUS(-27.2%)相当",
    ("E03", "3y"): "加 10% 债券：回撤从 -14.0% 降到 -12.5%，年化少 1.8 个百分点，夏普几乎不变(1.73)。债券只是降杠杆，没有额外好处",
    ("E03", "10y"): "十年同样结论：年化少 1.2 个百分点换回撤少 2.5 个百分点。可选项，不纳入默认配置",
    ("E04", "3y"): "动量倾斜 ±50%：三年结果与静态配置几乎一样(22.6% vs 22.7%)，换手从 0.5 升到 1.6 倍/年。牛市里信号没有帮助也没有伤害",
    ("E04", "10y"): "十年年化 +0.3 个百分点、回撤少 1.3 个百分点。方向正确但幅度很小",
    ("E05", "3y"): "倾斜加到 ±100%：三年仍与静态持平(22.7%)，回撤略小(-13.2%)，但换手 3.1 倍/年，且会把弱势资产权重降到接近 0",
    ("E05", "10y"): "十年年化 14.4%(+0.7)、回撤 -25.3%(少 2.9)。长期最好的纯动量版本，但持仓摆动大，偏离战略配置较远",
    ("E06", "3y"): "趋势过滤(全减)：三年年化 19.1%，比静态少 3.6 个百分点，回撤只改善 1 个百分点。牛市中被短暂回调震出去又追回，代价明显",
    ("E06", "10y"): "十年回撤从 -28.2% 降到 -13.2%，波动 9.7%，夏普 1.11(静态 1.00)；代价是年化降到 10.9%。防守有效但踏空成本高",
    ("E07", "3y"): "两个信号叠加：年化 19.9%，仍明显低于静态。全减式趋势过滤是主要拖累",
    ("E07", "10y"): "十年年化 11.7%、回撤 -13.9%、夏普 1.13。比单用趋势过滤好 0.8 个百分点",
    ("E08", "3y"): "趋势改为减半：年化 21.2%(比静态少 1.5)，回撤 -13.4%，夏普 1.66。牛市代价降到可接受范围",
    ("E08", "10y"): "十年年化 12.9%、回撤 -19.5%(静态 -28.2%)、夏普 1.12。收益少 0.8 个百分点换回撤少约三分之一。推荐的信号版方案",
    ("E09", "3y"): "波动率倒数加权：三年夏普最高(1.78)、年化 23.2%，优于静态",
    ("E09", "10y"): "十年反而略差于静态(13.3% vs 13.7%，回撤 -28.4%)。三年的优势不稳健，不采用",
    ("E10", "3y"): "强倾斜+减半趋势：年化 21.8%，略好于 E08",
    ("E10", "10y"): "十年年化 13.5%、回撤 -19.1%、夏普 1.13，数字略优于 E08，但换手 3.2 倍/年且常把个别 ETF 降到接近 0。备选，适合能接受持仓大幅摆动的情况",
    ("E11", "3y"): "固定止盈 20%：年化 20.4%，比不止盈(E02 22.7%)少 2.3 个百分点、终值少约 5000。波动降到 9.8%、夏普 1.95，但原因是平均约 34% 的资金停在短债里，等于降低了仓位，而不是选时有效",
    ("E11", "10y"): "十年年化 12.3%(E02 13.7%)，回撤 -27.7% 几乎没改善，夏普 1.03 与不止盈(1.00)基本相同，换手 2.6 倍/年。长期没有优势",
    ("E12", "3y"): "止盈 40%：年化 20.1%，回撤 -7.4%(恰好在 2025 年 4 月下跌前已卖出部分仓位)。这个回撤改善取决于卖出时点，带有运气成分",
    ("E12", "10y"): "十年年化 12.7%、回撤 -19.5%、夏普 1.03。回撤数字与 E08 相同但年化略低，且规则对止盈线和买回条件很敏感。结论：宽基 ETF 不采用固定止盈",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qlib-dir", default="data/qlib")
    parser.add_argument("--journal", default=DEFAULT_JOURNAL)
    parser.add_argument("--only", nargs="+", help="experiment ids to run, e.g. E04 E05")
    parser.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=list(WINDOWS))
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    init_qlib(args.qlib_dir)
    print(f"{'experiment':34s} {'win':4s} {'final':>8s} {'CAGR':>7s} {'vol':>6s} {'sharpe':>6s} {'maxDD':>7s} {'turn':>5s}")
    for exp in EXPERIMENTS:
        if args.only and exp["id"] not in args.only:
            continue
        for window in args.windows:
            start, end = WINDOWS[window]
            result = run_backtest(
                exp["strategy"], UNIVERSE, start, end,
                benchmark=BENCHMARK, capital=CAPITAL, cost_bps=COST_BPS,
                weights=exp["weights"], **exp["params"],
            )  # fmt: skip
            row = log_run(
                result,
                experiment=f"{exp['id']} {exp['name']} [{window}]",
                change=exp["change"], rationale=exp["rationale"],
                verdict=VERDICTS.get((exp["id"], window), ""),
                weights=exp["weights"], params=exp["params"],
                start=start, end=end, capital=CAPITAL, cost_bps=COST_BPS, benchmark=BENCHMARK,
                path=args.journal,
            )  # fmt: skip
            print(
                f"{exp['id'] + ' ' + exp['name']:34s} {window:4s} {row['final_value']:8.0f} {row['cagr']:7.1%} "
                f"{row['ann_vol']:6.1%} {row['sharpe']:6.2f} {row['max_drawdown']:7.1%} {row['ann_turnover']:5.2f}"
            )
    print(f"\njournal -> {args.journal}")
    if not args.only:
        save_summary(args)


def save_summary(args) -> None:
    """Equity chart of the key candidates (3y) and today's target holdings for the recommended one."""
    from pathlib import Path

    import pandas as pd

    from backtest_tool.report import _plot_equity

    out = Path(args.journal).parent
    start, end = WINDOWS["3y"]
    by_id = {e["id"]: e for e in EXPERIMENTS}
    results = []
    for exp_id in ("E01", "E02", "E06", RECOMMENDED):
        exp = by_id[exp_id]
        res = run_backtest(
            exp["strategy"], UNIVERSE, start, end,
            benchmark=BENCHMARK, capital=CAPITAL, cost_bps=COST_BPS, weights=exp["weights"], **exp["params"],
        )  # fmt: skip
        res.name = f"{exp_id} {CHART_LABELS[exp_id]}"  # ASCII: CJK fonts are not always installed
        results.append(res)
    _plot_equity(results, BENCHMARK, out / "equity_3y.png")

    latest = results[-1].weights.iloc[-1]
    base = pd.Series(by_id[RECOMMENDED]["weights"]).reindex(latest.index).fillna(0.0)
    table = pd.DataFrame({"base_weight": base, "target_weight": latest.round(4), "target_value": (latest * CAPITAL).round(0)})
    table = table[(table["base_weight"] > 0) | (table["target_weight"] > 0)]
    table.index.name = f"as_of_{results[-1].weights.index[-1]:%Y-%m-%d}"
    table.to_csv(out / "current_target_weights.csv")
    print(f"\n{RECOMMENDED} target holdings on {CAPITAL:,} as of {results[-1].weights.index[-1]:%Y-%m-%d}:")
    print(table.to_string())


if __name__ == "__main__":
    main()
