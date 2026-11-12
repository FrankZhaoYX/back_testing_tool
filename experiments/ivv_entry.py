"""IVV entry-timing research: which buy signals beat simply buying one share a month?

    python -m backtest_tool fetch --symbols IVV.US --data-start 2013-01-01 --csv-dir data/csv_usd
    python -m experiments.ivv_entry          # appends every run to journal/ivv_entry_journal.csv

Every variant gets the same money (one share's worth on the first trading day of each
month, plus dividends) and never sells; only the purchase timing differs. Four groups:

  A  baseline   buy 1 share on the first trading day of every month
  B  wait       hold the cash until the signal fires, then buy up to 2 shares
  C  in-month   wait for the signal, but buy at month end anyway if it never fired
  D  advance    buy monthly as in A; when the signal fires, also spend up to N future
                months' budget early (2 shares per signal day)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_tool import entry as en
from backtest_tool.journal import append_rows

CSV = "data/csv_usd/IVV.csv"
JOURNAL = "journal/ivv_entry_journal.csv"
WINDOWS = {"10y": ("2016-11-10", "2026-11-10"), "first5y": ("2016-11-10", "2021-11-10"), "last5y": ("2021-11-10", "2026-11-10")}

# name -> (signal, rule in plain words)
SIGNALS = {
    "down3": (en.down_streak(3), "连续 3 天收盘下跌"),
    "down4": (en.down_streak(4), "连续 4 天收盘下跌"),
    "down5": (en.down_streak(5), "连续 5 天收盘下跌"),
    "below_ma21": (en.below_ma(21), "收盘低于 21 日均线"),
    "below_ma21_2pct": (en.below_ma(21, 0.02), "收盘低于 21 日均线 2% 以上"),
    "below_ma252": (en.below_ma(252), "收盘低于 52 周(252 日)均线"),
    "down3_below_ma21": (en.all_of(en.down_streak(3), en.below_ma(21)), "连跌 3 天 且 低于 21 日均线"),
    "down3_below_ma252": (en.all_of(en.down_streak(3), en.below_ma(252)), "连跌 3 天 且 低于 52 周均线"),
    "below_ma21_below_ma252": (en.all_of(en.below_ma(21), en.below_ma(252)), "同时低于 21 日均线和 52 周均线"),
    "pullback_in_uptrend": (en.all_of(en.below_ma(21), en.above_ma(252)), "低于 21 日均线 但 高于 52 周均线(上升趋势中的回调)"),
    "dd5": (en.drawdown(0.05), "较 52 周最高收盘价回撤 ≥5%"),
    "dd10": (en.drawdown(0.10), "较 52 周最高收盘价回撤 ≥10%"),
    "dd15": (en.drawdown(0.15), "较 52 周最高收盘价回撤 ≥15%"),
    "dd10_down3": (en.all_of(en.drawdown(0.10), en.down_streak(3)), "回撤 ≥10% 且 连跌 3 天"),
    "rsi14_30": (en.rsi_below(14, 30), "RSI(14) 低于 30"),
    "rsi2_10": (en.rsi_below(2, 10), "RSI(2) 低于 10"),
    "bollinger": (en.bollinger_low(20, 2.0), "收盘跌破布林带下轨(20 日, 2σ)"),
    "drop2": (en.day_drop(0.02), "单日下跌 ≥2%"),
}
WAIT = list(SIGNALS)
IN_MONTH = ["down3", "below_ma21", "down3_below_ma21", "dd5", "bollinger", "drop2"]
ADVANCE = ["below_ma21", "down3", "down3_below_ma21", "below_ma252", "below_ma21_below_ma252",
           "dd5", "dd10", "dd15", "dd10_down3", "rsi14_30", "bollinger", "drop2"]  # fmt: skip
ADVANCE_SHARES = (2, 4, 6)

GROUP_CONCLUSION = {
    "A": "基线：不择时，资金到账当天就买",
    "B": "【B 组结论】等信号才买没有一个明显跑赢基线(最好的只多 0.1%，多数落后)：信号挑到的价格确实更好，但等待期间现金闲置的损失更大",
    "C": "【C 组结论】月内择时与基线几乎没有差别(±1% 以内)，不值得为此盯盘",
    "D": "【D 组结论】月买不停 + 大跌时提前买，全部跑赢基线；但对照组(不看信号直接提前买)赢得更多，"
         "说明主要收益来自更早入场。信号的价值在于效率：每多投入 1 个百分点资金，深跌信号换来的收益是无信号的 2-3 倍",
}  # fmt: skip

COLUMNS = [
    "group", "experiment", "rule", "window", "signal_days", "buy_days", "shares", "contributed", "avg_cost",
    "price_vs_month_avg", "avg_cash_pct", "min_cash", "final_value", "profit", "irr",
    "vs_baseline_10y", "vs_baseline_first5y", "vs_baseline_last5y", "extra_exposure_pp", "gain_per_exposure_pp",
    "fwd21_mean", "fwd21_win", "fwd63_mean", "fwd63_win", "fwd252_mean", "fwd252_win",
    "fwd21_all_days", "fwd63_all_days", "fwd252_all_days", "verdict",
]  # fmt: skip


def build_runs(df: pd.DataFrame) -> list[dict]:
    monthly = en.monthly(df)
    always = pd.Series(True, index=df.index)
    runs = [dict(group="A", name="A00 每月第一个交易日买 1 股", rule="每月第一个交易日收盘买 1 股", signal=monthly, kwargs={})]
    for key in WAIT:
        sig, rule = SIGNALS[key]
        runs.append(dict(group="B", name=f"B 等信号 {key}", rule=f"{rule} → 次日收盘买入，每次最多 2 股；无信号不买",
                         signal=sig(df), event=sig(df), kwargs={}))  # fmt: skip
    for key in IN_MONTH:
        sig, rule = SIGNALS[key]
        runs.append(dict(group="C", name=f"C 月内择时 {key}", rule=f"{rule} → 次日买入；当月未出现信号则月末最后一天买入",
                         signal=sig(df), event=sig(df), kwargs={"month_end_sweep": True}))  # fmt: skip
    for n in ADVANCE_SHARES:
        runs.append(dict(group="D", name=f"D 对照 无信号直接提前 {n} 股", rule=f"每月照常买 1 股；第一天起就把后面 {n} 个月的额度提前用掉",
                         signal=monthly, kwargs={"advance_signal": always, "advance_shares": n}))  # fmt: skip
    for key in ADVANCE:
        sig, rule = SIGNALS[key]
        for n in ADVANCE_SHARES:
            runs.append(dict(group="D", name=f"D 月买+提前 {key} 上限{n}股",
                             rule=f"每月照常买 1 股；{rule} 时次日加买(每个信号日最多 2 股)，累计最多提前用掉 {n} 个月额度",
                             signal=monthly, event=sig(df), kwargs={"advance_signal": sig(df), "advance_shares": n}))  # fmt: skip
    return runs


def verdict(group: str, row: dict) -> str:
    diff = row["vs_baseline_10y"]
    word = "多" if diff >= 0 else "少"
    parts = [f"十年终值比基线{word} {abs(diff):.2%}（前五年 {row['vs_baseline_first5y']:+.2%}，后五年 {row['vs_baseline_last5y']:+.2%}）"]
    if group == "A":
        return f"基线：投入 {row['contributed']:.0f} 美元，买到 {row['shares']} 股，终值 {row['final_value']:.0f}，资金加权年化 {row['irr']:.1%}"
    if pd.notna(row.get("fwd63_mean")):
        parts.append(f"信号后 63 日平均涨 {row['fwd63_mean']:.1%}（全部交易日 {row['fwd63_all_days']:.1%}）")
    if group == "B":
        parts.append(f"平均 {row['avg_cash_pct']:.0%} 的资金闲置")
    if group == "D" and row["gain_per_exposure_pp"] != "":
        parts.append(f"平均多投入 {row['extra_exposure_pp']:.1f} 个百分点资金，每个百分点换来 {row['gain_per_exposure_pp']:.2f}% 收益")
    return "；".join(parts) + "。" + GROUP_CONCLUSION[group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=CSV)
    parser.add_argument("--journal", default=JOURNAL)
    args = parser.parse_args()

    df = en.load_bars(args.csv)
    runs = build_runs(df)
    base = {w: en.simulate(df, runs[0]["signal"], *WINDOWS[w]).stats for w in WINDOWS}
    rows = []
    for run in runs:
        stats = {w: en.simulate(df, run["signal"], *WINDOWS[w], **run["kwargs"]).stats for w in WINDOWS}
        s = stats["10y"]
        row = {
            "group": run["group"], "experiment": run["name"], "rule": run["rule"],
            "window": " → ".join(WINDOWS["10y"]),
            **{k: s[k] for k in ("signal_days", "buy_days", "shares", "contributed", "avg_cost", "price_vs_month_avg",
                                 "min_cash", "final_value", "profit", "irr")},
            "avg_cash_pct": s["avg_idle_cash_pct"],
        }  # fmt: skip
        if "advance_signal" in run["kwargs"]:  # report how often the dip signal fired, not the monthly one
            row["signal_days"] = int(run["kwargs"]["advance_signal"].shift(1, fill_value=False).loc[slice(*WINDOWS["10y"])].sum())
        for w in WINDOWS:
            row[f"vs_baseline_{w}"] = round(stats[w]["final_value"] / base[w]["final_value"] - 1, 4)
        extra = (base["10y"]["avg_idle_cash_pct"] - s["avg_idle_cash_pct"]) * 100
        row["extra_exposure_pp"] = round(extra, 2)
        row["gain_per_exposure_pp"] = round(row["vs_baseline_10y"] * 100 / extra, 2) if extra > 0.5 else ""
        if "event" in run:
            row.update(en.forward_returns(df, run["event"], *WINDOWS["10y"]))
        row["verdict"] = verdict(run["group"], row)
        rows.append(row)

    append_rows(args.journal, COLUMNS, rows)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(table[["experiment", "buy_days", "shares", "avg_cost", "avg_cash_pct", "final_value", "vs_baseline_10y",
                 "vs_baseline_first5y", "vs_baseline_last5y", "gain_per_exposure_pp", "fwd63_mean"]].to_string(index=False))  # fmt: skip
    out = Path(args.journal).parent
    save_chart(table, out / "ivv_entry_vs_baseline.png")
    status = signal_status(df)
    status.to_csv(out / "ivv_signal_status.csv", encoding="utf-8-sig")
    print(f"\nsignal status as of {df.index[-1]:%Y-%m-%d}:\n{status.to_string()}")
    print(f"\njournal -> {args.journal}")


def signal_status(df: pd.DataFrame) -> pd.DataFrame:
    """Which signals are firing on the latest bar, plus the levels they are measured against."""
    close = df["close"]
    last = close.iloc[-1]
    rows = {
        "收盘价": (round(last, 2), ""),
        "21 日均线": (round(close.rolling(21).mean().iloc[-1], 2), f"{last / close.rolling(21).mean().iloc[-1] - 1:+.2%}"),
        "52 周均线": (round(close.rolling(252).mean().iloc[-1], 2), f"{last / close.rolling(252).mean().iloc[-1] - 1:+.2%}"),
        "52 周最高收盘": (round(close.rolling(252).max().iloc[-1], 2), f"{last / close.rolling(252).max().iloc[-1] - 1:+.2%}"),
    }
    for key, (sig, rule) in SIGNALS.items():
        rows[f"信号 {key}: {rule}"] = ("触发" if bool(sig(df).iloc[-1]) else "未触发", "")
    out = pd.DataFrame(rows, index=["value", "close_vs_level"]).T
    out.index.name = f"as_of_{df.index[-1]:%Y-%m-%d}"
    return out


def save_chart(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B": "#eb6834", "C": "#eda100", "D": "#2a78d6", "control": "#52514e"}
    labels = {"B": "B: wait for signal", "C": "C: signal, else month end", "D": "D: monthly + buy ahead on signal",
              "control": "D control: buy ahead, no signal"}  # fmt: skip
    data = table[table["group"] != "A"].copy()
    data["kind"] = [("control" if "对照" in e else g) for g, e in zip(data["group"], data["experiment"])]
    # ASCII labels: signal key plus the advance cap (CJK fonts are not always installed)
    data["label"] = [
        (f"no signal, cap {e.split()[-2]}" if k == "control" else e.split()[2] + (f", cap {e.split('上限')[1][0]}" if "上限" in e else ""))
        for k, e in zip(data["kind"], data["experiment"])
    ]
    data = data.sort_values("vs_baseline_10y")
    fig, ax = plt.subplots(figsize=(9, 0.24 * len(data) + 1.5))
    ax.barh(range(len(data)), data["vs_baseline_10y"] * 100, color=[colors[k] for k in data["kind"]], height=0.72)
    ax.set_yticks(range(len(data)), data["label"], fontsize=8)
    ax.axvline(0, color="#0b0b0b", linewidth=0.8)
    ax.set_xlim(max(data["vs_baseline_10y"].min() * 100, -8) - 0.5, data["vs_baseline_10y"].max() * 100 + 1)
    for i, v in enumerate(data["vs_baseline_10y"] * 100):
        if v < ax.get_xlim()[0]:
            ax.text(ax.get_xlim()[0] + 0.1, i, f"{v:.1f}% (off scale)", va="center", fontsize=7, color="white")
    ax.set_xlabel("Final wealth after 10 years vs. buying 1 share on the first trading day of each month (%)", fontsize=9)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in labels]
    ax.legend(handles, labels.values(), frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
