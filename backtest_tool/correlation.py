"""Return-correlation analysis between ETFs, rendered as a self-contained HTML report.

    r_t = (P_t - P_{t-1}) / P_{t-1}         daily simple return (P = adjusted close)
    rho = Cov(X, Y) / (sigma_X * sigma_Y)   Pearson correlation
    beta(Y on X) = Cov(X, Y) / sigma_X^2    sensitivity of Y to moves in X
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
# Reference categorical palette, slots 1-3 (validated all-pairs for up to 3 series).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_prices(csv_dir: str | Path, symbols: list[str]) -> pd.DataFrame:
    """Adjusted close (split/dividend adjusted) per symbol, aligned on common trading dates."""
    cols = {}
    for sym in symbols:
        df = pd.read_csv(Path(csv_dir) / f"{sym.upper()}.csv", parse_dates=["date"])
        cols[sym.upper()] = df.set_index("date")["adjusted_close"]
    return pd.DataFrame(cols).sort_index().dropna()


def last_years(prices: pd.DataFrame, years: float) -> pd.DataFrame:
    """Prices covering the last ``years`` years of returns (one extra leading day for the first return)."""
    cutoff = prices.index[-1] - pd.Timedelta(days=round(years * 365.25))
    before = prices.index[prices.index < cutoff]
    return prices[prices.index >= before[-1]] if len(before) else prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return (prices / prices.shift(1) - 1).dropna()


def beta_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """beta[Y, X] = Cov(X, Y) / Var(X): row asset regressed on column asset."""
    cov = returns.cov()
    var = np.diag(cov)
    return cov / var[np.newaxis, :]


def pair_list(symbols: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(symbols) for b in symbols[i + 1 :]]


def analyze(prices: pd.DataFrame, rolling_window: int = 63) -> dict:
    rets = daily_returns(prices)
    syms = list(rets.columns)
    pairs = pair_list(syms)
    corr = rets.corr()
    beta = beta_matrix(rets)
    years = len(rets) / TRADING_DAYS
    total = (1 + rets).prod() - 1
    equity = (1 + rets).cumprod()
    stats = pd.DataFrame(
        {
            "年化收益 CAGR": (1 + total) ** (1 / years) - 1,
            "年化波动率 σ": rets.std() * np.sqrt(TRADING_DAYS),
            "日波动率 σ": rets.std(),
            "最大回撤": (equity / equity.cummax() - 1).min(),
            "累计收益": total,
        }
    )
    by_year = pd.DataFrame(
        {f"{a}–{b}": rets.groupby(rets.index.year).apply(lambda g, a=a, b=b: g[a].corr(g[b])) for a, b in pairs}
    )
    by_year.index.name = "年份"
    rolling = pd.DataFrame({f"{a}–{b}": rets[a].rolling(rolling_window).corr(rets[b]) for a, b in pairs}).dropna()
    # Correlation on the market's down days vs up days (first symbol = market).
    mkt = syms[0]
    regime = pd.DataFrame(
        {
            f"{a}–{b}": {
                f"{mkt} 下跌日": rets[rets[mkt] < 0][[a, b]].corr().iloc[0, 1],
                f"{mkt} 上涨日": rets[rets[mkt] > 0][[a, b]].corr().iloc[0, 1],
                f"{mkt} 跌幅 > 2%": rets[rets[mkt] < -0.02][[a, b]].corr().iloc[0, 1],
            }
            for a, b in pairs
        }
    )
    return {
        "returns": rets,
        "prices": prices,
        "corr": corr,
        "beta": beta,
        "r2": corr**2,
        "stats": stats,
        "by_year": by_year,
        "rolling": rolling,
        "regime": regime,
        "window": rolling_window,
    }


# ---------------------------------------------------------------- charts


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight", facecolor="white")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _charts(res: dict) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out = {}

    # Growth of $1, log scale
    fig, ax = plt.subplots(figsize=(10, 3.8))
    growth = res["prices"] / res["prices"].iloc[0]
    for color, sym in zip(SERIES_COLORS, growth.columns):
        ax.plot(growth.index, growth[sym], color=color, linewidth=1.6, label=sym)
        ax.annotate(f"{sym} {growth[sym].iloc[-1]:.1f}x", (growth.index[-1], growth[sym].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    out["growth"] = _fig_to_data_uri(fig)
    plt.close(fig)

    # Rolling correlation
    fig, ax = plt.subplots(figsize=(10, 3.8))
    roll = res["rolling"]
    for color, col in zip(SERIES_COLORS, roll.columns):
        ax.plot(roll.index, roll[col], color=color, linewidth=1.4, label=col)
    ax.set_ylim(min(0, roll.min().min() - 0.05), 1.0)
    ax.set_ylabel(f"{res['window']} 日滚动 ρ", color=INK_2, fontsize=9)
    _style(ax)
    ax.legend(frameon=False, loc="lower left", ncol=len(roll.columns), fontsize=9)
    out["rolling"] = _fig_to_data_uri(fig)
    plt.close(fig)

    # Scatter of daily returns vs the first symbol, with beta fit line
    rets = res["returns"]
    mkt, others = rets.columns[0], list(rets.columns[1:])
    fig, axes = plt.subplots(1, len(others), figsize=(5 * len(others), 4), squeeze=False)
    for ax, sym, color in zip(axes[0], others, SERIES_COLORS[1:]):
        ax.scatter(rets[mkt] * 100, rets[sym] * 100, s=6, alpha=0.35, color=color, edgecolors="none")
        b = res["beta"].loc[sym, mkt]
        a = rets[sym].mean() - b * rets[mkt].mean()
        xs = np.linspace(rets[mkt].min(), rets[mkt].max(), 2)
        ax.plot(xs * 100, (a + b * xs) * 100, color=INK, linewidth=1.2)
        ax.set_title(f"{sym} vs {mkt}   β = {b:.2f}   ρ = {res['corr'].loc[sym, mkt]:.2f}", fontsize=10, color=INK)
        ax.set_xlabel(f"{mkt} 日收益 (%)", color=INK_2, fontsize=9)
        ax.set_ylabel(f"{sym} 日收益 (%)", color=INK_2, fontsize=9)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    out["scatter"] = _fig_to_data_uri(fig)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- HTML


def _matrix_html(df: pd.DataFrame, fmt: str = "{:.3f}", shade: bool = True, diag_note: str = "") -> str:
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for col, v in row.items():
            style = ""
            if shade and pd.notna(v):
                # single-hue sequential shade (blue), 0 -> light, 1 -> dark
                t = float(np.clip(v, 0, 1))
                style = f' style="background:rgba(42,120,214,{0.08 + 0.55 * t:.2f})"'
            text = "" if pd.isna(v) else fmt.format(v)
            cells.append(f"<td{style}>{text}</td>")
        rows.append(f"<tr><th>{escape(str(idx))}</th>{''.join(cells)}</tr>")
    note = f"<p class='note'>{diag_note}</p>" if diag_note else ""
    return f"<table class='mx'><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}"


def _table_html(df: pd.DataFrame, formats: dict[str, str] | None = None, default: str = "{:.3f}") -> str:
    formats = formats or {}
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = []
    for idx, row in df.iterrows():
        cells = "".join(
            f"<td>{'' if pd.isna(v) else formats.get(c, default).format(v)}</td>" for c, v in row.items()
        )
        body.append(f"<tr><th>{escape(str(idx))}</th>{cells}</tr>")
    name = escape(str(df.index.name or ""))
    return f"<table><thead><tr><th>{name}</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _level(rho: float) -> str:
    if rho >= 0.9:
        return "极高度相关"
    if rho >= 0.7:
        return "高度相关"
    if rho >= 0.4:
        return "中度相关"
    return "弱相关"


def _findings(res: dict) -> list[str]:
    corr, beta, rets = res["corr"], res["beta"], res["returns"]
    syms = list(corr.columns)
    mkt = syms[0]
    lines = []
    for a, b in pair_list(syms):
        rho = corr.loc[a, b]
        lines.append(
            f"<b>{a} vs {b}</b>：ρ = {rho:.3f}（{_level(rho)}），R² = {rho**2:.1%}，"
            f"即 {b} 每日波动中约 {rho**2:.0%} 可由 {a} 解释；β({b}→{a}) = {beta.loc[b, a]:.2f}。"
        )
    for sym in syms[1:]:
        lines.append(
            f"{sym} 相对 {mkt} 的 β = {beta.loc[sym, mkt]:.2f}：{mkt} 涨跌 1% 时，{sym} 平均涨跌约 "
            f"{beta.loc[sym, mkt]:.2f}%。"
        )
    reg = res["regime"]
    down, up = reg.iloc[0], reg.iloc[1]
    rises = [c for c in reg.columns if down[c] > up[c]]
    if rises:
        lines.append(
            f"市场下跌日的相关性高于上涨日（{', '.join(rises)}）——分散化效果在下跌时会减弱，正是最需要它的时候。"
        )
    by_year = res["by_year"]
    lo_pair = by_year.min().idxmin()
    lines.append(
        f"年度相关性最低的是 {lo_pair} 在 {by_year[lo_pair].idxmin()} 年（ρ = {by_year[lo_pair].min():.2f}），"
        f"最高为 {by_year[lo_pair].idxmax()} 年（ρ = {by_year[lo_pair].max():.2f}），相关性并非恒定。"
    )
    return lines


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--card:#ffffff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei","Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 12px;padding-top:8px;border-top:1px solid var(--grid)}
.sub{color:var(--ink2);margin:0 0 24px}.note{color:var(--ink2);font-size:13px;margin:6px 0 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums}.kpi .l{color:var(--ink2);font-size:13px}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:8px 0;background:var(--card)}
th,td{padding:7px 14px;border-bottom:1px solid var(--grid);text-align:right}
thead th{color:var(--ink2);font-weight:500;font-size:13px}tbody th{text-align:left;font-weight:600}
.mx td{min-width:84px;text-align:center}.scroll{overflow-x:auto}
img{max-width:100%;height:auto;display:block;border:1px solid var(--grid);border-radius:10px;background:#fff}
ul{padding-left:20px}li{margin:6px 0}code{background:#f0efeb;padding:1px 5px;border-radius:4px}
.formula{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px 16px;font-family:ui-monospace,Menlo,monospace;font-size:14px}
"""


def render_html(res: dict, title: str = "ETF 相关性分析报告") -> str:
    rets = res["returns"]
    syms = list(rets.columns)
    charts = _charts(res)
    start, end = rets.index[0], rets.index[-1]
    pairs = pair_list(syms)
    kpis = "".join(
        f"<div class='kpi'><div class='v'>{res['corr'].loc[a, b]:.3f}</div><div class='l'>ρ {a} – {b}</div></div>"
        for a, b in pairs
    )
    pct = {c: "{:.2%}" for c in res["stats"].columns}
    findings = "".join(f"<li>{s}</li>" for s in _findings(res))
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head>
<body><main>
<h1>{escape(title)}：{" / ".join(syms)}</h1>
<p class="sub">样本区间 {start:%Y-%m-%d} → {end:%Y-%m-%d}，共 {len(rets):,} 个交易日；数据来源 EODHD（复权收盘价 adjusted close）</p>
<div class="kpis">{kpis}</div>

<h2>核心结论</h2><ul>{findings}</ul>

<h2>方法</h2>
<div class="formula">r<sub>t</sub> = (P<sub>t</sub> − P<sub>t−1</sub>) / P<sub>t−1</sub><br>
ρ = Cov(X, Y) / (σ<sub>X</sub> × σ<sub>Y</sub>)<br>β = Cov(X, Y) / σ<sub>X</sub>²</div>
<p class="note">P 使用复权收盘价，避免拆股和分红造成的虚假"涨跌"；三只 ETF 只保留共同交易日。协方差、方差均为样本统计量（ddof=1）。</p>

<h2>相关系数矩阵 ρ</h2><div class="scroll">{_matrix_html(res["corr"])}</div>

<h2>β 矩阵</h2><div class="scroll">{_matrix_html(res["beta"], "{:.3f}", shade=False,
    diag_note="读法：行资产对列资产的 β。例如第 2 行第 1 列 = 该行 ETF 相对第 1 列 ETF 的 β = Cov(X,Y)/σ<sub>X</sub>²，X 为列资产。β 不对称：β(Y→X) ≠ β(X→Y)。")}</div>

<h2>R²（可解释波动比例 = ρ²）</h2><div class="scroll">{_matrix_html(res["r2"], "{:.1%}")}</div>

<h2>收益与风险</h2><div class="scroll">{_table_html(res["stats"], pct)}</div>

<h2>累计走势（1 美元起，对数坐标）</h2><img src="{charts["growth"]}" alt="累计收益走势">

<h2>每日收益散点与 β 拟合</h2><img src="{charts["scatter"]}" alt="日收益散点图">

<h2>滚动相关性（{res["window"]} 个交易日 ≈ 一个季度）</h2><img src="{charts["rolling"]}" alt="滚动相关性">

<h2>分年度相关性</h2><div class="scroll">{_table_html(res["by_year"])}</div>

<h2>涨跌市场下的相关性</h2><div class="scroll">{_table_html(res["regime"])}</div>
<p class="note">按 {syms[0]} 当日涨跌分组后分别计算 ρ。注意按条件切分样本会压缩波动区间，单独看数值会偏低，适合横向比较。</p>
</main></body></html>"""


def write_report(res: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res["corr"].to_csv(out / "correlation.csv")
    res["beta"].to_csv(out / "beta.csv")
    res["stats"].to_csv(out / "stats.csv")
    res["by_year"].to_csv(out / "correlation_by_year.csv")
    res["rolling"].to_csv(out / "rolling_correlation.csv")
    res["returns"].to_csv(out / "daily_returns.csv")
    path = out / "correlation_report.html"
    path.write_text(render_html(res), encoding="utf-8")
    return path
