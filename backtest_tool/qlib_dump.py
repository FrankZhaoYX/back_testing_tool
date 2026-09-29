"""Convert EODHD CSVs into Qlib's binary provider layout.

Layout written under ``qlib_dir``::

    calendars/day.txt                 one trading date per line
    instruments/all.txt               SYMBOL<TAB>start<TAB>end
    features/<symbol>/<field>.day.bin float32 LE: [calendar start index, values...]

Prices follow Qlib's convention: they are split/dividend adjusted, and
``factor = adjusted_close / close`` lets Qlib recover the raw traded price.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "factor", "change", "vwap"]


def to_qlib_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw EODHD bars (CSV_COLUMNS) into adjusted Qlib fields indexed by date."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[df["close"] > 0]
    factor = df["adjusted_close"] / df["close"]
    out = pd.DataFrame(index=df.index)
    for col in ["open", "high", "low", "close"]:
        out[col] = df[col] * factor
    out["volume"] = df["volume"] / factor
    out["factor"] = factor
    out["change"] = out["close"].pct_change()
    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3
    return out[FIELDS]


def dump_to_qlib(csv_dir: str | Path, qlib_dir: str | Path) -> list[str]:
    """Write every ``*.csv`` in ``csv_dir`` into a Qlib data directory. Returns the symbols."""
    csv_dir, qlib_dir = Path(csv_dir), Path(qlib_dir)
    frames = {p.stem.upper(): to_qlib_frame(pd.read_csv(p)) for p in sorted(csv_dir.glob("*.csv"))}
    if not frames:
        raise FileNotFoundError(f"no CSV files found in {csv_dir}")

    calendar = sorted(set().union(*(f.index for f in frames.values())))
    cal_pos = {d: i for i, d in enumerate(calendar)}
    (qlib_dir / "calendars").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "instruments").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "calendars" / "day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in calendar) + "\n")

    inst_lines = []
    for symbol, frame in frames.items():
        start, end = frame.index[0], frame.index[-1]
        # Qlib expects a contiguous series over the calendar between start and end.
        frame = frame.reindex(calendar[cal_pos[start] : cal_pos[end] + 1])
        feat_dir = qlib_dir / "features" / symbol.lower()
        feat_dir.mkdir(parents=True, exist_ok=True)
        for field in FIELDS:
            values = np.concatenate([[cal_pos[start]], frame[field].to_numpy(dtype=np.float64)])
            values.astype("<f4").tofile(feat_dir / f"{field}.day.bin")
        inst_lines.append(f"{symbol}\t{start:%Y-%m-%d}\t{end:%Y-%m-%d}")
    (qlib_dir / "instruments" / "all.txt").write_text("\n".join(inst_lines) + "\n")
    return list(frames)
