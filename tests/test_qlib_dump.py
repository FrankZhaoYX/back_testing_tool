import numpy as np
import pandas as pd
from qlib.data import D

from backtest_tool.qlib_dump import to_qlib_frame
from conftest import make_bars


def test_split_is_adjusted_out():
    frame = to_qlib_frame(make_bars("X", "2021-05-03", "2021-06-30", split_on="2021-06-01"))
    jump = frame["close"].pct_change().abs().max()
    assert jump < 0.2  # the raw 50% split drop must not show up in adjusted prices
    pre = frame.loc[:"2021-05-31", "factor"]
    assert np.allclose(pre, 0.5) and np.allclose(frame.loc["2021-06-01":, "factor"], 1.0)


def test_qlib_reads_dumped_data(qlib_dir):
    raw = pd.read_csv(qlib_dir.parent / "csv" / "SPY.csv")
    df = D.features(["SPY"], ["$close", "$factor"], start_time="2019-01-02", end_time="2019-01-10")
    expected = raw.set_index("date").loc["2019-01-02":"2019-01-10", "adjusted_close"].to_numpy()
    assert np.allclose(df["$close"].to_numpy(), expected, rtol=1e-6)
    assert set(D.list_instruments(D.instruments("all"), as_list=True)) == {"SPY", "TQQQ", "SOXX", "SMH"}
