# back_testing_tool

Backtest ETF strategies with **[EODHD](https://eodhd.com/)** market data and the
**[Microsoft Qlib](https://github.com/microsoft/qlib)** backtest engine.

Default universe: **SPY** (S&P 500), **TQQQ** (3x Nasdaq-100), **SOXX** and **SMH** (semiconductors).
Benchmark: SPY.

## How it works

```
EODHD /api/eod  ──►  data/csv/*.csv  ──►  data/qlib/ (Qlib binary format)  ──►  Qlib backtest  ──►  results/
   eodhd.py             (raw cache)          qlib_dump.py                       strategies.py + engine.py   report.py
```

1. **`eodhd.py`** downloads daily OHLCV + `adjusted_close` for each ticker (`SPY` → `SPY.US`).
2. **`qlib_dump.py`** writes Qlib's on-disk format (`calendars/`, `instruments/`, `features/*.day.bin`).
   Prices are split/dividend adjusted and `factor = adjusted_close / close` is stored, following Qlib's convention.
3. **`strategies.py`** computes signals with Qlib expressions (e.g. `Mean($close, 200)`) and turns them
   into target portfolio weights. Anything not allocated stays in cash.
4. **`engine.py`** runs those weights through Qlib's `SimulatorExecutor` + `Exchange` (fills, costs,
   positions). A signal decided on day *t*'s close is traded on day *t+1*, so there is no look-ahead.
5. **`report.py`** prints a summary table and saves CSVs plus an equity/drawdown chart.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your EODHD key in .env
```

## Usage

```bash
# one shot: download → convert → backtest every strategy
python -m backtest_tool run

# or step by step
python -m backtest_tool fetch --symbols SPY TQQQ SOXX SMH --data-start 2008-01-01
python -m backtest_tool dump
python -m backtest_tool backtest --strategy sma_trend momentum --start 2015-01-01 --cost-bps 2
```

Useful `backtest` options:

| option | default | meaning |
|---|---|---|
| `--strategy` | `all` | any of `buy_and_hold equal_weight sma_trend momentum` |
| `--start / --end` | `2012-01-01` / today | backtest window (2 extra years are loaded for indicator warm-up) |
| `--symbols` | `SPY TQQQ SOXX SMH` | universe |
| `--cost-bps` | `5` | commission + slippage per side, in basis points |
| `--deal-price` | `close` | fill at next day's `close` or `open` |
| `--sma-window` | `200` | SMA length for `sma_trend` |
| `--mom-lookback` / `--top-k` | `63` / `1` | momentum lookback days and number of ETFs held |
| `--rebalance` | `M` | `W`, `M` or `Q` for `equal_weight` and `momentum` |

Output in `results/`: `summary.csv`, `<strategy>_daily.csv` (Qlib's daily account report),
`<strategy>_rebalances.csv` (target weights on each rebalance) and `equity.png`.

## Built-in strategies

| name | rule |
|---|---|
| `buy_and_hold` | 1/N of capital in each ETF on day one, never rebalanced |
| `equal_weight` | 1/N in each ETF, rebalanced every period |
| `sma_trend` | each ETF gets 1/N while its close is above its N-day SMA, otherwise that slice sits in cash |
| `momentum` | each period hold the top-k ETFs by trailing return, only if that return is positive (else cash) |

### Adding a strategy

Write a function returning a DataFrame of target weights (index = decision date, columns = symbols,
rows sum to ≤ 1) and register it in `STRATEGIES`:

```python
def dip_buy(symbols, start, end, **_):
    data = load_fields(symbols, {"close": "$close", "ma": "Mean($close, 10)"}, start, end)
    weights = (data["close"] < data["ma"] * 0.95).astype(float) / len(symbols)
    return keep_changes(weights)   # only emit rows where the allocation changes

STRATEGIES["dip_buy"] = dip_buy
```

## Notes

- `SPY500` in the original request is interpreted as **SPY**, the SPDR S&P 500 ETF.
- TQQQ starts in Feb 2010; with the 2-year warm-up the default `--start 2012-01-01` gives every ETF full history.
- Qlib needs one trading day after the backtest end, so `--end` is clamped to the second-to-last available day.
- Returns in the summary are compounded, net of costs, annualised with 252 days.

## Tests

```bash
python -m pytest
```

Tests use synthetic price data (including a stock split) so they run without an API key or network.
