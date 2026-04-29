# Extreme-Price Strategy Pipeline (pykalshi)

## Architecture

```
fetch.py          →  filters.py          →  backtest.py
(pykalshi API)       (scan candles)          (simulate trades)
     ↓                    ↓                       ↓
  Dataset.pkl        opportunities_df       BacktestResults
  (markets_df +      (ticker, side,         (trades_df, equity,
   candles dict)      entry_price, ...)      win rate, Sharpe)
```

Three files, one runner (`run.py`). Each step is independently importable.

## Quick start

```bash
pip install pykalshi[dataframe] pandas numpy

# 1. Demo (no network, no pykalshi needed)
python run.py demo

# 2. Fetch real data
python run.py fetch --series KXNFL,KXOSC -o data.pkl

# 3. Backtest
python run.py backtest data.pkl

# 4. Sweep thresholds
python run.py sweep data.pkl
```

## Step 1: Fetch (`fetch.py`)

Uses pykalshi's bound method `market.get_candlesticks(start, end, period)` and
`.to_dataframe()` for pandas-native output. Saves a `Dataset` pickle containing
`markets_df` (one row per market with settlement result) and `candles` (dict
mapping ticker → DataFrame of OHLC candles).

Three ways to select markets:

```bash
# By series (all settled NFL + Oscar markets)
python run.py fetch --series KXNFL,KXOSC -o data.pkl

# By event (all markets in one specific event)
python run.py fetch --events KXNFL-25SB -o superbowl.pkl

# By individual ticker
python run.py fetch --tickers KXNFL-25SB-ROMO-STARBUCKS,KXOSC-26-BESTPIC -o picks.pkl
```

Options: `--limit 500` (max markets per query), `--candle-period 1h` (1m/1h/1d).

**You only fetch once.** After that, everything is offline from the pickle.

No API credentials needed — market data and candlesticks are public endpoints.
pykalshi handles rate limiting and pagination automatically.

## Step 2: Filter (`filters.py`)

Scans every market's candlestick history and returns a DataFrame of trade
opportunities — specific candle periods where all filters pass.

### Filter dimensions

| Filter | Flag | Default | What it does |
|--------|------|---------|--------------|
| Price (low) | `--low-threshold` | 0.15 | Buy NO when YES ≤ this |
| Price (high) | `--high-threshold` | 0.85 | Buy YES when YES ≥ this |
| Candle volume | `--min-candle-vol` | 0 | Skip candles with no activity |
| Open interest | `--min-oi` | 0 | Skip illiquid periods |
| Market volume | `--min-market-vol` | 500 | Skip thin markets entirely |
| Min hours to exp | `--min-hours` | 1.0 | Don't enter right before settlement |
| Max hours to exp | `--max-hours` | 0 (off) | Only enter near expiry |

### Entry modes

| Mode | Flag | Behavior |
|------|------|----------|
| First signal | `--entry-mode first` | Enter on the first qualifying candle (default) |
| Best price | `--entry-mode best` | Enter at the most extreme price seen |
| All signals | `--entry-mode all` | Every qualifying candle is a separate trade |

### Programmatic usage

```python
from fetch import load_dataset
from filters import FilterConfig, scan_opportunities

ds = load_dataset("data.pkl")
opps = scan_opportunities(ds.markets_df, ds.candles, FilterConfig(
    yes_max=0.10,
    yes_min=0.90,
    min_candle_volume=50,
    min_hours_to_expiry=4,
    entry_mode="best",
))
print(opps[["ticker", "side", "yes_mid", "entry_price", "hours_to_expiry"]])
```

## Step 3: Backtest (`backtest.py`)

Takes the opportunities DataFrame, sizes positions, deducts fees, resolves
against settlement outcomes. Assumes hold-to-expiration.

### Position sizing modes

| Mode | Flag | How it works |
|------|------|-------------|
| Fixed % | `--sizing fixed_pct --fixed-pct 0.05` | 5% of current bankroll per trade |
| Fixed $ | `--sizing fixed_dollar --fixed-dollar 20` | $20 per trade |
| Kelly | `--sizing kelly --kelly-frac 0.25` | Quarter-Kelly with assumed 5% edge |

### Fee modes

| Role | Flag | Fee |
|------|------|-----|
| Taker | `--role taker` | `ceil(0.07 × P × (1-P) × 100) / 100` per contract |
| Maker | `--role maker` | $0 (limit orders on most markets) |

### Output

The report shows: summary line, side breakdown (NO vs YES performance),
series breakdown (which market categories are profitable), sample trade log,
and equity curve.

### Programmatic usage

```python
from backtest import BacktestConfig, run_backtest, print_report

results = run_backtest(opps, BacktestConfig(
    bankroll=2000,
    sizing_mode="fixed_pct",
    fixed_pct=0.03,
    role="maker",
))
print_report(results)

# Access raw data
results.trades_df  # full DataFrame of every trade
results.win_rate
results.sharpe
results.max_drawdown
results.profit_factor
```

## Parameter sweep

Tests all combinations of low (0.05–0.20) × high (0.80–0.95) thresholds:

```bash
python run.py sweep data.pkl --sizing fixed_pct --fixed-pct 0.03
```

Prints a grid showing trades, win rate, P&L, ROI, Sharpe, and profit factor
for each combination. Use this to find which thresholds are robust vs.
overfit to one dataset.

## What the synthetic data tells you (and doesn't)

The `demo` command generates markets where prices drift toward the true outcome
in the final 30% of candles. This means the extreme-price signals are
realistic in structure but the base rates, correlations, and fill dynamics
are fabricated. A strategy that works on synthetic data may fail on real data
(and vice versa). The synthetic demo exists to verify the pipeline mechanics,
not to validate edge. Always confirm on real data via `fetch` → `backtest`.
