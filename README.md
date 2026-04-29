# KalshiBot

Two parallel pipelines for evaluating short-side / insurance-writing strategies on Kalshi prediction markets:

1. **Historical backtest pipeline** — fetch settled markets, save to disk, replay through filters + sizing engine.
2. **Live paper-trading pipeline** — pull live production market data, log simulated trades to SQLite, resolve on settlement.

The two pipelines share `config.py` and a few helpers but are independently runnable.

---

## File Map

### Pipeline 1 — Historical Backtest

| File | Role |
|---|---|
| `fetch.py` | Fetches market metadata + candle history via `pykalshi`, packages into a `Dataset`, pickles to disk |
| `filters.py` | `FilterConfig` + `scan_opportunities()` — applies price/volume/time-to-expiry filters and emits an opportunities DataFrame |
| `backtest.py` | `BacktestConfig` + `run_backtest()` — sizes positions, simulates resolution, tracks bankroll with **proper capital locking** (open positions hold capital until `close_ts`) |
| `diagnose.py` | Inspect a `.pkl` dataset to see why filters aren't matching |
| `run.py` | CLI orchestrator: `demo`, `fetch`, `backtest`, `sweep` |
| `*.pkl` | Saved datasets (`Sports.pkl`, `Entertainment.pkl`, `SoccerUFC.pkl`, `Trump.pkl`, …) |

### Pipeline 2 — Live Paper Trading

| File | Role |
|---|---|
| `client.py` | `KalshiClient` — wraps the production REST API |
| `scanner.py` | `MarketScanner` — filters live markets by volume, time-to-expiry, YES price band |
| `trade_log.py` | `TradeLog` — SQLite persistence for simulated trades + open positions |
| `paper_trader.py` | CLI: `scan`, `resolve`, `stats`, `positions`, `watch`, `export` |
| `paper_trades.db` | SQLite store (with WAL/SHM sidecars) |

### Shared

| File | Role |
|---|---|
| `config.py` | `ScannerConfig`, `PositionConfig`, `PaperTradingConfig`, `DB_PATH` |
| `models.py` | `Market` and related dataclasses |
| `Docs/` | `GUIDE.md`, `READ_ONLY.txt` — strategy notes |

> **Note on layout:** the two pipelines are intentionally left flat (no `historical/` and `live/` subfolders) to keep imports simple. A future refactor could split them — see the **Suggested Reorganization** section at the bottom.

---

## Installation

Python 3.10+. Core deps:

```bash
pip install pandas numpy pykalshi requests
```

`pykalshi` is required for `fetch.py` and the live client.

---

## Pipeline 1 — Historical Backtest

End-to-end flow: **fetch → save `.pkl` → filter → backtest → report**.

### Step 1 — Fetch & Save a Dataset

```bash
python run.py fetch -o Sports.pkl --series KXNFL KXNFLGAME --max-markets 500
```

`fetch.py` pulls market metadata + per-market candle history through `pykalshi` and pickles a `Dataset(markets_df, candles_dict)` to disk.

### Step 2 — Backtest

```bash
python run.py backtest Sports.pkl --bankroll 1000 --fixed-pct 0.05
```

Loads the dataset, runs `scan_opportunities()` to produce a candidate list, then runs `run_backtest()` to simulate trade-by-trade resolution.

**Key backtest behavior (recently fixed):** capital from an entered trade is **locked** in `open_positions` until that market's `close_ts` (= `entry_ts + hours_to_expiry × 3600`). Before sizing each new entry, settled positions are credited back to the bankroll. This eliminates the hidden leverage that existed when payouts were credited instantly. New trade-log fields: `close_ts`, `bankroll_after_entry`, `open_cost_at_entry`.

### Step 3 — Parameter Sweep

```bash
python run.py sweep Sports.pkl
```

Runs the backtest over a grid of filter / sizing thresholds and prints a comparison table.

### Step 4 — Diagnose

```bash
python diagnose.py SoccerUFC.pkl
```

Dumps per-market filter pass/fail counts so you can see which threshold is too tight.

### Offline Demo (no network)

```bash
python run.py demo
```

Generates a synthetic dataset and runs the full pipeline. Useful for verifying installation.

### Tuning Knobs (`FilterConfig` / `BacktestConfig`)

| Knob | File | Effect |
|---|---|---|
| `cutoff`, `yes_min`, `yes_max` | `FilterConfig` | YES-price band that defines NO-buy vs YES-buy zones |
| `min_candle_volume`, `min_open_interest`, `min_market_volume` | `FilterConfig` | Liquidity floors |
| `min_hours_to_expiry`, `max_hours_to_expiry` | `FilterConfig` | Don't enter too close to / too far from settlement |
| `entry_mode` | `FilterConfig` | `all` / `first` / `best` — how many entries per market |
| `sizing_mode` | `BacktestConfig` | `fixed_pct` / `fixed_dollar` / `kelly` |
| `fixed_pct`, `fixed_dollar`, `kelly_fraction`, `kelly_edge_assumption` | `BacktestConfig` | Sizing parameters |
| `max_position_pct`, `max_exposure_pct` | `BacktestConfig` | Per-trade and total exposure caps (against equity = cash + open cost) |
| `role` | `BacktestConfig` | `taker` charges fees per contract; `maker` does not |

---

## Pipeline 2 — Live Paper Trading

End-to-end flow: **scan live → log to SQLite → resolve on settlement → report**.

### One-Shot Scan

```bash
python paper_trader.py scan
python paper_trader.py scan --series KXNFL KXOSC
```

Pulls live markets via `KalshiClient`, applies `MarketScanner`, and inserts simulated trades into `paper_trades.db`. Side is determined by where YES sits within the configured band:
- `cutoff ≤ YES ≤ yes_max` → buy NO
- `yes_min ≤ YES ≤ 1 − cutoff` → buy YES

### Continuous Mode

```bash
python paper_trader.py watch --interval 300
```

Loops the scanner every N seconds.

### Resolve Settled Trades

```bash
python paper_trader.py resolve
```

Walks open positions, asks the API for settlement, computes P&L, updates the DB.

### Reporting

```bash
python paper_trader.py stats        # win rate, P&L, ROI
python paper_trader.py positions    # currently-open positions
python paper_trader.py export trades.csv
```

### Configuration

All knobs live in `config.py`:

| Class | Controls |
|---|---|
| `ScannerConfig` | Price band, liquidity floors, time-to-expiry window |
| `PositionConfig` | Flat dollar size, fee model |
| `PaperTradingConfig` | Series filter, scan cadence |
| `DB_PATH` | SQLite file location (`paper_trades.db`) |

---

## Common Recipes

```bash
# Refetch sports + entertainment, save separately
python run.py fetch -o Sports.pkl --series KXNFL KXNFLGAME
python run.py fetch -o Entertainment.pkl --series KXOSC KXYT

# Backtest with tight filters and Kelly sizing
python run.py backtest Sports.pkl \
    --cutoff 0.08 --yes-max 0.20 \
    --min-volume 5000 --min-hours 12 \
    --sizing kelly --kelly-fraction 0.25

# Live: continuous paper trading every 5 minutes, then resolve hourly
python paper_trader.py watch --interval 300 &
# in another shell:
python paper_trader.py resolve
```

---

## Suggested Reorganization (not applied)

If you ever want to split the two pipelines into subpackages, a clean split would be:

```
KalshiBot/
├── shared/
│   ├── config.py
│   ├── models.py
│   └── client.py            # used by both (live data + fetch)
├── historical/
│   ├── fetch.py
│   ├── filters.py
│   ├── backtest.py
│   ├── diagnose.py
│   ├── run.py
│   └── data/                # *.pkl files
├── live/
│   ├── scanner.py
│   ├── trade_log.py
│   ├── paper_trader.py
│   └── paper_trades.db
├── Docs/
└── README.md
```

This would require updating ~11 imports and the `sys.path.insert` shims in `run.py` / `paper_trader.py` / `diagnose.py`. Skipped per request.

---

## Tests / Sanity Checks

```bash
python run.py demo                     # synthetic end-to-end
python diagnose.py Sports.pkl          # show filter funnel
python paper_trader.py stats           # check DB integrity
```
