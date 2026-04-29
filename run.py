#!/usr/bin/env python3
"""run.py — Full pipeline: fetch → filter → backtest.

Commands:
    python run.py demo                      Run on synthetic data (no network)
    python run.py fetch -o data.pkl         Fetch real data via pykalshi
    python run.py backtest data.pkl         Backtest against saved data
    python run.py sweep data.pkl            Sweep threshold parameters

All commands accept filter + backtest flags (see --help).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import random
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import Dataset, save_dataset, load_dataset
from filters import FilterConfig, scan_opportunities, summarize_opportunities
from backtest import BacktestConfig, run_backtest, print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic data (for demo without pykalshi/network)
# ─────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_markets: int = 200, seed: int = 42) -> Dataset:
    """Generate a realistic Dataset for offline testing."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    series_pool = ["KXNFL", "KXOSC", "KXNFL", "KXSOTU", "KXYT", "KXNFLGAME", "KXEARN"]
    market_rows = []
    candles: dict[str, pd.DataFrame] = {}

    for i in range(n_markets):
        series = rng.choice(series_pool)
        ticker = f"{series}-SYN-{i:04d}"
        true_p = rng.uniform(0.03, 0.45)
        result = "yes" if rng.random() < true_p else "no"
        volume = rng.randint(500, 25000)

        # Close time spread across 3 months
        close_dt = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 90))
        close_ts = close_dt.timestamp()

        # Generate 15-40 daily candles
        n_candles = rng.randint(15, 40)
        price = true_p + rng.gauss(0, 0.06)
        price = max(0.03, min(0.97, price))

        candle_rows = []
        for day in range(n_candles):
            # Drift toward outcome in final 30% of candles
            if day > n_candles * 0.7:
                target = 0.93 if result == "yes" else 0.07
                price += (target - price) * rng.uniform(0.08, 0.35)
            else:
                price += rng.gauss(0, 0.04)
            price = max(0.02, min(0.98, price))

            spread = rng.uniform(0.02, 0.06)
            ts = close_ts - (n_candles - day) * 86400

            candle_rows.append({
                "timestamp": ts,
                "yes_bid_close": max(0.01, price - spread / 2),
                "yes_ask_close": min(0.99, price + spread / 2),
                "price_close": price,
                "volume": rng.randint(20, 800),
                "open_interest": rng.randint(10, 400),
            })

        candles[ticker] = pd.DataFrame(candle_rows)

        market_rows.append({
            "ticker": ticker,
            "title": f"Synthetic {series} #{i}",
            "subtitle": "",
            "status": "settled",
            "result": result,
            "series_ticker": series,
            "event_ticker": f"EVT-{i:04d}",
            "close_time": close_dt.isoformat(),
            "yes_bid": 0.0,
            "yes_ask": 0.0,
            "volume": volume,
            "open_interest": rng.randint(50, 500),
        })

    markets_df = pd.DataFrame(market_rows)
    return Dataset(markets_df=markets_df, candles=candles, fetched_at="synthetic")


# ─────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────

def cmd_demo(args):
    """Full pipeline on synthetic data."""
    logger.info("Generating synthetic dataset (%d markets)...", args.n)
    ds = generate_synthetic_dataset(n_markets=args.n, seed=args.seed)
    print(f"\n{ds.summary()}")

    fc = _filter_config(args)
    opps = scan_opportunities(ds.markets_df, ds.candles, fc)
    summarize_opportunities(opps, fc)

    if opps.empty:
        return

    bc = _backtest_config(args)
    results = run_backtest(opps, bc)
    print_report(results)


def cmd_fetch(args):
    """Fetch data from Kalshi via pykalshi."""
    from fetch import fetch_dataset

    kwargs = {}
    if args.tickers:
        kwargs["tickers"] = args.tickers.split(",")
    if args.series:
        kwargs["series_tickers"] = args.series.split(",")
    if args.events:
        kwargs["event_tickers"] = args.events.split(",")

    if not kwargs:
        print("Provide at least one of: --tickers, --series, --events")
        print("Example: python run.py fetch --series KXNFL,KXOSC -o data.pkl")
        sys.exit(1)

    fetch_dataset(
        **kwargs,
        market_limit=args.limit,
        candle_period=args.candle_period,
        output_path=args.output,
    )
    print(f"\nSaved to {args.output}")
    print(f"Next: python run.py backtest {args.output}")


def cmd_backtest(args):
    """Backtest against a saved dataset."""
    ds = load_dataset(args.data_path)

    fc = _filter_config(args)
    opps = scan_opportunities(ds.markets_df, ds.candles, fc)
    summarize_opportunities(opps, fc)

    if opps.empty:
        return

    bc = _backtest_config(args)
    results = run_backtest(opps, bc)
    print_report(results)


def cmd_sweep(args):
    """Sweep across filter thresholds."""
    if os.path.exists(args.data_path):
        ds = load_dataset(args.data_path)
    else:
        logger.info("No dataset found at %s, using synthetic data.", args.data_path)
        ds = generate_synthetic_dataset(300, seed=42)

    bc = _backtest_config(args)

    low_vals = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    high_vals = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]

    print(f"\n{'Low':>6} {'High':>6} {'Trades':>7} {'Win%':>7} {'P&L':>10} {'ROI':>8} "
          f"{'Sharpe':>7} {'PF':>6}")
    print("-" * 65)

    best_roi, best_params = -999, (0, 0)

    for low in low_vals:
        for high in high_vals:
            fc = FilterConfig(yes_max=low, yes_min=high,
                              min_market_volume=args.min_market_vol,
                              entry_mode=args.entry_mode)
            opps = scan_opportunities(ds.markets_df, ds.candles, fc)
            if opps.empty:
                continue
            r = run_backtest(opps, bc)
            if r.total_trades == 0:
                continue
            print(f"{low:>6.2f} {high:>6.2f} {r.total_trades:>7} {r.win_rate:>7.1%} "
                  f"${r.total_pnl:>+9.2f} {r.roi:>+7.1%} {r.sharpe:>7.2f} "
                  f"{r.profit_factor:>6.2f}")
            if r.roi > best_roi:
                best_roi = r.roi
                best_params = (low, high)

    print(f"\nBest: low={best_params[0]:.2f}, high={best_params[1]:.2f} → ROI={best_roi:+.1%}")


# ─────────────────────────────────────────────────────────────────────────
# Arg helpers
# ─────────────────────────────────────────────────────────────────────────

def _filter_config(args) -> FilterConfig:
    return FilterConfig(
        yes_max=args.low_threshold,
        yes_min=args.high_threshold,
        min_candle_volume=args.min_candle_vol,
        min_open_interest=args.min_oi,
        min_market_volume=args.min_market_vol,
        min_hours_to_expiry=args.min_hours,
        max_hours_to_expiry=args.max_hours,
        entry_mode=args.entry_mode,
    )


def _backtest_config(args) -> BacktestConfig:
    return BacktestConfig(
        bankroll=args.bankroll,
        sizing_mode=args.sizing,
        fixed_pct=args.fixed_pct,
        fixed_dollar=args.fixed_dollar,
        kelly_fraction=args.kelly_frac,
        role=args.role,
    )


def _add_common(p):
    # Filter args
    p.add_argument("--low-threshold", type=float, default=0.15, help="Buy NO when YES ≤ this (default: 0.15)")
    p.add_argument("--high-threshold", type=float, default=0.85, help="Buy YES when YES ≥ this (default: 0.85)")
    p.add_argument("--min-candle-vol", type=int, default=0, help="Min volume per candle (default: 0)")
    p.add_argument("--min-oi", type=int, default=0, help="Min open interest (default: 0)")
    p.add_argument("--min-market-vol", type=int, default=500, help="Min total market volume (default: 500)")
    p.add_argument("--min-hours", type=float, default=1.0, help="Min hours to expiry (default: 1)")
    p.add_argument("--max-hours", type=float, default=0, help="Max hours to expiry; 0=no limit (default: 0)")
    p.add_argument("--entry-mode", choices=["first", "best", "all"], default="first")

    # Backtest args
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--sizing", choices=["fixed_pct", "fixed_dollar", "kelly"], default="fixed_pct")
    p.add_argument("--fixed-pct", type=float, default=0.05, help="Pct of bankroll per trade (default: 0.05)")
    p.add_argument("--fixed-dollar", type=float, default=20.0, help="$/trade for fixed_dollar mode")
    p.add_argument("--kelly-frac", type=float, default=0.25, help="Kelly multiplier (default: 0.25)")
    p.add_argument("--role", choices=["taker", "maker"], default="taker")


def main():
    parser = argparse.ArgumentParser(description="Extreme-price strategy pipeline (pykalshi)")
    sub = parser.add_subparsers(dest="command", required=True)

    # demo
    demo_p = sub.add_parser("demo", help="Run full pipeline on synthetic data")
    demo_p.add_argument("-n", type=int, default=200)
    demo_p.add_argument("--seed", type=int, default=42)
    _add_common(demo_p)

    # fetch
    fetch_p = sub.add_parser("fetch", help="Fetch data from Kalshi API via pykalshi")
    fetch_p.add_argument("--tickers", type=str, default="", help="Comma-separated market tickers")
    fetch_p.add_argument("--series", type=str, default="", help="Comma-separated series tickers")
    fetch_p.add_argument("--events", type=str, default="", help="Comma-separated event tickers")
    fetch_p.add_argument("-o", "--output", type=str, default="data.pkl")
    fetch_p.add_argument("--limit", type=int, default=200, help="Max markets per query")
    fetch_p.add_argument("--candle-period", choices=["1m", "1h", "1d"], default="1d")
    _add_common(fetch_p)

    # backtest
    bt_p = sub.add_parser("backtest", help="Backtest against saved dataset")
    bt_p.add_argument("data_path", type=str, help="Path to .pkl dataset")
    _add_common(bt_p)

    # sweep
    sw_p = sub.add_parser("sweep", help="Sweep threshold parameters")
    sw_p.add_argument("data_path", type=str, nargs="?", default="data.pkl")
    _add_common(sw_p)

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "sweep":
        cmd_sweep(args)


if __name__ == "__main__":
    main()
