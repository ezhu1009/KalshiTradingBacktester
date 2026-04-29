"""backtest.py — Backtest engine for the extreme-price strategy.

Takes the opportunity DataFrame from filters.scan_opportunities(),
applies position sizing and fee logic, resolves each trade against
settlement data, and produces a full performance report.

Assumes hold-to-expiration (no early exits).

Usage:
    from backtest import BacktestConfig, run_backtest, print_report

    results = run_backtest(opportunities_df, BacktestConfig(bankroll=1000))
    print_report(results)
"""
from __future__ import annotations

import logging
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """All backtest parameters."""
    bankroll: float = 1000.0

    # Position sizing
    sizing_mode: str = "fixed_pct"   # "fixed_pct", "fixed_dollar", "kelly"
    fixed_pct: float = 0.05          # 5% of bankroll per trade (for fixed_pct)
    fixed_dollar: float = 20.0       # $/trade (for fixed_dollar)
    kelly_fraction: float = 0.25     # quarter-Kelly multiplier (for kelly)
    kelly_edge_assumption: float = 0.05  # assumed edge for Kelly calc

    # Fees
    role: str = "taker"              # "taker" or "maker"
    # Kalshi taker fee: ceil(0.07 * P * (1-P) * 100) / 100

    # Risk limits
    max_position_pct: float = 0.25   # hard cap per trade
    max_exposure_pct: float = 0.95   # max total deployed


# ─────────────────────────────────────────────────────────────────────────
# Fee calc (same formula as edge_calculator.py)
# ─────────────────────────────────────────────────────────────────────────

def taker_fee(price: float) -> float:
    return math.ceil(0.07 * price * (1 - price) * 100) / 100


# ─────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    trades_df: pd.DataFrame          # full trade log
    config: BacktestConfig
    initial_bankroll: float
    final_bankroll: float

    @property
    def total_trades(self) -> int:
        return len(self.trades_df)

    @property
    def wins(self) -> int:
        return int(self.trades_df["won"].sum()) if self.total_trades else 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades else 0

    @property
    def total_pnl(self) -> float:
        return float(self.trades_df["pnl"].sum()) if self.total_trades else 0

    @property
    def roi(self) -> float:
        return self.total_pnl / self.initial_bankroll if self.initial_bankroll else 0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades else 0

    @property
    def max_drawdown(self) -> float:
        if not self.total_trades:
            return 0
        equity = self.initial_bankroll + self.trades_df["pnl"].cumsum()
        peak = equity.cummax()
        dd = (peak - equity) / peak
        return float(dd.max())

    @property
    def sharpe(self) -> float:
        if self.total_trades < 2:
            return 0
        rets = self.trades_df["pnl"] / (self.trades_df["entry_price"] * self.trades_df["contracts"])
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) < 2 or rets.std() == 0:
            return 0
        return float((rets.mean() / rets.std()) * (250 ** 0.5))

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_win = float(self.trades_df.loc[self.trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss = float(self.trades_df.loc[self.trades_df["pnl"] < 0, "pnl"].sum())
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0
        return abs(gross_win / gross_loss)

    def summary(self) -> str:
        return (
            f"{self.total_trades} trades | "
            f"Win rate: {self.win_rate:.1%} | "
            f"P&L: ${self.total_pnl:+.2f} | "
            f"ROI: {self.roi:+.1%} | "
            f"Max DD: {self.max_drawdown:.1%} | "
            f"Sharpe: {self.sharpe:.2f} | "
            f"Profit factor: {self.profit_factor:.2f} | "
            f"Final: ${self.final_bankroll:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────

def run_backtest(
    opportunities: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResults:
    """Run the backtest.

    Takes the DataFrame from scan_opportunities() (must have: ticker, side,
    entry_price, yes_mid, market_result, timestamp). Processes trades in
    timestamp order. Assumes hold-to-expiration.
    """
    if config is None:
        config = BacktestConfig()

    if opportunities.empty:
        empty = pd.DataFrame()
        return BacktestResults(empty, config, config.bankroll, config.bankroll)

    # Sort by timestamp for chronological processing
    opps = opportunities.sort_values("timestamp").reset_index(drop=True)

    bankroll = config.bankroll  # free cash
    trades = []
    # Each entry: {"close_ts": unix_seconds, "payout": float, "cost": float}
    open_positions: list[dict] = []

    def settle_due(now_ts: float) -> None:
        """Credit payouts of any positions whose close time has passed."""
        nonlocal bankroll, open_positions
        still_open = []
        for pos in open_positions:
            if pos["close_ts"] <= now_ts:
                bankroll += pos["payout"]
            else:
                still_open.append(pos)
        open_positions = still_open

    for _, row in opps.iterrows():
        side = row["side"]
        entry_price = float(row["entry_price"])
        yes_mid = float(row["yes_mid"])
        result = row["market_result"]
        ticker = row["ticker"]
        entry_ts = float(row["timestamp"])
        hours_to_expiry = float(row.get("hours_to_expiry", 0) or 0)
        close_ts = entry_ts + hours_to_expiry * 3600

        if entry_price <= 0 or entry_price >= 1:
            continue

        # Free capital from any positions that have settled by this entry time
        settle_due(entry_ts)

        # Equity = free cash + capital locked in open positions (at cost basis)
        open_cost = sum(p["cost"] for p in open_positions)
        equity = bankroll + open_cost

        # ── Position sizing ─────────────────────────────────────────
        if config.sizing_mode == "fixed_pct":
            bet = config.fixed_pct * equity
        elif config.sizing_mode == "fixed_dollar":
            bet = config.fixed_dollar
        elif config.sizing_mode == "kelly":
            # Rough Kelly using assumed edge
            if side == "NO":
                win_p = (1 - yes_mid) + config.kelly_edge_assumption
            else:
                win_p = yes_mid + config.kelly_edge_assumption
            win_p = min(0.99, max(0.01, win_p))
            odds = (1 - entry_price) / entry_price if entry_price > 0 else 0
            kf = max(0, (win_p * odds - (1 - win_p)) / odds) if odds > 0 else 0
            bet = kf * config.kelly_fraction * equity
        else:
            bet = config.fixed_pct * equity

        # Caps (per-trade and total exposure)
        bet = min(bet, config.max_position_pct * equity)
        max_total_exposure = config.max_exposure_pct * equity
        remaining_exposure = max(0.0, max_total_exposure - open_cost)
        bet = min(bet, remaining_exposure)
        # Hard cap: cannot bet more free cash than we have
        bet = min(bet, bankroll)

        if bet <= 0:
            continue

        contracts = int(bet / entry_price)
        if contracts <= 0:
            continue

        cost = contracts * entry_price
        fee = taker_fee(entry_price) * contracts if config.role == "taker" else 0.0

        # Fee is paid up front; if we can't afford fee + cost, skip
        if cost + fee > bankroll:
            continue

        # ── Entry (deduct cost + fee immediately, lock capital) ─────
        bankroll -= cost
        bankroll -= fee

        won = (side == "NO" and result == "no") or (side == "YES" and result == "yes")
        payout = 1.0 * contracts if won else 0.0
        pnl = payout - cost - fee

        # Lock the position — payout is credited on settle_due()
        open_positions.append({
            "close_ts": close_ts,
            "payout": payout,
            "cost": cost,
        })

        trades.append({
            "ticker": ticker,
            "timestamp": entry_ts,
            "close_ts": close_ts,
            "side": side,
            "entry_price": entry_price,
            "yes_mid": yes_mid,
            "contracts": contracts,
            "cost": cost,
            "fee": fee,
            "market_result": result,
            "won": won,
            "pnl": pnl,
            "bankroll_after_entry": bankroll,
            "open_cost_at_entry": open_cost + cost,
            "series_ticker": row.get("series_ticker", ""),
            "hours_to_expiry": row.get("hours_to_expiry", 0),
        })

    # Drain any positions still open at the end of the run
    settle_due(float("inf"))

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return BacktestResults(
        trades_df=trades_df,
        config=config,
        initial_bankroll=config.bankroll,
        final_bankroll=bankroll,
    )


# ─────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────

def print_report(results: BacktestResults) -> None:
    """Print full backtest report."""
    cfg = results.config
    df = results.trades_df

    print(f"\n{'=' * 72}")
    print("BACKTEST REPORT")
    print(f"{'=' * 72}")

    print(f"\nConfig: sizing={cfg.sizing_mode}, role={cfg.role}, "
          f"bankroll=${cfg.bankroll:.0f}")
    if cfg.sizing_mode == "fixed_pct":
        print(f"        {cfg.fixed_pct:.0%} of bankroll per trade")
    elif cfg.sizing_mode == "fixed_dollar":
        print(f"        ${cfg.fixed_dollar:.0f} per trade")
    elif cfg.sizing_mode == "kelly":
        print(f"        {cfg.kelly_fraction:.0%} Kelly, edge assumption={cfg.kelly_edge_assumption:.1%}")

    print(f"\n{results.summary()}")

    if df.empty:
        return

    # ── By side ─────────────────────────────────────────────────────
    print(f"\nBy side:")
    print(f"  {'Side':<6} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'P&L':>10} {'Avg Entry':>10}")
    print(f"  {'-' * 50}")
    for side in ("NO", "YES"):
        sub = df[df["side"] == side]
        if sub.empty:
            continue
        wins = int(sub["won"].sum())
        print(f"  {side:<6} {len(sub):>7} {wins:>6} {wins/len(sub):>7.1%} "
              f"${sub['pnl'].sum():>+9.2f} {sub['entry_price'].mean():>10.3f}")

    # ── By series ───────────────────────────────────────────────────
    if "series_ticker" in df.columns and df["series_ticker"].nunique() > 1:
        print(f"\nBy series:")
        print(f"  {'Series':<15} {'Trades':>7} {'Win%':>7} {'P&L':>10}")
        print(f"  {'-' * 42}")
        for series, sub in df.groupby("series_ticker"):
            if not series:
                continue
            wins = int(sub["won"].sum())
            print(f"  {series:<15} {len(sub):>7} {wins/len(sub):>7.1%} "
                  f"${sub['pnl'].sum():>+9.2f}")

    # ── Sample trades ───────────────────────────────────────────────
    n = min(15, len(df))
    print(f"\nFirst {n} trades:")
    print(f"  {'Ticker':<35} {'Side':<4} {'Entry':>6} {'YES':>6} {'W':>2} {'P&L':>8} {'Bank':>8}")
    print(f"  {'-' * 75}")
    for _, t in df.head(n).iterrows():
        w = "✓" if t["won"] else "✗"
        print(f"  {t['ticker']:<35} {t['side']:<4} {t['entry_price']:>6.3f} "
              f"{t['yes_mid']:>6.3f} {w:>2} ${t['pnl']:>+7.2f} ${t['bankroll_after_entry']:>7.0f}")

    # ── Equity curve ────────────────────────────────────────────────
    print(f"\nEquity curve:")
    equity = results.initial_bankroll + df["pnl"].cumsum()
    step = max(1, len(equity) // 15)
    for i in range(0, len(equity), step):
        e = equity.iloc[i]
        bar = "█" * max(1, int(e / results.initial_bankroll * 30))
        print(f"  Trade {i+1:>4}: ${e:>8.2f} {bar}")
    # Always show final
    if len(equity) > 0:
        e = equity.iloc[-1]
        bar = "█" * max(1, int(e / results.initial_bankroll * 30))
        print(f"  Trade {len(equity):>4}: ${e:>8.2f} {bar}")
