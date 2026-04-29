"""paper_trader.py — Phase A paper trading engine (filter-only mode).

Reads LIVE production market data from Kalshi, filters by volume +
time-to-expiry + YES price, and logs simulated trades to SQLite.
Side is determined by price position:
  - YES in [cutoff, yes_max]     → buy NO
  - YES in [yes_min, 1-cutoff]   → buy YES
No base-rate assumptions. Flat position sizing.

Usage:
    python paper_trader.py scan                          # one-shot scan
    python paper_trader.py scan --series KXNFL KXOSC     # scan specific series
    python paper_trader.py resolve                       # resolve settled trades
    python paper_trader.py stats                         # show summary
    python paper_trader.py positions                     # show open positions
    python paper_trader.py watch --interval 300          # continuous mode
    python paper_trader.py export trades.csv             # export to CSV
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from config import ScannerConfig, PositionConfig, PaperTradingConfig, DB_PATH
from models import Market
from client import KalshiClient
from scanner import MarketScanner
from trade_log import TradeLog

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("paper_trader")


class PaperTrader:
    """Orchestrates the paper trading pipeline.

    1. Pull live market data from Kalshi production API
    2. Filter: volume ≥ threshold, closes within window, YES in price zone
    3. Determine side from price (NO for low YES, YES for high YES)
    4. Log simulated trades (flat position size) to SQLite
    5. Resolve settled trades and track P&L
    """

    def __init__(
        self,
        config: PaperTradingConfig | None = None,
        scanner_config: ScannerConfig | None = None,
        position_config: PositionConfig | None = None,
        db_path: Path | str = DB_PATH,
    ):
        self._config = config or PaperTradingConfig()
        self._pos_config = position_config or PositionConfig()

        sc = scanner_config or ScannerConfig()
        self._scanner = MarketScanner(
            cutoff=sc.cutoff,
            yes_max=sc.yes_max,
            yes_min=sc.yes_min,
            min_volume=sc.min_volume,
            min_oi=sc.min_oi,
            max_hours=sc.max_hours_to_close,
            min_hours=sc.min_hours_to_close,
        )

        self._client = KalshiClient()
        self._log = TradeLog(db_path)

    # -------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------

    def scan_and_trade(self, series_tickers: list[str] | None = None) -> int:
        """Run one scan cycle. Returns number of simulated trades placed."""
        logger.info("=" * 60)
        logger.info("Starting scan cycle")

        # Check current open positions
        open_tickers = self._log.get_tickers_with_open_trades()
        open_count = len(open_tickers)
        logger.info("Currently %d open positions", open_count)

        if open_count >= self._pos_config.max_positions:
            logger.info("At max positions (%d). Skipping scan.", self._pos_config.max_positions)
            return 0

        # 1. Pull live markets
        all_markets: list[Market] = []
        if series_tickers:
            for st in series_tickers:
                logger.info("Fetching markets for series: %s", st)
                for m in self._client.iter_markets(status="open", series_ticker=st):
                    all_markets.append(m)
        else:
            logger.info("Fetching all open markets...")
            for m in self._client.iter_markets(status="open"):
                all_markets.append(m)

        logger.info("Fetched %d open markets", len(all_markets))

        # 2. Filter
        candidates = self._scanner.scan(all_markets)
        logger.info("Scanner passed %d candidates (of %d)", len(candidates), len(all_markets))

        # 3. Log simulated trades
        trades_placed = 0
        for result in candidates:
            market = result.market
            side = result.side
            entry_price = result.entry_price

            # Skip if we already have an open position on this ticker
            if market.ticker in open_tickers:
                continue

            # Skip if we've hit the per-event limit
            event_count = self._log.get_event_trade_count(market.event_ticker)
            if event_count >= self._pos_config.max_per_event:
                continue

            # Skip if at max total positions
            if open_count + trades_placed >= self._pos_config.max_positions:
                break

            # Position sizing: flat number of contracts
            size = self._pos_config.contracts_per_trade
            position_dollars = size * entry_price

            # Grab orderbook snapshot
            try:
                orderbook = self._client.get_orderbook(market.ticker)
            except Exception as e:
                logger.warning("Failed to fetch orderbook for %s: %s", market.ticker, e)
                orderbook = None

            # Log the simulated trade
            trade_id = self._log.log_trade(
                market_ticker=market.ticker,
                market_title=market.title,
                event_ticker=market.event_ticker,
                series_ticker=market.series_ticker,
                category=market.category.value,
                side=side,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                yes_mid=market.yes_mid,
                no_mid=market.no_mid,
                spread=market.spread,
                volume=market.volume,
                open_interest=market.open_interest,
                close_time=market.close_time.isoformat(),
                hours_to_close=result.hours_to_close,
                entry_price=entry_price,
                position_size=size,
                position_dollars=position_dollars,
                orderbook_snapshot=orderbook,
            )

            open_tickers.add(market.ticker)
            trades_placed += 1

            logger.info(
                "  → #%d: %s %s | %d @ $%.2f ($%.2f) | YES=%.0f¢ | vol=%d | %.1fh left",
                trade_id, side.upper(), market.ticker, size, entry_price,
                position_dollars, market.yes_mid * 100, market.volume,
                result.hours_to_close,
            )

        # Log scan summary
        self._log.log_scan(len(all_markets), len(candidates), trades_placed)
        logger.info("Scan complete: %d trades placed", trades_placed)
        logger.info("=" * 60)
        return trades_placed

    def resolve_settled(self) -> int:
        """Check open trades against the API for settlement."""
        open_trades = self._log.get_open_trades()
        if not open_trades:
            logger.info("No open trades to resolve.")
            return 0

        logger.info("Checking %d open trades for settlement...", len(open_trades))
        resolved_count = 0

        for trade in open_trades:
            try:
                market = self._client.get_market(trade["market_ticker"])
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", trade["market_ticker"], e)
                continue

            if market.result is None:
                continue

            # P&L depends on which side we took
            entry_price = trade["entry_price"]
            size = trade["position_size"]
            side = trade["side"]

            won = (market.result == side)  # we win if the result matches our side

            if won:
                pnl = (1.0 - entry_price) * size
            else:
                pnl = -entry_price * size

            self._log.resolve_trade(trade["id"], market.result, pnl)
            resolved_count += 1

            logger.info(
                "  %s → %s | P&L: $%+.2f",
                trade["market_ticker"],
                "WIN" if won else "LOSS",
                pnl,
            )

        return resolved_count

    def print_stats(self):
        stats = self._log.get_stats()
        print("\n" + "=" * 50)
        print("  PAPER TRADING SUMMARY")
        print("=" * 50)
        print(f"  Total trades:       {stats['total_trades']}")
        print(f"  Resolved:           {stats['resolved']}")
        print(f"  Open:               {stats['open']}")
        print(f"  Wins:               {stats['wins']}")
        print(f"  Losses:             {stats['losses']}")
        print(f"  Win rate:           {stats['win_rate']:.1%}")
        print(f"  Total P&L:          ${stats['total_pnl']:+.2f}")
        print(f"  Avg YES price:      {stats['avg_yes_mid']*100:.1f}¢")
        print(f"  Avg hours to close: {stats['avg_hours_to_close']:.1f}h")
        print("=" * 50 + "\n")

    def print_positions(self):
        open_trades = self._log.get_open_trades()
        if not open_trades:
            print("\nNo open positions.\n")
            return

        total_deployed = sum(t["position_dollars"] for t in open_trades)
        print(f"\n{'='*90}")
        print(f"  OPEN POSITIONS ({len(open_trades)}) — ${total_deployed:.2f} deployed")
        print(f"{'='*90}")
        print(f"  {'Ticker':<40} {'Side':>4} {'Size':>5} {'Entry':>6} {'YES':>5} {'Spread':>6} {'Hours':>6}")
        print(f"  {'-'*40} {'-'*4} {'-'*5} {'-'*6} {'-'*5} {'-'*6} {'-'*6}")
        for t in open_trades:
            print(
                f"  {t['market_ticker']:<40} "
                f"{t['side'].upper():>4} "
                f"{t['position_size']:>5} "
                f"${t['entry_price']:.2f} "
                f"{t['yes_mid']*100:>4.0f}¢ "
                f"{t['spread']*100:>5.1f}¢ "
                f"{t['hours_to_close']:>5.1f}h"
            )
        print(f"{'='*90}\n")

    def export_csv(self, output_path: str):
        trades = self._log.get_all_trades()
        if not trades:
            print("No trades to export.")
            return

        fields = [k for k in trades[0].keys() if k != "orderbook_snapshot"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for t in trades:
                row = {k: v for k, v in t.items() if k != "orderbook_snapshot"}
                writer.writerow(row)

        print(f"Exported {len(trades)} trades to {output_path}")

    def watch(self, interval: int = 300, series_tickers: list[str] | None = None):
        """Continuous mode — scan every `interval` seconds."""
        logger.info("Starting watch mode (interval=%ds). Ctrl+C to stop.", interval)
        try:
            while True:
                self.scan_and_trade(series_tickers)
                self.resolve_settled()
                logger.info("Sleeping %d seconds...", interval)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Watch mode stopped.")

    def close(self):
        self._log.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kalshi Paper Trader — filter-based Phase A automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_trader.py scan                          # one-shot scan
  python paper_trader.py scan --series KXNFL KXOSC     # scan specific series
  python paper_trader.py scan --min-vol 500            # lower volume threshold
  python paper_trader.py resolve                       # resolve settled trades
  python paper_trader.py stats                         # show summary
  python paper_trader.py positions                     # show open positions
  python paper_trader.py watch --interval 300          # continuous mode
  python paper_trader.py export trades.csv             # export to CSV
        """,
    )
    parser.add_argument(
        "command",
        choices=["scan", "resolve", "stats", "positions", "watch", "export"],
    )
    parser.add_argument("--series", nargs="*", default=None,
                        help="Series tickers to scan (e.g., KXNFL KXOSC)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between scans in watch mode (default: 300)")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Initial bankroll in dollars (default: 1000)")
    parser.add_argument("--contracts", type=int, default=10,
                        help="Flat contracts per trade (default: 10)")
    parser.add_argument("--cutoff", type=float, default=0.05,
                        help="Min distance from 0 or 1 (default: 0.05)")
    parser.add_argument("--yes-max", type=float, default=0.20,
                        help="Buy NO when YES ≤ this (default: 0.20)")
    parser.add_argument("--yes-min", type=float, default=0.80,
                        help="Buy YES when YES ≥ this (default: 0.80)")
    parser.add_argument("--min-vol", type=int, default=1000,
                        help="Min volume filter (default: 1000)")
    parser.add_argument("--min-oi", type=int, default=1000,
                        help="Min open interest filter (default: 1000)")
    parser.add_argument("--max-hours", type=float, default=72.0,
                        help="Max hours to close (default: 72)")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help="Path to SQLite database")
    parser.add_argument("output", nargs="?", default="trades.csv",
                        help="Output file (for export command)")

    args = parser.parse_args()

    config = PaperTradingConfig(initial_bankroll=args.bankroll)
    scanner_config = ScannerConfig(
        cutoff=args.cutoff,
        yes_max=args.yes_max,
        yes_min=args.yes_min,
        min_volume=args.min_vol,
        min_oi=args.min_oi,  # using same threshold for open interest
        max_hours_to_close=args.max_hours,
    )
    position_config = PositionConfig(contracts_per_trade=args.contracts)

    trader = PaperTrader(
        config=config,
        scanner_config=scanner_config,
        position_config=position_config,
        db_path=args.db,
    )

    try:
        if args.command == "scan":
            trader.scan_and_trade(args.series)
            trader.print_stats()

        elif args.command == "resolve":
            n = trader.resolve_settled()
            print(f"Resolved {n} trades.")
            trader.print_stats()

        elif args.command == "stats":
            trader.print_stats()

        elif args.command == "positions":
            trader.print_positions()

        elif args.command == "watch":
            trader.watch(args.interval, args.series)

        elif args.command == "export":
            trader.export_csv(args.output)

    finally:
        trader.close()


if __name__ == "__main__":
    main()
