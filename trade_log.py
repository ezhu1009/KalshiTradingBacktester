"""trade_log.py — SQLite-backed trade log for paper trading.

Stores every simulated trade with market snapshot data at entry time.
No base-rate fields — the engine operates on pure filter criteria.
"""
from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import DB_PATH

logger = logging.getLogger(__name__)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeLog:
    """SQLite-backed log for paper trades and market snapshots."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("Trade log opened: %s", self._db_path)

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL,
                market_ticker   TEXT NOT NULL,
                market_title    TEXT NOT NULL,
                event_ticker    TEXT NOT NULL,
                series_ticker   TEXT NOT NULL,
                category        TEXT NOT NULL,
                side            TEXT NOT NULL DEFAULT 'no',

                -- Prices at time of simulated entry
                yes_bid         REAL NOT NULL,
                yes_ask         REAL NOT NULL,
                yes_mid         REAL NOT NULL,
                no_mid          REAL NOT NULL,
                spread          REAL NOT NULL,
                volume          INTEGER NOT NULL,
                open_interest   INTEGER NOT NULL,

                -- Time context
                close_time      TEXT,
                hours_to_close  REAL NOT NULL,

                -- Position
                entry_price     REAL NOT NULL,
                position_size   INTEGER NOT NULL,
                position_dollars REAL NOT NULL,

                -- Orderbook snapshot (JSON blob)
                orderbook_snapshot TEXT,

                -- Resolution (filled in later by 'resolve' command)
                resolved_at     TEXT,
                result          TEXT,          -- 'yes' or 'no'
                outcome         TEXT,          -- 'win' or 'loss'
                pnl             REAL,
                notes           TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS scan_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                markets_scanned INTEGER NOT NULL,
                candidates_found INTEGER NOT NULL,
                trades_placed   INTEGER NOT NULL,
                notes           TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_trades_ticker
                ON paper_trades(market_ticker);
            CREATE INDEX IF NOT EXISTS idx_trades_result
                ON paper_trades(result);
            CREATE INDEX IF NOT EXISTS idx_trades_created
                ON paper_trades(created_at);
        """)
        self._conn.commit()

    # -------------------------------------------------------------------
    # Write operations
    # -------------------------------------------------------------------

    def log_trade(
        self,
        market_ticker: str,
        market_title: str,
        event_ticker: str,
        series_ticker: str,
        category: str,
        side: str,
        yes_bid: float,
        yes_ask: float,
        yes_mid: float,
        no_mid: float,
        spread: float,
        volume: int,
        open_interest: int,
        close_time: str,
        hours_to_close: float,
        entry_price: float,
        position_size: int,
        position_dollars: float,
        orderbook_snapshot: dict | None = None,
        notes: str = "",
    ) -> int:
        """Log a simulated trade. Returns the trade ID."""
        cursor = self._conn.execute(
            """
            INSERT INTO paper_trades (
                created_at, market_ticker, market_title, event_ticker,
                series_ticker, category, side,
                yes_bid, yes_ask, yes_mid, no_mid, spread,
                volume, open_interest,
                close_time, hours_to_close,
                entry_price, position_size, position_dollars,
                orderbook_snapshot, notes
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?
            )
            """,
            (
                _now_utc(), market_ticker, market_title, event_ticker,
                series_ticker, category, side,
                yes_bid, yes_ask, yes_mid, no_mid, spread,
                volume, open_interest,
                close_time, hours_to_close,
                entry_price, position_size, position_dollars,
                json.dumps(orderbook_snapshot) if orderbook_snapshot else None,
                notes,
            ),
        )
        self._conn.commit()
        trade_id = cursor.lastrowid
        logger.info(
            "Paper trade #%d: %s NO %d @ $%.2f | %.1fh to close",
            trade_id, market_ticker, position_size, entry_price, hours_to_close,
        )
        return trade_id

    def resolve_trade(self, trade_id: int, result: str, pnl: float, notes: str = ""):
        """Update a trade with its resolution outcome."""
        outcome = "win" if pnl > 0 else "loss"
        self._conn.execute(
            """
            UPDATE paper_trades
            SET resolved_at = ?, result = ?, outcome = ?, pnl = ?,
                notes = notes || ?
            WHERE id = ?
            """,
            (_now_utc(), result, outcome, pnl, f" | {notes}" if notes else "", trade_id),
        )
        self._conn.commit()
        logger.info("Trade #%d resolved: %s (P&L: $%.2f)", trade_id, outcome, pnl)

    def log_scan(self, markets_scanned: int, candidates_found: int, trades_placed: int, notes: str = ""):
        self._conn.execute(
            "INSERT INTO scan_log (timestamp, markets_scanned, candidates_found, trades_placed, notes) VALUES (?, ?, ?, ?, ?)",
            (_now_utc(), markets_scanned, candidates_found, trades_placed, notes),
        )
        self._conn.commit()

    # -------------------------------------------------------------------
    # Read operations
    # -------------------------------------------------------------------

    def get_open_trades(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM paper_trades WHERE result IS NULL ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_trades(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM paper_trades ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_resolved_trades(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM paper_trades WHERE result IS NOT NULL ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tickers_with_open_trades(self) -> set[str]:
        """Get set of tickers that have unresolved trades."""
        rows = self._conn.execute(
            "SELECT DISTINCT market_ticker FROM paper_trades WHERE result IS NULL"
        ).fetchall()
        return {r[0] for r in rows}

    def get_event_trade_count(self, event_ticker: str) -> int:
        """Count open trades for a specific event."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE event_ticker = ? AND result IS NULL",
            (event_ticker,),
        ).fetchone()
        return row[0]

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
        resolved = self._conn.execute("SELECT COUNT(*) FROM paper_trades WHERE result IS NOT NULL").fetchone()[0]
        wins = self._conn.execute("SELECT COUNT(*) FROM paper_trades WHERE outcome = 'win'").fetchone()[0]
        losses = self._conn.execute("SELECT COUNT(*) FROM paper_trades WHERE outcome = 'loss'").fetchone()[0]
        total_pnl = self._conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM paper_trades WHERE pnl IS NOT NULL").fetchone()[0]
        avg_yes_mid = self._conn.execute("SELECT COALESCE(AVG(yes_mid), 0) FROM paper_trades").fetchone()[0]
        avg_hours = self._conn.execute("SELECT COALESCE(AVG(hours_to_close), 0) FROM paper_trades").fetchone()[0]

        return {
            "total_trades": total,
            "resolved": resolved,
            "open": total - resolved,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / resolved if resolved > 0 else 0,
            "total_pnl": total_pnl,
            "avg_yes_mid": avg_yes_mid,
            "avg_hours_to_close": avg_hours,
        }

    def close(self):
        self._conn.close()
