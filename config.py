"""config.py — All tunable constants in one place.

Reference: insurance_writing_guide.md
"""
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# API URLs
# ---------------------------------------------------------------------------
PRODUCTION_API_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_API_URL = "https://demo-api.kalshi.co/trade-api/v2"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "paper_trades.db"

# ---------------------------------------------------------------------------
# Scanner thresholds (the only three filters)
# ---------------------------------------------------------------------------

@dataclass
class ScannerConfig:
    cutoff: float = 0.05              # 5¢ — minimum distance from 0 or 1
    yes_max: float = 0.20             # buy NO when YES ≤ 20¢ (and ≥ cutoff)
    yes_min: float = 0.80             # buy YES when YES ≥ 80¢ (and ≤ 1-cutoff)
    min_volume: int = 1000            # thick-market proxy
    min_oi: int = 1000                # open interest proxy
    max_hours_to_close: float = 72.0  # must close within 72 hours
    min_hours_to_close: float = 0.5   # must not close within 30 minutes


# ---------------------------------------------------------------------------
# Position sizing (flat, no Kelly / no base rate needed)
# ---------------------------------------------------------------------------

@dataclass
class PositionConfig:
    contracts_per_trade: int = 10     # flat number of contracts per trade
    max_positions: int = 50           # max simultaneous open positions
    max_per_event: int = 5            # max positions in one event


# ---------------------------------------------------------------------------
# Paper trading config
# ---------------------------------------------------------------------------

@dataclass
class PaperTradingConfig:
    initial_bankroll: float = 1000.0
    poll_interval_seconds: int = 300  # 5 minutes between scans
