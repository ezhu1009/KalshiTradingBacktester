"""scanner.py — Filter Kalshi markets using three criteria.

Filters:
  1. Market volume (thick market proxy)
  2. Time left to expiry (how soon the market closes)
  3. YES price — determines side:
     - YES in [cutoff, yes_max]         → buy NO  (sell the longshot)
     - YES in [yes_min, 1 - cutoff]     → buy YES (buy the favorite)
     - everything else                  → skip

No base-rate assumptions, no category tier requirements.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from models import Market, MarketStatus

logger = logging.getLogger(__name__)

MarketFilter = Callable[[Market], bool]


def is_open(market: Market) -> bool:
    return market.status == MarketStatus.OPEN


def meets_volume_threshold(market: Market, min_volume: int = 1000) -> bool:
    """Thick-market proxy: total contracts traded > threshold."""
    return market.volume >= min_volume

def meets_oi_threshold(market: Market, min_oi: int = 100) -> bool:
    """Thick-market proxy: open interest > threshold."""
    return market.open_interest >= min_oi


def closes_within(market: Market, max_hours: float = 72.0) -> bool:
    """Market closes within the specified number of hours."""
    now = datetime.now(timezone.utc)
    remaining = (market.close_time - now).total_seconds() / 3600.0
    return 0 < remaining <= max_hours


def closes_after(market: Market, min_hours: float = 0.5) -> bool:
    """Market doesn't close for at least min_hours (avoid last-second entries)."""
    now = datetime.now(timezone.utc)
    remaining = (market.close_time - now).total_seconds() / 3600.0
    return remaining >= min_hours


def determine_side(
    yes_mid: float,
    cutoff: float = 0.05,
    yes_max: float = 0.20,
    yes_min: float = 0.80,
) -> tuple[str, float] | None:
    """Determine which side to trade and the entry price.

    Returns:
        ("no", entry_price)  if YES is low  → buy NO at 1 - yes_mid
        ("yes", entry_price) if YES is high → buy YES at yes_mid
        None                 if YES is in the dead zone (no trade)
    """
    if yes_mid <= yes_max and yes_mid >= cutoff:
        return ("no", 1.0 - yes_mid)

    if yes_mid >= yes_min and yes_mid <= 1.0 - cutoff:
        return ("yes", yes_mid)

    return None


@dataclass
class ScanResult:
    market: Market
    passed_filters: list[str]
    failed_filters: list[str]
    hours_to_close: float = 0.0
    side: str | None = None          # "yes" or "no"
    entry_price: float | None = None

    @property
    def is_candidate(self) -> bool:
        return len(self.failed_filters) == 0 and self.side is not None


class MarketScanner:
    """Market scanner with three configurable filters.

    Args:
        cutoff:     Minimum distance from 0 or 1 (default 5¢)
        yes_max:    Buy NO when YES ≤ this (default 20¢)
        yes_min:    Buy YES when YES ≥ this (default 80¢)
        min_volume: Minimum total contracts traded (default 1000)
        max_hours:  Market must close within this many hours (default 72)
        min_hours:  Market must NOT close sooner than this (default 0.5)
    """

    def __init__(
        self,
        cutoff: float = 0.05,
        yes_max: float = 0.20,
        yes_min: float = 0.80,
        min_volume: int = 1000,
        min_oi: int = 1000,
        max_hours: float = 72.0,
        min_hours: float = 0.5,
    ):
        self._cutoff = cutoff
        self._yes_max = yes_max
        self._yes_min = yes_min

        self._filters: list[tuple[str, MarketFilter]] = [
            ("is_open", is_open),
            ("volume_threshold", lambda m: meets_volume_threshold(m, min_volume)),
            ("oi_threshold", lambda m: meets_oi_threshold(m, min_oi)),
            ("closes_within", lambda m: closes_within(m, max_hours)),
            ("closes_after", lambda m: closes_after(m, min_hours)),
        ]

    def add_filter(self, name: str, fn: MarketFilter) -> None:
        """Add a custom filter to the pipeline."""
        self._filters.append((name, fn))

    def scan(self, markets: list[Market]) -> list[ScanResult]:
        """Return only candidates that pass ALL filters and have a valid side."""
        return [r for r in self.scan_with_details(markets) if r.is_candidate]

    def scan_with_details(self, markets: list[Market]) -> list[ScanResult]:
        """Return ScanResult for every market (useful for debugging)."""
        now = datetime.now(timezone.utc)
        results = []
        for market in markets:
            passed, failed = [], []
            for name, fn in self._filters:
                (passed if fn(market) else failed).append(name)

            # Determine side from price filter
            side_result = determine_side(
                market.yes_mid, self._cutoff, self._yes_max, self._yes_min
            )
            if side_result is not None:
                side, entry_price = side_result
                passed.append("price_side")
            else:
                side, entry_price = None, None
                failed.append("price_side")

            hours_to_close = max(0.0, (market.close_time - now).total_seconds() / 3600.0)
            results.append(ScanResult(
                market, passed, failed, hours_to_close, side, entry_price
            ))
        return results
