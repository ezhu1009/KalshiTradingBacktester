"""client.py — Lightweight Kalshi API client for read-only production data.

Uses raw requests (no SDK dependency). Authentication is NOT required
for public market data endpoints, which is all we need for paper trading.

API Reference: https://docs.kalshi.com
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterator

import requests

from config import PRODUCTION_API_URL
from models import Market, MarketStatus, MarketCategory, classify_market

logger = logging.getLogger(__name__)


class KalshiClient:
    """Read-only client for Kalshi production market data.

    No authentication required — only uses public GET endpoints.
    Handles pagination and basic rate limiting.
    """

    def __init__(self, base_url: str = PRODUCTION_API_URL):
        self._base_url = base_url
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "kalshi-paper-trader/1.0",
        })

    # -------------------------------------------------------------------
    # Price / timestamp parsing
    # -------------------------------------------------------------------

    @staticmethod
    def _parse_price(val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return 0.0
        if isinstance(val, (int, float)):
            return val / 100.0 if val > 1.0 else float(val)
        return 0.0

    @staticmethod
    def _parse_ts(val) -> datetime:
        if val is None:
            return datetime.now(timezone.utc)
        if isinstance(val, str):
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        return datetime.now(timezone.utc)

    # -------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------

    @classmethod
    def _to_market(cls, raw: dict) -> Market:
        """Convert raw API dict → domain Market dataclass."""
        series = raw.get("series_ticker", "")
        return Market(
            ticker=raw.get("ticker", ""),
            title=raw.get("title", ""),
            subtitle=raw.get("subtitle", ""),
            yes_bid=cls._parse_price(
                raw.get("yes_bid_dollars") or raw.get("yes_bid")
            ),
            yes_ask=cls._parse_price(
                raw.get("yes_ask_dollars") or raw.get("yes_ask")
            ),
            volume=int(raw.get("volume", 0)),
            open_interest=int(raw.get("open_interest", 0)),
            status=MarketStatus(raw.get("status", "open")),
            close_time=cls._parse_ts(raw.get("close_time")),
            result=raw.get("result") or None,
            category=classify_market(series),
            event_ticker=raw.get("event_ticker", ""),
            series_ticker=series,
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Make a GET request with basic retry on 429."""
        url = f"{self._base_url}{path}"
        for attempt in range(5):
            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited, waiting %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Rate limited after 5 retries: {url}")

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> tuple[list[Market], str | None]:
        """Fetch a page of markets. Returns (markets, next_cursor)."""
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        data = self._get("/markets", params)
        markets = [self._to_market(m) for m in (data.get("markets") or [])]
        next_cursor = data.get("cursor") or None
        return markets, next_cursor

    def iter_markets(
        self,
        status: str | None = None,
        series_ticker: str | None = None,
        limit: int = 200,
        max_pages: int = 50,
    ) -> Iterator[Market]:
        """Iterate all markets with automatic pagination."""
        cursor = None
        for _ in range(max_pages):
            markets, cursor = self.get_markets(
                series_ticker=series_ticker,
                status=status,
                limit=limit,
                cursor=cursor,
            )
            for m in markets:
                yield m
            if not cursor:
                break
            time.sleep(0.1)

    def get_market(self, ticker: str) -> Market:
        """Fetch a single market by ticker."""
        data = self._get(f"/markets/{ticker}")
        return self._to_market(data.get("market", data))

    def get_orderbook(self, ticker: str) -> dict:
        """Fetch the current orderbook for a market.

        Returns raw dict with 'yes' and 'no' arrays of [price, size] pairs.
        This is where you see real spread and depth.
        """
        data = self._get(f"/markets/{ticker}/orderbook")
        return data.get("orderbook", data)

    def get_trades(self, ticker: str, limit: int = 50) -> list[dict]:
        """Fetch recent trades for a market."""
        data = self._get(f"/markets/{ticker}/trades", {"limit": limit})
        return data.get("trades", [])
