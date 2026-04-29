"""fetch.py — Fetch market + candlestick data via pykalshi and cache locally.

Two-pass design:
  Pass 1 — fetch market listings (fast, one API call per page)
  Pass 2 — apply MarketPreFilter on metadata, then pull candlesticks only
            for markets that pass (saves API calls and time)

Requires: pip install pykalshi[dataframe]

Usage:
    from fetch import fetch_dataset, load_dataset, MarketPreFilter

    fetch_dataset(
        series_tickers=["KXUFCFIGHT", "KXMLSGAME"],
        pre_filter=MarketPreFilter(
            require_result=True,     # settled binary markets only
            min_duration_hours=2,    # skip markets live for < 2 hours
        ),
        output_path="data.pkl",
    )

    ds = load_dataset("data.pkl")
    ds.markets_df        # DataFrame of markets that passed pre-filter
    ds.candles           # {ticker: DataFrame of OHLC candles}
"""
from __future__ import annotations

import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Dataset container
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Dataset:
    """Everything the backtest needs, in one serializable object."""
    markets_df: pd.DataFrame
    candles: dict[str, pd.DataFrame] = field(default_factory=dict)
    fetched_at: str = ""
    pre_filter_applied: str = ""   # human-readable description of filters used

    def summary(self) -> str:
        n_markets = len(self.markets_df)
        n_with = sum(1 for df in self.candles.values() if len(df) > 0)
        n_candles = sum(len(df) for df in self.candles.values())
        return (
            f"Dataset: {n_markets} markets, {n_with} with candles "
            f"({n_candles} total candle rows), fetched {self.fetched_at}"
        )


def save_dataset(ds: Dataset, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved dataset to %s", path)


def load_dataset(path: str | list[str]) -> Dataset:
    """Load one or more dataset pickles and merge into a single Dataset.

    Args:
        path: A single path string, or a list of paths.
              e.g. "ufc.pkl" or ["ufc.pkl", "mls.pkl", "nfl.pkl"]

    Returns:
        A single Dataset with deduplicated markets and merged candles.
    """
    if isinstance(path, str):
        paths = [path]
    else:
        paths = list(path)

    datasets = []
    for p in paths:
        with open(p, "rb") as f:
            datasets.append(pickle.load(f))
        logger.info("Loaded %s — %s", p, datasets[-1].summary())

    if len(datasets) == 1:
        return datasets[0]

    # Merge markets_df — concatenate and deduplicate by ticker
    all_markets = pd.concat(
        [ds.markets_df for ds in datasets if len(ds.markets_df) > 0],
        ignore_index=True,
    )
    all_markets = all_markets.drop_duplicates(subset="ticker", keep="first")

    # Merge candles — first dataset's candles take priority on duplicates
    merged_candles: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        for ticker, df in ds.candles.items():
            if ticker not in merged_candles and len(df) > 0:
                merged_candles[ticker] = df

    sources = [os.path.basename(p) for p in paths]
    fetched_at = ", ".join(ds.fetched_at for ds in datasets if ds.fetched_at)

    merged = Dataset(
        markets_df=all_markets,
        candles=merged_candles,
        fetched_at=fetched_at,
        pre_filter_applied=f"merged from {sources}",
    )
    logger.info("Merged result — %s", merged.summary())
    return merged


# ─────────────────────────────────────────────────────────────────────────
# Pre-filter — applied to market metadata before fetching candlesticks
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MarketPreFilter:
    """Filters on market metadata applied before fetching candlesticks.

    These only use fields from the market listing response (no candle data),
    so they eliminate candlestick API calls entirely for filtered-out markets.

    Important: Kalshi returns volume=0 for many finalized markets even when
    real trading occurred. Keep min_volume=0 (the default) and use
    filters.FilterConfig.min_candle_volume / min_market_volume for volume
    filtering after candles are loaded.
    """

    # ── Result ─────────────────────────────────────────────────────────
    require_result: bool = True
    # True = only keep markets with result "yes" or "no"
    # Drops "scalar" markets and unresolved open markets

    # ── Volume / OI (from market snapshot) ─────────────────────────────
    min_volume: float = 0
    # Minimum lifetime volume. Default 0 because Kalshi reports 0 for
    # most settled markets regardless of actual volume.
    min_open_interest: float = 0
    # Minimum open interest at listing time.

    # ── Price range (at listing time snapshot) ──────────────────────────
    min_yes_bid: float = 0.0
    # Skip markets where YES bid < this at snapshot time.
    # Useful to exclude markets that have already resolved to near-certainty.
    max_yes_ask: float = 1.0
    # Skip markets where YES ask > this at snapshot time.

    # ── Duration ────────────────────────────────────────────────────────
    min_duration_hours: float = 0
    # Skip very short-lived markets (e.g. < 2 hours).
    # Requires open_time to be available on the market object.
    max_duration_hours: float = 0
    # 0 = no upper limit.

    # ── Series allowlist ─────────────────────────────────────────────────
    allowed_series: list[str] = field(default_factory=list)
    # If non-empty, only keep markets in these series.
    # e.g. ["KXUFCFIGHT", "KXMLSGAME"]

    def describe(self) -> str:
        parts = []
        if self.require_result:
            parts.append("result=yes/no only")
        if self.min_volume > 0:
            parts.append(f"volume≥{self.min_volume}")
        if self.min_open_interest > 0:
            parts.append(f"OI≥{self.min_open_interest}")
        if self.min_yes_bid > 0:
            parts.append(f"yes_bid≥{self.min_yes_bid:.2f}")
        if self.max_yes_ask < 1.0:
            parts.append(f"yes_ask≤{self.max_yes_ask:.2f}")
        if self.min_duration_hours > 0:
            parts.append(f"duration≥{self.min_duration_hours}h")
        if self.max_duration_hours > 0:
            parts.append(f"duration≤{self.max_duration_hours}h")
        if self.allowed_series:
            parts.append(f"series∈{self.allowed_series}")
        return ", ".join(parts) if parts else "none"

    def passes(self, m) -> bool:
        """Return True if a pykalshi Market object passes all pre-filters."""

        # Result
        if self.require_result:
            if getattr(m, "result", None) not in ("yes", "no"):
                return False

        # Volume
        if self.min_volume > 0:
            if _safe_float(getattr(m, "volume", 0)) < self.min_volume:
                return False

        # Open interest
        if self.min_open_interest > 0:
            if _safe_float(getattr(m, "open_interest", 0)) < self.min_open_interest:
                return False

        # Price range
        if self.min_yes_bid > 0:
            yes_bid = _safe_float(
                getattr(m, "yes_bid_dollars", None) or getattr(m, "yes_bid", None)
            )
            if yes_bid < self.min_yes_bid:
                return False

        if self.max_yes_ask < 1.0:
            yes_ask = _safe_float(
                getattr(m, "yes_ask_dollars", None) or getattr(m, "yes_ask", None)
            )
            if yes_ask > self.max_yes_ask:
                return False

        # Duration
        if self.min_duration_hours > 0 or self.max_duration_hours > 0:
            close_time = getattr(m, "close_time", None)
            open_time = getattr(m, "open_time", None)
            if close_time and open_time:
                try:
                    duration_h = (_to_unix(close_time) - _to_unix(open_time)) / 3600
                    if self.min_duration_hours > 0 and duration_h < self.min_duration_hours:
                        return False
                    if self.max_duration_hours > 0 and duration_h > self.max_duration_hours:
                        return False
                except Exception:
                    pass

        # Series allowlist
        if self.allowed_series:
            series = getattr(m, "series_ticker", "") or ""
            if series not in self.allowed_series:
                return False

        return True


# ─────────────────────────────────────────────────────────────────────────
# Main fetch function
# ─────────────────────────────────────────────────────────────────────────

def fetch_dataset(
    tickers: list[str] | None = None,
    series_tickers: list[str] | None = None,
    event_tickers: list[str] | None = None,
    market_limit: int = 200,
    candle_period: str = "1d",
    output_path: str = "data.pkl",
    include_open: bool = False,
    pre_filter: MarketPreFilter | None = None,
) -> Dataset:
    """Fetch markets and candlestick history via pykalshi.

    Pass 1: list markets from the API.
    Pass 2: apply pre_filter to eliminate thin/invalid markets.
    Pass 3: pull candlesticks only for markets that passed.

    Args:
        tickers: Specific market tickers to fetch.
        series_tickers: Series identifiers, e.g. ["KXUFCFIGHT"].
        event_tickers: Event identifiers, e.g. ["KXNFL-25SB"].
        market_limit: Max markets per listing query.
        candle_period: "1m", "1h", or "1d".
        output_path: Where to save the pickle.
        include_open: Also include open (unsettled) markets.
        pre_filter: Metadata filters applied before pulling candles.
                    Defaults to MarketPreFilter() — require_result=True only.

    Returns:
        Dataset saved to output_path.
    """
    from pykalshi import KalshiClient, MarketStatus, CandlestickPeriod

    period_map = {
        "1m": CandlestickPeriod.ONE_MINUTE,
        "1h": CandlestickPeriod.ONE_HOUR,
        "1d": CandlestickPeriod.ONE_DAY,
    }
    period = period_map.get(candle_period, CandlestickPeriod.ONE_DAY)

    if pre_filter is None:
        pre_filter = MarketPreFilter()

    client = KalshiClient()
    statuses = [MarketStatus.SETTLED]
    if include_open:
        statuses.append(MarketStatus.OPEN)

    # ── Pass 1: Collect market listings ──────────────────────────────────
    all_markets: list = []

    if tickers:
        logger.info("Fetching %d individual markets...", len(tickers))
        for ticker in tickers:
            try:
                all_markets.append(client.get_market(ticker))
            except Exception as e:
                logger.warning("Could not fetch %s: %s", ticker, e)

    if series_tickers:
        for series in series_tickers:
            for status in statuses:
                logger.info("Listing %s markets in series %s...", status.value, series)
                try:
                    batch = client.get_markets(
                        series_ticker=series, status=status, limit=market_limit
                    )
                    all_markets.extend(batch)
                    logger.info("  → %d markets listed", len(batch))
                except Exception as e:
                    logger.warning("Failed listing series %s: %s", series, e)

    if event_tickers:
        for event in event_tickers:
            for status in statuses:
                logger.info("Listing %s markets in event %s...", status.value, event)
                try:
                    batch = client.get_markets(
                        event_ticker=event, status=status, limit=market_limit
                    )
                    all_markets.extend(batch)
                    logger.info("  → %d markets listed", len(batch))
                except Exception as e:
                    logger.warning("Failed listing event %s: %s", event, e)

    # Deduplicate by ticker
    seen: set[str] = set()
    unique_markets = []
    for m in all_markets:
        if m.ticker not in seen:
            seen.add(m.ticker)
            unique_markets.append(m)
    logger.info("Total unique markets listed: %d", len(unique_markets))

    # ── Pass 2: Pre-filter on metadata ───────────────────────────────────
    filter_desc = pre_filter.describe()
    logger.info("Applying pre-filter: [%s]", filter_desc if filter_desc else "none")

    filtered = [m for m in unique_markets if pre_filter.passes(m)]
    n_dropped = len(unique_markets) - len(filtered)
    logger.info(
        "Pre-filter result: %d/%d markets pass (%d dropped)",
        len(filtered), len(unique_markets), n_dropped,
    )

    if not filtered:
        logger.warning("No markets passed pre-filter. Try relaxing the filters.")
        ds = Dataset(
            markets_df=pd.DataFrame(),
            candles={},
            fetched_at=datetime.now(timezone.utc).isoformat(),
            pre_filter_applied=filter_desc,
        )
        save_dataset(ds, output_path)
        print(ds.summary())
        return ds

    # Build markets DataFrame
    rows = []
    for m in filtered:
        rows.append({
            "ticker": m.ticker,
            "title": m.title,
            "subtitle": getattr(m, "subtitle", ""),
            "status": m.status.value if hasattr(m.status, "value") else str(m.status),
            "result": m.result,
            "series_ticker": getattr(m, "series_ticker", ""),
            "event_ticker": getattr(m, "event_ticker", ""),
            "close_time": str(getattr(m, "close_time", "")),
            "yes_bid": _safe_float(
                getattr(m, "yes_bid_dollars", None) or getattr(m, "yes_bid", None)
            ),
            "yes_ask": _safe_float(
                getattr(m, "yes_ask_dollars", None) or getattr(m, "yes_ask", None)
            ),
            "volume": _safe_float(getattr(m, "volume", 0)),
            "open_interest": _safe_float(getattr(m, "open_interest", 0)),
        })
    markets_df = pd.DataFrame(rows)

    # ── Pass 3: Fetch candlesticks for filtered markets only ──────────────
    candles: dict[str, pd.DataFrame] = {}
    total = len(filtered)

    for i, m in enumerate(filtered):
        if i == 0 or (i + 1) % 10 == 0 or i == total - 1:
            logger.info("Fetching candles: %d/%d (%s)...", i + 1, total, m.ticker)

        try:
            close_time = getattr(m, "close_time", None)
            end_ts = int(_to_unix(close_time)) + 86400 if close_time else int(time.time())
            start_ts = end_ts - 730 * 86400  # up to 2 years back

            raw_candles = m.get_candlesticks(start_ts, end_ts, period=period)

            if hasattr(raw_candles, "to_dataframe"):
                df = raw_candles.to_dataframe()
            else:
                df = _candles_to_df(raw_candles, m.ticker)

            if len(df) > 0:
                df["ticker"] = m.ticker
                candles[m.ticker] = df
            else:
                candles[m.ticker] = pd.DataFrame()

        except Exception as e:
            logger.debug("No candles for %s: %s", m.ticker, e)
            candles[m.ticker] = pd.DataFrame()

        time.sleep(0.05)

    n_with = sum(1 for df in candles.values() if len(df) > 0)
    logger.info("Candlesticks: %d/%d markets have data", n_with, total)

    ds = Dataset(
        markets_df=markets_df,
        candles=candles,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        pre_filter_applied=filter_desc,
    )
    save_dataset(ds, output_path)
    print(f"\n{ds.summary()}")
    if filter_desc:
        print(f"Pre-filter applied: {filter_desc}")
    return ds


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _to_unix(val) -> float:
    """Convert a datetime-like value to a unix timestamp."""
    if val is None:
        return 0.0
    if hasattr(val, "timestamp"):
        return float(val.timestamp())
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(pd.Timestamp(val).timestamp())
        except Exception:
            return 0.0
    return 0.0


def _candles_to_df(candles_list, ticker: str) -> pd.DataFrame:
    """Fallback: convert a list of candle objects to a DataFrame."""
    rows = []
    for c in candles_list:
        row = {"ticker": ticker}
        if hasattr(c, "__dict__"):
            row.update({k: v for k, v in c.__dict__.items()})
        elif isinstance(c, dict):
            row.update(c)
        rows.append(row)
    return pd.DataFrame(rows)