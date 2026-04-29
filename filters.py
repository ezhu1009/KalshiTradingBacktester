"""filters.py — Scan candlestick data and return trade opportunity windows.

Takes a Dataset (markets_df + candles dict) and applies configurable filters
across three dimensions: price, volume/OI, and time-to-expiry. Returns a
DataFrame of opportunity windows — specific candle periods where ALL filters
pass simultaneously.

Usage:
    from filters import FilterConfig, scan_opportunities

    opps = scan_opportunities(dataset, FilterConfig(
        yes_max=0.15,       # buy NO when YES ≤ 15%
        yes_min=0.85,       # buy YES when YES ≥ 85%
        min_volume=100,     # per-candle volume floor
        min_oi=50,          # open interest floor
        min_hours_to_expiry=2,  # don't enter in final 2 hours
    ))

    print(opps)  # DataFrame: ticker, timestamp, side, yes_mid, entry_price, ...
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """All filter thresholds in one place."""

    # ── Price filters ───────────────────────────────────────────────────
    # Buy NO when yes_mid ≤ yes_max  (longshot — market says unlikely)
    yes_max: float = 0.15

    # Buy YES when yes_mid ≥ yes_min  (favorite — market says very likely)
    yes_min: float = 0.85

    # Cutoff 
    cutoff: float = 0.02

    # Dead-zone: ignore prices between yes_max and yes_min (no signal)

    # ── Volume / liquidity filters ──────────────────────────────────────
    min_candle_volume: int = 0       # minimum volume within the candle period
    min_open_interest: int = 0       # minimum open interest
    min_market_volume: int = 500     # minimum total market lifetime volume

    # ── Time-to-expiry filters ──────────────────────────────────────────
    min_hours_to_expiry: float = 1.0   # don't enter too close to settlement
    max_hours_to_expiry: float = 0     # 0 = no upper limit

    # ── Entry mode ──────────────────────────────────────────────────────
    entry_mode: str = "first"   # "first" = first candle crossing threshold
                                # "best"  = most extreme candle
                                # "all"   = every candle that qualifies


def scan_opportunities(
    markets_df: pd.DataFrame,
    candles: dict[str, pd.DataFrame],
    config: FilterConfig | None = None,
) -> pd.DataFrame:
    """Scan all markets' candlestick histories and return trade opportunities.

    Returns a DataFrame with one row per opportunity:
        ticker, timestamp, side, yes_mid, entry_price, volume, oi,
        hours_to_expiry, market_result, market_volume, series_ticker
    """
    if config is None:
        config = FilterConfig()

    opportunities = []

    for _, market_row in markets_df.iterrows():
        ticker = market_row.get("ticker", "")
        result = market_row.get("result", "")
        close_time = market_row.get("close_time", "")
        market_vol = _safe_float(market_row.get("volume", 0))
        series = market_row.get("series_ticker", "")

        # Must have a settlement result
        # Note: Kalshi uses both "yes"/"no" and "scalar" for some markets.
        # We only support binary markets here.
        if result not in ("yes", "no"):
            continue

        # Must have candle data
        candle_df = candles.get(ticker)
        if candle_df is None or len(candle_df) == 0:
            continue

        # Market volume: fall back to summing candle volumes if the
        # market-level field is 0 (Kalshi often returns 0 for finalized markets)
        if market_vol <= 0:
            for vol_col in ("volume_fp", "volume"):
                if vol_col in candle_df.columns:
                    try:
                        market_vol = float(pd.to_numeric(
                            candle_df[vol_col], errors="coerce"
                        ).fillna(0).sum())
                        break
                    except Exception:
                        pass

        # Market-level volume filter
        if market_vol < config.min_market_volume:
            continue

        # Parse close_time for time-to-expiry calculation
        close_ts = _parse_timestamp(close_time)
        if close_ts is None:
            continue

        # Find opportunities in this market's candles
        market_opps = _scan_market_candles(
            candle_df=candle_df,
            ticker=ticker,
            result=result,
            close_ts=close_ts,
            market_vol=market_vol,
            series=series,
            config=config,
        )
        opportunities.extend(market_opps)

    if not opportunities:
        logger.warning("No opportunities found with current filter settings.")
        return pd.DataFrame()

    df = pd.DataFrame(opportunities)

    logger.info(
        "Found %d opportunities across %d markets (NO: %d, YES: %d)",
        len(df), df["ticker"].nunique(),
        (df["side"] == "NO").sum(), (df["side"] == "YES").sum(),
    )
    return df


def _scan_market_candles(
    candle_df: pd.DataFrame,
    ticker: str,
    result: str,
    close_ts: float,
    market_vol: float,
    series: str,
    config: FilterConfig,
) -> list[dict]:
    """Scan one market's candles for entry signals."""

    opps = []
    best_no: dict | None = None   # track best NO opportunity
    best_yes: dict | None = None  # track best YES opportunity

    for _, candle in candle_df.iterrows():
        # ── Compute yes_mid ─────────────────────────────────────────
        yes_mid = _get_yes_mid(candle)
        if yes_mid is None or yes_mid <= 0 or yes_mid >= 1:
            continue

        # ── Timestamp and time-to-expiry ────────────────────────────
        # Prefer end_period_ts (unix int) over timestamp (pandas Timestamp)
        ts = _candle_timestamp(candle)
        if ts <= 0:
            continue

        hours_to_expiry = (close_ts - ts) / 3600
        if hours_to_expiry < 0:
            continue  # candle is after settlement

        if config.min_hours_to_expiry > 0 and hours_to_expiry < config.min_hours_to_expiry:
            continue

        if config.max_hours_to_expiry > 0 and hours_to_expiry > config.max_hours_to_expiry:
            continue

        # ── Volume / OI filters ─────────────────────────────────────
        candle_vol = _safe_float(candle.get("volume_fp",
                                 candle.get("volume", 0)))
        oi = _safe_float(candle.get("open_interest_fp",
                          candle.get("open_interest", 0)))

        if candle_vol < config.min_candle_volume:
            continue
        if oi < config.min_open_interest:
            continue

        # ── Price signal ────────────────────────────────────────────
        side = None
        entry_price = None

        if (yes_mid <= config.yes_max) and (yes_mid >= config.cutoff):
            side = "NO"
            entry_price = 1.0 - yes_mid

        elif (yes_mid >= config.yes_min) and (yes_mid <= 1 - config.cutoff):
            side = "YES"
            entry_price = yes_mid

        if side is None:
            continue

        opp = {
            "ticker": ticker,
            "timestamp": ts,
            "side": side,
            "yes_mid": round(yes_mid, 4),
            "entry_price": round(entry_price, 4),
            "candle_volume": candle_vol,
            "open_interest": oi,
            "hours_to_expiry": round(hours_to_expiry, 1),
            "market_result": result,
            "market_volume": market_vol,
            "series_ticker": series,
        }

        if config.entry_mode == "all":
            opps.append(opp)

        elif config.entry_mode == "first":
            # Return the first qualifying candle and stop
            return [opp]

        elif config.entry_mode == "best":
            if side == "NO":
                if best_no is None or yes_mid < best_no["yes_mid"]:
                    best_no = opp
            else:
                if best_yes is None or yes_mid > best_yes["yes_mid"]:
                    best_yes = opp

    # For "best" mode, return the single best opportunity per side
    if config.entry_mode == "best":
        if best_no:
            opps.append(best_no)
        if best_yes:
            opps.append(best_yes)

    return opps


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _get_yes_mid(candle) -> float | None:
    """Extract yes_mid (or yes price proxy) from a candle row.

    pykalshi candlesticks return trade-price OHLC, not bid/ask.
    Column names depend on pykalshi version — try all known patterns.
    """
    import math

    def is_valid(val) -> bool:
        if val is None:
            return False
        try:
            f = float(val)
            return not math.isnan(f)
        except (ValueError, TypeError):
            return False

    # Pattern 1: separate yes_bid/yes_ask columns (older pykalshi versions)
    for bid_key, ask_key in [
        ("yes_bid_close", "yes_ask_close"),
        ("yes_bid_dollars_close", "yes_ask_dollars_close"),
        ("yes_bid", "yes_ask"),
    ]:
        bid = candle.get(bid_key) if hasattr(candle, "get") else None
        ask = candle.get(ask_key) if hasattr(candle, "get") else None
        if is_valid(bid) and is_valid(ask):
            try:
                return (float(bid) + float(ask)) / 2
            except (ValueError, TypeError):
                continue

    # Pattern 2: trade-price OHLC columns (current pykalshi default)
    # mean_dollars is the volume-weighted average — best single price proxy
    for key in (
        "mean_dollars", "close_dollars",
        "price_mean", "price_close",
        "close", "mean", "price",
    ):
        val = candle.get(key) if hasattr(candle, "get") else None
        if is_valid(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                continue

    return None


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _candle_timestamp(candle) -> float:
    """Get a unix timestamp from a candle row.

    Prefers end_period_ts (always a unix int) over timestamp (which may
    be a pandas Timestamp object or a string).
    """
    val = candle.get("end_period_ts") if hasattr(candle, "get") else None
    if val is not None:
        try:
            f = float(val)
            if f > 1e9:  # plausible unix timestamp
                return f
        except (ValueError, TypeError):
            pass

    val = candle.get("timestamp") if hasattr(candle, "get") else None
    if val is None:
        return 0.0

    # pandas Timestamp
    if hasattr(val, "timestamp"):
        try:
            return float(val.timestamp())
        except Exception:
            return 0.0

    # ISO string
    if isinstance(val, str):
        try:
            return float(pd.Timestamp(val).timestamp())
        except Exception:
            return 0.0

    # Plain number
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_timestamp(val) -> float | None:
    """Parse a close_time value into a unix timestamp."""
    if isinstance(val, (int, float)) and val > 1e9:
        return float(val)
    if isinstance(val, str):
        if not val:
            return None
        try:
            # ISO format
            dt = pd.Timestamp(val)
            return dt.timestamp()
        except Exception:
            pass
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────
# Pretty-print
# ─────────────────────────────────────────────────────────────────────────

def summarize_opportunities(opps_df: pd.DataFrame, config: FilterConfig) -> None:
    """Print a human-readable summary of scan results."""
    if opps_df.empty:
        print("No opportunities found.")
        return

    print(f"\n{'=' * 70}")
    print("OPPORTUNITY SCAN RESULTS")
    print(f"{'=' * 70}")
    print(f"Filters: YES ≤ {config.yes_max:.0%} (buy NO) | YES ≥ {config.yes_min:.0%} (buy YES)")
    print(f"         volume ≥ {config.min_candle_volume} | OI ≥ {config.min_open_interest} | "
          f"time-to-expiry ≥ {config.min_hours_to_expiry}h")
    print(f"Mode:    {config.entry_mode}\n")

    print(f"Total opportunities: {len(opps_df)}")
    print(f"Unique markets:      {opps_df['ticker'].nunique()}")

    for side in ("NO", "YES"):
        sub = opps_df[opps_df["side"] == side]
        if sub.empty:
            continue
        print(f"\n  {side} side: {len(sub)} entries")
        print(f"    Avg YES mid:       {sub['yes_mid'].mean():.3f}")
        print(f"    Avg entry price:   {sub['entry_price'].mean():.3f}")
        print(f"    Avg hours to exp:  {sub['hours_to_expiry'].mean():.1f}")
        # Win rate preview (using settlement data)
        if "market_result" in sub.columns:
            if side == "NO":
                wins = (sub["market_result"] == "no").sum()
            else:
                wins = (sub["market_result"] == "yes").sum()
            print(f"    Win rate (preview): {wins / len(sub):.1%}")

    # Sample
    print(f"\nSample opportunities:")
    cols = ["ticker", "side", "yes_mid", "entry_price", "hours_to_expiry", "market_result"]
    cols = [c for c in cols if c in opps_df.columns]
    print(opps_df[cols].head(10).to_string(index=False))