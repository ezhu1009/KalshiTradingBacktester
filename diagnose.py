"""diagnose.py — Inspect a dataset to see why filters aren't matching.

Usage:
    python diagnose.py SoccerUFC.pkl
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch import load_dataset
from filters import _get_yes_mid, _safe_float, _parse_timestamp


def diagnose(path: str):
    ds = load_dataset(path)
    print(f"\n{ds.summary()}\n")

    # ── Markets DataFrame ────────────────────────────────────────────
    print("=" * 70)
    print("MARKETS DATAFRAME")
    print("=" * 70)
    print(f"Columns: {list(ds.markets_df.columns)}")
    print(f"\nFirst 3 rows:")
    print(ds.markets_df.head(3).to_string())
    print(f"\nResult value counts:")
    print(ds.markets_df["result"].value_counts(dropna=False))
    print(f"\nVolume stats:")
    print(ds.markets_df["volume"].describe())

    # ── Candles structure ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CANDLES STRUCTURE")
    print("=" * 70)

    # Find a ticker with candles
    sample_ticker = None
    for ticker, df in ds.candles.items():
        if len(df) > 0:
            sample_ticker = ticker
            break

    if sample_ticker is None:
        print("⚠ No tickers have any candles!")
        return

    sample = ds.candles[sample_ticker]
    print(f"Sample ticker: {sample_ticker}")
    print(f"Candles: {len(sample)} rows")
    print(f"Columns: {list(sample.columns)}")
    print(f"\nFirst 3 rows:")
    print(sample.head(3).to_string())
    print(f"\nDtypes:")
    print(sample.dtypes)

    # ── Test yes_mid extraction ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("YES_MID EXTRACTION TEST")
    print("=" * 70)

    extracted = []
    for ticker, df in ds.candles.items():
        for _, candle in df.iterrows():
            ym = _get_yes_mid(candle)
            if ym is not None:
                extracted.append(ym)

    if not extracted:
        print("⚠ Could NOT extract yes_mid from ANY candle!")
        print("\nThe filters._get_yes_mid() function expects one of these column patterns:")
        print("  - yes_bid_close + yes_ask_close")
        print("  - yes_bid_dollars_close + yes_ask_dollars_close")
        print("  - yes_bid + yes_ask")
        print("  - price_close, price_mean, close, or price")
        print(f"\nYour candles have these columns instead: {list(sample.columns)}")
    else:
        import numpy as np
        arr = np.array(extracted)
        print(f"Successfully extracted {len(arr)} yes_mid values")
        print(f"\nDistribution:")
        print(f"  min:    {arr.min():.4f}")
        print(f"  25%:    {np.percentile(arr, 25):.4f}")
        print(f"  median: {np.median(arr):.4f}")
        print(f"  75%:    {np.percentile(arr, 75):.4f}")
        print(f"  max:    {arr.max():.4f}")
        print(f"\nHow many candles cross common thresholds:")
        for low in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            n = (arr <= low).sum()
            print(f"  yes_mid ≤ {low:.2f}: {n} candles ({100*n/len(arr):.1f}%)")
        for high in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            n = (arr >= high).sum()
            print(f"  yes_mid ≥ {high:.2f}: {n} candles ({100*n/len(arr):.1f}%)")

    # ── Time-to-expiry sanity check ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TIME-TO-EXPIRY CHECK")
    print("=" * 70)
    sample_market = ds.markets_df[ds.markets_df["ticker"] == sample_ticker].iloc[0]
    close_ts = _parse_timestamp(sample_market.get("close_time", ""))
    print(f"Market close_time: {sample_market['close_time']}")
    print(f"Parsed as unix ts: {close_ts}")

    if close_ts:
        sample_candles = ds.candles[sample_ticker]
        # Try to find a timestamp column
        ts_col = None
        for col in ["timestamp", "end_period_ts", "ts"]:
            if col in sample_candles.columns:
                ts_col = col
                break

        if ts_col:
            candle_ts = sample_candles[ts_col].iloc[0]
            hours = (close_ts - float(candle_ts)) / 3600
            print(f"First candle ts: {candle_ts}")
            print(f"Hours to expiry at first candle: {hours:.1f}")
        else:
            print(f"⚠ No timestamp column found in candles!")
            print(f"  Looked for: timestamp, end_period_ts, ts")
            print(f"  Available: {list(sample_candles.columns)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data.pkl"
    diagnose(path)