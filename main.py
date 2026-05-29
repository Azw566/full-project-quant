"""
main.py — Entry point for Phase 0.

Run this to verify the data layer works end-to-end:
    python main.py

What it does:
    1. Loads config from config.yaml
    2. Calls get_bars() twice with the same arguments
    3. Prints a sample of the data
    4. Confirms both calls returned identical results (reproducibility check)

Expected output:
    First call  → downloads from Yahoo Finance, prints progress
    Second call → reads from cache (near-instant), same data
"""

import logging

import pandas as pd
import yaml

from data.loader import get_bars

# ── Logging setup ────────────────────────────────────────────────────────────
# Format: timestamp | level | module | message
# This pattern carries through every phase — structured logs are searchable.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    # ── Load config ──────────────────────────────────────────────────────────
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    symbol    = cfg["symbol"]
    start     = cfg["start"]
    end       = cfg["end"]
    data_path = cfg["data_path"]

    # ── First call: expect a network download ────────────────────────────────
    print("\n" + "-"*60)
    print(" CALL 1 -- expecting a download from Yahoo Finance")
    print("-"*60)
    bars = get_bars(symbol, start, end, data_path)

    print(f"\nShape : {bars.shape}  ({bars.shape[0]} trading days x {bars.shape[1]} columns)")
    print(f"Dates : {bars.index[0].date()}  to  {bars.index[-1].date()}")
    print("\nFirst 3 rows:")
    print(bars.head(3).to_string())
    print("\nLast 3 rows:")
    print(bars.tail(3).to_string())

    # ── Second call: expect an instant cache read ────────────────────────────
    print("\n" + "-"*60)
    print(" CALL 2 -- expecting a cache hit (no network)")
    print("-"*60)
    bars2 = get_bars(symbol, start, end, data_path)

    # ── Reproducibility check ────────────────────────────────────────────────
    # pd.DataFrame.equals() checks values, dtypes, and index — everything.
    if bars.equals(bars2):
        print("\n[PASS] Reproducibility check PASSED -- both calls returned identical data.")
    else:
        print("\n[FAIL] Reproducibility check FAILED -- investigate data/loader.py.")


if __name__ == "__main__":
    main()
