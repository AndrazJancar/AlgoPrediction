from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import timedelta
from entsoe import EntsoePandasClient

# Ensure repo root is on sys.path for `from src import ...` when executed as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.util_env import get_entsoe_api_key
from src.features import TZ, BZN


def main() -> int:
    key = get_entsoe_api_key()
    print("ENTSOE key present:", bool(key))
    if not key:
        print("Warning: no ENTSOE_API_KEY; only cache/baseline will work.")
        return 0

    try:
        c = EntsoePandasClient(api_key=key)
        end = pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=2)
        df = c.query_day_ahead_prices(BZN, start=start, end=end)
        print("Fetched rows:", len(df))
        return 0
    except Exception as e:
        print("ENTSOE fetch failed:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


