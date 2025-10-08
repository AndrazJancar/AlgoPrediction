from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Ljubljana")
BZN = "SI"  # za EntsoePandasClient uporabi "SI" (ne EIC kodo)

def _window(days: int):
    end_local = pd.Timestamp(datetime.now(TZ).date(), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    # EntsoePandasClient sprejme TZ-aware Timestamp; sprejeto je tudi EU/Ljubljana
    return start_local, end_local

def load_da_prices(api_key: str, days: int = 120) -> pd.DataFrame:
    c = EntsoePandasClient(api_key=api_key)
    # za DA želimo do jutri (D+1)
    end_local = pd.Timestamp(datetime.now(TZ).date() + timedelta(days=1), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    da = c.query_day_ahead_prices(BZN, start=start_local, end=end_local)
    return da.tz_convert(TZ).to_frame("da_price")

def load_imbalance(api_key: str, days: int = 120) -> pd.DataFrame:
    c = EntsoePandasClient(api_key=api_key)
    start_local, end_local = _window(days)
    # KLJUČNO: poimenski argumenti start=, end=
    imb = c.query_imbalance_prices(BZN, start=start_local, end=end_local)
    return imb.tz_convert(TZ).to_frame("imb_price")
