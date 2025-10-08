from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Ljubljana")
BZN = "SI"  # za EntsoePandasClient uporabi "SI"

def _to_df(x: pd.DataFrame | pd.Series, colname: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(colname)
    # x je DataFrame
    if colname not in x.columns and x.shape[1] == 1:
        x = x.rename(columns={x.columns[0]: colname})
    return x

def load_da_prices(api_key: str, days: int = 120) -> pd.DataFrame:
    c = EntsoePandasClient(api_key=api_key)
    end_local = pd.Timestamp(datetime.now(TZ).date() + timedelta(days=1), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    da = c.query_day_ahead_prices(BZN, start=start_local, end=end_local)  # zahteva start=, end= :contentReference[oaicite:1]{index=1}
    df = _to_df(da, "da_price")
    return df.tz_convert(TZ)

def load_imbalance(api_key: str, days: int = 120) -> pd.DataFrame:
    c = EntsoePandasClient(api_key=api_key)
    end_local = pd.Timestamp(datetime.now(TZ).date(), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    imb = c.query_imbalance_prices(BZN, start=start_local, end=end_local)  # zahteva start=, end= :contentReference[oaicite:2]{index=2}
    df = _to_df(imb, "imb_price")
    return df.tz_convert(TZ)
