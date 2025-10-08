from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Ljubljana")
BZN = "10YSI-ELES-----O"  # SI bidding zone EIC

def load_da_prices(api_key:str, days:int=120)->pd.DataFrame:
    c = EntsoePandasClient(api_key=api_key)
    end = pd.Timestamp(datetime.now(TZ).date()+timedelta(days=1), tz=TZ)
    start = end - pd.Timedelta(days=days)
    da = c.query_day_ahead_prices(BZN, start.tz_convert("UTC"), end.tz_convert("UTC"))
    return da.tz_convert(TZ).to_frame("da_price")

def load_imbalance(api_key:str, days:int=120)->pd.DataFrame:
    c = EntsoePandasClient(api_key=api_key)
    end = pd.Timestamp(datetime.now(TZ).date(), tz=TZ)
    start = end - pd.Timedelta(days=days)
    imb = c.query_imbalance_prices(BZN, start.tz_convert("UTC"), end.tz_convert("UTC"))
    return imb.tz_convert(TZ).to_frame("imb_price")
