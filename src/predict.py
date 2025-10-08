import os, json
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.etl import load_da_prices, load_imbalance

TZ = ZoneInfo("Europe/Ljubljana")
API = os.environ["ENTSOE_API_KEY"]

def main():
    da = load_da_prices(API, 60)
    imb = load_imbalance(API, 60)
    df = da.join(imb, how="inner")
    tomorrow = (datetime.now(TZ)+timedelta(days=1)).date()
    yday = (datetime.now(TZ)-timedelta(days=1)).date()
    p50 = df[df.index.date==yday]["da_price"].groupby(df.index.hour).mean()
    out = {"date": str(tomorrow),
           "P50": [float(p50.get(h, p50.mean())) for h in range(24)]}
    os.makedirs("out", exist_ok=True)
    with open(f"out/forecast_{tomorrow}.json","w") as f: json.dump(out, f)
    print("Wrote", f"out/forecast_{tomorrow}.json")

if __name__ == "__main__":
    main()
