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

    # poravnaj na skupni časovni indeks (lahko 15-min, DA je 60-min; ok)
    df = da.join(imb, how="inner").sort_index()

    TZ = ZoneInfo("Europe/Ljubljana")
    tomorrow = (datetime.now(TZ)+timedelta(days=1)).date()
    yday = (datetime.now(TZ)-timedelta(days=1)).date()

    # izberi samo včeraj in potem agregiraj po urah
    sel = df.index.date == yday

    # možnost A (najbolj čisto): resample po uri
    # pozor: resample deluje na časovnem indeksu izbrane podmnožice
    p50_series = df.loc[sel, ['da_price']].resample('H').mean()['da_price']

    # če slučajno manjka kakšna ura, zapolni s povprečjem dneva
    fallback = float(p50_series.mean()) if not p50_series.empty else float(df['da_price'].tail(24).mean())
    p50 = [float(p50_series.get(pd.Timestamp(yday, tz=TZ)+pd.Timedelta(hours=h), fallback))
           for h in range(24)]

    os.makedirs("out", exist_ok=True)
    out = {"date": str(tomorrow), "P50": p50}
    with open(f"out/forecast_{tomorrow}.json", "w") as f:
        json.dump(out, f)
    print("Wrote", f"out/forecast_{tomorrow}.json")


if __name__ == "__main__":
    main()
