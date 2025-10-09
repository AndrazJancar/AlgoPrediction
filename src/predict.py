# src/predict.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
import os
import pandas as pd
import joblib
from entsoe import EntsoePandasClient

from .features import fetch_history, build_inference_frame, TZ, BZN

OUT_DIR = Path("out")
MODELS_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

def _client():
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_KEY")
    return EntsoePandasClient(api_key=api_key)

def _load_models():
    p50 = MODELS_DIR / "lgbm_p50.pkl"
    p10 = MODELS_DIR / "lgbm_p10.pkl"
    p90 = MODELS_DIR / "lgbm_p90.pkl"
    if p50.exists() and p10.exists() and p90.exists():
        return (joblib.load(p10), joblib.load(p50), joblib.load(p90))
    return None

def _persistence_forecast(df: pd.DataFrame, target_date: pd.Timestamp):
    """Fallback baseline: povprečje včerajšnje ure (kar imaš že delujoče)."""
    yday = (pd.Timestamp(target_date).tz_localize(TZ) - pd.Timedelta(days=1)).date()
    s = df.loc[df.index.date == yday, "da_price"]
    return s.values.tolist()

def forecast_next_day():
    client = _client()
    bundle = fetch_history(client, days_back=120)
    tomorrow = (pd.Timestamp.now(tz=TZ) + pd.Timedelta(days=1)).normalize()

    models = _load_models()
    if models is None:
        print("Models not found – using persistence baseline.")
        p50 = _persistence_forecast(bundle.history, tomorrow)
        out = {"date": str(tomorrow.date()), "P50": [float(x) for x in p50]}
        OUT_DIR.joinpath(f"forecast_{tomorrow.date()}.json").write_text(json.dumps(out))
        print(f"Wrote {OUT_DIR}/forecast_{tomorrow.date()}.json")
        return

    mdl_p10, mdl_p50, mdl_p90 = models
    X_pred = build_inference_frame(bundle.history, tomorrow)
    p10 = mdl_p10.predict(X_pred)
    p50 = mdl_p50.predict(X_pred)
    p90 = mdl_p90.predict(X_pred)

    # Uredi, da je P10 <= P50 <= P90
    dfq = pd.DataFrame({"p10": p10, "p50": p50, "p90": p90})
    dfq = dfq.apply(sorted, axis=1, result_type="broadcast")  # vrstično sortiranje

    out = {
        "date": str(tomorrow.date()),
        "P10": [float(x) for x in dfq["p10"].values],
        "P50": [float(x) for x in dfq["p50"].values],
        "P90": [float(x) for x in dfq["p90"].values],
    }
    OUT_DIR.joinpath(f"forecast_{tomorrow.date()}.json").write_text(json.dumps(out))
    print(f"Wrote {OUT_DIR}/forecast_{tomorrow.date()}.json")

def main():
    forecast_next_day()

if __name__ == "__main__":
    main()
