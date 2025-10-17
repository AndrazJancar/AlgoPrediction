# src/backtest.py
from __future__ import annotations

import math
from pathlib import Path
from datetime import timedelta
import os
import json
import pandas as pd
import joblib
from entsoe import EntsoePandasClient
from .util_env import get_entsoe_api_key

from .features import fetch_history, build_inference_frame, TZ, BZN

OUT_DIR = Path("out")
MODELS_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

def _client():
    api_key = get_entsoe_api_key()
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_KEY")
    return EntsoePandasClient(api_key=api_key)

def smape(y_true, y_pred):
    # 100/n * Σ |F-A| / ((|A|+|F|)/2)
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    denom = (y_true.abs() + y_pred.abs()) / 2.0
    # zaščita pred deljenjem z 0
    denom = denom.replace(0, 1e-9)
    return float(100 * (y_pred.sub(y_true).abs() / denom).mean())

def backtest(last_n_days: int = 14):
    client = _client()
    # zgodovina za obdobje + buffer za lage
    bundle = fetch_history(client, days_back=120 + last_n_days + 14)
    end_date = bundle.history.index.normalize().max().date()
    dates = [pd.Timestamp(end_date, tz=TZ) - pd.Timedelta(days=i) for i in range(last_n_days, 0, -1)]

    # naloži, če obstajajo; sicer baseline
    have_models = (MODELS_DIR / "lgbm_p50.pkl").exists()
    if have_models:
        mdl_p10 = joblib.load(MODELS_DIR / "lgbm_p10.pkl")
        mdl_p50 = joblib.load(MODELS_DIR / "lgbm_p50.pkl")
        mdl_p90 = joblib.load(MODELS_DIR / "lgbm_p90.pkl")

    rows = []
    for d in dates:
        X_pred = build_inference_frame(bundle.history.loc[: d - pd.Timedelta(seconds=1)], d)
        if have_models:
            p50 = mdl_p50.predict(X_pred)
        else:
            # baseline: včeraj = jutri
            yday = (d - pd.Timedelta(days=1)).date()
            p50 = bundle.history.loc[bundle.history.index.date == yday, "da_price"].values

        # resnica za dan d
        y_true = bundle.history.loc[bundle.history.index.date == d.date(), "da_price"].values
        if len(y_true) != 24 or len(p50) != 24:
            continue
        mae = float(pd.Series(p50).sub(y_true).abs().mean())
        rows.append({"date": str(d.date()), "MAE": mae, "SMAPE": smape(y_true, p50)})

    dfm = pd.DataFrame(rows)
    dfm.to_csv(OUT_DIR / f"backtest_{dates[0].date()}_{dates[-1].date()}.csv", index=False)
    print(dfm.describe(numeric_only=True))

if __name__ == "__main__":
    backtest()
