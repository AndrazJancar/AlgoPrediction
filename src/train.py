# src/train.py
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

from entsoe import EntsoePandasClient
from .features import fetch_history, make_supervised, TZ, BZN

MODELS_DIR = Path("models")
OUT_DIR = Path("out")
MODELS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def _client() -> EntsoePandasClient:
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ENTSOE_API_KEY")
    return EntsoePandasClient(api_key=api_key)

def train_models(days_back: int = 400):
    client = _client()
    bundle = fetch_history(client, days_back=days_back)
    X, y = make_supervised(bundle.history)

    # Časovni CV (vzdrži monotono časovno delitev)
    # Pazi: TimeSeriesSplit zahteva približno enakomerni korak vzorčenja
    tss = TimeSeriesSplit(n_splits=5)
    last_split = list(tss.split(X))[ -1 ]
    tr_idx, va_idx = last_split
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    # P50 (median)
    mdl_p50 = LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=64, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    mdl_p50.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Kvantili – objective='quantile', alpha določa kvantil
    mdl_p10 = LGBMRegressor(
        objective="quantile", alpha=0.10,
        n_estimators=800, learning_rate=0.05, num_leaves=64, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    mdl_p90 = LGBMRegressor(
        objective="quantile", alpha=0.90,
        n_estimators=800, learning_rate=0.05, num_leaves=64, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    mdl_p10.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    mdl_p90.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Preprosta validacija
    p50_va = mdl_p50.predict(X_va)
    mae = float(mean_absolute_error(y_va, p50_va))

    # Shrani modele + metapodatke
    joblib.dump(mdl_p50, MODELS_DIR / "lgbm_p50.pkl")
    joblib.dump(mdl_p10, MODELS_DIR / "lgbm_p10.pkl")
    joblib.dump(mdl_p90, MODELS_DIR / "lgbm_p90.pkl")

    report = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(len(X)),
        "val_mae": mae,
        "features": list(X.columns),
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved models to {MODELS_DIR.resolve()}")
    print(f"Validation MAE (P50): {mae:.3f}")

def main():
    train_models(days_back=400)

if __name__ == "__main__":
    main()
