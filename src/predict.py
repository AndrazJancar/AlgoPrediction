# src/predict.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from entsoe import EntsoePandasClient
from .util_env import get_entsoe_api_key

from .features import fetch_history, build_inference_frame, TZ, BZN

OUT_DIR = Path("out")
MODELS_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

def _client():
    api_key = get_entsoe_api_key()
    if not api_key:
        # Return a lightweight object with api_key=None; fetch_history will fallback to cache
        class _Dummy:
            api_key = None
        return _Dummy()
    return EntsoePandasClient(api_key=api_key)

def _load_models():
    """Load all trained models including baseline"""
    baseline_path = MODELS_DIR / "baseline.pkl"
    p50_path = MODELS_DIR / "lgbm_p50.pkl"
    p10_path = MODELS_DIR / "lgbm_p10.pkl"
    p90_path = MODELS_DIR / "lgbm_p90.pkl"
    
    if all(path.exists() for path in [baseline_path, p50_path, p10_path, p90_path]):
        return {
            'baseline': joblib.load(baseline_path),
            'p10': joblib.load(p10_path),
            'p50': joblib.load(p50_path),
            'p90': joblib.load(p90_path)
        }
    return None

def _persistence_forecast(df: pd.DataFrame, target_date: pd.Timestamp):
    """Fallback baseline: povprečje včerajšnje ure (kar imaš že delujoče)."""
    if target_date.tz is None:
        target_date = target_date.tz_localize(TZ)
    else:
        target_date = target_date.tz_convert(TZ)
    yday = (target_date - pd.Timedelta(days=1)).date()
    s = df.loc[df.index.date == yday, "da_price"]
    return s.values.tolist()

def forecast_next_day(freq: str = "15T"):
    client = _client()
    bundle = fetch_history(client, days_back=120, freq=freq)
    tomorrow = (pd.Timestamp.now(tz=TZ) + pd.Timedelta(days=1)).normalize()

    models = _load_models()
    if models is None:
        print("Models not found – using persistence baseline.")
        p50 = _persistence_forecast(bundle.history, tomorrow)
        out = {"date": str(tomorrow.date()), "P50": [float(x) for x in p50]}
        OUT_DIR.joinpath(f"forecast_{tomorrow.date()}.json").write_text(json.dumps(out))
        print(f"Wrote {OUT_DIR}/forecast_{tomorrow.date()}.json")
        return

    X_pred = build_inference_frame(bundle.history, tomorrow, freq=freq)
    
    # Handle NaN values in X_pred before prediction
    print(f"X_pred shape: {X_pred.shape}, NaN count: {X_pred.isna().sum().sum()}")
    
    # Impute NaN values using mean strategy
    imputer = SimpleImputer(strategy='mean')
    X_pred_imputed = pd.DataFrame(
        imputer.fit_transform(X_pred), 
        columns=X_pred.columns, 
        index=X_pred.index
    )
    
    print(f"X_pred_imputed shape: {X_pred_imputed.shape}, NaN count: {X_pred_imputed.isna().sum().sum()}")
    
    # Get predictions from all models using imputed data
    baseline_pred = models['baseline'].predict(X_pred_imputed)
    p10 = models['p10'].predict(X_pred_imputed)
    p50 = models['p50'].predict(X_pred_imputed)
    p90 = models['p90'].predict(X_pred_imputed)
    
    # Ensemble: blend baseline and LightGBM (simple weighted average)
    ensemble_weight = 0.3  # 30% baseline, 70% LightGBM
    p50_ensemble = ensemble_weight * baseline_pred + (1 - ensemble_weight) * p50
    
    # Ensure quantile ordering: P10 <= P50 <= P90
    dfq = pd.DataFrame({"p10": p10, "p50": p50_ensemble, "p90": p90})
    dfq = dfq.apply(sorted, axis=1, result_type="broadcast")  # row-wise sorting
    
    # Calculate prediction intervals and confidence
    prediction_interval = dfq["p90"].values - dfq["p10"].values
    # Better confidence score: based on relative interval width
    relative_interval = prediction_interval / (dfq["p50"].values + 1e-8)
    confidence_score = np.clip(1.0 - (relative_interval / 2.0), 0.0, 1.0)  # Scale to 0-1
    
    out = {
        "date": str(tomorrow.date()),
        "P10": [float(x) for x in dfq["p10"].values],
        "P50": [float(x) for x in dfq["p50"].values],
        "P90": [float(x) for x in dfq["p90"].values],
        "baseline": [float(x) for x in baseline_pred],
        "prediction_interval": [float(x) for x in prediction_interval],
        "confidence_score": [float(x) for x in confidence_score],
        "model_info": {
            "ensemble_weight": ensemble_weight,
            "features_count": len(X_pred.columns),
            "generated_at": datetime.now().isoformat()
        }
    }
    OUT_DIR.joinpath(f"forecast_{tomorrow.date()}.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_DIR}/forecast_{tomorrow.date()}.json")
    print(f"Average prediction interval: {np.mean(prediction_interval):.2f} EUR/MWh")
    print(f"Average confidence score: {np.mean(confidence_score):.3f}")

def main():
    forecast_next_day()

if __name__ == "__main__":
    main()
