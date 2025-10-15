# src/train.py
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
import numpy as np

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

    # Check if we have enough samples for time series split
    min_samples = 100  # minimum samples needed
    if len(X) < min_samples:
        print(f"Warning: Only {len(X)} samples available, using simple train/validation split")
        # Simple 80/20 split
        split_point = int(0.8 * len(X))
        train_idx = list(range(split_point))
        val_idx = list(range(split_point, len(X)))
        splits = [(train_idx, val_idx)]
    else:
        # Enhanced time series validation with multiple splits
        n_splits = min(5, len(X) // 50)  # Adaptive number of splits
        tss = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tss.split(X))
        print(f"Using {n_splits} time series splits with {len(X)} samples")
    
    # Use last 2 splits for validation (more robust) or single split if not enough data
    val_scores = []
    all_models = {'baseline': [], 'lgbm_p50': [], 'lgbm_p10': [], 'lgbm_p90': []}
    
    for train_idx, val_idx in splits[-2:]:  # Last 2 splits or single split
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Baseline model (ElasticNet) - fast and interpretable
        baseline = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)
        baseline.fit(X_tr, y_tr)
        all_models['baseline'].append(baseline)
        
        # 2. LightGBM P50 (main model)
        lgbm_p50 = LGBMRegressor(
            n_estimators=1000, learning_rate=0.03, num_leaves=128, 
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1
        )
        lgbm_p50.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], 
                     callbacks=[lgbm_p50.early_stopping(50, verbose=False)])
        all_models['lgbm_p50'].append(lgbm_p50)
        
        # 3. Quantile models
        lgbm_p10 = LGBMRegressor(
            objective="quantile", alpha=0.10,
            n_estimators=1000, learning_rate=0.03, num_leaves=128,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1
        )
        lgbm_p90 = LGBMRegressor(
            objective="quantile", alpha=0.90,
            n_estimators=1000, learning_rate=0.03, num_leaves=128,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1
        )
        
        lgbm_p10.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], 
                     callbacks=[lgbm_p10.early_stopping(50, verbose=False)])
        lgbm_p90.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], 
                     callbacks=[lgbm_p90.early_stopping(50, verbose=False)])
        
        all_models['lgbm_p10'].append(lgbm_p10)
        all_models['lgbm_p90'].append(lgbm_p90)
        
        # Validation score
        p50_pred = lgbm_p50.predict(X_va)
        mae = mean_absolute_error(y_va, p50_pred)
        val_scores.append(mae)
    
    # Final model selection - use the best performing model from validation
    best_split_idx = np.argmin(val_scores)
    print(f"Best validation MAE: {val_scores[best_split_idx]:.3f}")
    
    # Save the best models
    joblib.dump(all_models['baseline'][best_split_idx], MODELS_DIR / "baseline.pkl")
    joblib.dump(all_models['lgbm_p50'][best_split_idx], MODELS_DIR / "lgbm_p50.pkl")
    joblib.dump(all_models['lgbm_p10'][best_split_idx], MODELS_DIR / "lgbm_p10.pkl")
    joblib.dump(all_models['lgbm_p90'][best_split_idx], MODELS_DIR / "lgbm_p90.pkl")

    # Training report with more metrics
    final_X_tr, final_y_tr = X.iloc[splits[-1][0]], y.iloc[splits[-1][0]]
    final_X_va, final_y_va = X.iloc[splits[-1][1]], y.iloc[splits[-1][1]]
    
    baseline_pred = all_models['baseline'][best_split_idx].predict(final_X_va)
    lgbm_pred = all_models['lgbm_p50'][best_split_idx].predict(final_X_va)
    
    report = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "rows": int(len(X)),
        "validation_scores": val_scores,
        "best_mae": float(min(val_scores)),
        "baseline_mae": float(mean_absolute_error(final_y_va, baseline_pred)),
        "lgbm_mae": float(mean_absolute_error(final_y_va, lgbm_pred)),
        "features": list(X.columns),
        "feature_count": len(X.columns)
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved models to {MODELS_DIR.resolve()}")
    print(f"Best validation MAE: {min(val_scores):.3f}")
    print(f"Baseline MAE: {mean_absolute_error(final_y_va, baseline_pred):.3f}")
    print(f"LightGBM MAE: {mean_absolute_error(final_y_va, lgbm_pred):.3f}")

def main():
    train_models(days_back=500)  # More data for better training

if __name__ == "__main__":
    main()
