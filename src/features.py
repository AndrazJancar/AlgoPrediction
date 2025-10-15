# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, datetime
from pathlib import Path
import os
import pandas as pd
import numpy as np

# Lokalni import – uporabljaš tvoj obstoječi etl.load_da_prices
from .etl import load_da_prices, TZ, BZN

@dataclass
class DataBundle:
    history: pd.DataFrame  # index = tz-aware hourly, col 'da_price'

def fetch_history(client, days_back: int = 400) -> DataBundle:
    """Povleče zgodovino DA cen in jo normalizira na urni time-index."""
    df = load_da_prices(client.api_key, days=days_back).copy()
    # poskrbi, da je urni raster in brez lukenj
    df = df.resample("h").mean()  # 'h' (mala) – velika 'H' je deprecated
    df = df.rename(columns={"DA_price": "da_price"}) if "DA_price" in df.columns else df
    assert "da_price" in df.columns, "Pričakujem stolpec 'da_price'"
    return DataBundle(history=df)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced calendar features for better seasonality capture"""
    out = pd.DataFrame(index=df.index)
    
    # Basic time features
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["month"] = out.index.month
    out["dayofyear"] = out.index.dayofyear
    out["quarter"] = out.index.quarter
    
    # Weekend and holiday features
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_monday"] = (out["dow"] == 0).astype(int)
    out["is_friday"] = (out["dow"] == 4).astype(int)
    
    # Peak hours (business hours 8-18)
    out["is_peak_hour"] = ((out["hour"] >= 8) & (out["hour"] <= 18)).astype(int)
    out["is_night_hour"] = ((out["hour"] <= 6) | (out["hour"] >= 22)).astype(int)
    
    # Cyclical encoding for better ML performance
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    
    return out

def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced price lags and rolling statistics"""
    out = pd.DataFrame(index=df.index)
    s = df["da_price"]
    
    # Basic lags
    out["lag_24"] = s.shift(24)
    out["lag_48"] = s.shift(48)
    out["lag_7d"] = s.shift(24 * 7)
    out["lag_14d"] = s.shift(24 * 14)
    out["lag_21d"] = s.shift(24 * 21)
    
    # Same hour previous days
    out["lag_same_hour_1d"] = s.shift(24)
    out["lag_same_hour_7d"] = s.shift(24 * 7)
    out["lag_same_hour_14d"] = s.shift(24 * 14)
    
    # Rolling statistics
    out["roll24_mean"] = s.shift(1).rolling(24, min_periods=1).mean()
    out["roll24_std"] = s.shift(1).rolling(24, min_periods=1).std()
    out["roll24_min"] = s.shift(1).rolling(24, min_periods=1).min()
    out["roll24_max"] = s.shift(1).rolling(24, min_periods=1).max()
    
    # Weekly rolling
    out["roll7d_mean"] = s.shift(1).rolling(24 * 7, min_periods=1).mean()
    out["roll7d_std"] = s.shift(1).rolling(24 * 7, min_periods=1).std()
    
    # Price volatility (rolling coefficient of variation)
    out["price_volatility"] = out["roll24_std"] / (out["roll24_mean"] + 1e-8)
    
    return out

def make_supervised(df: pd.DataFrame):
    """Pripravi (X, y) za učenje: cilj je cena čez 24h na isti uri."""
    X = pd.concat([add_calendar_features(df), add_price_lags(df)], axis=1)
    y = df["da_price"].shift(-24)   # cilj = D+1 na isti uri
    XY = pd.concat([X, y.rename("y")], axis=1).dropna()
    return XY.drop(columns=["y"]), XY["y"]

def build_inference_frame(df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Zgradi X za 24 ur dneva `target_date` tako, da dodamo 'placeholder' vrstice
    za jutri in ponovno izračunamo lag-e, ki gledajo nazaj v zgodovino.
    """
    target_date = pd.Timestamp(target_date).tz_localize(TZ) if target_date.tz is None else target_date
    # zgradi 24-urni index za target dan
    idx_tomorrow = pd.date_range(
        start=target_date.normalize(), periods=24, freq="h", tz=TZ
    )
    # pripni prazne vrstice za izračun lagov
    df_ext = df.reindex(df.index.union(idx_tomorrow)).sort_index()
    X_ext = pd.concat([add_calendar_features(df_ext), add_price_lags(df_ext)], axis=1)
    X_pred = X_ext.loc[idx_tomorrow]
    return X_pred
