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

def fetch_history(client, days_back: int = 400, freq: str = "15T") -> DataBundle:
    """Povleče zgodovino DA cen in jo normalizira na 15-minutni time-index."""
    df = load_da_prices(client.api_key, days=days_back, freq=freq).copy()
    
    # Ensure 15-minute resolution
    if freq == "15T":
        df = df.resample("15T").ffill()  # Forward fill for 15-min intervals
    else:
        df = df.resample("h").mean()  # Hourly for backward compatibility
    
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

def add_price_lags(df: pd.DataFrame, freq: str = "15T") -> pd.DataFrame:
    """Enhanced price lags and rolling statistics for 15-minute or hourly data"""
    out = pd.DataFrame(index=df.index)
    s = df["da_price"]
    
    # Determine periods based on frequency
    if freq == "15T":
        # 15-minute intervals: 96 periods = 1 day, 672 = 1 week
        periods_1d = 96  # 24h * 4 (15-min intervals)
        periods_1w = 672  # 7 days * 96
        periods_2w = 1344  # 14 days * 96
    else:
        # Hourly intervals (default)
        periods_1d = 24
        periods_1w = 168  # 24 * 7
        periods_2w = 336  # 24 * 14
    
    # Basic lags
    out["lag_1d"] = s.shift(periods_1d)
    out["lag_2d"] = s.shift(periods_1d * 2)
    out["lag_1w"] = s.shift(periods_1w)
    out["lag_2w"] = s.shift(periods_2w)
    
    # Same time slot previous days
    out["lag_same_slot_1d"] = s.shift(periods_1d)
    out["lag_same_slot_7d"] = s.shift(periods_1w)
    out["lag_same_slot_14d"] = s.shift(periods_2w)
    
    # Rolling statistics (adaptive to frequency)
    out["roll1d_mean"] = s.shift(1).rolling(periods_1d, min_periods=1).mean()
    out["roll1d_std"] = s.shift(1).rolling(periods_1d, min_periods=1).std()
    out["roll1d_min"] = s.shift(1).rolling(periods_1d, min_periods=1).min()
    out["roll1d_max"] = s.shift(1).rolling(periods_1d, min_periods=1).max()
    
    # Weekly rolling
    out["roll1w_mean"] = s.shift(1).rolling(periods_1w, min_periods=1).mean()
    out["roll1w_std"] = s.shift(1).rolling(periods_1w, min_periods=1).std()
    
    # Price volatility (rolling coefficient of variation)
    out["price_volatility"] = out["roll1d_std"] / (out["roll1d_mean"] + 1e-8)
    
    # Intraday patterns (for 15-min data)
    if freq == "15T":
        out["lag_4h"] = s.shift(16)  # 4 hours back (16 * 15min)
        out["lag_8h"] = s.shift(32)  # 8 hours back
        out["roll4h_mean"] = s.shift(1).rolling(16, min_periods=1).mean()
    
    return out

def make_supervised(df: pd.DataFrame, freq: str = "15T"):
    """Pripravi (X, y) za učenje: cilj je cena čez 1 dan na istem času."""
    # Determine prediction horizon based on frequency
    if freq == "15T":
        periods_ahead = 96  # 1 day in 15-min intervals
    else:
        periods_ahead = 24  # 1 day in hourly intervals
    
    X = pd.concat([add_calendar_features(df), add_price_lags(df, freq=freq)], axis=1)
    y = df["da_price"].shift(-periods_ahead)   # cilj = D+1 na istem času
    XY = pd.concat([X, y.rename("y")], axis=1).dropna()
    return XY.drop(columns=["y"]), XY["y"]

def build_inference_frame(df: pd.DataFrame, target_date: pd.Timestamp, freq: str = "15T") -> pd.DataFrame:
    """
    Zgradi X za 1 dan `target_date` tako, da dodamo 'placeholder' vrstice
    za jutri in ponovno izračunamo lag-e, ki gledajo nazaj v zgodovino.
    """
    target_date = pd.Timestamp(target_date).tz_localize(TZ) if target_date.tz is None else target_date
    
    # Determine periods based on frequency
    if freq == "15T":
        periods_per_day = 96  # 24h * 4 (15-min intervals)
        freq_str = "15T"
    else:
        periods_per_day = 24  # 24 hours
        freq_str = "h"
    
    # zgradi index za target dan
    idx_tomorrow = pd.date_range(
        start=target_date.normalize(), periods=periods_per_day, freq=freq_str, tz=TZ
    )
    # pripni prazne vrstice za izračun lagov
    df_ext = df.reindex(df.index.union(idx_tomorrow)).sort_index()
    X_ext = pd.concat([add_calendar_features(df_ext), add_price_lags(df_ext, freq=freq)], axis=1)
    X_pred = X_ext.loc[idx_tomorrow]
    return X_pred
