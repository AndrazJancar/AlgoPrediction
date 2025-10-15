# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, datetime
from pathlib import Path
import os
import pandas as pd

# Lokalni import – uporabljaš tvoj obstoječi etl.load_da_prices
from .etl import load_da_prices

TZ = "Europe/Ljubljana"     # prilagodi, če v etl že definiraš
BZN = "SI"                  # bidding zone / country code

@dataclass
class DataBundle:
    history: pd.DataFrame  # index = tz-aware hourly, col 'da_price'

def fetch_history(client, days_back: int = 400) -> DataBundle:
    """Povleče zgodovino DA cen in jo normalizira na urni time-index."""
    df = load_da_prices(os.environ["ENTSOE_API_KEY"], days=days_back).copy()
    # poskrbi, da je urni raster in brez lukenj
    df = df.resample("h").mean()  # 'h' (mala) – velika 'H' je deprecated
    df = df.rename(columns={"DA_price": "da_price"}) if "DA_price" in df.columns else df
    assert "da_price" in df.columns, "Pričakujem stolpec 'da_price'"
    return DataBundle(history=df)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = out.index.month
    out["dayofyear"] = out.index.dayofyear
    return out

def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Osnovni EPF lag-i: 24h, 7*24h, 14*24h + 24h rolling povprečje."""
    out = pd.DataFrame(index=df.index)
    s = df["da_price"]
    out["lag_24"] = s.shift(24)
    out["lag_7d"] = s.shift(24 * 7)
    out["lag_14d"] = s.shift(24 * 14)
    out["roll24_mean"] = s.shift(1).rolling(24, min_periods=1).mean()
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
