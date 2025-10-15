from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import os

TZ = ZoneInfo("Europe/Ljubljana")
BZN = "10Y1001A1001A47J"  # SI bidding zone code for ENTSO-E
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def _to_df(x: pd.DataFrame | pd.Series, colname: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(colname)
    # x je DataFrame
    if colname not in x.columns and x.shape[1] == 1:
        x = x.rename(columns={x.columns[0]: colname})
    return x

def _cache_path(data_type: str, date: datetime) -> Path:
    """Generate cache path for data type and date"""
    date_str = date.strftime("%Y-%m-%d")
    cache_dir = DATA_DIR / "raw" / data_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{date_str}.parquet"

def _load_from_cache(cache_path: Path, max_age_hours: int = 24) -> pd.DataFrame | None:
    """Load data from cache if it exists and is fresh enough"""
    if not cache_path.exists():
        return None
    
    # Check if cache is fresh enough
    cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
    if cache_age > max_age_hours * 3600:
        return None
    
    try:
        return pd.read_parquet(cache_path)
    except:
        return None

def _save_to_cache(df: pd.DataFrame, cache_path: Path):
    """Save DataFrame to cache"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

def load_da_prices(api_key: str, days: int = 120, use_cache: bool = True) -> pd.DataFrame:
    """Load day-ahead prices with caching and better error handling"""
    c = EntsoePandasClient(api_key=api_key)
    end_local = pd.Timestamp(datetime.now(TZ).date() + timedelta(days=1), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    
    # Try to load from cache first
    if use_cache:
        cache_path = _cache_path("da_prices", datetime.now())
        cached_df = _load_from_cache(cache_path)
        if cached_df is not None:
            return cached_df
    
    try:
        da = c.query_day_ahead_prices(BZN, start=start_local, end=end_local)
        df = _to_df(da, "da_price")
        df = df.tz_convert(TZ)
        
        # Cache the result
        if use_cache:
            _save_to_cache(df, cache_path)
        
        return df
    except Exception as e:
        print(f"Error loading DA prices: {e}")
        # Return empty DataFrame with proper structure
        return pd.DataFrame(columns=["da_price"], index=pd.date_range(start_local, end_local, freq="h", tz=TZ))

def load_load_forecast(api_key: str, days: int = 30) -> pd.DataFrame:
    """Load load forecast data for better predictions"""
    c = EntsoePandasClient(api_key=api_key)
    end_local = pd.Timestamp(datetime.now(TZ).date() + timedelta(days=1), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    
    try:
        load_fc = c.query_load_forecast(BZN, start=start_local, end=end_local)
        df = _to_df(load_fc, "load_forecast")
        return df.tz_convert(TZ)
    except Exception as e:
        print(f"Error loading load forecast: {e}")
        return pd.DataFrame()

def load_generation_forecast(api_key: str, days: int = 30) -> pd.DataFrame:
    """Load generation forecast data"""
    c = EntsoePandasClient(api_key=api_key)
    end_local = pd.Timestamp(datetime.now(TZ).date() + timedelta(days=1), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    
    try:
        gen_fc = c.query_generation_forecast(BZN, start=start_local, end=end_local)
        df = _to_df(gen_fc, "generation_forecast")
        return df.tz_convert(TZ)
    except Exception as e:
        print(f"Error loading generation forecast: {e}")
        return pd.DataFrame()

def load_imbalance(api_key: str, days: int = 120) -> pd.DataFrame:
    """Load imbalance prices for risk assessment"""
    c = EntsoePandasClient(api_key=api_key)
    end_local = pd.Timestamp(datetime.now(TZ).date(), tz=TZ)
    start_local = end_local - pd.Timedelta(days=days)
    
    try:
        imb = c.query_imbalance_prices(BZN, start=start_local, end=end_local)
        df = _to_df(imb, "imb_price")
        return df.tz_convert(TZ)
    except Exception as e:
        print(f"Error loading imbalance prices: {e}")
        return pd.DataFrame()
