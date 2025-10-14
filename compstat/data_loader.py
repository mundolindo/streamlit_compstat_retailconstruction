from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

STORE_FILE = DATA_DIR / "home_improvement_stores.csv"
OFFENSE_FILE = DATA_DIR / "store_offenses.parquet"


def load_store_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load curated store locations."""
    data_path = path or STORE_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Store data not found at {data_path}. Run scripts/fetch_data.py first."
        )

    df = pd.read_csv(data_path)
    numeric_cols = ["latitude", "longitude", "radius_meters"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["store_label"] = df["brand"] + " — " + df["city"].fillna("")
    return df


def load_offense_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load pre-pulled offense data and ensure datetime fields."""
    data_path = path or OFFENSE_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Offense data not found at {data_path}. Run scripts/fetch_data.py first."
        )

    df = pd.read_parquet(data_path)
    if "occurred_local" in df.columns:
        if pd.api.types.is_datetime64tz_dtype(df["occurred_local"]):
            local_ts = df["occurred_local"]
        else:
            local_ts = pd.to_datetime(
                df["occurred_local"], errors="coerce"
            ).dt.tz_localize("America/Chicago")
    else:
        utc_ts = pd.to_datetime(df["occurred_datetime"], errors="coerce", utc=True)
        local_ts = utc_ts.dt.tz_convert("America/Chicago")

    df["occurred_local"] = local_ts
    df["occurred_datetime"] = local_ts.dt.tz_convert("UTC")
    df = df.dropna(subset=["occurred_local"])

    df["occurred_date"] = df["occurred_local"].dt.date
    localized = df["occurred_local"].dt.tz_localize(None)
    df["occurred_week"] = localized.dt.to_period("W").astype(str)
    df["occurred_month"] = localized.dt.to_period("M").astype(str)
    df["day_of_week"] = df["occurred_local"].dt.day_name()
    df["hour_of_day"] = df["occurred_local"].dt.hour

    # Ensure string columns don't contain NaN
    string_fields = [
        "nibrs_crime",
        "nibrs_crime_category",
        "nibrs_crime_against",
        "beat",
        "division",
        "sector",
        "district",
        "zip_code",
        "incident_address",
    ]
    for field in string_fields:
        if field in df.columns:
            df[field] = df[field].fillna("").astype(str)

    df["pull_timestamp_utc"] = pd.to_datetime(
        df["pull_timestamp_utc"], errors="coerce", utc=True
    ).fillna(datetime.utcnow())

    return df
