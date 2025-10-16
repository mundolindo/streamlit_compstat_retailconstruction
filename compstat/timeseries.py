from __future__ import annotations

from typing import Dict

import pandas as pd


def _ensure_local_timestamp(df: pd.DataFrame) -> pd.Series:
    if "occurred_local" in df.columns:
        return df["occurred_local"]
    if "occurred_datetime" in df.columns:
        ts = pd.to_datetime(df["occurred_datetime"], utc=True)
        return ts.dt.tz_convert("America/Chicago")
    raise ValueError("Dataframe must include occurred timestamps.")


def build_time_series_views(offenses_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return commonly used time-series slices for downstream visualizations."""
    if offenses_df.empty:
        raise ValueError("Offense dataframe is empty.")

    local_ts = _ensure_local_timestamp(offenses_df)
    df = offenses_df.copy()
    df["occurred_local"] = local_ts
    df["occurred_day"] = df["occurred_local"].dt.normalize()

    daily_counts = (
        df.groupby(["store_id", "occurred_day"])
        .size()
        .rename("count")
        .reset_index()
    )
    daily_counts["occurred_day"] = daily_counts["occurred_day"].dt.date

    weekly_counts = (
        df.set_index("occurred_local")
        .groupby("store_id")
        .resample("W", label="left", closed="left")
        .size()
        .rename("count")
        .reset_index()
    )
    weekly_counts["occurred_week"] = weekly_counts["occurred_local"].dt.strftime("%Y-%m-%d")
    weekly_counts = weekly_counts.drop(columns=["occurred_local"])

    offense_categories = (
        df.groupby(["store_id", "nibrs_crime_category"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    df["weekday_num"] = df["occurred_local"].dt.weekday
    df["weekday_name"] = df["occurred_local"].dt.day_name()
    df["hour"] = df["occurred_local"].dt.hour
    hourly_pattern = (
        df.groupby(["store_id", "weekday_num", "weekday_name", "hour"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["store_id", "weekday_num", "hour"])
    )

    return {
        "daily_counts": daily_counts,
        "weekly_counts": weekly_counts,
        "offense_categories": offense_categories,
        "hourly_pattern": hourly_pattern,
    }
