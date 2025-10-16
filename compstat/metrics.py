from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import poisson


@dataclass(frozen=True)
class PeriodDefinition:
    key: str
    label: str
    days: int
    type: str = "rolling"  # "rolling" or "ytd"


PERIODS: Tuple[PeriodDefinition, ...] = (
    PeriodDefinition(key="seven_day", label="7-Day", days=7),
    PeriodDefinition(key="four_week", label="28-Day", days=28),
    PeriodDefinition(key="ytd", label="YTD", days=0, type="ytd"),
)


def _ensure_reference_date(
    offenses_df: pd.DataFrame, reference_date: date | None
) -> date:
    if reference_date:
        return reference_date
    if "occurred_date" in offenses_df.columns:
        max_date = offenses_df["occurred_date"].max()
        if pd.isna(max_date):
            return date.today()
        if isinstance(max_date, pd.Timestamp):
            return max_date.date()
        return max_date
    return date.today()


def _build_daily_matrix(
    offenses_df: pd.DataFrame,
    store_columns: Iterable[str] = ("store_id", "store_brand", "store_name", "store_city"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_counts, store_metadata)."""
    df = offenses_df.copy()
    if "occurred_local" in df.columns:
        occurred_day = df["occurred_local"].dt.tz_convert("America/Chicago").dt.normalize()
    elif "occurred_datetime" in df.columns:
        occurred_day = pd.to_datetime(df["occurred_datetime"], utc=True).dt.tz_convert(
            "America/Chicago"
        ).dt.normalize()
    elif "occurred_date" in df.columns:
        occurred_day = pd.to_datetime(df["occurred_date"])
    else:
        raise ValueError("DataFrame must include an occurred datetime or date column.")

    df = df.assign(occurred_day=occurred_day.dt.date)

    store_meta = (
        df[list(store_columns)]
        .drop_duplicates("store_id")
        .set_index("store_id")
    )

    daily_counts = (
        df.groupby(["store_id", "occurred_day"])
        .size()
        .rename("count")
        .reset_index()
    )

    return daily_counts, store_meta


def _expand_daily_counts(
    daily_counts: pd.DataFrame,
    reference_date: date,
    store_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    min_date = daily_counts["occurred_day"].min()
    if pd.isna(min_date):
        raise ValueError("Daily counts are empty.")

    min_date = pd.to_datetime(min_date).date()
    all_dates = pd.date_range(start=min_date, end=reference_date, freq="D")

    if store_ids is None:
        store_ids = daily_counts["store_id"].unique()
    else:
        store_ids = pd.Index(store_ids).unique()
    full_index = pd.MultiIndex.from_product(
        [store_ids, all_dates.date], names=["store_id", "occurred_day"]
    )

    expanded = (
        daily_counts.set_index(["store_id", "occurred_day"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return expanded


def _period_bounds(
    reference_date: date, period: PeriodDefinition
) -> Tuple[Tuple[date, date, int], Tuple[date, date, int]]:
    if period.type == "ytd":
        current_start = date(reference_date.year, 1, 1)
        current_end = reference_date
        previous_end = reference_date - relativedelta(years=1)
        previous_start = date(previous_end.year, 1, 1)
    else:
        current_end = reference_date
        current_start = reference_date - timedelta(days=period.days - 1)
        previous_end = current_start - timedelta(days=1)
        previous_start = previous_end - timedelta(days=period.days - 1)

    current_length = (current_end - current_start).days + 1
    previous_length = (previous_end - previous_start).days + 1
    return (current_start, current_end, current_length), (
        previous_start,
        previous_end,
        previous_length,
    )


def _aggregate_period_counts(
    daily_counts: pd.DataFrame,
    start: date,
    end: date,
) -> pd.Series:
    mask = (daily_counts["occurred_day"] >= start) & (daily_counts["occurred_day"] <= end)
    filtered = daily_counts.loc[mask]
    if filtered.empty:
        return pd.Series(dtype=float)
    return filtered.groupby("store_id")["count"].sum()


def _compute_baselines(daily_counts: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    grouped = daily_counts.groupby("store_id")["count"]
    mean_daily = grouped.mean()
    std_daily = grouped.std(ddof=0).fillna(0.0)
    return mean_daily, std_daily


def _calc_z_score(
    observed: float,
    mean_daily: float,
    std_daily: float,
    period_days: int,
) -> float | None:
    if std_daily == 0 or math.isnan(std_daily):
        return None
    expected_total = mean_daily * period_days
    period_std = std_daily * math.sqrt(period_days)
    if period_std == 0:
        return None
    return (observed - expected_total) / period_std


def _calc_poisson_p(
    observed: float,
    mean_daily: float,
    period_days: int,
) -> float | None:
    lam = mean_daily * period_days
    if lam <= 0:
        return None
    # Survival function gives P(X >= observed)
    return float(poisson.sf(observed - 1, lam))


def compute_compstat_summary(
    offenses_df: pd.DataFrame,
    reference_date: date | None = None,
    stores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute CompStat metrics for each store.

    Returns a dataframe with one row per store and multi-period metrics.
    """
    if offenses_df.empty:
        raise ValueError("Offense dataframe is empty.")

    ref_date = _ensure_reference_date(offenses_df, reference_date)
    daily_counts, store_meta = _build_daily_matrix(offenses_df)
    observed_store_ids = pd.Index(daily_counts["store_id"].unique())
    if stores_df is not None and "store_id" in stores_df.columns:
        provided_ids = pd.Index(stores_df["store_id"].astype(str).unique())
        store_ids = observed_store_ids.union(provided_ids)
    else:
        store_ids = observed_store_ids

    expanded_daily = _expand_daily_counts(daily_counts, ref_date, store_ids=store_ids)
    mean_daily, std_daily = _compute_baselines(expanded_daily)

    metrics_rows: list[Dict] = []
    for store_id in store_ids:
        row: Dict[str, object] = {"store_id": store_id}
        if store_id in store_meta.index:
            meta = store_meta.loc[store_id]
            row.update(meta.to_dict())

        for period in PERIODS:
            (curr_start, curr_end, curr_len), (prev_start, prev_end, prev_len) = _period_bounds(
                ref_date, period
            )

            current_counts = _aggregate_period_counts(
                expanded_daily, curr_start, curr_end
            )
            previous_counts = _aggregate_period_counts(expanded_daily, prev_start, prev_end)

            prev_year_start = curr_start - relativedelta(years=1)
            prev_year_end = curr_end - relativedelta(years=1)
            prev_year_counts = _aggregate_period_counts(
                expanded_daily, prev_year_start, prev_year_end
            )

            current_value = float(current_counts.get(store_id, 0.0))
            previous_value = float(previous_counts.get(store_id, 0.0))
            prev_year_value = float(prev_year_counts.get(store_id, 0.0))
            change = current_value - previous_value
            pct_change = (
                (change / previous_value) * 100
                if previous_value > 0
                else (100.0 if current_value > 0 else 0.0)
            )
            change_prev_year = current_value - prev_year_value
            pct_change_prev_year = (
                (change_prev_year / prev_year_value) * 100
                if prev_year_value > 0
                else (100.0 if current_value > 0 else 0.0)
            )

            mean_val = float(mean_daily.get(store_id, 0.0))
            std_val = float(std_daily.get(store_id, 0.0))

            period_days = curr_len

            z_score = _calc_z_score(current_value, mean_val, std_val, period_days)
            poisson_p = _calc_poisson_p(current_value, mean_val, period_days)
            poisson_z = 2 * (math.sqrt(current_value) - math.sqrt(previous_value))

            row[f"{period.key}_current"] = current_value
            row[f"{period.key}_previous"] = previous_value
            row[f"{period.key}_previous_year"] = prev_year_value
            row[f"{period.key}_change"] = change
            row[f"{period.key}_pct_change"] = pct_change
            row[f"{period.key}_change_prev_year"] = change_prev_year
            row[f"{period.key}_pct_change_prev_year"] = pct_change_prev_year
            row[f"{period.key}_z_score"] = z_score
            row[f"{period.key}_poisson_p"] = poisson_p
            row[f"{period.key}_poisson_z"] = poisson_z
            row[f"{period.key}_poisson_z_prev_year"] = (
                2 * (math.sqrt(current_value) - math.sqrt(prev_year_value))
                if prev_year_value >= 0
                else None
            )
            row[f"{period.key}_window"] = (
                f"{curr_start.isoformat()} – {curr_end.isoformat()}"
            )
            row[f"{period.key}_window_prev_year"] = (
                f"{prev_year_start.isoformat()} – {prev_year_end.isoformat()}"
            )

        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["reference_date"] = ref_date
    return metrics_df
