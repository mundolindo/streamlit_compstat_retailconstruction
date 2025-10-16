import sys
import types
from typing import Dict, List

import numpy as np
import pytest
import pandas as pd


class _StreamlitStub:
    """Minimal stub to satisfy streamlit imports during tests."""

    def __init__(self) -> None:
        self._decorator = lambda **_: (lambda fn: fn)

    def cache_data(self, **kwargs):
        return self._decorator

    def set_page_config(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


# Ensure streamlit is stubbed before importing the app
sys.modules.setdefault("streamlit", _StreamlitStub())

from streamlit_app import (  # noqa: E402  (import after stubbing)
    COMPARISON_LABELS,
    aggregate_period,
    get_comparison_columns,
)


def _base_metrics(period_key: str = "seven_day") -> List[Dict[str, object]]:
    """Return minimal metrics rows with both prior-period and prior-year fields populated."""
    rows: List[Dict[str, object]] = []
    data = [
        ("A", 7.0, 3.0, 5.0),
        ("B", 0.0, 2.0, 2.0),
    ]
    for store_id, current, previous, previous_year in data:
        change = current - previous
        pct_change = (change / previous * 100) if previous else (100.0 if current else 0.0)
        change_py = current - previous_year
        pct_change_py = (
            (change_py / previous_year * 100) if previous_year else (100.0 if current else 0.0)
        )
        rows.append(
            {
                "store_id": store_id,
                f"{period_key}_current": current,
                f"{period_key}_previous": previous,
                f"{period_key}_previous_year": previous_year,
                f"{period_key}_change": change,
                f"{period_key}_pct_change": pct_change,
                f"{period_key}_change_prev_year": change_py,
                f"{period_key}_pct_change_prev_year": pct_change_py,
                f"{period_key}_z_score": 0.0,
                f"{period_key}_poisson_p": 1.0,
                f"{period_key}_poisson_z": 2 * (np.sqrt(max(current, 0)) - np.sqrt(max(previous, 0))),
                f"{period_key}_poisson_z_prev_year": 2
                * (np.sqrt(max(current, 0)) - np.sqrt(max(previous_year, 0))),
                f"{period_key}_window": "2024-03-01 – 2024-03-07",
                f"{period_key}_window_prev_year": "2023-03-01 – 2023-03-07",
            }
        )
    return rows


def _metrics_df() -> pd.DataFrame:
    base = _base_metrics("seven_day")
    df = pd.DataFrame(base)
    # Populate other period keys with the same values to ease aggregation in tests
    for key in ("four_week", "ytd"):
        alt_rows = _base_metrics(key)
        alt_df = pd.DataFrame(alt_rows)
        df = df.join(
            alt_df.drop(columns=["store_id"]),
            how="left",
        )
    return df


def test_get_comparison_columns_prior_period():
    cols = get_comparison_columns("seven_day", "prior_period")
    assert cols["baseline"] == "seven_day_previous"
    assert cols["change"] == "seven_day_change"
    assert cols["pct_change"] == "seven_day_pct_change"
    assert cols["poisson_z"] == "seven_day_poisson_z"


def test_get_comparison_columns_prior_year():
    cols = get_comparison_columns("seven_day", "prior_year")
    assert cols["baseline"] == "seven_day_previous_year"
    assert cols["change"] == "seven_day_change_prev_year"
    assert cols["pct_change"] == "seven_day_pct_change_prev_year"
    assert cols["poisson_z"] == "seven_day_poisson_z_prev_year"


def test_aggregate_period_prior_period():
    df = _metrics_df()
    summary = aggregate_period(df, "seven_day", "prior_period")
    assert summary["current"] == pytest.approx(7.0)
    assert summary["baseline"] == pytest.approx(5.0)
    assert summary["change"] == pytest.approx(2.0)
    assert summary["pct_change"] == pytest.approx(40.0)
    assert summary["risk_level"] == "Stable"


def test_aggregate_period_prior_year():
    df = _metrics_df()
    summary = aggregate_period(df, "seven_day", "prior_year")
    assert summary["current"] == pytest.approx(7.0)
    assert summary["baseline"] == pytest.approx(7.0)
    assert summary["change"] == pytest.approx(0.0)
    assert summary["pct_change"] == pytest.approx(0.0)


def test_validation_spot_checks():
    filtered_offenses = pd.DataFrame(
        [
            {
                "store_id": "A",
                "store_brand": "Home Depot",
                "occurred_date": "2024-03-01",
                "nibrs_crime_category": "Theft",
            },
            {
                "store_id": "B",
                "store_brand": "Home Depot",
                "occurred_date": "2024-03-02",
                "nibrs_crime_category": "Assault",
            },
        ]
    )
    metrics = _metrics_df()
    metrics["brand"] = ["Home Depot", "Home Depot"]
    metrics["city"] = ["Dallas", "Dallas"]
    metrics["latitude"] = [32.8, np.nan]
    metrics["longitude"] = [-96.8, np.nan]

    comparison_mode = "prior_period"
    summary_rows = []
    for period in ("seven_day", "four_week", "ytd"):
        aggregate = aggregate_period(metrics, period, comparison_mode)
        summary_rows.append(
            {
                "Period": period,
                "Current": aggregate["current"],
                COMPARISON_LABELS.get(comparison_mode, "Baseline"): aggregate["baseline"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    assert not summary_df.empty

    missing_geo = metrics[metrics[["latitude", "longitude"]].isna().any(axis=1)]
    assert not missing_geo.empty

    critical_columns = ["store_id", "store_brand", "occurred_date", "nibrs_crime_category"]
    null_report = (
        filtered_offenses[critical_columns]
        .isna()
        .mean()
        .rename("Null Rate")
        .mul(100)
        .reset_index()
        .rename(columns={"index": "Column"})
    )
    assert set(null_report.columns) == {"Column", "Null Rate"}

    focus_cols = get_comparison_columns("seven_day", comparison_mode)
    spot_columns = [
        "store_id",
        "brand",
        "city",
        "seven_day_current",
        focus_cols["baseline"],
        focus_cols["change"],
        focus_cols["pct_change"],
        "seven_day_z_score",
    ]
    spot_frame = metrics.assign(
        **{
            focus_cols["change"]: metrics["seven_day_change"],
            focus_cols["pct_change"]: metrics["seven_day_pct_change"],
        }
    )[spot_columns].head(10)
    assert not spot_frame.empty
