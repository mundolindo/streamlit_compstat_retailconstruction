from __future__ import annotations

import math
import textwrap
from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from compstat.data_loader import load_offense_data, load_store_data
from compstat.metrics import PERIODS, compute_compstat_summary
from compstat.timeseries import build_time_series_views


st.set_page_config(
    page_title="Dallas Home Improvement CompStat",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>.main { padding-top: 1.5rem; }</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data() -> Dict[str, pd.DataFrame]:
    stores = load_store_data()
    offenses = load_offense_data()
    views = build_time_series_views(offenses)
    return {"stores": stores, "offenses": offenses, **views}


@st.cache_data(show_spinner=False)
def compute_metrics_frame(
    filtered_offenses: pd.DataFrame,
    stores: pd.DataFrame,
    reference_date: date,
) -> pd.DataFrame:
    metrics = compute_compstat_summary(filtered_offenses, reference_date, stores)
    store_merge_cols = ["store_id"]
    for candidate in ["brand", "city", "latitude", "longitude", "street_address"]:
        if candidate in stores.columns:
            store_merge_cols.append(candidate)
    metrics = metrics.merge(
        stores[store_merge_cols],
        on="store_id",
        how="left",
    )
    if "brand" not in metrics.columns and "store_brand" in metrics.columns:
        metrics["brand"] = metrics["store_brand"]
    if "city" not in metrics.columns and "store_city" in metrics.columns:
        metrics["city"] = metrics["store_city"]
    if "street_address" not in metrics.columns:
        metrics["street_address"] = ""
    return metrics


@st.cache_data(show_spinner=False)
def build_filtered_views_cached(filtered_offenses: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return build_time_series_views(filtered_offenses)


def get_reference_date(offenses: pd.DataFrame) -> date:
    max_date = offenses["occurred_date"].max()
    if isinstance(max_date, pd.Timestamp):
        return max_date.date()
    return max_date


def format_delta(change: float, pct: float) -> str:
    arrow = "▲" if change > 0 else "▼" if change < 0 else "■"
    return f"{arrow} {change:.0f} ({pct:.1f}%)"


COMPARISON_LABELS = {
    "prior_period": "Prior period",
    "prior_year": "Prior year",
}


def normalize_comparison_mode(selection: str) -> str:
    if selection.lower().startswith("prior year"):
        return "prior_year"
    return "prior_period"


def get_comparison_columns(period_key: str, mode: str) -> Dict[str, str]:
    if mode == "prior_year":
        return {
            "baseline": f"{period_key}_previous_year",
            "change": f"{period_key}_change_prev_year",
            "pct_change": f"{period_key}_pct_change_prev_year",
            "poisson_z": f"{period_key}_poisson_z_prev_year",
            "window": f"{period_key}_window",
            "baseline_window": f"{period_key}_window_prev_year",
        }
    return {
        "baseline": f"{period_key}_previous",
        "change": f"{period_key}_change",
        "pct_change": f"{period_key}_pct_change",
        "poisson_z": f"{period_key}_poisson_z",
        "window": f"{period_key}_window",
        "baseline_window": f"{period_key}_window",
    }


def z_to_color(z: float | None) -> List[int]:
    if z is None or math.isnan(z):
        return [128, 128, 128, 190]
    scale = max(min(z, 3), -3) / 3
    if scale >= 0:
        r = int(255 * scale)
        g = int(180 * (1 - scale))
        b = 60
    else:
        b = int(255 * abs(scale))
        g = int(180 * (1 - abs(scale)))
        r = 60
    return [r, g, b, 210]


def rate_signal(z: float | None, p_value: float | None) -> str:
    if z is None:
        return "Stable"
    if p_value is not None and p_value < 0.01:
        return "Critical"
    if p_value is not None and p_value < 0.05:
        return "Elevated"
    if z >= 2:
        return "Elevated"
    if z <= -2:
        return "Improving"
    return "Stable"


def derive_action_items(row: pd.Series, period_key: str, comparison_mode: str) -> List[str]:
    takeaways: List[str] = []
    columns = get_comparison_columns(period_key, comparison_mode)
    z = row.get(f"{period_key}_z_score")
    p_val = row.get(f"{period_key}_poisson_p")
    pct = row.get(columns["pct_change"])
    change = row.get(columns["change"])
    current = row.get(f"{period_key}_current")

    if current is None:
        return takeaways

    if z is not None and z >= 2:
        takeaways.append("Spike exceeds historical norm (z ≥ 2). Prioritize short-term deployment.")
    if p_val is not None and p_val < 0.05:
        takeaways.append("Poisson trigger: observed volume unlikely under baseline. Treat as anomaly.")
    if pct is not None and pct >= 25 and current >= 5:
        takeaways.append("Sustained growth vs prior period. Validate staffing and alarm compliance.")
    if change and change > 0 and (pct or 0) > 10:
        takeaways.append("Engage store leadership on repeat-offender deterrence and target-hardening.")
    if not takeaways:
        takeaways.append("Trend remains within typical bounds. Maintain periodic directed patrol checks.")

    return takeaways


def build_map_layer(
    metrics: pd.DataFrame,
    period_key: str,
    comparison_mode: str,
) -> pdk.Deck:
    layer_data = metrics.dropna(subset=["latitude", "longitude"]).copy()
    comp_cols = get_comparison_columns(period_key, comparison_mode)
    baseline_label = COMPARISON_LABELS.get(comparison_mode, "Prior period")
    keep_cols = list(
        {
            "longitude",
            "latitude",
            "store_id",
            "brand",
            "city",
            "store_name",
            f"{period_key}_current",
            f"{period_key}_z_score",
            comp_cols["baseline"],
            comp_cols["change"],
            comp_cols["pct_change"],
            comp_cols["poisson_z"],
        }
    )
    max_current = layer_data[f"{period_key}_current"].max()
    layer_data = layer_data[keep_cols]
    layer_data["store_display"] = (
        layer_data["store_name"].fillna("").replace({"": np.nan}).combine_first(layer_data["store_id"])
    )
    layer_data["color"] = layer_data[f"{period_key}_z_score"].apply(z_to_color)
    if pd.isna(max_current) or max_current <= 0:
        layer_data["radius"] = 60.0
    else:
        layer_data["radius"] = (
            layer_data[f"{period_key}_current"]
            .clip(lower=0)
            .apply(lambda val: 30.0 + (math.sqrt(val / max_current) * 140.0))
        )

    def _fmt_delta(change: float | None) -> str:
        if change is None or pd.isna(change):
            return "n/a"
        return f"{change:+.0f}"

    def _fmt_poisson_z(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:+.2f}"

    def _fmt_z(value: float | None) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{value:+.2f}"

    layer_data["current_display"] = (
        layer_data[f"{period_key}_current"].fillna(0).map("{:.0f}".format)
    )
    layer_data["baseline_display"] = (
        layer_data[comp_cols["baseline"]].fillna(0).map("{:.0f}".format)
    )
    layer_data["change_display"] = layer_data[comp_cols["change"]].apply(_fmt_delta)
    layer_data["delta_badge"] = layer_data.apply(
        lambda row: format_delta(row[comp_cols["change"]], row[comp_cols["pct_change"]]),
        axis=1,
    )
    layer_data["poisson_z_display"] = layer_data[comp_cols["poisson_z"]].apply(_fmt_poisson_z)
    layer_data["z_score_display"] = layer_data[f"{period_key}_z_score"].apply(_fmt_z)
    layer_data["comparison_label"] = baseline_label

    def _sanitize(record: Dict[str, object]) -> Dict[str, object]:
        cleaned: Dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                value = value.isoformat()
            elif isinstance(value, pd.Series):
                value = value.squeeze().tolist()
            elif isinstance(value, (np.generic,)):
                value = value.item()
            elif isinstance(value, (list, tuple, dict)):
                cleaned[key] = value
                continue
            try:
                if pd.isna(value):
                    cleaned[key] = None
                    continue
            except Exception:
                pass
            cleaned[key] = value
        return cleaned

    def _coerce_float(value: object, fallback: float = 0.0) -> float:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return fallback
            return float(value)
        except Exception:
            return fallback

    records = []
    for rec in layer_data.to_dict(orient="records"):
        enriched = {**rec}
        enriched["radius"] = _coerce_float(rec.get("radius"), 60.0)
        enriched["poisson_z_value"] = _coerce_float(rec.get(comp_cols["poisson_z"]), 0.0)
        records.append(_sanitize(enriched))

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=records,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius="radius",
        radius_scale=1,
        radius_min_pixels=24,
        radius_max_pixels=140,
        pickable=True,
        elevation_scale=2,
        stroked=True,
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=1,
    )

    tooltip = {
        "html": "<b>{store_display}</b><br/>"
        "Brand: {brand}<br/>"
        "City: {city}<br/>"
        "Current incidents: {current_display}<br/>"
        "Baseline ({comparison_label}): {baseline_display}<br/>"
        "Δ vs baseline: {delta_badge}<br/>"
        "Poisson Z: {poisson_z_display}<br/>"
        "Baseline z-score: {z_score_display}",
        "style": {"backgroundColor": "rgba(15,17,22,0.85)", "color": "white"},
    }

    view_state = pdk.ViewState(
        latitude=32.80,
        longitude=-96.80,
        zoom=9.6,
        min_zoom=5,
        max_zoom=16,
        pitch=30,
    )

    return pdk.Deck(
        layers=[scatter],
        initial_view_state=view_state,
        tooltip=tooltip,
        height=360,
    )


def render_kpis(metrics: pd.DataFrame, comparison_mode: str) -> None:
    comparison_label = COMPARISON_LABELS.get(comparison_mode, "Prior period")
    cols = st.columns(len(PERIODS))
    for col, period in zip(cols, PERIODS):
        summary = aggregate_period(metrics, period.key, comparison_mode)
        with col:
            st.metric(
                f"{period.label} Incidents",
                f"{summary['current']:.0f}",
                format_delta(summary["change"], summary["pct_change"]),
            )
            st.caption(
                f"{comparison_label}: {summary['baseline']:.0f}"
            )
            st.caption(
                f"Daily avg {summary['daily_mean']:.2f} (Δ {summary['daily_change']:+.2f})"
            )
            st.markdown(
                f"**Risk:** {summary['risk_level']} (z̄ {summary['z_score']:.2f})"
            )


def aggregate_period(
    metrics: pd.DataFrame,
    period_key: str,
    comparison_mode: str,
) -> Dict[str, float]:
    if metrics.empty:
        return {
            "current": 0.0,
            "baseline": 0.0,
            "change": 0.0,
            "pct_change": 0.0,
            "daily_mean": 0.0,
            "daily_change": 0.0,
            "z_score": 0.0,
            "risk_level": "Stable",
            "baseline_label": COMPARISON_LABELS.get(comparison_mode, "Prior period"),
        }

    comp_cols = get_comparison_columns(period_key, comparison_mode)
    current = metrics[f"{period_key}_current"].sum()
    baseline_total = metrics[comp_cols["baseline"]].sum()
    change = current - baseline_total
    pct_change = (
        (change / baseline_total * 100)
        if baseline_total > 0
        else (100 if current > 0 else 0)
    )

    window_str = metrics[f"{period_key}_window"].iloc[0]
    start_str, end_str = window_str.split(" – ")
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    period_days = max((end - start).days + 1, 1)

    daily_mean = current / period_days
    daily_change = change / period_days

    z_mean = metrics[f"{period_key}_z_score"].replace({np.nan: 0}).mean()
    p_min = metrics[f"{period_key}_poisson_p"].replace({np.nan: 1}).min()

    if p_min < 0.01 or z_mean >= 2:
        risk = "Critical"
    elif p_min < 0.05 or z_mean >= 1:
        risk = "Elevated"
    else:
        risk = "Stable"

    return {
        "current": current,
        "baseline": baseline_total,
        "change": change,
        "pct_change": pct_change,
        "daily_mean": daily_mean,
        "daily_change": daily_change,
        "z_score": z_mean,
        "risk_level": risk,
        "baseline_label": COMPARISON_LABELS.get(comparison_mode, "Prior period"),
    }


def build_metrics_table(
    metrics: pd.DataFrame,
    stores: pd.DataFrame,
    period_key: str,
    comparison_mode: str,
) -> pd.DataFrame:
    comp_cols = get_comparison_columns(period_key, comparison_mode)
    display_cols = [
        "store_id",
        "brand",
        "city",
        f"{period_key}_current",
        comp_cols["baseline"],
        comp_cols["change"],
        comp_cols["pct_change"],
        comp_cols["poisson_z"],
    ]
    optional_cols = []
    for candidate in ["street_address"]:
        if candidate in metrics.columns:
            optional_cols.append(candidate)

    available_cols = [col for col in display_cols if col in metrics.columns]
    table = metrics[available_cols + optional_cols].copy()

    rename_map = {
        "brand": "Brand",
        "city": "City",
        f"{period_key}_current": "Current",
        comp_cols["baseline"]: COMPARISON_LABELS.get(comparison_mode, "Baseline"),
        comp_cols["change"]: "Δ",
        comp_cols["pct_change"]: "%Δ",
        comp_cols["poisson_z"]: "Poisson Z",
        "street_address": "Street Address",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in table.columns}
    table = table.rename(columns=rename_map)
    return table


def plot_weekly_trend(weekly: pd.DataFrame, store_id: str) -> go.Figure:
    data = weekly[weekly["store_id"] == store_id].copy()
    if data.empty:
        return go.Figure()
    fig = px.line(
        data,
        x="occurred_week",
        y="count",
        title="Weekly Incident Volume",
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        xaxis_title="Week starting",
        yaxis_title="Incidents",
        margin=dict(l=10, r=10, t=60, b=20),
        height=320,
    )
    return fig


def plot_category_distribution(categories: pd.DataFrame, store_id: str) -> go.Figure:
    data = categories[categories["store_id"] == store_id].nlargest(8, "count")
    fig = px.bar(
        data,
        x="count",
        y="nibrs_crime_category",
        orientation="h",
        title="Top NIBRS Categories",
        text_auto=True,
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=20),
        yaxis_title="Category",
        xaxis_title="Incidents",
    )
    return fig


def plot_heatmap(hourly: pd.DataFrame, store_id: str) -> go.Figure:
    scope = hourly[hourly["store_id"] == store_id]
    if scope.empty:
        return go.Figure()
    pivot = scope.pivot_table(
        index="weekday_num", columns="hour", values="count", fill_value=0
    )
    pivot = pivot.reindex(index=range(7), fill_value=0)
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = weekday_labels

    heat = go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=weekday_labels,
        colorscale="YlOrRd",
        hovertemplate="Hour %{x}:00<br>Weekday %{y}<br>Count %{z}<extra></extra>",
    )
    layout = go.Layout(
        title="Day & Hour Concentration",
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(title="", autorange="reversed"),
        height=320,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    return go.Figure(data=[heat], layout=layout)


def main():
    data = get_data()
    stores = data["stores"]
    offenses = data["offenses"]
    reference_date = get_reference_date(offenses)

    period_labels = {p.label: p.key for p in PERIODS}
    brand_options = sorted(stores["brand"].unique())
    offense_categories = sorted(
        offenses["nibrs_crime_category"].unique()
    )

    comparison_options = [
        COMPARISON_LABELS["prior_period"],
        COMPARISON_LABELS["prior_year"],
    ]

    with st.sidebar:
        st.header("Control Panel")
        comparison_selection = st.radio(
            "Comparison baseline",
            comparison_options,
            index=0,
            help="Switch the executive summary between prior-period and prior-year baselines.",
        )
        comparison_mode = normalize_comparison_mode(comparison_selection)

        selected_period_label = st.radio(
            "Focus period (table & deep dive)",
            list(period_labels.keys()),
            index=0,
            help="Drives the store table ordering and deep-dive metrics.",
        )
        focus_period_key = period_labels[selected_period_label]

        selected_brands = st.multiselect(
            "Retailers",
            options=brand_options,
            default=brand_options,
        )

        category_filter = st.multiselect(
            "NIBRS categories",
            options=offense_categories,
            help="Filter to specific offense categories of interest.",
        )

        st.caption(
            "Data: Dallas Police Department NIBRS offenses (Open Data) "
            "and OpenStreetMap store locations. Pulled through Oct "
            f"{reference_date.strftime('%d, %Y')}."
        )

    filtered_offenses = offenses[
        offenses["store_brand"].isin(selected_brands)
    ].copy()

    if category_filter:
        filtered_offenses = filtered_offenses[
            filtered_offenses["nibrs_crime_category"].isin(category_filter)
        ]

    if filtered_offenses.empty:
        st.warning("No offenses match the selected filters. Adjust the controls to view data.")
        return

    metrics = compute_metrics_frame(filtered_offenses, stores, reference_date).copy()
    filtered_views = build_filtered_views_cached(filtered_offenses)
    metrics = metrics.sort_values(f"{focus_period_key}_z_score", ascending=False)

    overview_tab, viz_tab, validation_tab, about_tab = st.tabs(
        ["Executive Overview", "Store Visualizations", "Data Validation", "About the Data"]
    )

    with overview_tab:
        st.title("Dallas Home Improvement CompStat")
        st.subheader(
            "Operational intelligence for Home Depot and Lowe's locations "
            "across Dallas–Fort Worth."
        )

        st.markdown("### Multi-Window Snapshot")
        render_kpis(metrics, comparison_mode)

        map_col, list_col = st.columns([3, 1])
        with map_col:
            st.markdown("#### Hotspot Map")
            map_period_label = st.radio(
                "Map period",
                list(period_labels.keys()),
                index=list(period_labels.keys()).index(selected_period_label),
                horizontal=True,
                key="map_period_selector",
            )
            map_period_key = period_labels[map_period_label]
            map_deck = build_map_layer(metrics, map_period_key, comparison_mode)
            st.pydeck_chart(map_deck, use_container_width=True)
            legend_html = """
            <div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;">
              <div style="display:flex;align-items:center;gap:6px;">
                <span style="width:14px;height:14px;background:rgba(220,72,65,0.85);display:inline-block;border-radius:3px;"></span>
                <span style="font-size:0.85rem;">Elevated spike (z ≥ 2)</span>
              </div>
              <div style="display:flex;align-items:center;gap:6px;">
                <span style="width:14px;height:14px;background:rgba(128,128,128,0.85);display:inline-block;border-radius:3px;"></span>
                <span style="font-size:0.85rem;">Stable trend</span>
              </div>
              <div style="display:flex;align-items:center;gap:6px;">
                <span style="width:14px;height:14px;background:rgba(60,120,255,0.85);display:inline-block;border-radius:3px;"></span>
                <span style="font-size:0.85rem;">Improving (z ≤ -2)</span>
              </div>
            </div>
            <div style="margin-top:6px;font-size:0.8rem;color:#7f8a9c;">
              Circle size scales with current-period incident volume; outline highlights clickable locations.
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
            st.caption(
                f"Tip: hover a store to see current incidents, change vs {COMPARISON_LABELS[comparison_mode].lower()}, the Wheeler Poisson Z-score, and the baseline anomaly z-score."
            )
        with list_col:
            st.write("")

        st.markdown("### Store CompStat Table")
        table_tabs = st.tabs([p.label for p in PERIODS])
        baseline_header = COMPARISON_LABELS.get(comparison_mode, "Baseline")
        for tab, period in zip(table_tabs, PERIODS):
            with tab:
                table = build_metrics_table(metrics, stores, period.key, comparison_mode)
                st.dataframe(
                    table,
                    column_config={
                        "Current": st.column_config.NumberColumn(format="%.0f"),
                        baseline_header: st.column_config.NumberColumn(format="%.0f"),
                        "Δ": st.column_config.NumberColumn(format="%.0f"),
                        "%Δ": st.column_config.NumberColumn(format="%.1f%%"),
                        "Poisson Z": st.column_config.NumberColumn(format="%.2f"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

        store_labels = (
            metrics["store_id"].astype(str)
            + " • "
            + metrics["brand"].astype(str)
            + " ("
            + metrics["city"].astype(str)
            + ")"
        )
        store_lookup = dict(zip(store_labels, metrics["store_id"]))

        st.markdown("### Deep Dive")
        selected_store_label = st.selectbox(
            "Select a store for tactical review",
            options=list(store_lookup.keys()),
        )
        selected_store_id = store_lookup[selected_store_label]
        selected_row = metrics[metrics["store_id"] == selected_store_id].iloc[0]

        focus_cols = get_comparison_columns(focus_period_key, comparison_mode)
        cols = st.columns(3)
        cols[0].metric(
            "Current incidents",
            f"{selected_row[f'{focus_period_key}_current']:.0f}",
            format_delta(
                selected_row[focus_cols["change"]],
                selected_row[focus_cols["pct_change"]],
            ),
        )
        z_val = selected_row.get(f"{focus_period_key}_z_score") or 0
        cols[1].metric("Z-Score", f"{z_val:.2f}")
        poisson_z_val = selected_row.get(focus_cols["poisson_z"]) or 0
        cols[2].metric(
            f"{COMPARISON_LABELS[comparison_mode]} Poisson Z",
            f"{poisson_z_val:+.2f}",
        )
        p_val = selected_row.get(f"{focus_period_key}_poisson_p") or 1
        cols[2].caption(f"Poisson p-value: {p_val:.3f}")

        charts_col1, charts_col2 = st.columns(2)
        with charts_col1:
            st.plotly_chart(
                plot_weekly_trend(filtered_views["weekly_counts"], selected_store_id),
                use_container_width=True,
                key=f"deep-dive-weekly-{selected_store_id}",
            )
            st.plotly_chart(
                plot_category_distribution(
                    filtered_views["offense_categories"], selected_store_id
                ),
                use_container_width=True,
                key=f"deep-dive-category-{selected_store_id}",
            )
        with charts_col2:
            st.plotly_chart(
                plot_heatmap(filtered_views["hourly_pattern"], selected_store_id),
                use_container_width=True,
                key=f"deep-dive-heatmap-{selected_store_id}",
            )

        st.markdown("#### Tactical Notes")
        for note in derive_action_items(selected_row, focus_period_key, comparison_mode):
            st.write(f"- {note}")

        st.markdown("#### Strategic Considerations")
        st.write(
            "- Contrast store performance with same-brand peers to identify "
            "enterprise-level vulnerabilities."
        )
        st.write(
            "- Leverage directed patrols, safety audits, and CCTV intelligence "
            "within active anomalies to validate mitigation."
        )
        st.write(
            "- Coordinate cross-functional reviews (loss prevention, property "
            "management, DPD) for sustained trend lines."
        )

        with st.expander("Data Quality & Methodology"):
            st.markdown(
                "- **Source refresh:** `scripts/fetch_data.py` pulls OpenStreetMap store "
                "geometry and Dallas NIBRS incidents (rolling 2024–present).\n"
                "- **CompStat windows:** 7-day and 28-day compare against the preceding "
                "matching duration; YTD compares to the same span from the prior calendar year.\n"
                "- **Anomaly scoring:** z-scores derived from store-specific daily baselines; "
                "Poisson survival probabilities < 0.05 highlight statistically rare spikes."
            )

    with viz_tab:
        st.header("Store-Level Visual Analytics")
        st.caption(
            "Compare each Dallas-area Home Depot and Lowe's location using synchronized trend, "
            "category, and day/hour visuals."
        )

        store_display = (
            metrics["store_id"].astype(str)
            + " – "
            + metrics["store_name"].fillna(metrics["brand"])
            + " ("
            + metrics["city"].astype(str)
            + ")"
        )
        viz_options = dict(zip(store_display, metrics["store_id"]))

        default_selection = list(viz_options.keys())[:3]
        selected_viz_labels = st.multiselect(
            "Stores to visualize",
            options=list(viz_options.keys()),
            default=default_selection,
        )

        if not selected_viz_labels:
            st.info("Select one or more stores to load visual comparisons.")
        else:
            for label in selected_viz_labels:
                store_id = viz_options[label]
                store_row = metrics[metrics["store_id"] == store_id].iloc[0]
                st.markdown(f"### {label}")
                kpi_cols = st.columns(4)
                kpi_cols[0].metric(
                    "7-Day",
                    f"{store_row['seven_day_current']:.0f}",
                    format_delta(
                        store_row["seven_day_change"],
                        store_row["seven_day_pct_change"],
                    ),
                )
                kpi_cols[1].metric(
                    "28-Day",
                    f"{store_row['four_week_current']:.0f}",
                    format_delta(
                        store_row["four_week_change"],
                        store_row["four_week_pct_change"],
                    ),
                )
                kpi_cols[2].metric(
                    "YTD",
                    f"{store_row['ytd_current']:.0f}",
                    format_delta(
                        store_row["ytd_change"],
                        store_row["ytd_pct_change"],
                    ),
                )
                z_score = store_row["seven_day_z_score"]
                kpi_cols[3].metric(
                    "7-Day z-score",
                    f"{(z_score or 0):.2f}",
                )

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.plotly_chart(
                        plot_weekly_trend(filtered_views["weekly_counts"], store_id),
                        use_container_width=True,
                        key=f"compare-weekly-{store_id}",
                    )
                with chart_col2:
                    st.plotly_chart(
                        plot_category_distribution(
                            filtered_views["offense_categories"], store_id
                        ),
                        use_container_width=True,
                        key=f"compare-category-{store_id}",
                    )

                st.plotly_chart(
                    plot_heatmap(filtered_views["hourly_pattern"], store_id),
                    use_container_width=True,
                    key=f"compare-heatmap-{store_id}",
                )
                st.divider()

    with validation_tab:
        st.header("Data Validation & Spot Checks")
        total_filtered_incidents = int(filtered_offenses.shape[0])
        total_unique_stores = filtered_offenses["store_id"].nunique()
        min_date = pd.to_datetime(filtered_offenses["occurred_date"]).min()
        max_date = pd.to_datetime(filtered_offenses["occurred_date"]).max()
        validation_metrics = st.columns(4)
        validation_metrics[0].metric("Filtered Incidents", f"{total_filtered_incidents:,}")
        validation_metrics[1].metric("Stores in Scope", f"{total_unique_stores}")
        validation_metrics[2].metric(
            "First Incident",
            min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "n/a",
        )
        validation_metrics[3].metric(
            "Last Incident",
            max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "n/a",
        )

        run_validation = st.checkbox(
            "Run detailed validation summaries",
            value=False,
            help="Enable to materialize baseline comparisons, null checks, and store spot checks.",
        )

        if not run_validation:
            st.info("Enable the checkbox above to compute validation summaries.")
        else:
            st.markdown("### Period Totals vs Baselines")
            summary_rows: List[Dict[str, object]] = []
            for period in PERIODS:
                aggregate = aggregate_period(metrics, period.key, comparison_mode)
                summary_rows.append(
                    {
                        "Period": period.label,
                        "Current": aggregate["current"],
                        COMPARISON_LABELS.get(comparison_mode, "Baseline"): aggregate["baseline"],
                        "Δ": aggregate["change"],
                        "%Δ": aggregate["pct_change"],
                        "Avg Daily": aggregate["daily_mean"],
                        "Risk": aggregate["risk_level"],
                    }
                )
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(
                summary_df,
                column_config={
                    "Current": st.column_config.NumberColumn(format="%.0f"),
                    COMPARISON_LABELS.get(comparison_mode, "Baseline"): st.column_config.NumberColumn(
                        format="%.0f"
                    ),
                    "Δ": st.column_config.NumberColumn(format="%.0f"),
                    "%Δ": st.column_config.NumberColumn(format="%.1f%%"),
                    "Avg Daily": st.column_config.NumberColumn(format="%.2f"),
                },
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Null & Geometry Checks")
            missing_geo = metrics[metrics[["latitude", "longitude"]].isna().any(axis=1)][
                ["store_id", "brand", "city"]
            ]
            if missing_geo.empty:
                st.success("All mapped stores have latitude and longitude values.")
            else:
                st.warning("Some stores are missing geometry; they will be excluded from the map.")
                st.dataframe(missing_geo, hide_index=True)

            critical_columns = [
                "store_id",
                "store_brand",
                "occurred_date",
                "nibrs_crime_category",
            ]
            null_report = (
                filtered_offenses[critical_columns]
                .isna()
                .mean()
                .rename("Null Rate")
                .mul(100)
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            st.dataframe(
                null_report,
                column_config={"Null Rate": st.column_config.NumberColumn(format="%.2f%%")},
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("### Store Spot Checks")
            focus_cols = get_comparison_columns(focus_period_key, comparison_mode)
            spot_columns = [
                "store_id",
                "brand",
                "city",
                f"{focus_period_key}_current",
                focus_cols["baseline"],
                focus_cols["change"],
                focus_cols["pct_change"],
                f"{focus_period_key}_z_score",
            ]
            spot_frame = metrics[spot_columns].head(10)
            st.dataframe(
                spot_frame,
                column_config={
                    f"{focus_period_key}_current": st.column_config.NumberColumn(format="%.0f"),
                    focus_cols["baseline"]: st.column_config.NumberColumn(format="%.0f"),
                    focus_cols["change"]: st.column_config.NumberColumn(format="%.0f"),
                    focus_cols["pct_change"]: st.column_config.NumberColumn(format="%.1f%%"),
                    f"{focus_period_key}_z_score": st.column_config.NumberColumn(format="%.2f"),
                },
                hide_index=True,
                use_container_width=True,
            )
        st.caption(
            "Spot checks list the first ten stores ordered by anomaly score for the focused period."
        )

    with about_tab:
        st.header("About the Data")
        st.markdown(
            """
            ### Sources & Attribution

            - **Store locations:** Queried from [OpenStreetMap Nominatim](https://nominatim.openstreetmap.org/) for Home Depot and Lowe's sites across the Dallas–Fort Worth metro. © OpenStreetMap contributors (ODbL 1.0).
            - **Crime data:** Dallas Police Department NIBRS incidents via the [Dallas Open Data Portal](https://www.dallasopendata.com/resource/qv6i-rri7.json). Public domain; see portal usage terms.

            ### Data Pipeline

            - `scripts/fetch_data.py` geocodes stores, downloads incidents, localizes timestamps, and writes:
              - `data/home_improvement_stores.csv`
              - `data/store_offenses.parquet`
            - `compstat/data_loader.py` reads the curated files, applies quality checks, and derives daily/week/month fields.
            - `compstat/metrics.py` builds rolling (7-day, 28-day) and year-to-date comparisons, including z-scores and Wheeler Poisson Z for anomaly signals.
            - `compstat/timeseries.py` produces daily, weekly, category, and hourly views for downstream charts.

            ### Refresh Cadence

            - Fetch script intended for on-demand updates; respect Nominatim rate limits (≥1s between calls) and Socrata API quotas.
            - After running the script, redeploy or restart the Streamlit session to load the refreshed files.

            ### Caveats & Interpretation

            - NIBRS offense data is as-published by DPD; geocoding accuracy depends on the portal’s `geocoded_column` field.
            - Anomaly scores are store-specific: z-scores compare current counts to historical store baselines; Poisson probabilities highlight statistically rare spikes.
            - Filter controls propagate to all visuals; use the Data Validation tab to inspect totals, null rates, and spot checks for the active slice.
            """
        )


if __name__ == "__main__":
    main()
