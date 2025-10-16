from __future__ import annotations

import math
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


def get_reference_date(offenses: pd.DataFrame) -> date:
    max_date = offenses["occurred_date"].max()
    if isinstance(max_date, pd.Timestamp):
        return max_date.date()
    return max_date


def format_delta(change: float, pct: float) -> str:
    arrow = "▲" if change > 0 else "▼" if change < 0 else "■"
    return f"{arrow} {change:.0f} ({pct:.1f}%)"


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


def derive_action_items(row: pd.Series, period_key: str) -> List[str]:
    takeaways: List[str] = []
    z = row.get(f"{period_key}_z_score")
    p_val = row.get(f"{period_key}_poisson_p")
    pct = row.get(f"{period_key}_pct_change")
    change = row.get(f"{period_key}_change")
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


def build_map_layer(metrics: pd.DataFrame, period_key: str) -> pdk.Deck:
    layer_data = metrics.dropna(subset=["latitude", "longitude"]).copy()
    keep_cols = [
        "longitude",
        "latitude",
        "brand",
        "city",
        "store_name",
        f"{period_key}_current",
        f"{period_key}_change",
        f"{period_key}_z_score",
        f"{period_key}_poisson_z",
    ]
    max_current = layer_data[f"{period_key}_current"].max()
    layer_data = layer_data[keep_cols]
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

    layer_data["current_display"] = layer_data[f"{period_key}_current"].fillna(0).map("{:.0f}".format)
    layer_data["change_display"] = layer_data[f"{period_key}_change"].apply(_fmt_delta)
    layer_data["poisson_z_display"] = layer_data[f"{period_key}_poisson_z"].apply(_fmt_poisson_z)
    layer_data["z_score_display"] = layer_data[f"{period_key}_z_score"].apply(_fmt_z)

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
        enriched[f"{period_key}_poisson_z"] = _coerce_float(
            rec.get(f"{period_key}_poisson_z"), 0.0
        )
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
        "html": "<b>{store_name}</b><br/>"
        "Brand: {brand}<br/>"
        "City: {city}<br/>"
        "Current incidents: {current_display}<br/>"
        "Δ vs prior: {change_display}<br/>"
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


def render_kpis(aggregated: Dict[str, float], period: PeriodDefinition) -> None:
    cols = st.columns(3)
    cols[0].metric(
        f"{period.label} Incidents",
        f"{aggregated['current']:.0f}",
        format_delta(aggregated["change"], aggregated["pct_change"]),
    )
    cols[1].metric(
        "Average Daily Volume",
        f"{aggregated['daily_mean']:.2f}",
        f"{aggregated['daily_change']:+.2f}",
    )
    risk_label = aggregated["risk_level"]
    cols[2].metric(
        "Risk Signal",
        risk_label,
        f"Anomaly score: {aggregated['z_score']:.2f}",
    )


def aggregate_period(metrics: pd.DataFrame, period_key: str) -> Dict[str, float]:
    if metrics.empty:
        return {
            "current": 0.0,
            "previous": 0.0,
            "change": 0.0,
            "pct_change": 0.0,
            "daily_mean": 0.0,
            "daily_change": 0.0,
            "z_score": 0.0,
            "risk_level": "Stable",
        }

    current = metrics[f"{period_key}_current"].sum()
    previous = metrics[f"{period_key}_previous"].sum()
    change = current - previous
    pct_change = (change / previous * 100) if previous > 0 else (100 if current > 0 else 0)

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
        "previous": previous,
        "change": change,
        "pct_change": pct_change,
        "daily_mean": daily_mean,
        "daily_change": daily_change,
        "z_score": z_mean,
        "risk_level": risk,
    }


def build_metrics_table(
    metrics: pd.DataFrame,
    stores: pd.DataFrame,
    period_key: str,
) -> pd.DataFrame:
    display_cols = [
        "store_id",
        "brand",
        "city",
        f"{period_key}_current",
        f"{period_key}_previous",
        f"{period_key}_change",
        f"{period_key}_pct_change",
        f"{period_key}_poisson_z",
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
        f"{period_key}_previous": "Previous",
        f"{period_key}_change": "Δ",
        f"{period_key}_pct_change": "%Δ",
        f"{period_key}_poisson_z": "Poisson Z",
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

    with st.sidebar:
        st.header("Control Panel")
        selected_period_label = st.radio(
            "Focus period",
            list(period_labels.keys()),
            index=0,
            help="Determines which CompStat window drives the KPIs and map colors.",
        )
        period_key = period_labels[selected_period_label]

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

    filtered_views = build_time_series_views(filtered_offenses)
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
    metrics = metrics.sort_values(f"{period_key}_z_score", ascending=False)

    overview_tab, viz_tab = st.tabs(
        ["Executive Overview", "Store Visualizations"]
    )

    with overview_tab:
        st.title("Dallas Home Improvement CompStat")
        st.subheader(
            "Operational intelligence for Home Depot and Lowe's locations "
            "across Dallas–Fort Worth."
        )

        aggregate = aggregate_period(metrics, period_key)
        render_kpis(aggregate, next(p for p in PERIODS if p.key == period_key))

        map_deck = build_map_layer(metrics, period_key)
        map_col, list_col = st.columns([3, 1])
        with map_col:
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
                "Tip: hover a store to see current incidents, change vs prior window, the Wheeler Poisson Z-score, and the baseline anomaly z-score."
            )
        with list_col:
            st.markdown("#### Priority Stores")
            priority = (
                metrics.dropna(subset=[f"{period_key}_z_score"])
                .sort_values(f"{period_key}_z_score", ascending=False)
                .head(5)
            )
            if priority.empty:
                st.caption("No positive anomalies for the selected filters.")
            else:
                for _, row in priority.iterrows():
                    label = row["store_id"]
                    st.metric(
                        label,
                        f"{row[f'{period_key}_current']:.0f}",
                        format_delta(
                            row[f"{period_key}_change"],
                            row[f"{period_key}_pct_change"],
                        ),
                    )

        st.markdown("### Store CompStat Table")
        table = build_metrics_table(metrics, stores, period_key)
        st.dataframe(
            table,
            column_config={
                "Current": st.column_config.NumberColumn(format="%.0f"),
                "Previous": st.column_config.NumberColumn(format="%.0f"),
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

        cols = st.columns(3)
        cols[0].metric(
            "Current incidents",
            f"{selected_row[f'{period_key}_current']:.0f}",
            format_delta(
                selected_row[f"{period_key}_change"],
                selected_row[f"{period_key}_pct_change"],
            ),
        )
        z_val = selected_row.get(f"{period_key}_z_score") or 0
        cols[1].metric("Z-Score", f"{z_val:.2f}")
        p_val = selected_row.get(f"{period_key}_poisson_p") or 1
        cols[2].metric("Poisson Trigger", f"{p_val:.3f}")

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
        for note in derive_action_items(selected_row, period_key):
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


if __name__ == "__main__":
    main()
