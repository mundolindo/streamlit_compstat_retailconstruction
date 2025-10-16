import math
import time
from datetime import datetime
from os import getenv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
SOC_DATASET_URL = "https://www.dallasopendata.com/resource/qv6i-rri7.json"

NOMINATIM_HEADERS = {
    "User-Agent": "streamlit-compstat-app/0.1 (+https://streamlit.io/)",
}

SOC_HEADERS = {
    "User-Agent": "streamlit-compstat-app/0.1 (+https://streamlit.io/)",
}
SOC_APP_TOKEN = getenv("DALLAS_OPEN_DATA_APP_TOKEN") or getenv("SOCRATA_APP_TOKEN")
if SOC_APP_TOKEN:
    SOC_HEADERS["X-App-Token"] = SOC_APP_TOKEN

ALLOWED_COUNTIES = {
    "Dallas County",
    "Collin County",
    "Tarrant County",
    "Denton County",
    "Rockwall County",
    "Ellis County",
    "Kaufman County",
}

STORE_QUERIES = [
    {"brand": "Home Depot", "query": "Home Depot, Dallas, Texas", "prefix": "HD"},
    {"brand": "Lowe's", "query": "Lowe's, Dallas, Texas", "prefix": "LW"},
]

DEFAULT_RADIUS_METERS = 400  # roughly a quarter-mile catchment
DEFAULT_START_DATE = "2024-01-01"
SOC_PAGINATION_LIMIT = 20000
MAX_RETRY_ATTEMPTS = 6
BASE_RETRY_DELAY = 1.0


def haversine_distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Return the distance in meters between two lat/lon pairs."""
    radius = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _request_with_backoff(
    url: str,
    params: Dict,
    headers: Dict,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
) -> requests.Response:
    attempt = 0
    delay = BASE_RETRY_DELAY

    while True:
        attempt += 1
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
        except requests.RequestException:
            if attempt >= max_attempts:
                raise
            time.sleep(delay)
            delay *= 1.5
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_attempts:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_time = float(retry_after)
                except ValueError:
                    wait_time = delay
            else:
                wait_time = delay
            time.sleep(wait_time)
            delay *= 1.5
            continue

        response.raise_for_status()
        return response


def fetch_store_candidates() -> List[Dict]:
    """Fetch store candidates from Nominatim and filter to the DFW area."""
    results: List[Dict] = []
    seen_positions: set[Tuple[str, str]] = set()

    for config in STORE_QUERIES:
        response = _request_with_backoff(
            NOMINATIM_URL,
            params={
                "format": "json",
                "addressdetails": 1,
                "limit": 50,
                "q": config["query"],
            },
            headers=NOMINATIM_HEADERS,
        )
        payload = response.json()

        for entry in payload:
            address = entry.get("address", {})
            county = address.get("county")
            state = address.get("state")
            if state != "Texas" or (county and county not in ALLOWED_COUNTIES):
                continue

            lat = entry.get("lat")
            lon = entry.get("lon")
            if not lat or not lon:
                continue

            key = (lat, lon)
            if key in seen_positions:
                continue
            seen_positions.add(key)

            street_parts = [
                address.get("house_number"),
                address.get("road") or address.get("pedestrian"),
            ]
            street_address = " ".join(part for part in street_parts if part)

            results.append(
                {
                    "brand": config["brand"],
                    "name": entry.get("name") or address.get("shop") or config["brand"],
                    "display_name": entry.get("display_name"),
                    "street_address": street_address,
                    "city": address.get("city")
                    or address.get("town")
                    or address.get("village"),
                    "county": county,
                    "state": address.get("state"),
                    "postal_code": address.get("postcode"),
                    "latitude": float(lat),
                    "longitude": float(lon),
                }
            )

        # be courteous to the API
        time.sleep(1)

    return results


def assign_store_ids(stores: List[Dict]) -> pd.DataFrame:
    """Assign deterministic store IDs and return a dataframe."""
    df = pd.DataFrame(stores)
    if df.empty:
        raise RuntimeError("No store locations found. Check the query filters.")

    df = df.sort_values(["brand", "city", "street_address"]).reset_index(drop=True)

    store_ids: List[str] = []
    counters: Dict[str, int] = {}
    for brand in df["brand"]:
        counters.setdefault(brand, 0)
        counters[brand] += 1
        prefix = next(
            conf["prefix"]
            for conf in STORE_QUERIES
            if conf["brand"] == brand
        )
        store_ids.append(f"{prefix}-{counters[brand]:03d}")

    df.insert(0, "store_id", store_ids)
    df["radius_meters"] = DEFAULT_RADIUS_METERS
    return df


def fetch_offenses_for_store(
    store: Dict,
    start_date: str = DEFAULT_START_DATE,
    radius_meters: int = DEFAULT_RADIUS_METERS,
) -> List[Dict]:
    """Fetch offenses for a single store within a given radius."""
    lat = store["latitude"]
    lon = store["longitude"]

    # Socrata expects ISO 8601 timestamp
    start_iso = f"{start_date}T00:00:00"

    enriched: List[Dict] = []
    offset = 0

    selected_columns = [
        "incidentnum",
        "incident_address",
        "date1",
        "nibrs_crime",
        "nibrs_crime_category",
        "nibrs_crimeagainst",
        "nibrs_code",
        "beat",
        "division",
        "sector",
        "district",
        "zip_code",
        "geocoded_column",
        "callorgdate",
        "time1",
        "time2",
    ]

    while True:
        params = {
            "$select": ",".join(selected_columns),
            "$where": (
                f"within_circle(geocoded_column, {lat}, {lon}, {radius_meters}) "
                f"AND date1 >= '{start_iso}'"
            ),
            "$limit": SOC_PAGINATION_LIMIT,
            "$offset": offset,
        }

        response = _request_with_backoff(SOC_DATASET_URL, params=params, headers=SOC_HEADERS)
        records = response.json()

        if not records:
            break

        for entry in records:
            loc = entry.get("geocoded_column") or {}
            incident_lat = loc.get("latitude")
            incident_lon = loc.get("longitude")
            if not incident_lat or not incident_lon:
                continue

            try:
                incident_lat_f = float(incident_lat)
                incident_lon_f = float(incident_lon)
            except ValueError:
                continue

            enriched.append(
                {
                    "store_id": store["store_id"],
                    "store_brand": store["brand"],
                    "store_name": store["name"],
                    "store_city": store["city"],
                    "radius_meters": radius_meters,
                    "incident_number": entry.get("incidentnum"),
                    "incident_address": entry.get("incident_address"),
                    "occurred_raw": entry.get("date1"),
                    "time1_raw": entry.get("time1"),
                    "time2_raw": entry.get("time2"),
                    "call_received_raw": entry.get("callorgdate"),
                    "nibrs_crime": entry.get("nibrs_crime"),
                    "nibrs_code": entry.get("nibrs_code"),
                    "nibrs_crime_category": entry.get("nibrs_crime_category"),
                    "nibrs_crime_against": entry.get("nibrs_crimeagainst"),
                    "beat": entry.get("beat"),
                    "division": entry.get("division"),
                    "sector": entry.get("sector"),
                    "district": entry.get("district"),
                    "zip_code": entry.get("zip_code"),
                    "incident_latitude": incident_lat_f,
                    "incident_longitude": incident_lon_f,
                    "distance_meters": haversine_distance_meters(
                        lat, lon, incident_lat_f, incident_lon_f
                    ),
                }
            )

        if len(records) < SOC_PAGINATION_LIMIT:
            break

        offset += SOC_PAGINATION_LIMIT
        time.sleep(0.25)

    return enriched


def fetch_offenses(
    stores: Iterable[Dict],
    start_date: str = DEFAULT_START_DATE,
    radius_meters: int = DEFAULT_RADIUS_METERS,
) -> pd.DataFrame:
    """Fetch offenses for all stores and return a dataframe."""
    offenses: List[Dict] = []
    for idx, store in enumerate(stores, start=1):
        print(f"Fetching offenses for {store['store_id']} ({idx})...")
        store_offenses = fetch_offenses_for_store(
            store,
            start_date=start_date,
            radius_meters=radius_meters,
        )
        offenses.extend(store_offenses)
        time.sleep(0.5)  # throttle requests to avoid hammering the API

    df = pd.DataFrame(offenses)
    if df.empty:
        raise RuntimeError("No offenses retrieved. Adjust the query parameters.")

    df["occurred_raw"] = pd.to_datetime(df["occurred_raw"], errors="coerce")
    df["call_received_raw"] = pd.to_datetime(df["call_received_raw"], errors="coerce")

    # Combine date and time components where both are provided
    def _parse_time_component(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return pd.NaT
        text = str(value).strip()
        if not text:
            return pd.NaT
        parts = text.split(":")
        if len(parts) == 2:
            hh, mm = parts
            ss = "0"
        elif len(parts) == 3:
            hh, mm, ss = parts
        else:
            return pd.NaT
        try:
            hours = int(hh)
            minutes = int(mm)
            seconds = int(ss)
        except ValueError:
            return pd.NaT
        return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)

    time_offsets = df["time1_raw"].apply(_parse_time_component)

    base_occurred = df["occurred_raw"]
    base_floor = base_occurred.dt.floor("D")
    occurred_naive = base_occurred.where(
        time_offsets.isna() | base_occurred.isna(),
        base_floor + time_offsets,
    )

    # Fall back to call received timestamp if occurred is missing
    occurred_naive = occurred_naive.combine_first(df["call_received_raw"])

    def _localize_central(series: pd.Series) -> pd.Series:
        if series.isna().all():
            return pd.Series(
                pd.NaT, index=series.index, dtype="datetime64[ns, America/Chicago]"
            )
        localized = series.dt.tz_localize(
            "America/Chicago", nonexistent="shift_forward", ambiguous="NaT"
        )
        if localized.isna().any():
            localized = localized.combine_first(
                series.dt.tz_localize(
                    "America/Chicago", nonexistent="shift_forward", ambiguous=False
                )
            )
            localized = localized.combine_first(
                series.dt.tz_localize(
                    "America/Chicago", nonexistent="shift_forward", ambiguous=True
                )
            )
        return localized

    occurred_local = _localize_central(occurred_naive)
    call_received_local = _localize_central(df["call_received_raw"])

    occurred_local = occurred_local.combine_first(call_received_local)

    df["occurred_local"] = occurred_local
    df["occurred_datetime"] = occurred_local.dt.tz_convert("UTC")
    df["call_received_local"] = call_received_local

    df = df.dropna(subset=["occurred_local"])
    df["occurred_date"] = df["occurred_local"].dt.date
    local_naive = df["occurred_local"].dt.tz_localize(None)
    df["occurred_week"] = local_naive.dt.to_period("W").astype(str)
    df["occurred_month"] = local_naive.dt.to_period("M").astype(str)
    df["pull_timestamp_utc"] = datetime.utcnow()
    df = df.drop(
        columns=["occurred_raw", "time1_raw", "time2_raw", "call_received_raw"]
    )
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching store locations...")
    stores = fetch_store_candidates()
    stores_df = assign_store_ids(stores)

    stores_path = DATA_DIR / "home_improvement_stores.csv"
    stores_df.to_csv(stores_path, index=False)
    print(f"Wrote {len(stores_df)} stores to {stores_path}")

    print("Fetching offenses near stores...")
    offenses_df = fetch_offenses(stores_df.to_dict(orient="records"))
    offenses_path = DATA_DIR / "store_offenses.parquet"
    offenses_df.to_parquet(offenses_path, index=False)
    print(f"Wrote {len(offenses_df)} offenses to {offenses_path}")


if __name__ == "__main__":
    main()
