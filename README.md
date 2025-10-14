# Dallas Home Improvement CompStat

Operational analytics for Home Depot and Lowe's locations across the Dallas–Fort Worth metro. The Streamlit dashboard fuses store-level geometry with Dallas Police Department NIBRS incident data to drive mobile-friendly CompStat reviews focused on tactical and strategic decision making.

## Features

- Store-centric CompStat KPIs (7-day, 28-day, YTD) with rolling vs. prior period comparisons, z-score anomaly checks, and Poisson probability triggers.
- Interactive map highlighting store risk posture, optimized for quick consumption on phones or tablets.
- Deep-dive store view with weekly trend lines, offense-category mix, and day/hour heatmaps to guide deployment.
- Tactical and strategic note generation to support heavy outcome-focused leaders.

## Data Sources

- **Store locations:** Queried from [OpenStreetMap Nominatim](https://nominatim.openstreetmap.org/) for Home Depot and Lowe's sites in the DFW area (Dallas, Collin, Denton, Tarrant, Rockwall, Ellis, Kaufman counties).
- **Crime data:** Dallas Police Department NIBRS incidents via [Dallas Open Data](https://www.dallasopendata.com/resource/qv6i-rri7.json), filtered to occurrences within 400 meters of each store (2024 onward).

> **Attribution:** OpenStreetMap data © OpenStreetMap contributors (ODbL 1.0). Dallas Police Department data provided via Dallas Open Data (public domain, see site terms).

## Project Structure

```
├─ compstat/             # Reusable data loaders, metrics, and time-series helpers
├─ data/                 # Generated store/offense extracts (git-friendly sample output)
├─ scripts/
│  └─ fetch_data.py      # Pull latest store geometry + offense incidents
├─ streamlit_app.py      # Streamlit application entrypoint
├─ requirements.txt      # Python dependencies
└─ README.md
```

## Getting Started

1. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

2. **Refresh data (optional)**

   ```bash
   python scripts/fetch_data.py
   ```

   - Respects Nominatim usage guidelines with lightweight query volume and throttling.
   - Pulls Dallas NIBRS incidents from 2024-01-01 forward for each store (≈4–5k rows).
   - Outputs `data/home_improvement_stores.csv` and `data/store_offenses.parquet`.

3. **Run Streamlit locally**

   ```bash
   streamlit run streamlit_app.py
   ```

   The dashboard defaults to a Dallas focus map and mobile-responsive view, with sidebar controls for retailer and NIBRS category filters.

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app and point it to the repo/branch with `streamlit_app.py` as the entry point.
3. Streamlit Cloud will install packages from `requirements.txt`. The cached dataset bundled in `data/` enables instant startup; schedule refreshes by re-running `scripts/fetch_data.py`, committing updated parquet/CSV outputs, and redeploying.
4. Optionally set up the in-app “Re-run” button after data refreshes or automate pulls via GitHub Actions.

## Methodology Notes

- **CompStat windows:** 7-day and 28-day metrics compare against immediately preceding periods; YTD compares against the same span in the prior calendar year.
- **Anomaly scoring:** z-scores leverage store-specific daily baselines. Poisson survival probabilities (< 0.05) highlight statistically unlikely surges.
- **Radius:** 400 meters (~0.25 miles) around each store balances catchment breadth with noise control.

## Next Ideas

- Integrate repeat-call clustering or recency-weighted scoring to spotlight problem anchors.
- Add rapid exporter (CSV/PDF) for command staff briefings.
- Wire in automated data refresh using scheduled GitHub Action.

---

Built with audit and field operations workflows in mind—fast to grasp on mobile, with deeper layers a tap away. PRs welcome!
