# Spatial wealth surfaces

Disaggregates official **district GDDP** to **H3 resolution 7** (~5 km²) using a hierarchical graph neural network, following Lee, Blankespoor, and Newhouse (World Bank, 2026), *Fine-Scale Spatial Disaggregation of Statistical Data via Graph Neural Networks*.

Train **one GNN per state**. Maharashtra remains the reference implementation; `run_maharashtra.py` is a wrapper around `run_state.py --state MH`.

## What the output is

`data/output/{STATE}/{state}_h3_r7_wealth.csv` has one row per H3 R7 cell:

| column | meaning |
|---|---|
| `gdp_crore` / `gdp_inr` | cell share of official district GDDP (dasymetric, sums to `Y_g`) |
| `gdp_per_capita_inr` | ensemble **mean** allocated ₹/person across 5 seeds |
| `gdp_per_capita_inr_min` / `_max` | min–max across seeds |
| `wealth_index` | within-state percentile of per-capita intensity, 0–100 |
| `wealth_index_min` / `_max` | min–max of per-seed percentile ranks |

**Cell GDP is an allocation of official district GDP, not a measurement.**  
**`wealth_index` is not a DHS survey score.** The paper’s validation is that disaggregated GDP intensity correlates with DHS wealth; this pipeline uses additive GDDP as `Y_g`.  
**Ensemble range is across random seeds, not a calibrated probability of error.**  
Post-scale district reconstruction error ≈ 0 is **conservation**, not accuracy.

Night-time lights (**VIIRS average_masked**) are in the model when the raster is present.

## Data (all saved under `data/raw/{layer}/{state or IN}/` with SOURCE.json)

| layer | source | year | role |
|---|---|---|---|
| District GDDP | State DES / Economic Survey via OpenCity or official PDF | see `data/raw/gddp/STATE_COVERAGE.md` | constraint `Y_g` |
| Boundaries | [geoBoundaries](https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/) IND ADM1+ADM2 | 2021 | H3 crosswalk (CC BY) |
| Population | [Kontur / HDX](https://data.humdata.org/dataset/kontur-population-india) India 2023 H3 (~R8, summed to R7) | 2023 | exposure `w_i` + feature (CC BY). WorldPop 2020 is `--population worldpop` |
| OSM | OpenStreetMap via Overpass | current extract | roads, building footprints, landuse shares, enriched POIs (ODbL) |
| VIIRS | EOG Annual VNL v2.2 **average_masked** | 2023 | mean radiance per R7 cell. Not unmasked, not median. Multi-GB rasters stay local |

Coverage, year, units, and merges: [`data/raw/gddp/STATE_COVERAGE.md`](data/raw/gddp/STATE_COVERAGE.md).  
NTL product choice: [`data/raw/nightlights/CHOICE.md`](data/raw/nightlights/CHOICE.md).

Place user-provided `*average_masked*` (and optional `*median_masked*`) under `data/raw/nightlights/IN/`. Training **fails** if the raster does not cover the state bbox.

## Method (mapped to the paper)

1. **H3 R7 nodes** with parent R6 / R5 cells.
2. **Features:** population, night_lights, OSM `road_km`, OSM POIs (kept), OSM **building area**, **landuse** shares, and OSM-only premium/mass POI counts. Zero-inflation split + `log10` + z-score (paper §4.2). No Google / MagicBricks / Zomato.
3. **Graph:** k=1 hex adjacency + parent–child edges.
4. **GNN:** GraphSAGE-style message passing, GCNII initial residual, `softplus` intensity `s_i ≥ 0`. Architecture unchanged except extra feature columns.
5. **Loss:** `mean((log10 Σ_i s_i w_i − log10 Y_g)²)` plus a light population-weighted floor (paper §4.4). Unchanged.
6. **Dasymetric rescale** inside each district so `Σ z_i = Y_g` exactly (paper §5.2).
7. **5 random seeds:** central estimate = mean; min–max stored on cells, CSV, UI, and `run_summary.json`.

Evaluation (also in `run_summary.json`): pre-scale district log-loss; spatial hold-out MAPE on district sums **before** scale; ablation with/without NTL; Spearman vs NTL/roads/POIs as sanity, not truth.

## Run

```bash
python -m pip install -r requirements.txt
python run_state.py --list-states
python run_state.py --state MH
python run_state.py --state KA
python run_state.py --state TG
python run_state.py --state DL
```

Maharashtra wrapper: `python run_maharashtra.py`.

Useful flags: `--epochs 50 --seeds 0 1 2 3 4 --population kontur --skip-eval --osm-step 1.0`.

Requires Python 3.10+ and a working rasterio/GDAL wheel on Windows. No API secrets. Do not commit multi-GB rasters; they stay in `data/raw/`.

PDF-only DES tables (UP, Bihar, Rajasthan, West Bengal): place `data/raw/gddp/{STATE}/gddp.csv` with `district,y_crore` from the official publication. Paywalled aggregators are not used.

## Web map + explainer + tiles

```bash
python web/prepare_data.py
python -m http.server 8000 --directory web
```

Open [http://localhost:8000](http://localhost:8000).

- State filter first; district and optional metro **cascade** to that state.
- Default colour = **within-state percentile** of the chosen metric (`wealth_index` default). Toggle **absolute** (₹/person, counts, radiance) — a warning is shown; do not mix states on one absolute scale.
- **Split map:** two linked panes, same camera and hover id, independent KPIs (wealth vs population vs lights vs roads vs POIs vs buildings).
- **Download:** per-state CSV of all R7 cells (inputs, engineered columns, intensity, scale, GDP, wealth, ensemble min/max, district `Y_g`) plus the data dictionary. `prepare_data.py` writes both `cells.csv` and `cells.csv.gz`; only the gzip is committed, and the button falls back to it when the plain file is absent.
- Hex geometry is generated in the browser from H3 ids, so there are no tiles to host. The state opens as an **R6 overview** (fast first paint) and switches to **R7 detail** when a district is picked, loading only `web/data/{STATE}/r7/{district}.json` (≤ 1.2 MB per district).
- Cell files are **columnar** — one array per field rather than one object per cell, with `y_crore` and `dasymetric_scale` hoisted to a per-district map. `expand()` in the page rebuilds row objects on load, which keeps the whole payload for 7 states near 50 MB instead of 190 MB.
- Each state shows its own **GDDP reference year and price basis** next to the state name, because official releases are for different years.
- **Explainer:** [http://localhost:8000/explainer.html](http://localhost:8000/explainer.html) — a nine-step animated walkthrough (play / pause / step / arrow keys) of `district Y_g → H3 cells → features → graph → message passing → embedding → s_i → z_i / Z_g → final ₹`. Presets (rich urban / poor rural) carry the logged layer activations of seed 0; clicking any hex on the map walks that cell through the same steps using its published values.

## Licenses

| dataset | license |
|---|---|
| EOG VIIRS VNL v2.2 | EOG / Payne Institute terms (free registration; rasters not in git) |
| Kontur Population | CC BY via HDX |
| OpenStreetMap | ODbL |
| geoBoundaries gbOpen | CC BY |
| State DES / Economic Survey tables | as published by the state; OpenCity extracts are public-domain redistributions of those tables |
