# Method and interview notes

This document is the long form of the project: **what we actually implemented**, why, and where it is weak. It matches `src/`, `run_state.py`, and the Maharashtra web artifacts — not a generic GNN tutorial.

If something is not in the code, it is marked as unknown.

**Related files**


| Piece                               | Where                                   |
| ----------------------------------- | --------------------------------------- |
| Pipeline                            | `run_state.py`                          |
| Config (H3, states, GDDP, features) | `src/config.py`                         |
| Polygons, H3, features, targets     | `src/features.py`                       |
| Fetch                               | `src/fetch.py`, `src/osm_pbf.py`        |
| Graph, GNN, train, scale            | `src/model.py`                          |
| Hold-out / ablation                 | `src/evaluate.py`                       |
| Explainer JSON                      | `src/explainer.py`                      |
| Static web assets                   | `web/prepare_data.py`, `web/index.html` |


---



## 1. Problem in one district: Pune



### Observed

Pune’s official GDDP (Maharashtra Economic Survey extract, 2023–24, **constant 2011–12** prices):

$$
Y_{\text{Pune}} = 281{,}986 \text{ crore rupees}
$$

In code: `src/features.py` → `load_gddp()`, then `y_inr = y_crore × 10^7`.

That number answers “how large is Pune’s economy?” It does **not** answer “which 5 km² patches produced it?”

Pune is filled with **3,235** H3 resolution-7 cells (`web/data/MH/summary.json`).

### Estimated

For each cell i we produce an **allocated** GDP GDP_i such that:

$$
\sum_{i \in \text{Pune}} GDP_i = Y_{\text{Pune}}
$$

That equality is **forced after** the model runs. It is not proof that the split is correct.

### One real cell (illustrative)

From the committed Maharashtra cell extract, hex `876081286ffffff`:


| Field              | Approx. value                                          |
| ------------------ | ------------------------------------------------------ |
| Population         | 9,585                                                  |
| Night lights       | 12.94                                                  |
| Road km            | 17.94                                                  |
| POI count          | 16                                                     |
| Building area      | 18,608 m²                                              |
| GNN intensity s_i  | 213,932                                                |
| Pune scale k       | 0.399                                                  |
| Allocated GDP      | ₹81.91 crore                                           |
| Allocated ₹/person | ₹85,458                                                |
| `wealth_index`     | 38.2 (percentile **in Maharashtra**, not in Pune only) |


₹81.91 crore is **not** surveyed. It is s_i \times pop_i \times k_{\text{Pune}}.

---



## 2. What “disaggregation” means

**Observed money:** one Y_g per reporting unit (district, or Delhi NCT).

**Observed proxies:** people, lights, OSM on each hex.

**Learned:** a function from proxies + neighbours → intensity s_i \ge 0.

**Imposed:** shares that add to Y_g.

The compact formula:

$$
GDP_i = Y_g \cdot \frac{s_i \cdot pop_i}{\sum_{j \in g} s_j \cdot pop_j}
$$

Two cells with the **same population** can still get different GDP if their s_i differ (lights, roads, POIs, neighbours). Two cells with the **same s_i** in the same district get the same ₹/person, even if population differs, because:

$$
\frac{GDP_i}{pop_i} = s_i \cdot k_g \qquad (pop_i > 0)
$$

with k_g = Y_g / \sum_{j \in g} s_j pop_j.

---



## 3. End-to-end order (what first)

`run_state.py` → `run()`:

1. **Fetch** (or reuse) GDDP, boundaries, population, lights, OSM (`src/fetch.py`).
2. **District polygons** + **GDDP table** (`load_district_polygons`, `load_gddp`).
3. **H3 R7 fill** (`build_h3_index`).
4. **Join proxies** (`aggregate_population`, `aggregate_nightlights`, `aggregate_osm`).
5. **Attach Y_g** (`attach_targets`) — join by district name; **not** a GNN feature.
6. **Engineer features** (`engineer_features`) — exists + log10 + z-score.
7. **Graph tensors** (`prepare_tensors`).
8. **Train 5 seeds** (`train_ensemble` → `train_gnn`).
9. **Predict + scale** (`predict_and_scale`).
10. **Evaluate** (optional): hold-out, NTL ablation, Spearman (`src/evaluate.py`).
11. **Export** CSV/parquet, explainer, `run_summary.json`.
12. **Web** (separate command): `web/prepare_data.py`.

The map never calls PyTorch.

---



## 4. H3 cells

`src/config.py`: `H3_RES = 7`, parents 6 and 5.

`build_h3_index()` polyfills each district polygon, keeps a cell if its **centroid** is in the polygon, stores `h3_r7`, `h3_r6`, `h3_r5`, lat/lon, area.

**Why H3:** comparable units, stable ids for the browser, parents for the graph, Kontur population already lives on H3.

**Not why:** hexes are not “the true geography of the economy.” Border hexes can stick out of the district. Assignment is centroid-based, not area-weighted clip.

**Hierarchy in the GNN is H3 R7↔R6↔R5**, not State→District→H3. District is a **label** for loss grouping and scaling. There is no “Pune node” in the network.

---



## 5. Features (only what exists)

Magnitudes (`MAGNITUDE_FEATURE_BASE` in `src/config.py`): population, night lights, `road_km`, `poi_count`, building area/count, per-tag POI counts and densities.

Shares: residential / commercial / industrial / retail landuse, `building_share`.

**Not used:** Google, MagicBricks, Zomato, DHS scores, `y_crore` as an input column.

### Why each family *might* track activity


| Feature             | Possible signal                                       | Bias / limit                                                          |
| ------------------- | ----------------------------------------------------- | --------------------------------------------------------------------- |
| Population          | Where people live; exposure                           | Not income or productivity; used **twice** (feature + s_i \times pop) |
| Night lights        | Electrification, night-time urban/industrial activity | Agriculture/daytime work can be dark; spillover; year ≠ GDDP year     |
| Roads               | Access                                                | Highway through a poor cell; OSM completeness                         |
| POIs                | Services / consumption                                | Volunteer OSM; urban over-mapping; 0 can mean unmapped                |
| Buildings / landuse | Built intensity, use mix                              | Footprint ≠ height or output; warehouses vs offices                   |




### Preprocessing (`engineer_features`)

For a magnitude x:

- `exists` = 1 if x > 0, else 0  
- `log` = \log_{10}(x) if x > 0, else 0  
- then **state-wide** z-score: (v - \mu) / \sigma

**Why exists + log:** \log_{10}(0) is undefined; we store 0. \log_{10}(1) = 0 as well, so **0 vs 1 POI** would collapse without `exists`.

R6/R5 **node features** are **means** of child R7 engineered columns (`prepare_tensors`), not extra surveys and not sums.

Maharashtra uses **94** z-scored columns when lights are present.

**Leakage note:** z-score \mu,\sigma use **all** cells in the state **before** hold-out retraining. Hold-out districts leak into normalisation.

---



## 6. Graph

**Nodes:** every R7 cell in the state; unique R6 and R5 parents of those cells.

**Same-level edges:** `build_edges()` — H3 `grid_disk(..., 1)`, undirected, no self-loops, only neighbours that exist in the node set. Interior hexes usually have 6 neighbours.

**Parent–child:** `build_hierarchy()` — R7→R6 and R6→R5.

An edge means **“share learned vectors.”** It does **not** mean GDP flows between hexes.

---



## 7. GNN (`HierarchicalGNN`)

Hyperparameters (`src/config.py`): 4 layers, hidden 64, \alpha = 0.1, Adam lr 10^{-2}, 50 epochs.

### Embeddings

Each resolution: linear map 94 → 64, then ELU.

### Same-level GraphSAGE (`GraphSageConv`)

Intended idea: average neighbour embeddings, mix with self:

$$
m_i = \mathrm{mean}*{j \in N(i)} h_j, \qquad
a_i = \mathrm{ELU}(W*{\mathrm{self}} h_i + W_{\mathrm{nb}} m_i)
$$

**Implementation detail:** `deg` starts at **1**, then adds neighbour count, but the summed message is **neighbours only**. So the code divides by |N(i)|+1, not |N(i)|. Neighbour signal is slightly damped. Treat this as an implementation quirk, not as a theorem.

### Up and down

- **Up:** mean-pool children into parent (`_pool_to_parent`).
- **Down:** linear transform of parent copied to each child (`_child_from_parent`). `_broadcast_from_parent` exists but is **unused**.

R7 mix: concatenate GraphSAGE output with downward R6 context, linear `64×2 → 64`, ELU.

R5 update has adjacency + upward pool, **no** downward-from-above term.

### Residual (GCNII-style)

$$
h \leftarrow (1-\alpha) h_{\mathrm{new}} + \alpha h^{(0)}, \quad \alpha = 0.1
$$

Keeps a slice of the **original** embedding so four layers do not wash the cell into its neighbourhood (oversmoothing risk, not proven solved).

### Output

$$
s_i = \mathrm{softplus}(W_{\mathrm{out}} h_i^{(4)} + b) > 0
$$

In the **loss**, s_i \times pop_i is compared to rupees, so s_i behaves like an **uncalibrated ₹/person**. Docs sometimes say “unitless”; dimensionally it is easier to call it raw per-person intensity.

After four layers, s_i can depend on own features, adjacent R7 cells, R6 parent (and through later layers, wider context). **GDP still does not travel.** Only numbers in \mathbb{R}^{64} do.

### Are R6/R5 required?

**No.** They are a multi-scale shortcut. The repo has **no** ablation of R7-only vs hierarchical GNN. Do not claim hierarchy “improves accuracy.”

---



## 8. Training — how rupees appear without putting Y_g in x_i



### Forward

$$
\widehat{Y}*g = \sum*{i: d(i)=g} s_i \cdot pop_i
$$

`train_gnn()`: `z = s * pop`, then `index_add_` by district.

### Tiny example


| Cell | pop | s   | z = s \times pop |
| ---- | --- | --- | ---------------- |
| A    | 100 | 10  | 1,000            |
| B    | 200 | 20  | 4,000            |
| C    | 300 | 10  | 3,000            |


\widehat{Y} = 8{,}000. Official Y = 10{,}000. Raw model is low.

### Why this is in rupees

Features (lights, roads) have **no inherent rupee unit**. Random initial s_i would make \widehat{Y}_g nonsense. **Official Y_g is in the loss every epoch**, so gradients push s_i onto a scale where \sum s_i pop_i is near Y_g. Same idea as a house-price model: bedrooms are not rupees; labels teach the scale.

Y_g is **not** in the 94-D feature vector (`prepare_tensors` keeps `y_g` separate). That is **not** “GDP unused until the end.”

### Loss

$$
L_{\mathrm{agg}} = \frac{1}{|G|} \sum_g \big(\log_{10}(\widehat{Y}*g+\varepsilon) - \log*{10}(Y_g+\varepsilon)\big)^2
$$

Log so Mumbai does not dominate Gadchiroli in raw rupees. \varepsilon = 10^{-6}.

Floor (usually small):

$$
L_{\mathrm{floor}} = \frac{\sum_i pop_i [\max(0, \tau - s_i)]^2}{\sum_i pop_i}, \quad \tau=1
$$

$$
L = L_{\mathrm{agg}} + 0.05 L_{\mathrm{floor}}
$$

### What the gradient does **not** say

If Pune’s \widehat{Y} is low, the loss does **not** name which hex is wrong. It says the **sum** is low. Shared weights \theta decide how each cell moves. Many maps can match 34 district totals (MH: **61,822** cells, **34** Y_g).

### Optimizer

Adam, cosine LR, grad clip 5, **full graph** (no mini-batches), checkpoint = **best training loss** (no validation set for early stopping).

### Five seeds

Seeds 0–4. Central s_i = mean. Scale is **recomputed** on the mean. Min/max stored. That is **init sensitivity**, not a probability of error.

---



## 9. After training: k_g and “why is Pune’s k \approx 0.4?”

`predict_and_scale()`:

$$
k_g = \frac{Y_g}{\sum_{i \in g} s_i pop_i}, \qquad GDP_i = s_i pop_i k_g
$$

k_g is **arithmetic**, not a neural weight.

If k_{\text{Pune}} \approx 0.399, then **raw** \widehat{Y}_{\text{Pune}} \approx Y / 0.399 \approx 7.06 \times 10^5 crore — the ensemble **over-predicted** Pune’s total before scaling. Multiplying every Pune z_i by 0.4 restores ₹281,986 crore **without changing Pune’s internal shares**:

$$
\mathrm{share}*i = \frac{s_i pop_i}{\sum*{j \in g} s_j pop_j}
$$

k_g is **not** “how important Pune is versus Thane.” Each district has its **own** k_g. Thane’s k does not reweight Pune cells.

**Why not force k \approx 1 in training?** Training **tries** to match \widehat{Y}_g \approx Y_g on average across districts. One shared GNN, 50 epochs, messy features, and five-seed averaging mean Pune need not land on k=1. Setting k=1 by hand would **violate** official Pune GDDP.

Two-cell example (shares vs ₹/person):

- A: s=2, pop=100 → z=200
- B: s=4, pop=50 → z=200
- Y=1000 → each gets 500 rupees of GDP; ₹/person is 5 vs 10.

---



## 10. `wealth_index`

Percentile of allocated ₹/person among **inhabited cells in the state** (`_wealth_from_pci`). Rank 38 means “below about 62% of Maharashtra hexes,” not “poor by NFHS.”

---



## 11. Identifiability (the core critique)

Constraint for two equal-pop cells: s_A + s_B = 10 has infinitely many solutions (5,5), (2,8), …  
Features + graph + optimisation **pick one** story. They do not **prove** it.

This is **ecological inference**: district-level fit ≠ cell-level truth.

---



## 12. Why a GNN vs simpler methods


| Approach                                   | Idea                                        | Honest take                                            |
| ------------------------------------------ | ------------------------------------------- | ------------------------------------------------------ |
| Share by population                        | s_i=1                                       | Strong, transparent baseline; **not compared in-repo** |
| Share by lights                            | s_i \propto radiance                        | Same; night-blind activity missed                      |
| Linear s_i = \mathrm{softplus}(w^\top x_i) | Same loss, no graph                         | Should be a baseline                                   |
| RF / XGBoost                               | Need labels or a custom aggregate objective | Spatial lags can mimic much GNN context                |
| This GNN                                   | Learns neighbour + parent mixing            | Reasonable **hypothesis**; **not shown to win**        |


Say in interview: *GNN can learn spatial mixing without hand-built lags; we lack baselines, so I will not claim it is necessary.*

---



## 13. Validation (what evidence we actually have)

**No classical train/val/test on cells.** No H3 GDP labels.


| Check                             | What it is                        | What it is not                                             |
| --------------------------------- | --------------------------------- | ---------------------------------------------------------- |
| Post-scale recon error ~ 10^{-16} | Scaling algebra                   | Accuracy                                                   |
| In-sample log-loss                | Fit of **raw** district sums      | H3 map quality                                             |
| Hold-out MAPE (MH ~ **0.57**)     | Raw sums on 20% districts         | H3 accuracy; graph + z-scores still see hold-out geography |
| Spearman vs NTL/roads/POIs        | Input–output rank corr            | Independent truth                                          |
| NTL ablation                      | Lights help **district** log-loss | Cell map is right                                          |
| Seed min–max                      | Optimisation spread               | Calibrated CI                                              |


Hold-out (`spatial_holdout_mape`): districts dropped from L_{\mathrm{agg}} only. Cells stay in the graph; `L_floor` uses all cells; features already z-scored on the full state. MH hold-out included Pune and Nagpur among others.

**Paper vs this repo:** the paper’s DHS-style check is **not implemented** here.

---



## 14. Limitations that actually apply

- No fine-scale economic ground truth  
- Temporal mismatch (GDDP / pop / lights / OSM)  
- Constant vs current prices across states  
- OSM urban completeness  
- Night-light urban/electricity bias  
- Population as feature **and** weight  
- Adjacency ≠ economic networks  
- Oversmoothing risk  
- Forced allocation of every rupee  
- H3 centroid assignment  
- Delhi: one Y_g for all NCT cells — internally almost unconstrained  
- District name merges (Mumbai City+Suburban, Thane+Palghar, …)  
- Distribution shift if you reuse weights across states (we train **per state** instead)



### Improvements (grouped)

**Easy:** population- and light-only baselines; linear / XGBoost; fix GraphSAGE degree; train-only z-scores; cut cross-boundary edges in hold-out; report k_g per district; rename `wealth_index`.

**More data:** ward GDP, firms/jobs, tax, aligned years, building height.

**Research:** real uncertainty, admin hierarchy with care not to leak Y_g into children, transport graphs, external validation.

---



## 15. Software system

```
User → static web (index.html)
         → catalog.json / summary.json / cells_r6.json / r7/{district}.json
         → h3-js draws hexes, MapLibre colours them

Training machine (earlier):
  fetch → features → GNN train/predict/scale → data/output → prepare_data.py → web/data
```

No Flask/FastAPI inference. Checkpoints `*.pt` are gitignored and not loaded in the browser.

Explainer: nine canned steps. Layer traces exist for two **preset** cells (seed 0). Other hexes: published columns only.

---



## 16. Quick Qna

**What problem?** Allocate official district (or NCT) GDDP to ~5 km² hexes using proxies. Not a new GDP survey.

**Why H3?** Regular grid, ids, parents, population product.

**Why GNN?** Neighbour + parent context in the model; unproven vs XGBoost or population shares.

**Node / edge?** R7 (and R6/R5) hexes; k=1 adjacency + parent–child. Messages are embeddings, not rupees.

**Features?** Pop, VIIRS average_masked, OSM roads/buildings/landuse/POIs; exists+log+z. Not `Y_g` in x_i.

**Message passing?** GraphSAGE mix of self and neighbours; pool up, broadcast down; residual; 4 layers; softplus s_i.

**What does it learn?** Shared \theta so \sum_i s_i pop_i matches district Y_g in log space.

**Why train?** No known formula mapping lights/roads to rupees; labels teach scale and combinations. Still underdetermined at cell level.

**Target / loss?** District Y_g; log-MSE of sums + small floor. Not H3 GDP.

**One cell?** s_i from GNN; GDP_i = Y_g \times (s_i pop_i) / \sum_j s_j pop_j.

**Learned vs imposed?** s_i learned; k_g and exact \sum GDP_i = Y_g imposed.

**Validation?** Weak district-level checks; no H3 truth.

**Limits?** Identifiability, leakage in hold-out, OSM/NTL bias, year mismatch, no baselines.

**Improve?** Baselines, cleaner CV, external microdata, fix aggregation bug, report k_g.

---



## 17. Possible questions and direct answers

**Q. If we already have population, why not just split GDP by population?**  
A. That assumes equal ₹/person inside the district. The GNN lets s_i vary. We have not shown it is better.

**Q. How can s_i be ~200,000 if GDP per capita is ~85,000?**  
A. Because k_{\text{Pune}} \approx 0.4. GDPPC_i = s_i k_g. Raw s_i is uncalibrated.

**Q. Is State→District→H3 in the neural net?**  
A. No. Separate model per state; district for loss/scale; H3 parents for the graph.

**Q. Does reconstruction error ≈ 0 mean the map is good?**  
A. No. That is the scaling identity.

**Q. Is there an API?**  
A. No. Offline batch + static files.