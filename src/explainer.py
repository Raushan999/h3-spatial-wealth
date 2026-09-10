"""Dump tensors/logs used by the interactive GNN explainer (no invented numbers)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import h3

from .config import POI_GROUPS


PRESET_RULES = {
    "rich_urban": "highest wealth_index among cells with population above the 80th percentile",
    "poor_rural": "lowest wealth_index among cells with population between the 20th and 60th percentiles and below-median road_km",
}


def _pick_presets(df: pd.DataFrame) -> dict[str, str]:
    pop = df["population"]
    rich_pool = df[pop >= pop.quantile(0.8)]
    rural_pool = df[
        (pop >= pop.quantile(0.2))
        & (pop <= pop.quantile(0.6))
        & (df["road_km"] <= df["road_km"].median())
    ]
    rich = rich_pool.sort_values("wealth_index", ascending=False).iloc[0]["h3_r7"] if len(rich_pool) else df.sort_values("wealth_index", ascending=False).iloc[0]["h3_r7"]
    poor = rural_pool.sort_values("wealth_index", ascending=True).iloc[0]["h3_r7"] if len(rural_pool) else df.sort_values("wealth_index", ascending=True).iloc[0]["h3_r7"]
    return {"rich_urban": str(rich), "poor_rural": str(poor)}


def _cell_payload(row: pd.Series, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    cell = str(row["h3_r7"])
    neighbors = [n for n in h3.grid_disk(cell, 1) if n != cell]
    present = set(df["h3_r7"])
    nb_rows = df[df["h3_r7"].isin([n for n in neighbors if n in present])]
    raw_feats = {}
    for col in [
        "population",
        "night_lights",
        "road_km",
        "poi_count",
        "building_area_m2",
        *[f"poi_{k}" for k in POI_GROUPS],
        *[f"landuse_{c}" for c in ("residential", "commercial", "industrial", "retail")],
    ]:
        if col in row.index:
            val = row[col]
            raw_feats[col] = None if pd.isna(val) else float(val)
    z_feats = {c: float(row[c]) for c in feature_cols if c in row.index}
    district = row["district"]
    dpart = df[df["district"] == district]
    return {
        "h3_r7": cell,
        "h3_r6": str(row["h3_r6"]),
        "h3_r5": str(row["h3_r5"]),
        "district": district,
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "inputs": raw_feats,
        "features_z": z_feats,
        "neighbors": [
            {
                "h3_r7": str(r.h3_r7),
                "population": float(r.population),
                "night_lights": float(getattr(r, "night_lights", 0) or 0),
                "road_km": float(r.road_km),
                "poi_count": float(r.poi_count),
                "wealth_index": float(r.wealth_index),
            }
            for r in nb_rows.itertuples(index=False)
        ],
        "parents": {"h3_r6": str(row["h3_r6"]), "h3_r5": str(row["h3_r5"])},
        "s_i": float(row["intensity_raw"]),
        "s_i_min": float(row["intensity_raw_min"]) if "intensity_raw_min" in row.index and pd.notna(row["intensity_raw_min"]) else None,
        "s_i_max": float(row["intensity_raw_max"]) if "intensity_raw_max" in row.index and pd.notna(row["intensity_raw_max"]) else None,
        "population": float(row["population"]),
        "z_i": float(row["gdp_inr_raw"]) if "gdp_inr_raw" in row.index else float(row["intensity_raw"] * row["population"]),
        "district_sum_raw": float(dpart["gdp_inr_raw"].sum()) if "gdp_inr_raw" in dpart.columns else None,
        "Y_g": float(row["y_inr"]),
        "dasymetric_scale": float(row["dasymetric_scale"]),
        "gdp_inr": float(row["gdp_inr"]),
        "gdp_per_capita_inr": None if pd.isna(row["gdp_per_capita_inr"]) else float(row["gdp_per_capita_inr"]),
        "gdp_per_capita_inr_min": None if "gdp_per_capita_inr_min" not in row.index or pd.isna(row["gdp_per_capita_inr_min"]) else float(row["gdp_per_capita_inr_min"]),
        "gdp_per_capita_inr_max": None if "gdp_per_capita_inr_max" not in row.index or pd.isna(row["gdp_per_capita_inr_max"]) else float(row["gdp_per_capita_inr_max"]),
        "wealth_index": float(row["wealth_index"]),
        "wealth_index_min": None if "wealth_index_min" not in row.index or pd.isna(row["wealth_index_min"]) else float(row["wealth_index_min"]),
        "wealth_index_max": None if "wealth_index_max" not in row.index or pd.isna(row["wealth_index_max"]) else float(row["wealth_index_max"]),
    }


def _district_context(df: pd.DataFrame, district: str) -> dict:
    part = df[df["district"] == district]
    raw = float(part["gdp_inr_raw"].sum()) if "gdp_inr_raw" in part.columns else None
    return {
        "district": district,
        "n_cells": int(len(part)),
        "Y_g_inr": float(part["y_inr"].iloc[0]) if "y_inr" in part.columns and len(part) else None,
        "Y_g_crore": float(part["y_crore"].iloc[0]) if "y_crore" in part.columns and len(part) else None,
        "district_sum_raw_inr": raw,
        "dasymetric_scale": float(part["dasymetric_scale"].iloc[0]) if "dasymetric_scale" in part.columns and len(part) else None,
        "population": float(part["population"].sum()),
        "gdp_inr_after_scale": float(part["gdp_inr"].sum()) if "gdp_inr" in part.columns else None,
    }


def _feature_attribution(row: pd.Series, feature_cols: list[str], top: int = 10) -> list[dict]:
    """Rank the z-scored inputs for one cell by |z|.

    This is a magnitude ranking of the model's inputs, not a causal attribution:
    it says which features are unusual for this cell, which is what makes the
    walkthrough legible without over-claiming.
    """
    rows = []
    for col in feature_cols:
        if col not in row.index or pd.isna(row[col]):
            continue
        rows.append({"feature": col, "z": round(float(row[col]), 4)})
    rows.sort(key=lambda r: -abs(r["z"]))
    return rows[:top]


def write_explainer_artifact(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    loss_histories: list | None = None,
    traces: dict | None = None,
    node_index: dict | None = None,
    extra_presets: dict | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    presets = _pick_presets(df)
    if extra_presets:
        presets.update(extra_presets)
    by_id = df.set_index("h3_r7", drop=False)
    log_cols = [c for c in df.columns if c.endswith("_log") or c.endswith("_exists")]
    cells = {}
    for key, hid in presets.items():
        row = by_id.loc[hid]
        payload = _cell_payload(row, df, feature_cols)
        payload["logged_values"] = {c: float(row[c]) for c in log_cols if c in row.index}
        payload["top_features"] = _feature_attribution(row, feature_cols)
        payload["district_context"] = _district_context(df, row["district"])
        if node_index is not None and traces is not None:
            idx = node_index.get(hid)
            if idx is not None and str(idx) in traces:
                payload["trace"] = traces[str(idx)]
        cells[key] = payload
    payload = {
        "presets": presets,
        "preset_rules": PRESET_RULES,
        "cells": cells,
        "feature_cols": list(feature_cols),
        "loss_curve": loss_histories or [],
        "steps": [
            {
                "id": "district",
                "title": "Official district GDP",
                "body": "Every district has one published GDDP total, Y_g. That number is the only monetary truth in the pipeline; the model never changes it.",
            },
            {
                "id": "cells",
                "title": "Split the district into H3 cells",
                "body": "The district polygon is filled with H3 resolution-7 hexagons (~5 km² each). Each hexagon becomes one node.",
            },
            {
                "id": "features",
                "title": "Attach features to each cell",
                "body": "Population (Kontur), night lights (VIIRS average_masked), OSM road km, building footprint area, landuse shares and POI counts by tag group. Magnitudes are split into an exists flag plus log10 value, then z-scored.",
            },
            {
                "id": "graph",
                "title": "Wire the graph",
                "body": "Each cell links to its 6 immediate H3 neighbours, and upward to its R6 and R5 parents. Economic activity spills across hexagon borders, so a cell is described partly by its surroundings.",
            },
            {
                "id": "message",
                "title": "Message passing",
                "body": "GraphSAGE averages neighbour embeddings and mixes them with the cell's own embedding; parent cells pool their children and broadcast context back down. A GCNII initial residual keeps the original features in play at every layer.",
            },
            {
                "id": "embedding",
                "title": "Node embedding",
                "body": "After 4 rounds the cell holds a 64-dimensional embedding that encodes both its own features and its neighbourhood.",
            },
            {
                "id": "score",
                "title": "Economic intensity",
                "body": "A linear readout plus softplus turns the embedding into s_i ≥ 0: an unitless per-person intensity, not rupees.",
            },
            {
                "id": "share",
                "title": "Normalised share",
                "body": "Weight the intensity by population, z_i = s_i × pop_i, and sum inside the district: Z_g. The dasymetric factor is scale_g = Y_g / Z_g.",
            },
            {
                "id": "final",
                "title": "Final cell GDP and wealth",
                "body": "gdp_i = z_i × scale_g, so cells in a district sum exactly to the official Y_g. Dividing by population gives ₹/person, and its within-state percentile rank is wealth_index.",
            },
        ],
        "formulas": [
            "s_i = softplus(GNN(x_i, neighbors, parents))  # intensity ≥ 0",
            "z_i = s_i * population_i",
            "Z_g = sum_{i in district g} z_i",
            "scale_g = Y_g / Z_g",
            "gdp_i = z_i * scale_g   # allocated ₹, not a measurement",
        ],
        "training": {
            "loss": "mean over districts of (log10 Σ_i s_i·pop_i − log10 Y_g)² + 0.05 · population-weighted floor hinge",
            "note": "The loss only ever sees district totals. Individual cell values are unidentified, which is why the range across seeds matters.",
        },
        "disclaimer": "Cell GDP is an allocation of official district GDP, not a measurement. Ensemble range is across random seeds, not a calibrated probability of error. wealth_index is not a DHS score. NTL is in the model when the average_masked composite is present.",
    }
    path = output_dir / "explainer.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
