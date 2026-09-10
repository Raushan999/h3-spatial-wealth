"""Build per-state web assets: compact H3 JSON, catalog, CSV copy, dictionary.

Layout written under `web/data/`:

    catalog.json              metric definitions + one entry per state
    {CODE}/summary.json       centre, districts, metros, source year, availability
    {CODE}/cells_r6.json      whole-state overview grid (fast first paint)
    {CODE}/r7/{slug}.json     full R7 detail, one file per district (loaded on demand)
    {CODE}/cells.csv[.gz]     every R7 cell with every column
    {CODE}/explainer.json     walkthrough payload written by the training run
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import h3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_ROOT = ROOT / "data" / "output"
WEB_DATA = ROOT / "web" / "data"

from src.config import LANDUSE_CLASSES, POI_GROUPS, STATES

# Short keys keep the JSON small; the UI maps them back to labels.
FIELDS: dict[str, str] = {
    "district": "d",
    "population": "p",
    "night_lights": "nl",
    "road_km": "rd",
    "poi_count": "poi",
    "building_area_m2": "bldg",
    "building_count": "bcnt",
    "building_share": "bsh",
    "area_km2": "ar",
    "wealth_index": "w",
    "wealth_index_min": "wmin",
    "wealth_index_max": "wmax",
    "gdp_per_capita_inr": "g",
    "gdp_per_capita_inr_min": "gmin",
    "gdp_per_capita_inr_max": "gmax",
    "gdp_crore": "c",
    "intensity_raw": "sr",
    "gdp_inr_raw": "zr",
    "dasymetric_scale": "s",
    "y_crore": "y",
}
for _cls in LANDUSE_CLASSES:
    FIELDS[f"landuse_{_cls}"] = f"lu_{_cls[:3]}"
for _grp in POI_GROUPS:
    FIELDS[f"poi_{_grp}"] = "pg_" + re.sub(r"[^a-z]", "", _grp)[:10]

ROUND = {
    "p": 1, "nl": 3, "rd": 2, "poi": 0, "bldg": 0, "bcnt": 0, "bsh": 4, "ar": 3,
    "w": 2, "wmin": 2, "wmax": 2, "c": 4, "sr": 6, "zr": 0, "s": 8, "y": 2,
}
INT_FIELDS = {"g", "gmin", "gmax"}
# Constant within a district, so these are hoisted out of the per-cell columns.
PER_DISTRICT = ("y", "s")

# How an R7 column collapses to its R6 parent for the overview grid.
AGG = {
    "district": "first", "population": "sum", "night_lights": "mean", "road_km": "sum",
    "poi_count": "sum", "building_area_m2": "sum", "building_count": "sum",
    "building_share": "mean", "area_km2": "sum", "wealth_index": "mean",
    "wealth_index_min": "min", "wealth_index_max": "max",
    "gdp_per_capita_inr": "median", "gdp_per_capita_inr_min": "min",
    "gdp_per_capita_inr_max": "max", "gdp_crore": "sum", "intensity_raw": "mean",
    "gdp_inr_raw": "sum", "dasymetric_scale": "mean", "y_crore": "first",
}
for _cls in LANDUSE_CLASSES:
    AGG[f"landuse_{_cls}"] = "mean"
for _grp in POI_GROUPS:
    AGG[f"poi_{_grp}"] = "sum"

BASE_METRICS = [
    {"id": "wealth_index", "label": "Wealth index", "units": "percentile 0â€“100", "group": "Model output", "default": True},
    {"id": "gdp_per_capita_inr", "label": "Allocated GDDP per capita", "units": "â‚¹/person", "group": "Model output"},
    {"id": "gdp_crore", "label": "Allocated cell GDDP", "units": "â‚¹ crore", "group": "Model output"},
    {"id": "intensity_raw", "label": "Economic intensity s_i (pre-scale)", "units": "index", "group": "Model output"},
    {"id": "population", "label": "Population", "units": "people", "group": "Input"},
    {"id": "night_lights", "label": "Night lights", "units": "nW/cmÂ²/sr", "group": "Input"},
    {"id": "road_km", "label": "Road length", "units": "km", "group": "Input"},
    {"id": "poi_count", "label": "OSM POIs (all groups)", "units": "count", "group": "Input"},
    {"id": "building_area_m2", "label": "Building footprint area", "units": "mÂ²", "group": "Input"},
    {"id": "building_count", "label": "Building count", "units": "buildings", "group": "Input"},
    {"id": "building_share", "label": "Built-up share of cell", "units": "share 0â€“1", "group": "Input"},
]
POI_LABELS = {
    "amenity_restaurant": "Restaurants", "amenity_cafe": "CafÃ©s", "amenity_bar": "Bars",
    "amenity_pub": "Pubs", "amenity_cinema": "Cinemas", "amenity_theatre": "Theatres",
    "amenity_nightclub": "Nightclubs", "amenity_university": "Universities",
    "amenity_hospital": "Hospitals", "amenity_bank": "Banks",
    "shop_mall": "Malls / department stores", "shop_luxury": "Jewellery & boutiques",
    "shop_supermarket": "Supermarkets", "shop_convenience": "Convenience stores",
    "tourism_hotel": "Hotels", "tourism_guest_house": "Guest houses",
    "leisure": "Leisure sites", "office": "Offices",
}
LANDUSE_LABELS = {c: f"Landuse: {c}" for c in LANDUSE_CLASSES}

# A metric is offered only if the underlying column is populated often enough to
# colour a map. Sparse OSM classes stay in the CSV but are not put in the picker.
MIN_NONZERO_SHARE = 0.005


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "unknown"


def _find_state_tables() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not OUT_ROOT.exists():
        return found
    for child in sorted(OUT_ROOT.iterdir()):
        if not child.is_dir():
            continue
        for name in (
            f"{child.name.lower()}_h3_r7_wealth.parquet",
            f"{child.name.lower()}_h3_r7_wealth.csv",
        ):
            p = child / name
            if p.exists():
                found[child.name.upper()] = p
                break
    return found


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _column(df: pd.DataFrame, col: str, key: str) -> list:
    series = pd.to_numeric(df[col], errors="coerce")
    if key in INT_FIELDS:
        return [None if pd.isna(v) else int(round(v)) for v in series]
    nd = ROUND.get(key)
    if nd == 0:
        return [None if pd.isna(v) else int(round(v)) for v in series]
    return [None if pd.isna(v) else (round(float(v), nd) if nd is not None else float(v)) for v in series]


def pack_columns(df: pd.DataFrame, id_col: str, res: int, district: str | None = None) -> dict:
    """Columnar payload: one array per field instead of one object per cell.

    Repeating ~40 short keys on every one of a quarter-million cells was most of
    the JSON. Storing columns and rebuilding row objects in the browser keeps the
    same fields available in the popup at roughly a third of the bytes.
    """
    cols: dict[str, list] = {"h": df[id_col].astype(str).tolist()}
    payload: dict = {"res": res, "n": int(len(df)), "columnar": True}

    if district is not None:
        payload["district"] = district
    elif "district" in df.columns:
        names = sorted({str(d) for d in df["district"].fillna("").unique()})
        idx = {n: i for i, n in enumerate(names)}
        payload["d_vals"] = names
        cols["d"] = [idx[str(d)] for d in df["district"].fillna("")]

    # y_crore and the dasymetric scale are district-level constants.
    keyed = {FIELDS[c]: c for c in ("y_crore", "dasymetric_scale") if c in df.columns}
    if keyed and "district" in df.columns:
        by_district: dict[str, dict[str, float | None]] = {}
        for name, part in df.groupby("district"):
            entry = {}
            for key, col in keyed.items():
                v = pd.to_numeric(part[col], errors="coerce").dropna()
                entry[key] = round(float(v.iloc[0]), ROUND.get(key, 6)) if len(v) else None
            by_district[str(name)] = entry
        payload["per_district"] = by_district

    for col, key in FIELDS.items():
        if col not in df.columns or key == "d" or key in PER_DISTRICT:
            continue
        vals = _column(df, col, key)
        if all(v is None or v == 0 for v in vals):
            continue
        cols[key] = vals
    payload["cols"] = cols
    return payload


def available_metrics(df: pd.DataFrame) -> list[dict]:
    metrics: list[dict] = []
    n = max(len(df), 1)

    def populated(col: str) -> bool:
        if col not in df.columns:
            return False
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return float((s != 0).sum()) / n >= MIN_NONZERO_SHARE

    for m in BASE_METRICS:
        if m["id"] in {"wealth_index", "gdp_per_capita_inr", "population"} or populated(m["id"]):
            metrics.append(dict(m))
    for cls in LANDUSE_CLASSES:
        col = f"landuse_{cls}"
        if populated(col):
            metrics.append({"id": col, "label": LANDUSE_LABELS[cls], "units": "share of cell area", "group": "Landuse"})
    for grp in POI_GROUPS:
        col = f"poi_{grp}"
        if populated(col):
            metrics.append({"id": col, "label": POI_LABELS.get(grp, grp), "units": "count", "group": "POI class"})
    return metrics


def write_json(path: Path, obj, indent=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":") if indent is None else None, indent=indent), encoding="utf-8")


def prepare_state(code: str, table: Path) -> dict:
    df = _read_table(table)
    dest = WEB_DATA / code
    dest.mkdir(parents=True, exist_ok=True)
    out_dir = table.parent

    # --- R6 overview grid -------------------------------------------------
    agg = {c: (c, how) for c, how in AGG.items() if c in df.columns}
    grouped = df.groupby("h3_r6", as_index=False).agg(**agg)
    write_json(dest / "cells_r6.json", pack_columns(grouped, "h3_r6", res=6))
    n_r6 = int(len(grouped))

    # --- R7 detail, split per district so the UI can load just one --------
    r7_dir = dest / "r7"
    if r7_dir.exists():
        shutil.rmtree(r7_dir)
    r7_dir.mkdir(parents=True, exist_ok=True)
    district_files: dict[str, str] = {}
    for district, part in df.groupby("district"):
        name = slug(district)
        write_json(r7_dir / f"{name}.json", pack_columns(part, "h3_r7", res=7, district=str(district)))
        district_files[str(district)] = f"r7/{name}.json"

    # --- full CSV ---------------------------------------------------------
    src_csv = table if table.suffix == ".csv" else table.with_suffix(".csv")
    if src_csv.exists():
        raw = src_csv.read_bytes()
        (dest / "cells.csv").write_bytes(raw)
        with gzip.open(dest / "cells.csv.gz", "wb") as f:
            f.write(raw)
    else:
        df.to_csv(dest / "cells.csv", index=False)

    for name in ("DATA_DICTIONARY.md", "explainer.json", "run_summary.json", "layer_trace.json"):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)

    run = {}
    if (dest / "run_summary.json").exists():
        run = json.loads((dest / "run_summary.json").read_text(encoding="utf-8"))

    cfg = STATES.get(code)
    districts = sorted({str(d) for d in df["district"].dropna().unique()})
    metros = {}
    if cfg:
        for mname, dlist in cfg.metros.items():
            present = [d for d in dlist if d in districts]
            if present:
                metros[mname] = present

    dist_rows = []
    for district, part in df.groupby("district"):
        dist_rows.append(
            {
                "district": str(district),
                "n_cells": int(len(part)),
                "population": float(part["population"].sum()),
                "y_crore": float(part["y_crore"].iloc[0]) if "y_crore" in part.columns else None,
                "gdp_crore": float(part["gdp_crore"].sum()) if "gdp_crore" in part.columns else None,
                "median_gdp_per_capita_inr": float(part["gdp_per_capita_inr"].median())
                if "gdp_per_capita_inr" in part.columns
                else None,
            }
        )
    dist_rows.sort(key=lambda r: -(r["y_crore"] or 0))

    gd = cfg.gddp if cfg and cfg.gddp else None
    summary = {
        "state": code,
        "name": cfg.name if cfg else code,
        "n_r7": int(len(df)),
        "n_r6": n_r6,
        "districts": districts,
        "district_files": district_files,
        "district_table": dist_rows,
        "metros": metros,
        "center": list(cfg.center) if cfg else [78.0, 22.0],
        "zoom": cfg.zoom if cfg else 5.5,
        "metrics": available_metrics(df),
        "source": {
            "gddp_year": gd.year if gd else run.get("gddp_year"),
            "price_basis": gd.price_basis if gd else run.get("price_basis"),
            "base_year": gd.base_year if gd else None,
            "units": gd.units if gd else "INR crore",
            "page": gd.page if gd else None,
            "notes": list(gd.notes) if gd else [],
            "total_crore": run.get("gddp_total_crore"),
            "population_product": run.get("population_product"),
            "total_population": run.get("total_population"),
            "nightlights_product": run.get("night_lights_product"),
            "nightlights_year": run.get("night_lights_year"),
            "nightlights_used": run.get("night_lights_used", bool(
                "night_lights" in df.columns and float(pd.to_numeric(df["night_lights"], errors="coerce").fillna(0).sum()) > 0
            )),
            "osm": "OpenStreetMap via Geofabrik extract (ODbL)",
        },
        "evaluation": {
            "max_abs_rel_recon_error": run.get("max_abs_rel_recon_error"),
            "ensemble": run.get("ensemble"),
            "holdout": (run.get("evaluation") or {}).get("holdout"),
            "ablation_ntl": (run.get("evaluation") or {}).get("ablation_ntl"),
            "spearman": (run.get("evaluation") or {}).get("spearman"),
        },
        "caption": "ensemble range across seeds, not a calibrated probability of error.",
        "disclaimer": "Cell GDP is an allocation of official district GDP, not a measurement. wealth_index is not a DHS score.",
    }
    write_json(dest / "summary.json", summary, indent=2)
    print(
        f"wrote {dest.name}: r7={len(df):,} r6={n_r6:,} districts={len(districts)} "
        f"metrics={len(summary['metrics'])} year={summary['source']['gddp_year']}"
    )
    return summary


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build web/data from data/output/{STATE}.")
    ap.add_argument(
        "--states",
        nargs="*",
        help="Rebuild only these state codes. The catalog still lists every state already in web/data.",
    )
    args = ap.parse_args()

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    tables = _find_state_tables()
    if not tables:
        raise SystemExit("No state output tables under data/output/. Run python run_state.py --state MH first.")
    wanted = {c.upper() for c in (args.states or [])}
    todo = {c: p for c, p in tables.items() if not wanted or c in wanted}
    if wanted and not todo:
        raise SystemExit(f"No output tables for {sorted(wanted)}; found {sorted(tables)}")
    states = [prepare_state(code, path) for code, path in sorted(todo.items())]
    # States skipped this run keep the summary they already have on disk.
    for code in sorted(set(tables) - set(todo)):
        prev = WEB_DATA / code / "summary.json"
        if prev.exists():
            states.append(json.loads(prev.read_text(encoding="utf-8")))
    states.sort(key=lambda s: s["state"])
    metric_defs: dict[str, dict] = {}
    for s in states:
        for m in s["metrics"]:
            metric_defs.setdefault(m["id"], m)
    catalog = {
        "metrics": list(metric_defs.values()),
        "states": [
            {
                "state": s["state"],
                "name": s["name"],
                "center": s["center"],
                "zoom": s["zoom"],
                "n_r7": s["n_r7"],
                "gddp_year": s["source"]["gddp_year"],
                "price_basis": s["source"]["price_basis"],
            }
            for s in states
        ],
        "default_metric": "wealth_index",
        "colour_mode_default": "percentile",
        "copy": {
            "ntl": "Night-time lights (VIIRS average_masked) are in the model wherever the composite covers the state.",
            "dhs": "wealth_index is a within-state percentile of allocated GDDP per capita, not a DHS wealth score.",
            "allocation": "Cell GDP is an allocation of official district GDP, not a measurement.",
            "ensemble": "ensemble range across seeds, not a calibrated probability of error.",
            "absolute_warning": "Absolute colour scales use native units (â‚¹/person, people, nW/cmÂ²/sr, â€¦). States are estimated for different years and price bases, so do not compare two states on one absolute scale.",
            "years": "Each state is disaggregated from its own official GDDP release, so the reference year differs by state. The year is shown next to the state name.",
        },
    }
    write_json(WEB_DATA / "catalog.json", catalog, indent=2)
    dict_src = next(
        (WEB_DATA / s["state"] / "DATA_DICTIONARY.md" for s in states
         if (WEB_DATA / s["state"] / "DATA_DICTIONARY.md").exists()),
        None,
    )
    if dict_src:
        shutil.copy2(dict_src, WEB_DATA / "DATA_DICTIONARY.md")
    print("catalog states", [(s["state"], s["source"]["gddp_year"]) for s in states])


if __name__ == "__main__":
    main()
