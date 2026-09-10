"""Train one hierarchical GNN per state: python run_state.py --state MH"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (
    EPOCHS,
    NIGHTLIGHTS_PRODUCT,
    NIGHTLIGHTS_YEAR,
    POPULATION_SOURCE,
    SEEDS,
    STATES,
    get_state,
    output_dir,
)
from src.evaluate import ntl_ablation, spatial_holdout_mape, spearman_sanity
from src.explainer import write_explainer_artifact
from src.features import (
    aggregate_nightlights,
    aggregate_osm,
    aggregate_population,
    attach_targets,
    build_h3_index,
    engineer_features,
    load_district_polygons,
    load_gddp,
    save_processed,
)
from src.fetch import fetch_all
from src.model import prepare_tensors, train_ensemble


EXPORT_ALWAYS = [
    "h3_r7",
    "h3_r6",
    "h3_r5",
    "district",
    "lat",
    "lon",
    "area_km2",
    "population",
    "population_source",
    "night_lights",
    "night_lights_source",
    "road_km",
    "poi_count",
    "building_area_m2",
    "building_count",
    "building_share",
    "intensity_raw",
    "intensity_raw_min",
    "intensity_raw_max",
    "dasymetric_scale",
    "gdp_inr_raw",
    "gdp_inr",
    "gdp_crore",
    "gdp_per_capita_inr",
    "gdp_per_capita_inr_mean",
    "gdp_per_capita_inr_min",
    "gdp_per_capita_inr_max",
    "wealth_index",
    "wealth_index_mean",
    "wealth_index_min",
    "wealth_index_max",
    "wealth_quintile",
    "y_crore",
    "y_inr",
    "gddp_year",
    "price_basis",
    "gddp_units",
]


def write_coverage_md() -> Path:
    from src.config import RAW

    lines = [
        "# State GDDP coverage",
        "",
        "Official open government / OpenCity / data.gov.in only. Paywalled tables (Dataful etc.) are skipped.",
        "",
        "| code | state | status | year | units | prices | source | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for code, st in STATES.items():
        src = st.gddp
        gdir = RAW / "gddp" / code
        has_csv = (gdir / "gddp.csv").exists()
        has_json = bool(src) and (gdir / src.filename).exists() and (gdir / src.filename).suffix.lower() == ".json"
        has_parsed = (gdir / "gddp.json").exists()
        has_pdf = bool(src) and (gdir / src.filename).exists() and (gdir / src.filename).suffix.lower() == ".pdf"
        if src is None or (st.skip_reason and not has_csv and not has_parsed):
            status = "skip"
        elif src.kind == "local_csv" and not has_csv:
            status = "skip"
        elif src.kind == "pdf" and not (has_csv or has_parsed):
            status = "skip (PDF persisted; needs gddp.csv)" if has_pdf else "skip (DES PDF download failed; drop-in CSV)"
        elif has_csv or has_json or has_parsed or src.kind in {"ckan_json", "opencity_csv", "gsdp_single"}:
            status = "in"
        else:
            status = "in (configured; run fetch)"
        year = src.year if src else ""
        units = src.units if src else ""
        prices = f"{src.price_basis} {src.base_year or ''}".strip() if src else ""
        page = src.page if src else ""
        notes = "; ".join(src.notes) if src else (st.skip_reason or "")
        if st.dissolve_to_single:
            notes += f" Reporting unit dissolved to `{st.dissolve_to_single}` (official GSDP, not an invented NCR super-district)."
        lines.append(
            f"| {code} | {st.name} | {status} | {year} | {units} | {prices} | {page} | {notes} |"
        )
    lines += [
        "",
        "## Merges",
        "",
        "- **MH:** Mumbai City + Mumbai Suburban → Mumbai; Thane + Palghar → Thane; Aurangabad → Chhatrapati Sambhajinagar; Osmanabad → Dharashiv.",
        "- **DL:** 11 NCT revenue districts dissolved to one Y_g = NCT GSDP because DES Delhi does not publish district GDDP.",
        "- **NCR in the UI:** Haryana/UP/Rajasthan NCR districts appear only if those states' GDDP loaded. No synthetic NCR district.",
        "- **WB / Kolkata:** Kolkata is a city; the model (if loaded) is West Bengal districts. Metro filter is UI-only.",
        "",
        "Drop-in for PDF-only DES tables: `data/raw/gddp/{STATE}/gddp.csv` with columns `district,y_crore`.",
        "",
    ]
    path = RAW / "gddp" / "STATE_COVERAGE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_data_dictionary(path: Path, state_code: str, feature_cols: list[str]) -> None:
    text = f"""# Data dictionary — {state_code} H3 R7 cells

Cell GDP is an **allocation** of official district GDP (`Y_g`), not a measurement.
Night-time lights (VIIRS **average_masked**) are in the model when the raster is present.
`wealth_index` is a within-state percentile of allocated GDDP per capita. It is **not** a DHS wealth score.

## Identifiers

| column | meaning |
|---|---|
| h3_r7 | H3 index at resolution 7 (~5 km²) |
| h3_r6 / h3_r5 | parent cells used by the hierarchical GNN |
| district | reporting unit matched to official GDDP (after configured merges) |
| lat, lon | cell centroid |
| area_km2 | H3 cell area |

## Inputs (persisted under data/raw/)

| column | units | source |
|---|---|---|
| population | people | Kontur India 2023 (H3; R8 summed to R7). WorldPop 2020 is a config fallback. |
| night_lights | nW/cm²/sr | VIIRS VNL v2.2 **average_masked** mean radiance |
| road_km | km | OSM major highways |
| poi_count | count | OSM POIs (amenity/shop/tourism/leisure/office tag groups) |
| poi_* | count | class-specific OSM counts (premium vs mass proxies). Sparse classes stay mostly zero. |
| poi_*_density | count/km² | optional density |
| building_area_m2 | m² | sum of OSM building footprints |
| landuse_* | share of cell area | OSM landuse residential/commercial/industrial/retail |

## Engineered (paper §4.2)

Zero-inflation split: `{{col}}_exists` and `{{col}}_log` = log10(value) if value>0 else 0, then z-scored (`*_z`).
Feature columns used by the GNN: {", ".join(feature_cols)}

## Model outputs

| column | meaning |
|---|---|
| intensity_raw | ensemble-mean softplus intensity `s_i` |
| intensity_raw_min / max | range across 5 random seeds |
| gdp_inr_raw | `s_i * population` before scale |
| dasymetric_scale | `Y_g / sum_i z_i` inside the district |
| gdp_inr / gdp_crore | allocated cell GDP after scale (sums to official district `Y_g`) |
| gdp_per_capita_inr | allocated ₹/person (ensemble central estimate) |
| gdp_per_capita_inr_min / max | min–max across seeds |
| wealth_index | within-state percentile of per-capita intensity, 0–100 |
| wealth_index_min / max | min–max of per-seed percentile ranks |
| y_crore / y_inr | official district GDDP (`Y_g`) |

Ensemble caption: **ensemble range across seeds, not a calibrated probability of error.**
Post-scale district reconstruction error ≈ 0 is conservation, not accuracy.
"""
    path.write_text(text, encoding="utf-8")


def run(state_code: str, args: argparse.Namespace) -> None:
    state = get_state(state_code)
    out_dir = output_dir(state.code)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {state.code} {state.name} ==", flush=True)

    if not args.skip_fetch:
        print("== 1. Fetch raw datasets ==", flush=True)
        paths = fetch_all(state.code)
    else:
        from src.fetch import (
            fetch_boundaries,
            fetch_gddp,
            fetch_kontur_population,
            fetch_osm_pbf,
            find_nightlights,
        )

        fetch_boundaries()
        paths = {
            "gddp": fetch_gddp(state),
            "population": fetch_kontur_population() if args.population == "kontur" else None,
            "nightlights": find_nightlights(),
            "osm": fetch_osm_pbf(state),
        }
        if args.population == "worldpop":
            from src.fetch import fetch_worldpop

            paths["population"] = fetch_worldpop()
    print({k: str(v) if v is not None else None for k, v in paths.items()})
    write_coverage_md()

    if paths.get("gddp") is None:
        raise RuntimeError(
            f"No official open GDDP table for {state.code}. See data/raw/gddp/STATE_COVERAGE.md. "
            f"Place data/raw/gddp/{state.code}/gddp.csv to load a local official extract."
        )

    print("== 2. District polygons + GDDP targets ==")
    districts = load_district_polygons(state)
    gddp = load_gddp(state)
    print(f"boundary units={len(districts)}  gddp units={len(gddp)}")
    print("districts:", sorted(districts["district"].tolist()))

    print("== 3. H3 R7 index ==")
    cells = build_h3_index(districts)
    print(f"R7 cells={len(cells)}")

    print("== 4. Features ==")
    pop_path = paths["population"]
    if pop_path is None:
        raise RuntimeError("Population layer missing")
    cells = aggregate_population(cells, pop_path, state, source=args.population)
    cells = aggregate_nightlights(cells, paths["nightlights"], state.bbox)
    osm_path = paths["osm"]
    if osm_path is None or not Path(osm_path).exists():
        raise RuntimeError("OSM extract missing")
    cells = aggregate_osm(cells, osm_path, state)
    cells = attach_targets(cells, gddp)
    cells = engineer_features(cells)
    feature_cols = cells.attrs["feature_cols"]
    save_processed(cells, state.code)
    print("features:", feature_cols)
    print(cells[["population", "road_km", "poi_count", "building_area_m2", "night_lights"]].describe())

    print("== 5. Train 5-seed hierarchical GNN ==")
    data = prepare_tensors(cells, feature_cols)
    out, ens_meta = train_ensemble(
        data, cells, epochs=args.epochs, seeds=tuple(args.seeds), output_dir=out_dir
    )

    print("== 6. Evaluation ==")
    eval_block = {"ensemble": ens_meta}
    if not args.skip_eval:
        eval_block["holdout"] = spatial_holdout_mape(
            cells, feature_cols, epochs=args.epochs, output_dir=out_dir
        )
        eval_block["ablation_ntl"] = ntl_ablation(
            cells, feature_cols, epochs=args.epochs, output_dir=out_dir
        )
    eval_block["spearman"] = spearman_sanity(out)
    print(json.dumps(eval_block, indent=2, default=str))

    print("== 7. Export ==")
    extra = [c for c in out.columns if c.startswith("poi_") or c.startswith("landuse_") or c.endswith("_z") or c.endswith("_log") or c.endswith("_exists")]
    cols = [c for c in EXPORT_ALWAYS if c in out.columns] + [c for c in extra if c not in EXPORT_ALWAYS]
    export = out.loc[:, ~out.columns.duplicated()][cols].sort_values(["district", "h3_r7"])
    csv_path = out_dir / f"{state.code.lower()}_h3_r7_wealth.csv"
    parquet_path = out_dir / f"{state.code.lower()}_h3_r7_wealth.parquet"
    export.to_csv(csv_path, index=False)
    export.to_parquet(parquet_path, index=False)
    write_data_dictionary(out_dir / "DATA_DICTIONARY.md", state.code, feature_cols)
    traces, node_index = _capture_traces(out, data, feature_cols, out_dir, args.seeds[0])
    write_explainer_artifact(
        out,
        feature_cols,
        out_dir,
        loss_histories=_load_loss(out_dir),
        traces=traces,
        node_index=node_index,
    )

    recon = export.groupby("district").agg(pred=("gdp_inr", "sum"), obs=("y_inr", "first"))
    recon["rel_err"] = (recon["pred"] - recon["obs"]).abs() / recon["obs"]
    summary = {
        "state": state.code,
        "state_name": state.name,
        "n_cells": int(len(export)),
        "n_districts": int(export["district"].nunique()),
        "max_abs_rel_recon_error": float(recon["rel_err"].max()),
        "mean_gdp_per_capita_inr": float(export["gdp_per_capita_inr"].mean()),
        "output_csv": str(csv_path),
        "night_lights_used": bool(float(export["night_lights"].fillna(0).sum()) > 0),
        "night_lights_product": NIGHTLIGHTS_PRODUCT,
        "night_lights_year": NIGHTLIGHTS_YEAR,
        "population_source": args.population,
        "population_product": "Kontur Population India 2023 (H3 R8 summed to R7)"
        if args.population == "kontur"
        else "WorldPop 2020 1 km",
        "total_population": float(export["population"].sum()),
        "osm_extract": str(paths.get("osm")),
        "feature_cols": feature_cols,
        "ensemble": ens_meta,
        "evaluation": eval_block,
        "gddp_year": state.gddp.year if state.gddp else None,
        "price_basis": state.gddp.price_basis if state.gddp else None,
        "base_year": state.gddp.base_year if state.gddp else None,
        "gddp_units": state.gddp.units if state.gddp else None,
        "gddp_source_page": state.gddp.page if state.gddp else None,
        "gddp_notes": list(state.gddp.notes) if state.gddp else [],
        "gddp_total_crore": float(export.groupby("district")["y_crore"].first().sum()),
        "caption": "ensemble range across seeds, not a calibrated probability of error.",
        "disclaimer": "Cell GDP is an allocation of official district GDP, not a measurement. wealth_index is not DHS.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    # legacy alias for MH
    if state.code == "MH":
        from src.config import OUTPUT

        (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_cells", "n_districts", "max_abs_rel_recon_error", "caption")}, indent=2))
    print("wrote", csv_path)


def _capture_traces(out, data, feature_cols: list[str], out_dir: Path, seed: int):
    """Replay one trained checkpoint to log real activations for the walkthrough."""
    import torch

    from src.explainer import _pick_presets
    from src.model import HierarchicalGNN, trace_nodes

    ckpt_path = out_dir / f"gnn_ensemble_seed{seed}.pt"
    if not ckpt_path.exists():
        return None, None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = HierarchicalGNN(in_dim=data["x7"].shape[1])
        model.load_state_dict(ckpt["state_dict"])
        node_index = {h: i for i, h in enumerate(out["h3_r7"].tolist())}
        presets = _pick_presets(out)
        idx = [node_index[h] for h in presets.values() if h in node_index]
        traces = trace_nodes(model, data, idx)
        (out_dir / "layer_trace.json").write_text(
            json.dumps({"seed": seed, "presets": presets, "traces": traces}, indent=2),
            encoding="utf-8",
        )
        return traces, node_index
    except Exception as exc:  # tracing is diagnostic only; never fail a run for it
        print("layer trace skipped:", exc)
        return None, None


def _load_loss(out_dir: Path) -> list:
    path = out_dir / "loss_ensemble.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("histories") or []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hierarchical GNN spatial disaggregation for one Indian state")
    p.add_argument("--state", default="MH", help="State code, e.g. MH, KA, TG, DL")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--population", choices=["kontur", "worldpop"], default=POPULATION_SOURCE)
    p.add_argument("--osm-step", type=float, default=1.0, help="Overpass tile size in degrees")
    p.add_argument("--skip-fetch", action="store_true")
    p.add_argument("--skip-osm", action="store_true")
    p.add_argument("--skip-eval", action="store_true", help="Skip hold-out and ablation retrains")
    p.add_argument("--list-states", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list_states:
        for code, st in STATES.items():
            flag = "skip" if st.skip_reason else "in"
            print(f"{code:3}  {st.name:24}  {flag}  {st.skip_reason or (st.gddp.year if st.gddp else '')}")
        write_coverage_md()
        return
    run(args.state, args)


if __name__ == "__main__":
    main()
