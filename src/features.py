"""H3 R7 feature table: population, lights, OSM, landuse, POIs, GDDP targets."""

from __future__ import annotations

import gzip
import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.windows import from_bounds
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from tqdm import tqdm

import h3

from .config import (
    H3_RES,
    LANDUSE_CLASSES,
    MAGNITUDE_FEATURE_BASE,
    NIGHTLIGHTS_H3_PARQUET,
    NIGHTLIGHTS_H3_VALUE,
    NIGHTLIGHTS_YEAR,
    POI_GROUPS,
    POPULATION_SOURCE,
    PROCESSED,
    RAW,
    SHARE_FEATURE_BASE,
    StateConfig,
    processed_dir,
    raw_layer,
)

GEOD = Geod(ellps="WGS84")


def _fold(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def _norm_name(name: str, rename_map: dict[str, str]) -> str:
    s = re.sub(r"[#$\*+]+", "", _fold(name))
    s = re.sub(r"\s+", " ", s).strip()
    if s in rename_map:
        return rename_map[s]
    return s.title() if s else s


def parse_inr_number(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    text = str(value).replace(",", "").replace("₹", "").strip()
    text = re.sub(r"[^\d.\-eE]", "", text)
    if text in {"", "-", "."}:
        return float("nan")
    return float(text)


def _load_geojson_features(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["features"]


def _props(feat: dict) -> dict:
    return feat.get("properties") or {}


def _admin_name(props: dict) -> str:
    for key in ("shapeName", "shapeNAME", "NAME_1", "NAME_2", "name", "shapeISO"):
        if props.get(key):
            return str(props[key])
    return ""


def _adm_paths() -> tuple[Path, Path]:
    adm1_candidates = [
        raw_layer("boundaries", "IN") / "geoBoundaries-IND-ADM1.geojson",
        raw_layer("boundaries", "IN") / "geoBoundaries-IND-adm1.geojson",
        RAW / "boundaries" / "geoBoundaries-IND-adm1.geojson",
    ]
    adm2_candidates = [
        raw_layer("boundaries", "IN") / "geoBoundaries-IND-ADM2.geojson",
        raw_layer("boundaries", "IN") / "geoBoundaries-IND-adm2.geojson",
        RAW / "boundaries" / "geoBoundaries-IND-adm2.geojson",
    ]
    adm1 = next((p for p in adm1_candidates if p.exists()), None)
    adm2 = next((p for p in adm2_candidates if p.exists()), None)
    if adm1 is None or adm2 is None:
        raise FileNotFoundError("geoBoundaries ADM1/ADM2 GeoJSON missing under data/raw/boundaries/")
    return adm1, adm2


def state_polygon(state: StateConfig) -> MultiPolygon:
    adm1, _ = _adm_paths()
    wanted = {_fold(n) for n in state.adm1_names}
    found = None
    for f in _load_geojson_features(adm1):
        if _fold(_admin_name(_props(f))) in wanted:
            found = shape(f["geometry"])
            break
    if found is None:
        raise RuntimeError(f"{state.name} not found in ADM1 geoBoundaries file (looked for {wanted})")
    if isinstance(found, Polygon):
        found = MultiPolygon([found])
    return found


def load_district_polygons(state: StateConfig) -> pd.DataFrame:
    poly = state_polygon(state)
    _, adm2 = _adm_paths()
    rows = []
    for f in _load_geojson_features(adm2):
        geom = shape(f["geometry"])
        if geom.is_empty:
            continue
        if not poly.contains(geom.centroid) and not poly.intersects(geom):
            continue
        raw_name = _admin_name(_props(f))
        district = _norm_name(raw_name, state.rename_map)
        if state.dissolve_to_single:
            district = state.dissolve_to_single
        rows.append({"district_raw": raw_name, "district": district, "geometry": geom})
    gdf = pd.DataFrame(rows)
    gdf = gdf[gdf["geometry"].apply(lambda g: poly.contains(g.centroid))]
    dissolved = []
    for district, part in gdf.groupby("district"):
        geoms = [g if g.is_valid else g.buffer(0) for g in part["geometry"].tolist()]
        merged = geoms[0]
        for g in geoms[1:]:
            merged = merged.union(g)
        if isinstance(merged, Polygon):
            merged = MultiPolygon([merged])
        dissolved.append({"district": district, "geometry": merged})
    return pd.DataFrame(dissolved)


def _records_from_ckan(payload: dict, state: StateConfig) -> list[dict]:
    src = state.gddp
    assert src is not None
    rows = []
    for rec in payload.get("result", payload).get("records", payload.get("records", [])):
        name = rec.get(src.district_field) or rec.get("District") or rec.get("district")
        if not name:
            continue
        if any(tok in str(name).lower() for tok in src.skip_tokens):
            continue
        value = rec.get(src.value_field)
        if value is None:
            continue
        y = parse_inr_number(value)
        rows.append(
            {
                "district": _norm_name(name, state.rename_map),
                "source_name": name,
                "y_crore": y,
                **{k: rec.get(k) for k in rec if k not in {src.district_field, src.value_field, "_id"}},
            }
        )
    return rows


def load_gddp(state: StateConfig) -> pd.DataFrame:
    src = state.gddp
    if src is None:
        raise RuntimeError(f"No GDDP source configured for {state.code}")
    out_dir = raw_layer("gddp", state.code)
    local_csv = out_dir / "gddp.csv"
    parsed = out_dir / "gddp.json"
    dest = out_dir / src.filename

    rows: list[dict] = []
    if local_csv.exists() and local_csv.stat().st_size > 0:
        raw = pd.read_csv(local_csv)
        name_col = "district" if "district" in raw.columns else raw.columns[0]
        val_col = "y_crore" if "y_crore" in raw.columns else raw.columns[1]
        for rec in raw.to_dict(orient="records"):
            rows.append(
                {
                    "district": _norm_name(rec[name_col], state.rename_map),
                    "source_name": rec[name_col],
                    "y_crore": parse_inr_number(rec[val_col]),
                }
            )
    elif parsed.exists() and src.kind in {
        "gsdp_single",
        "records_json",
        "pdf",
        "pdf_table",
        "xlsx_matrix",
        "xlsx_table",
    }:
        payload = json.loads(parsed.read_text(encoding="utf-8"))
        for rec in payload.get("records", []):
            rows.append(
                {
                    "district": _norm_name(rec.get("district") or rec.get("District"), state.rename_map),
                    "source_name": rec.get("source_name") or rec.get("district"),
                    "y_crore": parse_inr_number(rec.get("y_crore")),
                }
            )
    elif dest.exists() and dest.suffix.lower() == ".json":
        payload = json.loads(dest.read_text(encoding="utf-8"))
        rows = _records_from_ckan(payload, state)
    elif dest.exists() and dest.suffix.lower() == ".pdf":
        raise FileNotFoundError(
            f"Official GDDP PDF saved at {dest} but no machine-readable table. "
            f"Place {local_csv} with columns district,y_crore (from the DES publication) to enable {state.code}."
        )
    else:
        raise FileNotFoundError(
            f"No GDDP table for {state.code}. Expected {dest} or {local_csv}. "
            "Place an official open CSV at data/raw/gddp/{state}/gddp.csv if the download was skipped."
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"GDDP parsed to zero rows for {state.code}")
    df = df.dropna(subset=["y_crore"])
    df = df[df["y_crore"] > 0]
    df = df.groupby("district", as_index=False).agg(
        y_crore=("y_crore", "sum"),
        source_name=("source_name", "first"),
    )
    df["y_inr"] = df["y_crore"] * 1e7
    df["price_basis"] = src.price_basis
    df["gddp_year"] = src.year
    df["gddp_units"] = src.units
    return df


def build_h3_index(districts: pd.DataFrame) -> pd.DataFrame:
    records = []
    seen = set()
    for rec in tqdm(districts.itertuples(index=False), total=len(districts), desc="H3 polyfill"):
        geom = rec.geometry.buffer(0)
        try:
            cells = h3.geo_to_cells(mapping(geom), H3_RES)
        except Exception:
            cells = []
            for poly in getattr(geom, "geoms", [geom]):
                cells.extend(h3.geo_to_cells(mapping(poly), H3_RES))
        for cell in cells:
            if cell in seen:
                continue
            lat, lon = h3.cell_to_latlng(cell)
            pt = Point(lon, lat)
            if not geom.contains(pt) and not geom.covers(pt):
                continue
            seen.add(cell)
            records.append(
                {
                    "h3_r7": cell,
                    "district": rec.district,
                    "lat": lat,
                    "lon": lon,
                    "h3_r6": h3.cell_to_parent(cell, 6),
                    "h3_r5": h3.cell_to_parent(cell, 5),
                    "area_km2": float(h3.cell_area(cell, unit="km^2")),
                }
            )
    return pd.DataFrame(records).drop_duplicates("h3_r7")


def _iter_raster_to_h3(raster_path: Path, bbox, res: int = H3_RES):
    minx, miny, maxx, maxy = bbox
    acc = defaultdict(float)
    n = defaultdict(int)
    with rasterio.open(raster_path) as src:
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        window = window.round_lengths().round_offsets()
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        nodata = src.nodata
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata
        rows, cols = np.where(valid)
        for r, c in zip(rows, cols):
            x, y = rasterio.transform.xy(transform, r, c, offset="center")
            if x < minx or x > maxx or y < miny or y > maxy:
                continue
            cell = h3.latlng_to_cell(y, x, res)
            acc[cell] += float(data[r, c])
            n[cell] += 1
    return acc, n


def _maybe_unzip_tif(path: Path) -> Path:
    if path.suffix == ".gz" or str(path).endswith(".tif.gz"):
        out = path.with_suffix("").with_suffix(".tif") if path.name.endswith(".tif.gz") else path.with_suffix("")
        if not out.exists():
            with gzip.open(path, "rb") as src, out.open("wb") as dst:
                dst.write(src.read())
        return out
    return path


def _assert_raster_covers_bbox(raster_path: Path, bbox, label: str) -> None:
    minx, miny, maxx, maxy = bbox
    with rasterio.open(raster_path) as src:
        b = src.bounds
    if b.right < minx or b.left > maxx or b.top < miny or b.bottom > maxy:
        raise RuntimeError(
            f"{label} raster {raster_path} does not cover state bbox {bbox}. "
            f"Raster bounds={tuple(b)}."
        )


def aggregate_population_worldpop(cells: pd.DataFrame, raster_path: Path, bbox) -> pd.DataFrame:
    acc, _ = _iter_raster_to_h3(raster_path, bbox)
    cells = cells.copy()
    cells["population"] = cells["h3_r7"].map(acc).fillna(0.0)
    cells["population_source"] = "worldpop_2020"
    return cells


def _open_kontur_sqlite(gpkg_path: Path) -> tuple[sqlite3.Connection, str, str, str]:
    conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
        lower = {c.lower(): c for c in cols}
        h3_col = lower.get("h3") or lower.get("h3index") or lower.get("hex_id")
        pop_col = lower.get("population") or lower.get("pop")
        if h3_col and pop_col:
            return conn, table, h3_col, pop_col
    conn.close()
    raise RuntimeError(f"Kontur gpkg {gpkg_path} has no h3/population columns. tables={tables}")


def build_india_r7_population(gpkg_path: Path, force: bool = False) -> Path:
    """Sum Kontur H3 population to R7 once for all of India and cache it."""
    dest = PROCESSED / "india_r7_population.parquet"
    gpkg = Path(gpkg_path)
    if dest.exists() and not force and dest.stat().st_mtime >= gpkg.stat().st_mtime:
        return dest

    conn, table, h3_col, pop_col = _open_kontur_sqlite(gpkg)
    sample = conn.execute(f'SELECT "{h3_col}" FROM "{table}" LIMIT 32').fetchall()
    resolutions = {h3.get_resolution(str(r[0])) for r in sample if r and r[0]}
    if len(resolutions) != 1:
        conn.close()
        raise RuntimeError(f"Kontur H3 resolution is mixed or empty: {resolutions}")
    native_res = next(iter(resolutions))
    n_total = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    print(f"Kontur native H3 resolution={native_res} (R8 expected ~400 m), rows={n_total:,}")

    acc: dict[str, float] = defaultdict(float)
    kids: dict[str, int] = defaultdict(int)
    scanned = 0
    with tqdm(total=n_total, desc="kontur R8->R7", unit="cell") as bar:
        for hidx, pop in conn.execute(f'SELECT "{h3_col}", "{pop_col}" FROM "{table}"'):
            scanned += 1
            if scanned % 100_000 == 0:
                bar.update(100_000)
            if hidx is None:
                continue
            cell = str(hidx)
            res = h3.get_resolution(cell)
            if res > H3_RES:
                cell = h3.cell_to_parent(cell, H3_RES)
            elif res < H3_RES:
                continue  # coarser than the model grid: never split downward
            acc[cell] += float(pop or 0.0)
            kids[cell] += 1
    conn.close()

    out = pd.DataFrame(
        {
            "h3_r7": list(acc.keys()),
            "population": list(acc.values()),
            "kontur_children": [kids[k] for k in acc],
        }
    )
    out.attrs["native_res"] = native_res
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    (PROCESSED / "india_r7_population.json").write_text(
        json.dumps(
            {
                "source_gpkg": str(gpkg),
                "native_h3_res": native_res,
                "source_rows": n_total,
                "r7_cells": int(len(out)),
                "india_total_population": float(out["population"].sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"kontur: {n_total:,} R{native_res} cells -> {len(out):,} R7 cells, "
        f"India total {out['population'].sum() / 1e6:,.1f}M, cached at {dest}"
    )
    return dest


def aggregate_population_kontur(cells: pd.DataFrame, gpkg_path: Path, state: StateConfig) -> pd.DataFrame:
    r7 = pd.read_parquet(build_india_r7_population(Path(gpkg_path)))
    lookup = dict(zip(r7["h3_r7"], r7["population"]))
    cells = cells.copy()
    cells["population"] = cells["h3_r7"].map(lookup).astype(float).fillna(0.0)
    cells["population_source"] = "kontur_IN_2023_h3"
    total = float(cells["population"].sum())
    lo, hi = state.pop_sanity_million
    print(
        f"{state.code} Kontur population after R7 join: {total:,.0f} "
        f"({total / 1e6:.1f}M; sanity range {lo:.0f}–{hi:.0f}M)"
    )
    if not (lo * 1e6 * 0.5 <= total <= hi * 1e6 * 1.6):
        print(
            f"WARNING: {state.code} joined population {total/1e6:.1f}M is outside "
            f"the loose sanity window around {lo}–{hi}M."
        )
    return cells


def aggregate_population(cells: pd.DataFrame, pop_path: Path, state: StateConfig, source: str | None = None) -> pd.DataFrame:
    source = source or POPULATION_SOURCE
    path = Path(pop_path)
    if source == "kontur" or path.suffix.lower() in {".gpkg", ".gz"} or "kontur" in path.name.lower():
        return aggregate_population_kontur(cells, path, state)
    return aggregate_population_worldpop(cells, path, state.bbox)


def build_india_r7_nightlights(parquet_path: Path, force: bool = False) -> Path:
    """Collapse the India-wide H3 nightlight table to R7 once and cache it.

    The source table stores per-cell mean radiance plus the pixel count behind
    it, so the R7 parent value is the pixel-count weighted mean of its children
    (a plain mean of means would bias small partial cells upward).
    """
    import pyarrow.parquet as pq

    dest = PROCESSED / "india_r7_nightlights.parquet"
    if dest.exists() and not force and dest.stat().st_mtime >= Path(parquet_path).stat().st_mtime:
        return dest

    pf = pq.ParquetFile(parquet_path)
    names = set(pf.schema_arrow.names)
    value_col = NIGHTLIGHTS_H3_VALUE if NIGHTLIGHTS_H3_VALUE in names else "nightlight_mean"
    cols = ["h3_index", value_col]
    weighted = "pixel_count" in names
    if weighted:
        cols.append("pixel_count")

    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    native: set[int] = set()
    total = pf.metadata.num_rows
    with tqdm(total=total, desc="nightlights R8->R7", unit="cell") as bar:
        for batch in pf.iter_batches(batch_size=250_000, columns=cols):
            idx = batch.column(0).to_pylist()
            val = batch.column(1).to_pylist()
            wts = batch.column(2).to_pylist() if weighted else None
            for i, cell in enumerate(idx):
                v = val[i]
                if cell is None or v is None:
                    continue
                res = h3.get_resolution(cell)
                native.add(res)
                if res > H3_RES:
                    parent = h3.cell_to_parent(cell, H3_RES)
                elif res == H3_RES:
                    parent = cell
                else:
                    continue  # coarser than the model grid: never split downward
                w = float(wts[i]) if wts and wts[i] else 1.0
                num[parent] += float(v) * w
                den[parent] += w
            bar.update(len(idx))

    out = pd.DataFrame(
        {
            "h3_r7": list(num.keys()),
            "night_lights": [num[k] / den[k] for k in num],
            "ntl_pixels": [den[k] for k in num],
        }
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(
        f"nightlights: {total:,} source cells (H3 res {sorted(native)}) -> "
        f"{len(out):,} R7 cells, cached at {dest}"
    )
    return dest


def aggregate_nightlights(cells: pd.DataFrame, source: Path | None, bbox) -> pd.DataFrame:
    """Mean VIIRS average_masked radiance per R7 cell.

    Accepts either the pre-aggregated India H3 table (fast path) or the raw
    GeoTIFF. Both carry the same product; only the plumbing differs.
    """
    cells = cells.copy()
    if source is None or not Path(source).exists():
        raise FileNotFoundError(
            "Night lights source missing. Expected the India H3 table at "
            f"{NIGHTLIGHTS_H3_PARQUET} or a VIIRS average_masked GeoTIFF under data/raw/nightlights/."
        )
    path = Path(source)
    if path.suffix.lower() == ".parquet":
        r7 = pd.read_parquet(build_india_r7_nightlights(path))
        lookup = dict(zip(r7["h3_r7"], r7["night_lights"]))
        cells["night_lights"] = cells["h3_r7"].map(lookup).astype(float).fillna(0.0)
        cells["night_lights_source"] = f"viirs_average_masked_h3_{NIGHTLIGHTS_YEAR}"
    else:
        tif = _maybe_unzip_tif(path)
        _assert_raster_covers_bbox(tif, bbox, "VIIRS average_masked")
        acc, n = _iter_raster_to_h3(tif, bbox)
        mean = {k: acc[k] / max(n[k], 1) for k in acc}
        cells["night_lights"] = cells["h3_r7"].map(mean).fillna(0.0)
        cells["night_lights_source"] = f"viirs_average_masked_raster_{NIGHTLIGHTS_YEAR}"
    cells["night_lights_missing"] = (cells["night_lights"] <= 0).astype(int)
    covered = float((cells["night_lights"] > 0).mean())
    print(
        f"night lights: {covered * 100:.1f}% of R7 cells lit, "
        f"mean={cells['night_lights'].mean():.3f} max={cells['night_lights'].max():.1f} nW/cm²/sr"
    )
    if covered == 0.0:
        raise RuntimeError(
            f"No night-light values landed on this state's {len(cells):,} R7 cells. "
            f"The source {path} does not cover bbox {bbox}."
        )
    return cells


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _ring_area_m2(coords: list[dict]) -> float:
    if len(coords) < 3:
        return 0.0
    lons = [c["lon"] for c in coords]
    lats = [c["lat"] for c in coords]
    if lons[0] != lons[-1] or lats[0] != lats[-1]:
        lons.append(lons[0])
        lats.append(lats[0])
    area, _ = GEOD.polygon_area_perimeter(lons, lats)
    return abs(float(area))


def _centroid_latlon(coords: list[dict]) -> tuple[float, float] | None:
    if not coords:
        return None
    return (
        sum(c["lat"] for c in coords) / len(coords),
        sum(c["lon"] for c in coords) / len(coords),
    )


def _classify_poi(tags: dict) -> list[str]:
    hits = []
    for col, (key, values) in POI_GROUPS.items():
        raw = tags.get(key)
        if not raw:
            continue
        if values is None:
            hits.append(col)
        elif str(raw).lower() in values:
            hits.append(col)
    return hits


def _osm_files(osm_path: Path) -> list[Path]:
    p = Path(osm_path)
    if p.is_dir():
        tiles = sorted(p.glob("tile_*.json"))
        if tiles:
            return tiles
    if p.is_file():
        return [p]
    return []


def _finalise_osm_columns(
    cells: pd.DataFrame,
    road_km: dict,
    building_m2: dict,
    building_n: dict | None,
    landuse_m2: dict,
    poi_n: dict,
    poi_by: dict,
) -> pd.DataFrame:
    cells = cells.copy()
    cells["road_km"] = cells["h3_r7"].map(road_km).astype(float).fillna(0.0)
    cells["building_area_m2"] = cells["h3_r7"].map(building_m2).astype(float).fillna(0.0)
    if building_n:
        cells["building_count"] = cells["h3_r7"].map(building_n).astype(float).fillna(0.0)
    cells["poi_count"] = cells["h3_r7"].map(poi_n).astype(float).fillna(0.0)
    area = cells["area_km2"].replace(0, np.nan)
    for cls in LANDUSE_CLASSES:
        raw = cells["h3_r7"].map(landuse_m2.get(cls, {})).astype(float).fillna(0.0)
        cells[f"landuse_{cls}_m2"] = raw
        cells[f"landuse_{cls}"] = (
            (raw / (cells["area_km2"] * 1e6)).clip(lower=0, upper=1.5).fillna(0.0)
        )
    cells["building_share"] = (
        (cells["building_area_m2"] / (cells["area_km2"] * 1e6)).clip(lower=0, upper=1.5).fillna(0.0)
    )
    for k in POI_GROUPS:
        col = f"poi_{k}"
        cells[col] = cells["h3_r7"].map(poi_by.get(k, {})).astype(float).fillna(0.0)
        cells[f"{col}_density"] = (cells[col] / area).fillna(0.0)
    print(
        f"OSM joined: roads>0 in {int((cells['road_km'] > 0).sum()):,} cells, "
        f"buildings>0 in {int((cells['building_area_m2'] > 0).sum()):,}, "
        f"POIs>0 in {int((cells['poi_count'] > 0).sum()):,} of {len(cells):,} cells"
    )
    return cells


def aggregate_osm_pbf(cells: pd.DataFrame, pbf_path: Path, state: StateConfig, force: bool = False) -> pd.DataFrame:
    """Roads, building footprints, landuse and POIs from a Geofabrik PBF extract."""
    from .osm_pbf import load_or_build

    agg = load_or_build(
        state.code, Path(pbf_path), set(cells["h3_r7"]), bbox=state.bbox, force=force
    )
    return _finalise_osm_columns(
        cells,
        agg["road_km"],
        agg["building_m2"],
        agg.get("building_count"),
        agg["landuse_m2"],
        agg["poi_count"],
        agg["poi_by"],
    )


def aggregate_osm(cells: pd.DataFrame, osm_path: Path, state: StateConfig | None = None) -> pd.DataFrame:
    path = Path(osm_path)
    if path.suffix.lower() == ".pbf" or str(path).endswith(".osm.pbf"):
        if state is None:
            raise RuntimeError("aggregate_osm needs the StateConfig to read a PBF extract")
        return aggregate_osm_pbf(cells, path, state)
    files = _osm_files(path)
    if not files:
        raise FileNotFoundError(f"No OSM tiles at {osm_path}")
    road_km = defaultdict(float)
    building_m2 = defaultdict(float)
    landuse_m2 = {cls: defaultdict(float) for cls in LANDUSE_CLASSES}
    poi_n = defaultdict(int)
    poi_by = {k: defaultdict(int) for k in POI_GROUPS}
    valid = set(cells["h3_r7"])
    seen = set()
    for fpath in files:
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        for el in tqdm(payload.get("elements", []), desc=f"OSM {fpath.name}"):
            key = (el.get("type"), el.get("id"))
            if key in seen:
                continue
            seen.add(key)
            tags = el.get("tags") or {}
            lat = el.get("lat")
            lon = el.get("lon")
            geom = el.get("geometry") or []
            if lat is None and geom:
                c = _centroid_latlon(geom)
                if c:
                    lat, lon = c
            cell = None
            if lat is not None and lon is not None:
                cell = h3.latlng_to_cell(lat, lon, H3_RES)
                if cell not in valid:
                    cell = None

            if el.get("type") == "way" and "highway" in tags and geom:
                for a, b in zip(geom, geom[1:]):
                    d = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
                    mid_lat = 0.5 * (a["lat"] + b["lat"])
                    mid_lon = 0.5 * (a["lon"] + b["lon"])
                    rc = h3.latlng_to_cell(mid_lat, mid_lon, H3_RES)
                    if rc in valid:
                        road_km[rc] += d

            if el.get("type") == "way" and "building" in tags and geom:
                area = _ring_area_m2(geom)
                if cell is not None and area > 0:
                    building_m2[cell] += area

            lu = (tags.get("landuse") or "").lower()
            if el.get("type") == "way" and lu in LANDUSE_CLASSES and geom:
                area = _ring_area_m2(geom)
                if cell is not None and area > 0:
                    landuse_m2[lu][cell] += area

            hits = _classify_poi(tags)
            if cell is not None and hits:
                poi_n[cell] += 1
                for h in hits:
                    poi_by[h][cell] += 1

    return _finalise_osm_columns(
        cells, road_km, building_m2, None, landuse_m2, poi_n, poi_by
    )


def engineer_features(cells: pd.DataFrame) -> pd.DataFrame:
    """Paper-style zero-inflation split + log magnitudes (Section 4.2)."""
    df = cells.copy()
    magnitude_cols = [c for c in MAGNITUDE_FEATURE_BASE if c in df.columns]
    share_cols = [c for c in SHARE_FEATURE_BASE if c in df.columns]
    # Build every derived column into one dict, then attach in a single concat:
    # assigning ~200 columns one at a time fragments the frame badly.
    new: dict[str, np.ndarray] = {}
    engineered: list[str] = []
    for col in magnitude_cols:
        base = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        df[col] = base
        pos = base > 0
        mag = np.zeros(len(df), dtype=np.float32)
        mag[pos] = np.log10(base[pos])
        new[f"{col}_exists"] = pos.astype(np.float32)
        new[f"{col}_log"] = mag
        engineered.extend([f"{col}_exists", f"{col}_log"])
    for col in share_cols:
        base = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        df[col] = base
        new[f"{col}_exists"] = (base > 0).astype(np.float32)
        engineered.extend([col, f"{col}_exists"])
    if "night_lights" not in df or float(df["night_lights"].sum()) == 0:
        engineered = [c for c in engineered if not c.startswith("night_lights")]
    z_cols = []
    for c in engineered:
        vals = new[c] if c in new else df[c].to_numpy(dtype=np.float32)
        mu = float(vals.mean())
        sd = float(vals.std())
        new[f"{c}_z"] = ((vals - mu) / (sd if sd else 1.0)).astype(np.float32)
        z_cols.append(f"{c}_z")
    df = pd.concat(
        [df.drop(columns=[c for c in new if c in df.columns]), pd.DataFrame(new, index=df.index)],
        axis=1,
    )
    df.attrs["feature_cols"] = z_cols
    df.attrs["magnitude_cols"] = magnitude_cols
    df.attrs["share_cols"] = share_cols
    return df


def attach_targets(cells: pd.DataFrame, gddp: pd.DataFrame) -> pd.DataFrame:
    keep = ["district", "y_crore", "y_inr", "gddp_year", "price_basis", "gddp_units"]
    keep = [c for c in keep if c in gddp.columns]
    df = cells.merge(gddp[keep], on="district", how="left")
    extra = sorted(set(df.loc[df["y_inr"].isna(), "district"].unique()))
    if extra:
        df = df[df["y_inr"].notna()].copy()
    missing = sorted(set(gddp["district"]) - set(df["district"].unique()))
    if missing:
        raise RuntimeError(f"District target mismatch. missing={missing} dropped_unmatched={extra}")
    return df


def save_processed(cells: pd.DataFrame, state_code: str) -> Path:
    dest = processed_dir(state_code)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{state_code.lower()}_h3_r7_features.parquet"
    # attrs do not survive parquet; persist feature cols beside the table.
    feature_cols = cells.attrs.get("feature_cols", [])
    cells.to_parquet(path, index=False)
    cells.to_csv(dest / f"{state_code.lower()}_h3_r7_features.csv", index=False)
    (dest / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    return path
