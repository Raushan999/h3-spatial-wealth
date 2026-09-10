"""Download and persist raw inputs under data/raw/{layer}/{state or IN}/."""

from __future__ import annotations

import gzip
import json
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import (
    NIGHTLIGHTS_H3_PARQUET,
    NIGHTLIGHTS_H3_VALUE,
    NIGHTLIGHTS_PRODUCT,
    NIGHTLIGHTS_YEAR,
    RAW,
    SOURCES,
    STATES,
    StateConfig,
    get_state,
    raw_layer,
)

HEADERS = {"User-Agent": "spatialdisintegration/1.0 (research pipeline; open-data)"}
OVERPASS_ENDPOINTS = SOURCES["overpass_mirrors"]


def _get(url: str, timeout: int | tuple = (30, 120), **kwargs) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_source(
    path: Path,
    *,
    url: str,
    license: str,
    units: str,
    extra: dict | None = None,
) -> None:
    payload = {
        "url": url,
        "date": date.today().isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "license": license,
        "units": units,
    }
    if extra:
        payload.update(extra)
    save_json(payload, path)


def download_file(url: str, dest: Path, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with _get(url, timeout=timeout, stream=True) as r:
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def fetch_gddp(state: StateConfig) -> Path | None:
    """Persist official GDDP/GSDP for one state. Returns the table path or None if skipped."""
    src = state.gddp
    out_dir = raw_layer("gddp", state.code)
    out_dir.mkdir(parents=True, exist_ok=True)
    if src is None:
        write_source(
            out_dir / "SOURCE.json",
            url="",
            license="",
            units="",
            extra={"status": "skip", "reason": state.skip_reason or "no GDDP source configured"},
        )
        return None

    # Local CSV drop-in always wins if present.
    local_csv = out_dir / "gddp.csv"
    if local_csv.exists() and local_csv.stat().st_size > 0:
        write_source(
            out_dir / "SOURCE.json",
            url=src.page or src.url,
            license="as stated by the placing agency (must be official open government data)",
            units=src.units,
            extra={
                "status": "local_csv",
                "path": str(local_csv),
                "year": src.year,
                "price_basis": src.price_basis,
                "notes": list(src.notes),
            },
        )
        return local_csv

    dest = out_dir / src.filename
    if src.kind in {"ckan_json", "opencity_csv"}:
        if not dest.exists():
            payload = _get(src.url).json()
            save_json(payload, dest)
        write_source(
            out_dir / "SOURCE.json",
            url=src.url,
            license="OpenCity public-domain extract of state Economic Survey / DES table",
            units=src.units,
            extra={
                "status": "ok",
                "page": src.page,
                "year": src.year,
                "price_basis": src.price_basis,
                "base_year": src.base_year,
                "value_field": src.value_field,
                "notes": list(src.notes),
            },
        )
        return dest

    if src.kind in {"gsdp_single", "xlsx_matrix", "xlsx_table", "pdf_table"}:
        # These states ship an official publication on disk (or downloadable);
        # parse it into gddp.json so the training step reads a plain table.
        if src.url and not dest.exists():
            try:
                download_file(src.url, dest, timeout=180)
            except Exception as exc:
                save_json({"ok": False, "reason": str(exc), "url": src.url}, out_dir / "download_status.json")
        if not dest.exists():
            write_source(
                out_dir / "SOURCE.json",
                url=src.url or src.page,
                license="State DES / official statistical publication",
                units=src.units,
                extra={
                    "status": "missing_publication",
                    "expected": str(dest),
                    "notes": list(src.notes),
                },
            )
            return None
        from .gddp_parse import parse_official_source

        records, provenance = parse_official_source(
            state.code, dest, src, state.dissolve_to_single or state.name
        )
        parsed = out_dir / "gddp.json"
        save_json(
            {
                "records": records,
                "source_file": dest.name,
                "source_url": src.url,
                "page": src.page,
                "year": src.year,
                "price_basis": src.price_basis,
                "base_year": src.base_year,
                "units": src.units,
                "provenance": provenance,
                "notes": list(src.notes),
            },
            parsed,
        )
        write_source(
            out_dir / "SOURCE.json",
            url=src.url or src.page,
            license="State DES / official statistical publication (open government data)",
            units=src.units,
            extra={
                "status": "ok",
                "kind": src.kind,
                "year": src.year,
                "price_basis": src.price_basis,
                "base_year": src.base_year,
                "source_file": dest.name,
                "provenance": provenance,
                "notes": list(src.notes),
            },
        )
        return parsed

    if src.kind == "pdf":
        if src.url and not dest.exists():
            try:
                download_file(src.url, dest, timeout=180)
            except Exception as exc:
                save_json({"ok": False, "reason": str(exc), "url": src.url}, out_dir / "download_status.json")
        write_source(
            out_dir / "SOURCE.json",
            url=src.url or src.page,
            license="State DES / Finance Department official publication",
            units=src.units,
            extra={
                "status": "pdf_persisted" if dest.exists() else "download_failed",
                "page": src.page,
                "year": src.year,
                "price_basis": src.price_basis,
                "notes": list(src.notes),
                "drop_in": str(local_csv),
            },
        )
        parsed = out_dir / "gddp.json"
        if parsed.exists():
            return parsed
        return dest if dest.exists() else None

    if src.kind == "local_csv":
        write_source(
            out_dir / "SOURCE.json",
            url=src.page,
            license="",
            units=src.units,
            extra={
                "status": "skip_until_local_csv",
                "reason": state.skip_reason or "place official gddp.csv to enable this state",
                "drop_in": str(local_csv),
                "notes": list(src.notes),
            },
        )
        return None

    if src.kind == "records_json":
        if dest.exists():
            write_source(
                out_dir / "SOURCE.json",
                url=src.url or src.page,
                license="as documented in this SOURCE.json",
                units=src.units,
                extra={"status": "ok", "notes": list(src.notes)},
            )
            return dest
        return None

    return None


def _download_geoboundaries(level: str) -> Path:
    api = SOURCES[f"geoboundaries_{level}_api"]
    dest_dir = raw_layer("boundaries", "IN")
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dest_dir / f"geoboundaries_IND_{level}_meta.json"
    geo_path = dest_dir / f"geoBoundaries-IND-{level}.geojson"
    legacy_geo = RAW / "boundaries" / f"geoBoundaries-IND-{level}.geojson"
    legacy_meta = RAW / "boundaries" / f"geoboundaries_IND_{level}_meta.json"
    if not meta_path.exists() and legacy_meta.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_meta, meta_path)
    if not geo_path.exists() and legacy_geo.exists():
        shutil.copy2(legacy_geo, geo_path)
    if not meta_path.exists():
        meta = _get(api).json()
        save_json(meta, meta_path)
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    url = meta["gjDownloadURL"] if isinstance(meta, dict) else meta[0]["gjDownloadURL"]
    if isinstance(meta, list):
        url = meta[0]["gjDownloadURL"]
    download_file(url, geo_path)
    write_source(
        dest_dir / "SOURCE.json",
        url=api,
        license="CC BY 4.0 (geoBoundaries gbOpen)",
        units="polygon WGS84",
        extra={"adm_levels": ["ADM1", "ADM2"], "download": url},
    )
    return geo_path


def fetch_boundaries() -> tuple[Path, Path]:
    return _download_geoboundaries("adm1"), _download_geoboundaries("adm2")


def fetch_worldpop() -> Path:
    dest_dir = raw_layer("population", "IN")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_unadj = dest_dir / "ind_ppp_2020_1km_Aggregated_UNadj.tif"
    dest = dest_dir / "ind_ppp_2020_1km_Aggregated.tif"
    legacy_unadj = RAW / "population" / "ind_ppp_2020_1km_Aggregated_UNadj.tif"
    legacy = RAW / "population" / "ind_ppp_2020_1km_Aggregated.tif"
    for src, dst in ((legacy_unadj, dest_unadj), (legacy, dest)):
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    if dest_unadj.exists() and dest_unadj.stat().st_size > 0:
        used, url, unadj = dest_unadj, SOURCES["worldpop_1km_unadj"], True
    elif dest.exists() and dest.stat().st_size > 0:
        used, url, unadj = dest, SOURCES["worldpop_1km"], False
    else:
        download_file(SOURCES["worldpop_1km"], dest, timeout=(30, 1800))
        used, url, unadj = dest, SOURCES["worldpop_1km"], False
    write_source(
        dest_dir / "SOURCE_worldpop.json",
        url=url,
        license="CC BY 4.0 (WorldPop)",
        units="people per 1 km pixel (2020)",
        extra={"un_adjusted": unadj, "year": 2020, "role": "fallback w_i"},
    )
    return used


def fetch_kontur_population() -> Path:
    """HDX Kontur India H3 population (R8). Prefers a user-placed GeoPackage."""
    dest_dir = raw_layer("population", "IN")
    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_path = dest_dir / "kontur_population_IN_20231101.gpkg.gz"
    # A GeoPackage already on disk always wins: the biggest one is the full extract.
    candidates = [
        RAW / "population" / "kontur_population_india.gpkg",
        *sorted(dest_dir.glob("kontur_population_*.gpkg")),
        *sorted((RAW / "population").glob("kontur_population_*.gpkg")),
    ]
    seen: list[Path] = []
    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0 and cand not in seen:
            seen.append(cand)
    if seen:
        gpkg_path = max(seen, key=lambda p: p.stat().st_size)
    else:
        gpkg_path = dest_dir / "kontur_population_IN_20231101.gpkg"
        if not gz_path.exists():
            download_file(SOURCES["kontur_gpkg_gz"], gz_path, timeout=(30, 1800))
        with gzip.open(gz_path, "rb") as src, gpkg_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    write_source(
        dest_dir / "SOURCE.json",
        url=SOURCES["kontur_hdx"],
        license="CC BY (Kontur Population via HDX)",
        units="people per H3 cell (R8, ~400 m)",
        extra={
            "status": "ok",
            "path": str(gpkg_path),
            "size_bytes": gpkg_path.stat().st_size,
            "other_copies_on_disk": [str(p) for p in seen if p != gpkg_path],
            "download": SOURCES["kontur_gpkg_gz"],
            "role": "default exposure w_i and population feature",
            "year": 2023,
        },
    )
    return gpkg_path


def _iter_candidate_rasters(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not any(name.endswith(ext) for ext in (".tif", ".tiff", ".tif.gz", ".tiff.gz")):
            continue
        out.append(p)
    return out


def find_nightlights() -> Path:
    """
    Night lights input, in preference order:

    1. the India-wide H3 table derived from the VIIRS `average_masked` composite
       (``data/processed/h3_nightlights_2025.parquet``), which is the same
       product pre-aggregated so we do not rescan an 11 GB global GeoTIFF;
    2. a user-placed ``average_masked`` GeoTIFF under ``data/raw/nightlights/``.

    ``median_masked`` and ``unmasked`` rasters are never used for training.
    """
    dest_dir = raw_layer("nightlights", "IN")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if NIGHTLIGHTS_H3_PARQUET.exists() and NIGHTLIGHTS_H3_PARQUET.stat().st_size > 0:
        rasters = _iter_candidate_rasters(RAW / "nightlights")
        lineage = [p.name for p in rasters if "average_masked" in p.name.lower()]
        (dest_dir / "CHOICE.md").write_text(
            "# Night lights product\n\n"
            f"Training uses **{NIGHTLIGHTS_PRODUCT}**, read from the India-wide H3 table\n"
            f"`{NIGHTLIGHTS_H3_PARQUET.relative_to(RAW.parent.parent)}`"
            f" (column `{NIGHTLIGHTS_H3_VALUE}`), aggregated to H3 R7 by the pipeline.\n\n"
            f"- H3 table: `{NIGHTLIGHTS_H3_PARQUET.name}` "
            f"({NIGHTLIGHTS_H3_PARQUET.stat().st_size / 1e6:.0f} MB, H3 R8 mean/median/max radiance)\n"
            f"- raw `average_masked` composite kept in `data/raw/nightlights/`: {lineage or 'not present'}\n"
            "- `median_masked` and `unmasked` composites are ignored for training.\n\n"
            "average_masked is the cloud- and fire-screened annual mean, which matches the\n"
            "paper's lights feature (mean radiance per H3 cell, not a sum).\n",
            encoding="utf-8",
        )
        write_source(
            dest_dir / "SOURCE.json",
            url=SOURCES["viirs_eog_index"],
            license="EOG / Payne Institute VNL v2.2",
            units="nW/cm²/sr (annual mean radiance, average_masked)",
            extra={
                "status": "ok",
                "product_used": NIGHTLIGHTS_PRODUCT,
                "year": NIGHTLIGHTS_YEAR,
                "path": str(NIGHTLIGHTS_H3_PARQUET),
                "value_column": NIGHTLIGHTS_H3_VALUE,
                "native_h3_res": 8,
                "raw_composites_on_disk": [str(p) for p in rasters],
            },
        )
        shutil.copy2(dest_dir / "CHOICE.md", RAW / "nightlights" / "CHOICE.md")
        return NIGHTLIGHTS_H3_PARQUET
    return find_viirs_average_masked()


def find_viirs_average_masked() -> Path:
    """
    Use user-placed VIIRS **average_masked** (not unmasked, not median).

    Search data/raw/nightlights/ including IN/ subfolders. Fail later if the
    raster does not cover a state's bbox.
    """
    search_roots = [
        raw_layer("nightlights", "IN"),
        RAW / "nightlights",
    ]
    masked: list[Path] = []
    median: list[Path] = []
    unmasked: list[Path] = []
    other: list[Path] = []
    for root in search_roots:
        for p in _iter_candidate_rasters(root):
            n = p.name.lower()
            if "unmasked" in n and "average_masked" not in n:
                unmasked.append(p)
            elif "median_masked" in n:
                median.append(p)
            elif "average_masked" in n or "avg_rade9h" in n or "average_rade9h" in n:
                masked.append(p)
            else:
                other.append(p)
    dest_dir = raw_layer("nightlights", "IN")
    dest_dir.mkdir(parents=True, exist_ok=True)
    choice = dest_dir / "CHOICE.md"
    if not masked:
        note = (
            "VIIRS product choice: **average_masked** (EOG Annual VNL v2.2).\n\n"
            "Place a clipped or India/state GeoTIFF whose filename contains "
            "`average_masked` under `data/raw/nightlights/IN/` "
            "(or `data/raw/nightlights/`). Do **not** use `unmasked` or "
            "`median_masked` as the training raster. `median_masked` may sit "
            "alongside for reference but is not read.\n"
        )
        choice.write_text(note, encoding="utf-8")
        write_source(
            dest_dir / "SOURCE.json",
            url=SOURCES["viirs_eog_index"],
            license="EOG / Payne Institute VNL v2.2 (free registration; not committed)",
            units="nW/cm²/sr (annual mean radiance, average_masked)",
            extra={
                "status": "missing",
                "product_used": "average_masked",
                "reason": "no average_masked GeoTIFF on disk",
                "median_masked_seen": [str(p) for p in median],
                "unmasked_ignored": [str(p) for p in unmasked],
            },
        )
        raise FileNotFoundError(
            "VIIRS average_masked raster not found under data/raw/nightlights/. "
            "Place the user-provided average_masked GeoTIFF there (median_masked is ignored)."
        )
    used = max(masked, key=lambda p: p.stat().st_size)
    choice.write_text(
        "# Night lights product\n\n"
        "Training uses **average_masked** annual VIIRS VNL v2.2 mean radiance.\n\n"
        f"- used: `{used.name}`\n"
        f"- median_masked present (not used): {[p.name for p in median] or 'none'}\n"
        f"- unmasked present (ignored): {[p.name for p in unmasked] or 'none'}\n\n"
        "average_masked is the cloud- and fire-screened annual mean, which matches "
        "the paper's lights feature (mean radiance per H3 cell, not a sum).\n",
        encoding="utf-8",
    )
    write_source(
        dest_dir / "SOURCE.json",
        url=SOURCES["viirs_eog_index"],
        license="EOG / Payne Institute VNL v2.2",
        units="nW/cm²/sr (annual mean radiance, average_masked)",
        extra={
            "status": "ok",
            "product_used": "average_masked",
            "path": str(used),
            "median_masked_not_used": [str(p) for p in median],
            "unmasked_ignored": [str(p) for p in unmasked],
        },
    )
    # Also copy a pointer into the legacy folder for the README.
    (RAW / "nightlights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(choice, RAW / "nightlights" / "CHOICE.md")
    return used


def fetch_viirs_average_masked() -> Path:
    """Discover the on-disk average_masked product. Never downloads the multi-GB global file."""
    return find_nightlights()


def overpass_query(query: str, retries: int = 5) -> dict:
    last_err = None
    for i in range(retries):
        endpoint = OVERPASS_ENDPOINTS[i % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                headers=HEADERS,
                timeout=300,
            )
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(15 * (i + 1))
                last_err = RuntimeError(f"Overpass HTTP {r.status_code} at {endpoint}")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"Overpass failed: {last_err}") from last_err


def _osm_query(south: float, west: float, north: float, east: float) -> dict:
    amenity = "restaurant|cafe|bar|pub|cinema|theatre|theater|nightclub|university|hospital|bank"
    shop = "mall|department_store|jewelry|jewellery|watches|boutique|supermarket|convenience"
    query = f"""
    [out:json][timeout:180];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary"]({south},{west},{north},{east});
      way["building"]({south},{west},{north},{east});
      way["landuse"~"residential|commercial|industrial|retail"]({south},{west},{north},{east});
      node["amenity"~"{amenity}"]({south},{west},{north},{east});
      way["amenity"~"{amenity}"]({south},{west},{north},{east});
      node["shop"~"{shop}"]({south},{west},{north},{east});
      way["shop"~"{shop}"]({south},{west},{north},{east});
      node["tourism"~"hotel|guest_house"]({south},{west},{north},{east});
      way["tourism"~"hotel|guest_house"]({south},{west},{north},{east});
      node["leisure"]({south},{west},{north},{east});
      way["leisure"]({south},{west},{north},{east});
      node["office"]({south},{west},{north},{east});
      way["office"]({south},{west},{north},{east});
    );
    out geom;
    """
    return overpass_query(query)


def _bbox_tiles(south: float, west: float, north: float, east: float, step: float = 0.5):
    lat = south
    tiles = []
    while lat < north - 1e-9:
        lon = west
        lat2 = min(lat + step, north)
        while lon < east - 1e-9:
            lon2 = min(lon + step, east)
            tiles.append((lat, lon, lat2, lon2))
            lon = lon2
        lat = lat2
    return tiles


def fetch_osm_for_state(state: StateConfig, step: float = 0.5) -> Path:
    """Recent OSM extract for the state bbox: roads, buildings, landuse, enriched POIs."""
    minx, miny, maxx, maxy = state.bbox
    dest_dir = raw_layer("osm", state.code)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tiles = _bbox_tiles(miny, minx, maxy, maxx, step=step)
    paths = []
    for i, tile in enumerate(tiles, 1):
        tile_path = dest_dir / f"tile_{i:04d}.json"
        paths.append(tile_path)
        if tile_path.exists() and tile_path.stat().st_size > 500:
            continue
        payload = _osm_query(*tile)
        save_json(payload, tile_path)
        time.sleep(1.5)
    write_source(
        dest_dir / "SOURCE.json",
        url=SOURCES["overpass"],
        license="ODbL (OpenStreetMap contributors)",
        units="OSM geometries (roads km, building m², landuse shares, POI counts)",
        extra={
            "bbox": list(state.bbox),
            "filters": "major highways + building footprints + landuse + amenity/shop/tourism/leisure/office",
            "tiles": [p.name for p in paths],
            "n_tiles": len(paths),
        },
    )
    save_json({"tiles": [p.name for p in paths], "schema": "v2_buildings_landuse_pois"}, dest_dir / "tiles_complete.json")
    return dest_dir


def fetch_osm_pbf(state: StateConfig) -> Path:
    """Geofabrik India sub-region extract that contains this state."""
    if not state.osm_zone:
        raise RuntimeError(f"No Geofabrik zone configured for {state.code}")
    dest_dir = raw_layer("osm", "IN")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{state.osm_zone}-latest.osm.pbf"
    url = SOURCES["geofabrik_zone"].format(zone=state.osm_zone)
    if not (dest.exists() and dest.stat().st_size > 0):
        print(f"downloading {url}", flush=True)
        download_file(url, dest, timeout=(30, 3600))
    write_source(
        dest_dir / f"SOURCE_{state.osm_zone}.json",
        url=url,
        license="ODbL 1.0 (OpenStreetMap contributors, via Geofabrik)",
        units="OSM objects (roads, building footprints, landuse polygons, POI nodes/ways)",
        extra={
            "status": "ok",
            "zone": state.osm_zone,
            "page": SOURCES["geofabrik_india"],
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            "used_by_states": sorted(
                c for c, s in STATES.items() if s.osm_zone == state.osm_zone
            ),
            "reason": "state-wide building footprints are not retrievable from Overpass without timeouts",
        },
    )
    return dest


def fetch_all(state_code: str = "MH") -> dict[str, Path | None]:
    state = get_state(state_code)
    RAW.mkdir(parents=True, exist_ok=True)
    gddp = fetch_gddp(state)
    adm1, adm2 = fetch_boundaries()
    kontur = fetch_kontur_population()
    worldpop = None
    try:
        worldpop = fetch_worldpop()
    except Exception as exc:
        save_json({"ok": False, "reason": str(exc)}, raw_layer("population", "IN") / "worldpop_status.json")
    lights = None
    try:
        lights = fetch_viirs_average_masked()
    except FileNotFoundError as exc:
        save_json({"ok": False, "reason": str(exc)}, raw_layer("nightlights", "IN") / "download_status.json")
        raise
    osm = fetch_osm_pbf(state)
    return {
        "gddp": gddp,
        "adm1": adm1,
        "adm2": adm2,
        "population": kontur,
        "population_worldpop": worldpop,
        "nightlights": lights,
        "osm": osm,
    }
