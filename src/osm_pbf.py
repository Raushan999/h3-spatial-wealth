"""Aggregate an OpenStreetMap PBF extract into H3 R7 features.

Overpass can serve roads and POIs for a state bbox, but building footprints for a
whole Indian state run into millions of ways and repeatedly time out. A Geofabrik
sub-region PBF carries the same OSM data (same ODbL licence, same contributors)
and streams locally, so buildings and landuse become tractable.

Output is exactly the per-cell aggregate `features.aggregate_osm` produces:
road km, building footprint m², landuse m² by class, and POI counts by tag group.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from pyproj import Geod

import h3

from .config import H3_RES, LANDUSE_CLASSES, POI_GROUPS

GEOD = Geod(ellps="WGS84")

ROAD_CLASSES = frozenset(
    {"motorway", "trunk", "primary", "secondary", "tertiary",
     "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
)

# Keys worth decoding at all; everything else is skipped before tag inspection.
WANTED_KEYS = ("highway", "building", "landuse", "amenity", "shop", "tourism", "leisure", "office")


class OsmAggregate:
    """Per-R7-cell accumulators plus the counters used for the run summary."""

    def __init__(self, valid_cells: set[str]):
        self.valid = valid_cells
        self.road_km: dict[str, float] = defaultdict(float)
        self.building_m2: dict[str, float] = defaultdict(float)
        self.building_n: dict[str, int] = defaultdict(int)
        self.landuse_m2: dict[str, dict[str, float]] = {c: defaultdict(float) for c in LANDUSE_CLASSES}
        self.poi_n: dict[str, int] = defaultdict(int)
        self.poi_by: dict[str, dict[str, int]] = {k: defaultdict(int) for k in POI_GROUPS}
        self.stats = defaultdict(int)

    def cell(self, lat: float, lon: float) -> str | None:
        try:
            c = h3.latlng_to_cell(lat, lon, H3_RES)
        except Exception:
            return None
        return c if c in self.valid else None

    def to_dict(self) -> dict:
        return {
            "road_km": dict(self.road_km),
            "building_m2": dict(self.building_m2),
            "building_count": dict(self.building_n),
            "landuse_m2": {c: dict(v) for c, v in self.landuse_m2.items()},
            "poi_count": dict(self.poi_n),
            "poi_by": {k: dict(v) for k, v in self.poi_by.items()},
            "stats": dict(self.stats),
        }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _ring_area_m2(lats: list[float], lons: list[float]) -> float:
    if len(lats) < 3:
        return 0.0
    if lats[0] != lats[-1] or lons[0] != lons[-1]:
        lats = lats + [lats[0]]
        lons = lons + [lons[0]]
    area, _ = GEOD.polygon_area_perimeter(lons, lats)
    return abs(float(area))


def _classify_poi(tags) -> list[str]:
    hits = []
    for col, (key, values) in POI_GROUPS.items():
        raw = tags.get(key)
        if not raw:
            continue
        if values is None or str(raw).lower() in values:
            hits.append(col)
    return hits


def aggregate_pbf(pbf_path: Path, valid_cells: set[str], bbox=None, progress_every: int = 2_000_000) -> dict:
    """Stream a PBF once and bin roads, buildings, landuse and POIs into R7 cells."""
    import osmium
    import osmium.filter

    agg = OsmAggregate(valid_cells)
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
    else:
        minx = miny = -180.0
        maxx = maxy = 180.0

    processor = (
        osmium.FileProcessor(str(pbf_path))
        .with_locations()
        .with_filter(osmium.filter.KeyFilter(*WANTED_KEYS))
    )

    seen = 0
    for obj in processor:
        seen += 1
        if progress_every and seen % progress_every == 0:
            print(
                f"  ...{seen:,} tagged objects  "
                f"buildings={agg.stats['buildings']:,} pois={agg.stats['pois']:,} "
                f"road_km={sum(agg.road_km.values()):,.0f}",
                flush=True,
            )
        tags = obj.tags
        is_way = obj.is_way()

        if is_way:
            lats: list[float] = []
            lons: list[float] = []
            for node in obj.nodes:
                loc = node.location
                if loc.valid():
                    lats.append(loc.lat)
                    lons.append(loc.lon)
            if not lats:
                continue
            # Cheap bbox reject before any H3 or geodesic work.
            if max(lons) < minx or min(lons) > maxx or max(lats) < miny or min(lats) > maxy:
                continue
            clat = sum(lats) / len(lats)
            clon = sum(lons) / len(lons)
            cell = agg.cell(clat, clon)

            highway = tags.get("highway")
            if highway and highway in ROAD_CLASSES and len(lats) > 1:
                for i in range(len(lats) - 1):
                    mid = agg.cell(0.5 * (lats[i] + lats[i + 1]), 0.5 * (lons[i] + lons[i + 1]))
                    if mid is not None:
                        agg.road_km[mid] += _haversine_km(lats[i], lons[i], lats[i + 1], lons[i + 1])
                        agg.stats["road_segments"] += 1

            if "building" in tags and cell is not None:
                area = _ring_area_m2(lats, lons)
                if area > 0:
                    agg.building_m2[cell] += area
                    agg.building_n[cell] += 1
                    agg.stats["buildings"] += 1

            landuse = (tags.get("landuse") or "").lower()
            if landuse in LANDUSE_CLASSES and cell is not None:
                area = _ring_area_m2(lats, lons)
                if area > 0:
                    agg.landuse_m2[landuse][cell] += area
                    agg.stats[f"landuse_{landuse}"] += 1
        elif obj.is_node():
            loc = obj.location
            if not loc.valid():
                continue
            if not (minx <= loc.lon <= maxx and miny <= loc.lat <= maxy):
                continue
            cell = agg.cell(loc.lat, loc.lon)
        else:
            continue

        if cell is None:
            continue
        hits = _classify_poi(tags)
        if hits:
            agg.poi_n[cell] += 1
            agg.stats["pois"] += 1
            for h in hits:
                agg.poi_by[h][cell] += 1

    agg.stats["objects_scanned"] = seen
    agg.stats["cells_with_buildings"] = len(agg.building_m2)
    agg.stats["cells_with_pois"] = len(agg.poi_n)
    agg.stats["cells_with_roads"] = len(agg.road_km)
    return agg.to_dict()


def cache_path(state_code: str) -> Path:
    from .config import PROCESSED

    return PROCESSED / state_code.upper() / "osm_r7_aggregate.json"


def load_or_build(state_code: str, pbf_path: Path, valid_cells: set[str], bbox=None, force: bool = False) -> dict:
    """Cache the per-cell aggregate so re-runs never rescan the PBF."""
    dest = cache_path(state_code)
    if dest.exists() and not force and dest.stat().st_mtime >= Path(pbf_path).stat().st_mtime:
        payload = json.loads(dest.read_text(encoding="utf-8"))
        if payload.get("n_cells") == len(valid_cells):
            print(f"OSM aggregate cache hit: {dest}")
            return payload["aggregate"]
    print(f"OSM: streaming {pbf_path} ({Path(pbf_path).stat().st_size / 1e6:.0f} MB)", flush=True)
    aggregate = aggregate_pbf(Path(pbf_path), valid_cells, bbox=bbox)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"pbf": str(pbf_path), "n_cells": len(valid_cells), "aggregate": aggregate}),
        encoding="utf-8",
    )
    print("OSM stats:", json.dumps(aggregate["stats"], indent=2))
    return aggregate
