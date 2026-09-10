"""Pipeline configuration: H3 setup, shared sources, and per-state GDDP / bbox maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUT = DATA / "output"
WEB = ROOT / "web"

H3_RES = 7
PARENT_RES = [6, 5]

# Default exposure / feature population. WorldPop remains a config fallback.
POPULATION_SOURCE: Literal["kontur", "worldpop"] = "kontur"

# Night lights. The India-wide H3 table is the pre-aggregated derivative of the
# EOG VNL v2.2 `average_masked` annual composite; reading it beats re-scanning
# the 11 GB global GeoTIFF for every state. Falls back to the raster if absent.
NIGHTLIGHTS_H3_PARQUET = PROCESSED / "h3_nightlights_2025.parquet"
NIGHTLIGHTS_H3_VALUE = "nightlight_mean"  # average_masked mean radiance
NIGHTLIGHTS_YEAR = "2025"
NIGHTLIGHTS_PRODUCT = "VIIRS VNL v2.2 average_masked (annual mean radiance)"

N_SEEDS = 5
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 50
HIDDEN = 64
LAYERS = 4
GCNII_ALPHA = 0.1
LR = 1e-2
FLOOR_WEIGHT = 0.05  # do not change the loss formula
TAU_FLOOR = 1.0

PriceBasis = Literal["constant", "current"]


@dataclass(frozen=True)
class GddpSource:
    """How to load official district (or NCT-wide) GDDP for one state."""

    # ckan_json | opencity_csv | records_json | gsdp_single | local_csv | pdf
    # | xlsx_matrix (districts across columns) | xlsx_table (districts down rows)
    # | pdf_table (text-extractable district x year table)
    kind: str
    url: str
    page: str
    year: str
    units: str  # e.g. "INR crore"
    price_basis: PriceBasis
    base_year: str | None
    district_field: str
    value_field: str
    resource_id: str | None = None
    skip_tokens: tuple[str, ...] = ("div.", "includes", "revised", "estimates", "total", "state")
    notes: tuple[str, ...] = ()
    filename: str = "gddp.json"
    # Extra knobs used by the spreadsheet / PDF parsers.
    sheet: str | None = None
    row_label: str | None = None
    value_scale: float = 1.0  # multiply parsed value to reach INR crore
    drop_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class StateConfig:
    code: str
    name: str
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy (WGS84)
    adm1_names: tuple[str, ...]
    rename_map: dict[str, str]
    pop_sanity_million: tuple[float, float]
    center: tuple[float, float]
    zoom: float
    gddp: GddpSource | None = None
    metros: dict[str, tuple[str, ...]] = field(default_factory=dict)
    enabled: bool = True
    skip_reason: str | None = None
    dissolve_to_single: str | None = None  # e.g. NCT GSDP as one Y_g unit
    osm_zone: str | None = None  # Geofabrik India sub-region holding this state


SOURCES = {
    "geoboundaries_adm2_api": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/",
    "geoboundaries_adm1_api": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM1/",
    "worldpop_1km": (
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
        "2020/IND/ind_ppp_2020_1km_Aggregated.tif"
    ),
    "worldpop_1km_unadj": (
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
        "2020/IND/ind_ppp_2020_1km_Aggregated_UNadj.tif"
    ),
    "kontur_hdx": "https://data.humdata.org/dataset/kontur-population-india",
    "kontur_gpkg_gz": (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/"
        "kontur_datasets/kontur_population_IN_20231101.gpkg.gz"
    ),
    "viirs_eog_index": "https://eogdata.mines.edu/nighttime_light/annual/v22/2023/",
    "geofabrik_india": "https://download.geofabrik.de/asia/india.html",
    "geofabrik_zone": "https://download.geofabrik.de/asia/india/{zone}-latest.osm.pbf",
    "overpass": "https://overpass-api.de/api/interpreter",
    "overpass_mirrors": (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
    ),
}

# Paper-style POI tag groups that proxy premium vs mass consumption.
# Sparse classes stay as mostly-zero columns; we do not invent counts.
POI_GROUPS: dict[str, tuple[str, frozenset[str] | None]] = {
    "amenity_restaurant": ("amenity", frozenset({"restaurant"})),
    "amenity_cafe": ("amenity", frozenset({"cafe"})),
    "amenity_bar": ("amenity", frozenset({"bar"})),
    "amenity_pub": ("amenity", frozenset({"pub"})),
    "amenity_cinema": ("amenity", frozenset({"cinema"})),
    "amenity_theatre": ("amenity", frozenset({"theatre", "theater"})),
    "amenity_nightclub": ("amenity", frozenset({"nightclub"})),
    "amenity_university": ("amenity", frozenset({"university"})),
    "amenity_hospital": ("amenity", frozenset({"hospital"})),
    "amenity_bank": ("amenity", frozenset({"bank"})),
    "shop_mall": ("shop", frozenset({"mall", "department_store"})),
    "shop_luxury": ("shop", frozenset({"jewelry", "jewellery", "watches", "boutique"})),
    "shop_supermarket": ("shop", frozenset({"supermarket"})),
    "shop_convenience": ("shop", frozenset({"convenience"})),
    "tourism_hotel": ("tourism", frozenset({"hotel"})),
    "tourism_guest_house": ("tourism", frozenset({"guest_house"})),
    "leisure": ("leisure", None),
    "office": ("office", None),
}

LANDUSE_CLASSES = ("residential", "commercial", "industrial", "retail")

# Magnitude columns get exists + log10 + z (paper §4.2). Share columns are z-scored as-is.
MAGNITUDE_FEATURE_BASE = [
    "population",
    "night_lights",
    "road_km",
    "poi_count",
    "building_area_m2",
    "building_count",
] + [f"poi_{k}" for k in POI_GROUPS] + [f"poi_{k}_density" for k in POI_GROUPS]

SHARE_FEATURE_BASE = [f"landuse_{c}" for c in LANDUSE_CLASSES] + ["building_share"]


def raw_layer(layer: str, key: str = "IN") -> Path:
    return RAW / layer / key


MH_RENAME = {
    "mumbai": "Mumbai",
    "mumbai city": "Mumbai",
    "mumbai suburban": "Mumbai",
    "greater mumbai": "Mumbai",
    "thane": "Thane",
    "palghar": "Thane",
    "ahmednagar": "Ahmednagar",
    "ahmadnagar": "Ahmednagar",
    "ahilyanagar": "Ahmednagar",
    "aurangabad": "Chhatrapati Sambhajinagar",
    "chhatrapati sambhajinagar": "Chhatrapati Sambhajinagar",
    "osmanabad": "Dharashiv",
    "dharashiv": "Dharashiv",
    "beed": "Beed",
    "bid": "Beed",
    "buldhana": "Buldhana",
    "buldana": "Buldhana",
    "gondia": "Gondia",
    "gondiya": "Gondia",
    "raigad": "Raigad",
    "raigarh": "Raigad",
}

KA_RENAME = {
    "bangalore urban": "Bengaluru Urban",
    "bengaluru urban": "Bengaluru Urban",
    "bangalore": "Bengaluru Urban",
    "bengaluru": "Bengaluru Urban",
    # Ramanagara was renamed "Bengaluru South" in 2024; geoBoundaries still calls
    # the polygon Ramanagara, so the Economic Survey row maps onto it.
    "bangalore south": "Ramanagara",
    "bengaluru south": "Ramanagara",
    "ramanagara": "Ramanagara",
    "bangalore rural": "Bengaluru Rural",
    "bengaluru rural": "Bengaluru Rural",
    "belgaum": "Belagavi",
    "belagavi": "Belagavi",
    "bellary": "Ballari",
    "ballari": "Ballari",
    "vijayanagar": "Ballari",
    "vijayanagara": "Ballari",
    "bijapur": "Vijayapura",
    "vijayapura": "Vijayapura",
    "gulbarga": "Kalaburagi",
    "kalaburagi": "Kalaburagi",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "tumkur": "Tumakuru",
    "tumakuru": "Tumakuru",
    "shimoga": "Shivamogga",
    "shivamogga": "Shivamogga",
    "chikmagalur": "Chikkamagaluru",
    "chikkamagaluru": "Chikkamagaluru",
    "bagalkot": "Bagalkote",
    "bagalkote": "Bagalkote",
    "chikkaballapur": "Chikkaballapura",
    "chikkaballapura": "Chikkaballapura",
    "yadagiri": "Yadgir",
    "yadgir": "Yadgir",
}

TG_RENAME = {
    "bhadradri": "Bhadradri Kothagudem",
    "bhadradri kothagudem": "Bhadradri Kothagudem",
    "hanumakonda": "Hanamkonda",
    "hanamkonda": "Hanamkonda",
    "warangal (u)": "Hanamkonda",
    "warangal (urban)": "Hanamkonda",
    "warangal urban": "Hanamkonda",
    "warangal (r)": "Warangal",
    "warangal rural": "Warangal",
    "hydrabad": "Hyderabad",
    "hyderabad": "Hyderabad",
    "jayashankar": "Jayashankar Bhupalpally",
    "jayashankar bhupalpally": "Jayashankar Bhupalpally",
    "jogulamba": "Jogulamba Gadwal",
    "jogulamba gadwal": "Jogulamba Gadwal",
    "kumuram bheem": "Kumuram Bheem Asifabad",
    "komaram bheem": "Kumuram Bheem Asifabad",
    "asifabad": "Kumuram Bheem Asifabad",
    "medchal": "Medchal Malkajgiri",
    "medchal-malkajgiri": "Medchal Malkajgiri",
    "medchal malkajgiri": "Medchal Malkajgiri",
    "rangareddy": "Ranga Reddy",
    "ranga reddy": "Ranga Reddy",
    "yadadri": "Yadadri Bhuvanagiri",
    "yadadri bhuvanagiri": "Yadadri Bhuvanagiri",
    "yadadri bhongiri": "Yadadri Bhuvanagiri",
    "rajanna sircilla": "Rajanna Sircilla",
    "sircilla": "Rajanna Sircilla",
}

# geoBoundaries ADM2 still uses pre-rename UP names; the DES GDDP sheet uses a
# mix of old and new spellings. Both sides fold onto the same canonical string.
UP_RENAME = {
    "gautam buddha nagar": "Gautam Buddha Nagar",
    "gautambudh nagar": "Gautam Buddha Nagar",
    "gautambuddha nagar": "Gautam Buddha Nagar",
    "gb nagar": "Gautam Buddha Nagar",
    "prayagraj": "Prayagraj",
    "allahabad": "Prayagraj",
    "ayodhya": "Ayodhya",
    "faizabad": "Ayodhya",
    "sambhal": "Sambhal",
    "bhim nagar": "Sambhal",
    "amroha": "Amroha",
    "jyotiba phule nagar": "Amroha",
    "kasganj": "Kasganj",
    "kanshiram nagar": "Kasganj",
    "shamli": "Shamli",
    "samli": "Shamli",
    "hapur": "Hapur",
    "panchsheel nagar": "Hapur",
    "hathras": "Hathras",
    "mahamaya nagar": "Hathras",
    "muzaffar nagar": "Muzaffarnagar",
    "muzaffarnagar": "Muzaffarnagar",
    "buland shahar": "Bulandshahr",
    "bulandshahr": "Bulandshahr",
    "badaun": "Budaun",
    "budaun": "Budaun",
    "auraiyya": "Auraiya",
    "auraiya": "Auraiya",
    "bara banki": "Barabanki",
    "barabanki": "Barabanki",
    "raebareilly": "Rae Bareli",
    "rae bareli": "Rae Bareli",
    "kheri": "Lakhimpur Kheri",
    "lakhimpur kheri": "Lakhimpur Kheri",
    "kushi nagar": "Kushinagar",
    "kushinagar": "Kushinagar",
    "maharajganj": "Mahrajganj",
    "mahrajganj": "Mahrajganj",
    "sant kabeer nagar": "Sant Kabir Nagar",
    "sant kabir nagar": "Sant Kabir Nagar",
    "siddharth nagar": "Siddharthnagar",
    "siddharthnagar": "Siddharthnagar",
    "shravasti": "Shrawasti",
    "shrawasti": "Shrawasti",
    "bhadohi": "Bhadohi",
    "sant ravidas nagar (bhadohi)": "Bhadohi",
    "sant ravidas nagar": "Bhadohi",
}

WB_RENAME = {
    "north 24 parganas": "North 24 Parganas",
    "north twenty four parganas": "North 24 Parganas",
    "south 24 parganas": "South 24 Parganas",
    "south twenty four parganas": "South 24 Parganas",
    "howrah": "Howrah",
    "haora": "Howrah",
    "hooghly": "Hooghly",
    "hugli": "Hooghly",
    "purba bardhaman": "Purba Bardhaman",
    "barddhaman": "Purba Bardhaman",
    "burdwan": "Purba Bardhaman",
    "paschim bardhaman": "Paschim Bardhaman",
    "east midnapore": "Purba Medinipur",
    "purba medinipur": "Purba Medinipur",
    "west midnapore": "Paschim Medinipur",
    "paschim medinipur": "Paschim Medinipur",
    "cooch behar": "Cooch Behar",
    "koch bihar": "Cooch Behar",
    "darjeeling": "Darjeeling",
    "darjiling": "Darjeeling",
}

BR_RENAME = {
    "pashchim champaran": "West Champaran",
    "west champaran": "West Champaran",
    "purba champaran": "East Champaran",
    "east champaran": "East Champaran",
    "kaimur": "Kaimur",
    "kaimur (bhabua)": "Kaimur",
    "bhabua": "Kaimur",
    "nawada": "Nawada",
    "nawadah": "Nawada",
    "narwada": "Nawada",
    "saharsa": "Saharsa",
    "sahasra": "Saharsa",
    "purnia": "Purnia",
    "purnea": "Purnia",
}

DL_RENAME = {
    "nct of delhi": "Delhi",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "central": "Delhi",
    "central delhi": "Delhi",
    "east": "Delhi",
    "east delhi": "Delhi",
    "north": "Delhi",
    "north delhi": "Delhi",
    "north east": "Delhi",
    "north east delhi": "Delhi",
    "north west": "Delhi",
    "north west delhi": "Delhi",
    "south": "Delhi",
    "south delhi": "Delhi",
    "south east": "Delhi",
    "south east delhi": "Delhi",
    "south west": "Delhi",
    "south west delhi": "Delhi",
    "west": "Delhi",
    "west delhi": "Delhi",
    "shahdara": "Delhi",
}

RJ_RENAME = {
    "sri ganganagar": "Ganganagar",
    "ganganagar": "Ganganagar",
    "pratapgarh": "Pratapgarh",
    "karauli": "Karauli",
    "karouli": "Karauli",
    "sawai madhopur": "Sawai Madhopur",
    "s. madhopur": "Sawai Madhopur",
    "chittorgarh": "Chittaurgarh",
    "chittaurgarh": "Chittaurgarh",
    "dholpur": "Dhaulpur",
    "dhaulpur": "Dhaulpur",
    "jalore": "Jalor",
    "jalor": "Jalor",
    "jhunjhunu": "Jhunjhunun",
    "jhunjhunun": "Jhunjhunun",
}


STATES: dict[str, StateConfig] = {
    "MH": StateConfig(
        code="MH",
        osm_zone="western-zone",
        name="Maharashtra",
        bbox=(72.55, 15.55, 80.95, 22.15),
        adm1_names=("maharashtra",),
        rename_map=MH_RENAME,
        pop_sanity_million=(110.0, 140.0),
        center=(76.0, 19.05),
        zoom=6.15,
        metros={"Mumbai Metropolitan": ("Mumbai", "Thane", "Raigad", "Palghar")},
        gddp=GddpSource(
            kind="ckan_json",
            url=(
                "https://data.opencity.in/api/3/action/datastore_search"
                "?resource_id=4ea748f2-308b-42fa-a736-0fccc4707597&limit=100"
            ),
            page="https://data.opencity.in/dataset/economic-survey-of-maharashtra-2024-25",
            year="2023-24",
            units="INR crore",
            price_basis="constant",
            base_year="2011-12",
            district_field="District",
            value_field="Real GDDP 2023-24 (Constant prices from 2011-12)",
            resource_id="4ea748f2-308b-42fa-a736-0fccc4707597",
            filename="maharashtra_gddp_economic_survey_2024_25.json",
            notes=(
                "Mumbai# = Mumbai City + Mumbai Suburban",
                "Thane$ = Thane + Palghar",
                "Source: DES Maharashtra via OpenCity CKAN (public domain extract of Economic Survey 2024-25)",
            ),
        ),
    ),
    "UP": StateConfig(
        code="UP",
        osm_zone="central-zone",
        name="Uttar Pradesh",
        bbox=(77.05, 23.85, 84.65, 30.45),
        adm1_names=("uttar pradesh",),
        rename_map=UP_RENAME,
        pop_sanity_million=(199.0, 250.0),
        center=(80.9, 26.85),
        zoom=6.0,
        metros={"NCR (UP districts)": ("Gautam Buddha Nagar", "Ghaziabad", "Meerut", "Bulandshahr", "Hapur", "Baghpat")},
        gddp=GddpSource(
            kind="xlsx_matrix",
            url="",
            page="https://updes.up.nic.in/updes/dist_domestic_product.html",
            year="2021-22",
            units="INR crore",
            price_basis="constant",
            base_year="2011-12",
            district_field="district",
            value_field="y_crore",
            filename="GDDP Constant.xlsx",
            sheet="GDDP Constant",
            row_label="GROSS DISTRICT DOMESTIC PRODUCT",
            drop_names=("western region", "central region", "bundel khand region", "eastern region", "uttar pradesh"),
            notes=(
                "DES Uttar Pradesh, Gross District Domestic Product by economic activity 2021-22 (tentative), base 2011-12, constant prices, ₹ crore.",
                "Row used is 'GROSS DISTRICT DOMESTIC PRODUCT (At Market Prices)'. The four regional aggregates and the state total are dropped.",
                "A 'GDDP Current.xlsx' sits beside it; the constant-price sheet is used so the series is real.",
            ),
        ),
    ),
    "BR": StateConfig(
        code="BR",
        osm_zone="eastern-zone",
        name="Bihar",
        bbox=(83.25, 24.20, 88.55, 27.65),
        adm1_names=("bihar",),
        rename_map=BR_RENAME,
        pop_sanity_million=(100.0, 140.0),
        center=(85.5, 25.6),
        zoom=6.4,
        gddp=GddpSource(
            kind="xlsx_table",
            url="",
            page="https://state.bihar.gov.in/finance/CitizenHome.html",
            year="2023-24",
            units="INR crore",
            price_basis="current",
            base_year=None,
            district_field="District",
            value_field="Nominal GDP in Crore INR",
            filename="Bihar Districts GDDP.xlsx",
            sheet="Sheet1",
            drop_names=("bihar total", "bihar"),
            notes=(
                "Bihar district income estimates 2023-24, nominal (current-price) district GDP in ₹ crore, as compiled from the Bihar Economic Survey district income series.",
                "Current prices: do not compare levels against constant-price states without deflating.",
            ),
        ),
    ),
    "DL": StateConfig(
        code="DL",
        osm_zone="northern-zone",
        name="Delhi (NCT)",
        bbox=(76.82, 28.40, 77.35, 28.89),
        adm1_names=("nct of delhi", "delhi"),
        rename_map=DL_RENAME,
        pop_sanity_million=(16.0, 25.0),
        center=(77.1, 28.65),
        zoom=9.4,
        dissolve_to_single="Delhi",
        metros={"Delhi NCT": ("Delhi",)},
        gddp=GddpSource(
            kind="gsdp_single",
            url="https://des.delhi.gov.in/sites/default/files/DES/generic_multiple_files/statistical_abstract-2024.pdf",
            page="https://des.delhi.gov.in/",
            year="2023-24",
            units="INR crore",
            price_basis="constant",
            base_year="2011-12",
            district_field="district",
            value_field="y_crore",
            filename="delhi_statistical_abstract_2024.pdf",
            row_label="GSDP at constant",
            notes=(
                "Figure is **Delhi / NCT** only, not NCR. Source: Statistical Abstract of Delhi 2024 (DES, Govt of NCT of Delhi), Executive Summary A: 'GSDP at constant (2011-12) prices ... ₹672247 Crore in 2023-24'.",
                "DES Delhi does not publish district GDDP, so the 11 NCT revenue districts form a single Y_g reporting unit. No NCR super-district is invented.",
            ),
        ),
    ),
    "RJ": StateConfig(
        code="RJ",
        osm_zone="northern-zone",
        name="Rajasthan",
        bbox=(69.45, 23.03, 78.28, 30.20),
        adm1_names=("rajasthan",),
        rename_map=RJ_RENAME,
        pop_sanity_million=(68.0, 90.0),
        center=(74.2, 26.6),
        zoom=6.1,
        metros={"NCR (RJ districts)": ("Alwar",)},
        gddp=GddpSource(
            kind="pdf_table",
            url="https://desddp.rajasthan.gov.in/PDF/Publication_12102024%20102450%20AM.pdf",
            page="https://desddp.rajasthan.gov.in/Publications.aspx",
            year="2023-24",
            units="INR crore",
            price_basis="constant",
            base_year="2011-12",
            district_field="district",
            value_field="y_crore",
            filename="rajasthan_ddp_2023_24.pdf",
            row_label="Table 3: Gross District Domestic Product of Rajasthan at Constant (2011-12) Prices",
            value_scale=0.01,  # published in ₹ lakh -> ₹ crore
            drop_names=("state",),
            notes=(
                "DES Rajasthan, 'Estimates of District Domestic Product of Rajasthan 2011-12 to 2023-24' (Nov 2024), Table 3: GDDP at constant 2011-12 prices, ₹ lakh, 2023-24 (provisional).",
                "The publication uses the 33-district structure, which matches the geoBoundaries ADM2 layer one-to-one.",
            ),
        ),
    ),
    "WB": StateConfig(
        code="WB",
        osm_zone="eastern-zone",
        name="West Bengal",
        bbox=(85.75, 21.45, 89.90, 27.25),
        adm1_names=("west bengal",),
        rename_map=WB_RENAME,
        pop_sanity_million=(90.0, 115.0),
        center=(87.8, 23.8),
        zoom=6.3,
        metros={"Kolkata metro": ("Kolkata", "North 24 Parganas", "South 24 Parganas", "Howrah", "Hooghly")},
        enabled=False,
        gddp=GddpSource(
            kind="local_csv",
            url="",
            page="https://www.wbpspm.gov.in/",
            year="2020-21",
            units="INR crore",
            price_basis="current",
            base_year="2011-12",
            district_field="district",
            value_field="y_crore",
            filename="gddp.csv",
            notes=(
                "No recent machine-readable DDP on DES / data.gov.in / OpenCity. Skip unless a local official CSV is placed at data/raw/gddp/WB/gddp.csv. Do not use Wikipedia or paywalled tables.",
            ),
        ),
        skip_reason="No free machine-readable official district GDDP found (DES/OpenCity/data.gov.in).",
    ),
    "KA": StateConfig(
        code="KA",
        osm_zone="southern-zone",
        name="Karnataka",
        bbox=(74.05, 11.50, 78.60, 18.48),
        adm1_names=("karnataka",),
        rename_map=KA_RENAME,
        pop_sanity_million=(60.0, 80.0),
        center=(76.5, 15.0),
        zoom=6.2,
        metros={"Bengaluru metro": ("Bengaluru Urban", "Bengaluru Rural")},
        gddp=GddpSource(
            kind="opencity_csv",
            url=(
                "https://data.opencity.in/api/3/action/datastore_search"
                "?resource_id=27684453-1635-4571-b5f1-94c706303037&limit=200"
            ),
            page="https://data.opencity.in/dataset/economic-survey-of-karnataka-2025-26",
            year="2024-25",
            units="INR crore",
            price_basis="constant",
            base_year="2011-12",
            district_field="District",
            value_field="GDDP Constant Prices (Rs. Cr)",
            resource_id="27684453-1635-4571-b5f1-94c706303037",
            filename="karnataka_gddp_economic_survey_2025_26.json",
            notes=(
                "DES Karnataka via OpenCity, Economic Survey 2025-26 district GDDP (constant prices).",
                "geoBoundaries 2021 has no Bengaluru South or Vijayanagara polygons: those GDDP rows are merged into Bengaluru Urban and Ballari. Ramanagara has no GDDP row in this table so those ADM2 cells are dropped.",
            ),
        ),
    ),
    "TG": StateConfig(
        code="TG",
        osm_zone="southern-zone",
        name="Telangana",
        bbox=(77.20, 15.70, 81.80, 19.95),
        adm1_names=("telangana",),
        rename_map=TG_RENAME,
        pop_sanity_million=(32.0, 45.0),
        center=(79.0, 17.9),
        zoom=6.6,
        metros={"Hyderabad metro": ("Hyderabad", "Ranga Reddy", "Medchal Malkajgiri", "Sangareddy")},
        gddp=GddpSource(
            kind="opencity_csv",
            url=(
                "https://data.opencity.in/api/3/action/datastore_search"
                "?resource_id=9acd8282-8081-4a46-af8b-dbd1ba57adac&limit=200"
            ),
            page="https://data.opencity.in/dataset/telangana-socio-economic-outlook-2024",
            year="2022-23",
            units="INR crore",
            price_basis="current",
            base_year=None,
            district_field="District",
            value_field="GDDP (Cr) at Current Prices 2022-23",
            resource_id="9acd8282-8081-4a46-af8b-dbd1ba57adac",
            filename="telangana_gddp_seo_2024.json",
            notes=(
                "DES Telangana via OpenCity, Socio-Economic Outlook 2024. Latest free table is current-price GDDP 2022-23 (no constant-price column in the CSV).",
            ),
        ),
    ),
}

# Backward-compatible aliases used by older MH-only scripts.
STATE_NAME = STATES["MH"].name
TARGET_YEAR = STATES["MH"].gddp.year if STATES["MH"].gddp else "2023-24"
TARGET_COL = "y_crore"
MH_BBOX = STATES["MH"].bbox
COMBINE_TO = MH_RENAME
SKIP_DISTRICT_TOKENS = STATES["MH"].gddp.skip_tokens if STATES["MH"].gddp else ("div.",)


def get_state(code: str) -> StateConfig:
    key = code.strip().upper()
    if key not in STATES:
        known = ", ".join(sorted(STATES))
        raise KeyError(f"Unknown state {code!r}. Configured: {known}")
    return STATES[key]


def output_dir(code: str) -> Path:
    return OUTPUT / code.upper()


def processed_dir(code: str) -> Path:
    return PROCESSED / code.upper()
