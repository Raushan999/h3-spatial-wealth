# Data dictionary — KA H3 R7 cells

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

Zero-inflation split: `{col}_exists` and `{col}_log` = log10(value) if value>0 else 0, then z-scored (`*_z`).
Feature columns used by the GNN: population_exists_z, population_log_z, night_lights_exists_z, night_lights_log_z, road_km_exists_z, road_km_log_z, poi_count_exists_z, poi_count_log_z, building_area_m2_exists_z, building_area_m2_log_z, building_count_exists_z, building_count_log_z, poi_amenity_restaurant_exists_z, poi_amenity_restaurant_log_z, poi_amenity_cafe_exists_z, poi_amenity_cafe_log_z, poi_amenity_bar_exists_z, poi_amenity_bar_log_z, poi_amenity_pub_exists_z, poi_amenity_pub_log_z, poi_amenity_cinema_exists_z, poi_amenity_cinema_log_z, poi_amenity_theatre_exists_z, poi_amenity_theatre_log_z, poi_amenity_nightclub_exists_z, poi_amenity_nightclub_log_z, poi_amenity_university_exists_z, poi_amenity_university_log_z, poi_amenity_hospital_exists_z, poi_amenity_hospital_log_z, poi_amenity_bank_exists_z, poi_amenity_bank_log_z, poi_shop_mall_exists_z, poi_shop_mall_log_z, poi_shop_luxury_exists_z, poi_shop_luxury_log_z, poi_shop_supermarket_exists_z, poi_shop_supermarket_log_z, poi_shop_convenience_exists_z, poi_shop_convenience_log_z, poi_tourism_hotel_exists_z, poi_tourism_hotel_log_z, poi_tourism_guest_house_exists_z, poi_tourism_guest_house_log_z, poi_leisure_exists_z, poi_leisure_log_z, poi_office_exists_z, poi_office_log_z, poi_amenity_restaurant_density_exists_z, poi_amenity_restaurant_density_log_z, poi_amenity_cafe_density_exists_z, poi_amenity_cafe_density_log_z, poi_amenity_bar_density_exists_z, poi_amenity_bar_density_log_z, poi_amenity_pub_density_exists_z, poi_amenity_pub_density_log_z, poi_amenity_cinema_density_exists_z, poi_amenity_cinema_density_log_z, poi_amenity_theatre_density_exists_z, poi_amenity_theatre_density_log_z, poi_amenity_nightclub_density_exists_z, poi_amenity_nightclub_density_log_z, poi_amenity_university_density_exists_z, poi_amenity_university_density_log_z, poi_amenity_hospital_density_exists_z, poi_amenity_hospital_density_log_z, poi_amenity_bank_density_exists_z, poi_amenity_bank_density_log_z, poi_shop_mall_density_exists_z, poi_shop_mall_density_log_z, poi_shop_luxury_density_exists_z, poi_shop_luxury_density_log_z, poi_shop_supermarket_density_exists_z, poi_shop_supermarket_density_log_z, poi_shop_convenience_density_exists_z, poi_shop_convenience_density_log_z, poi_tourism_hotel_density_exists_z, poi_tourism_hotel_density_log_z, poi_tourism_guest_house_density_exists_z, poi_tourism_guest_house_density_log_z, poi_leisure_density_exists_z, poi_leisure_density_log_z, poi_office_density_exists_z, poi_office_density_log_z, landuse_residential_z, landuse_residential_exists_z, landuse_commercial_z, landuse_commercial_exists_z, landuse_industrial_z, landuse_industrial_exists_z, landuse_retail_z, landuse_retail_exists_z, building_share_z, building_share_exists_z

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
