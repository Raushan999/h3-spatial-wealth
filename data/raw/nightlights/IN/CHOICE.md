# Night lights product

Training uses **VIIRS VNL v2.2 average_masked (annual mean radiance)**, read from the India-wide H3 table
`data\processed\h3_nightlights_2025.parquet` (column `nightlight_mean`), aggregated to H3 R7 by the pipeline.

- H3 table: `h3_nightlights_2025.parquet` (149 MB, H3 R8 mean/median/max radiance)
- raw `average_masked` composite kept in `data/raw/nightlights/`: ['VNL_npp_2025_global_vcmslcfg_v2_c202604011200.average_masked.dat.tif']
- `median_masked` and `unmasked` composites are ignored for training.

average_masked is the cloud- and fire-screened annual mean, which matches the
paper's lights feature (mean radiance per H3 cell, not a sum).
