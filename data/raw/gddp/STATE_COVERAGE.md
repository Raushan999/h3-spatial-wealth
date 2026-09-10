# State GDDP coverage

Official open government / OpenCity / data.gov.in only. Paywalled tables (Dataful etc.) are skipped.

| code | state | status | year | units | prices | source | notes |
|---|---|---|---|---|---|---|---|
| MH | Maharashtra | in | 2023-24 | INR crore | constant 2011-12 | https://data.opencity.in/dataset/economic-survey-of-maharashtra-2024-25 | Mumbai# = Mumbai City + Mumbai Suburban; Thane$ = Thane + Palghar; Source: DES Maharashtra via OpenCity CKAN (public domain extract of Economic Survey 2024-25) |
| UP | Uttar Pradesh | in | 2021-22 | INR crore | constant 2011-12 | https://updes.up.nic.in/updes/dist_domestic_product.html | DES Uttar Pradesh, Gross District Domestic Product by economic activity 2021-22 (tentative), base 2011-12, constant prices, ₹ crore.; Row used is 'GROSS DISTRICT DOMESTIC PRODUCT (At Market Prices)'. The four regional aggregates and the state total are dropped.; A 'GDDP Current.xlsx' sits beside it; the constant-price sheet is used so the series is real. |
| BR | Bihar | in | 2023-24 | INR crore | current | https://state.bihar.gov.in/finance/CitizenHome.html | Bihar district income estimates 2023-24, nominal (current-price) district GDP in ₹ crore, as compiled from the Bihar Economic Survey district income series.; Current prices: do not compare levels against constant-price states without deflating. |
| DL | Delhi (NCT) | in | 2023-24 | INR crore | constant 2011-12 | https://des.delhi.gov.in/ | Figure is **Delhi / NCT** only, not NCR. Source: Statistical Abstract of Delhi 2024 (DES, Govt of NCT of Delhi), Executive Summary A: 'GSDP at constant (2011-12) prices ... ₹672247 Crore in 2023-24'.; DES Delhi does not publish district GDDP, so the 11 NCT revenue districts form a single Y_g reporting unit. No NCR super-district is invented. Reporting unit dissolved to `Delhi` (official GSDP, not an invented NCR super-district). |
| RJ | Rajasthan | in | 2023-24 | INR crore | constant 2011-12 | https://desddp.rajasthan.gov.in/Publications.aspx | DES Rajasthan, 'Estimates of District Domestic Product of Rajasthan 2011-12 to 2023-24' (Nov 2024), Table 3: GDDP at constant 2011-12 prices, ₹ lakh, 2023-24 (provisional).; The publication uses the 33-district structure, which matches the geoBoundaries ADM2 layer one-to-one. |
| WB | West Bengal | skip | 2020-21 | INR crore | current 2011-12 | https://www.wbpspm.gov.in/ | No recent machine-readable DDP on DES / data.gov.in / OpenCity. Skip unless a local official CSV is placed at data/raw/gddp/WB/gddp.csv. Do not use Wikipedia or paywalled tables. |
| KA | Karnataka | in | 2024-25 | INR crore | constant 2011-12 | https://data.opencity.in/dataset/economic-survey-of-karnataka-2025-26 | DES Karnataka via OpenCity, Economic Survey 2025-26 district GDDP (constant prices).; geoBoundaries 2021 has no Bengaluru South or Vijayanagara polygons: those GDDP rows are merged into Bengaluru Urban and Ballari. Ramanagara has no GDDP row in this table so those ADM2 cells are dropped. |
| TG | Telangana | in | 2022-23 | INR crore | current | https://data.opencity.in/dataset/telangana-socio-economic-outlook-2024 | DES Telangana via OpenCity, Socio-Economic Outlook 2024. Latest free table is current-price GDDP 2022-23 (no constant-price column in the CSV). |

## Merges

- **MH:** Mumbai City + Mumbai Suburban → Mumbai; Thane + Palghar → Thane; Aurangabad → Chhatrapati Sambhajinagar; Osmanabad → Dharashiv.
- **DL:** 11 NCT revenue districts dissolved to one Y_g = NCT GSDP because DES Delhi does not publish district GDDP.
- **NCR in the UI:** Haryana/UP/Rajasthan NCR districts appear only if those states' GDDP loaded. No synthetic NCR district.
- **WB / Kolkata:** Kolkata is a city; the model (if loaded) is West Bengal districts. Metro filter is UI-only.

Drop-in for PDF-only DES tables: `data/raw/gddp/{STATE}/gddp.csv` with columns `district,y_crore`.
