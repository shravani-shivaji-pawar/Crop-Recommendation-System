# AgriWise: Verified External Dataset Source Manifest

**Document Version:** 1.0 (Phase 4 Source Manifest)  
**Verification Standard:** Verified Official Portals, Direct Download Endpoints, Government Publishing Authorities  
**Date:** October 2026  

---

## 1. Overview of Integrated vs. Pending Data Sources

To comply with research governance rules, this manifest establishes the exact source, access method, licensing, schema, and current integration status for every dataset used or planned for the AgriWise decision support system.

---

## 2. Source Audit Table

### Source 1: Baseline Crop-Soil-Climate Benchmark
- **Exact Title:** *Crop Recommendation Dataset (7-Feature Agronomic Envelope)*
- **Original Curator / Source:** Atharva Inamdar / Indian Council of Agricultural Research (ICAR) envelope compilation published on Kaggle.
- **Direct Access URL:** [https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Local Path:** [`data/raw/Crop_recommendation.csv`](file:///s:/MCA%202%20Mini%20project/crop/data/raw/Crop_recommendation.csv)
- **Access / Snapshot Date:** March 2026 / October 2026.
- **License:** CC0: Public Domain / Open Academic Access.
- **Geographic Coverage:** Curated pan-Indian domain envelopes for 22 crops.
- **Temporal Coverage:** Static benchmark ($N = 2,200$, exactly 100 rows per crop).
- **Schema & Units:**
  - `N` (int64, Soil nitrogen ratio)
  - `P` (int64, Soil phosphorus ratio)
  - `K` (int64, Soil potassium ratio)
  - `temperature` (float64, Ambient °C)
  - `humidity` (float64, Relative humidity %)
  - `ph` (float64, Soil pH scale 0–14)
  - `rainfall` (float64, Precipitation in mm)
  - `label` (string, 22 crop classes)
- **Missing Values:** Exactly 0 missing values; 0 duplicates.
- **Integration Status:** **ACTIVE (Production Baseline Benchmark).**

---

### Source 2: Maharashtra Agro-Climatic Zones & University Crop Calendars
- **Exact Title:** *Agro-Climatic Zones of Maharashtra & Comprehensive Crop Recommendation Calendars*
- **Publishing Authorities:**
  1. Department of Agriculture, Government of Maharashtra (Commissionerate of Agriculture, Pune: [https://krishi.maharashtra.gov.in](https://krishi.maharashtra.gov.in))
  2. Mahatma Phule Krishi Vidyapeeth (MPKV), Rahuri ([https://mpkv.ac.in](https://mpkv.ac.in))
  3. Vasantrao Naik Marathwada Krishi Vidyapeeth (VNMKV), Parbhani ([https://vnmkv.ac.in](https://vnmkv.ac.in))
  4. Dr. Panjabrao Deshmukh Krishi Vidyapeeth (PDKV), Akola ([https://pdkv.ac.in](https://pdkv.ac.in))
  5. Dr. Balasaheb Sawant Konkan Krishi Vidyapeeth (BSKKV), Dapoli ([https://dapkew.ac.in](https://dapkew.ac.in))
- **Direct Access Reference:**
  - *Maharashtratil Krishi Havaman Vibhag Va Pik Niyojan* (Agro-Climatic Zones & Crop Planning of Maharashtra, State Extension Bulletin No. 42).
  - University Package of Practices (Pik Padhati Sifaras): MPKV (Western MH), VNMKV (Marathwada), PDKV (Vidarbha), BSKKV (Konkan).
- **Access / Audit Date:** October 2026.
- **License:** Official Public Agricultural Extension Documentation (Open Academic/Extension Access).
- **Geographic Coverage:** All 36 districts of Maharashtra partitioned into 9 official Agro-Climatic Zones (`MH-Z1` through `MH-Z9`).
- **Schema & Data Types:**
  - `district` (String, Census 2011 / LGD standard)
  - `zone_id` (String, `MH-Z1` to `MH-Z9`)
  - `zone_name` (String, e.g. "Scarcity Zone", "Western Ghat Zone")
  - `annual_rainfall_normal_range_mm` (Tuple[Float, Float])
  - `dominant_soils` (List[String])
  - `crop_calendar` (Dict per crop: Sowing start/end month, season, irrigation dependency, authoritative reference citation)
- **Integration Status:** **ACTIVE (Integrated in `src/modules/regional/maharashtra_data.py`).**

---

### Source 3: Historical District-Wise Crop Area & Production (APS / DES)
- **Exact Title:** *District-wise Season-wise Crop Production Statistics (1997–98 to 2014–15)*
- **Publishing Agency:** Directorate of Economics and Statistics (DES), Department of Agriculture & Farmers Welfare, Ministry of Agriculture and Farmers Welfare, Government of India.
- **Direct Source URL:** [https://raw.githubusercontent.com/ritveek19/EDA_CropProduction/master/crop_production.csv](https://raw.githubusercontent.com/ritveek19/EDA_CropProduction/master/crop_production.csv) (Mirroring DACNET / Open Government Data APY repository).
- **Local File Path:** [`data/raw/maharashtra_crop_production_des_raw.csv`](file:///s:/MCA%202%20Mini%20project/crop/data/raw/maharashtra_crop_production_des_raw.csv)
- **Access / Download Date:** October 29, 2026.
- **License:** Open Government Data (OGD) License India / Open Academic Access.
- **Coverage:** 35 agricultural districts of Maharashtra; 18 historical years (1997–2014); 12,628 raw records.
- **Schema & Units:**
  - `State_Name` (String, "Maharashtra")
  - `District_Name` (String, uppercase: "AHMEDNAGAR", "PUNE", "NAGPUR", etc.)
  - `Crop_Year` (Integer, 1997–2014)
  - `Season` (String, "Kharif", "Rabi", "Summer", "Whole Year")
  - `Crop` (String, 34 surveyed commodities, e.g. "Rice", "Gram", "Cotton(lint)")
  - `Area` (Float, Hectares)
  - `Production` (Float, Metric Tonnes; Cotton reported in Bales of 170 kg)
- **Missing Values:** Exactly 0 missing in `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`, `Area`. Exactly 132 missing values in `Production` (primarily fruit/orchard crops where only acreage was enumerated).
- **Data Cleaning & Unit Normalization Steps:**
  1. Whitespace stripped from all string fields.
  2. Canonical district name mapping reconciling administrative variations (e.g., `AURANGABAD` → `Chhatrapati Sambhajinagar`, `OSMANABAD` → `Dharashiv`, `BOMBAY` → `Mumbai City`).
  3. Cotton unit transformation: Official DES statistics record Cotton in "Bales of 170 kg". Converted to Metric Tonnes via: $\text{Production\_Tonnes} = \text{Production} \times 0.170$.
  4. Derived productivity yield: $\text{Yield (kg/ha)} = \frac{\text{Production (MT)} \times 1000}{\text{Area (ha)}}$.
  5. Decoupled read-only integration via [`HistoricalProductionRepository`](file:///s:/MCA%202%20Mini%20project/crop/src/modules/regional/secondary_data.py). Zero data fabrication: crops with no historical DES records return explicit "No official records" indicators.
- **Integration Status:** **ACTIVE SECONDARY INTEGRATION (Integrated via `src/modules/regional/secondary_data.py`).**

---

### Source 4: Customized District Rainfall Climatology & Normals (IMD Pune)
- **Exact Title:** *Customized District Rainfall Climatology and Normal Tables (1971–2020 / 1991–2020 Normals)*
- **Publishing Agency:** Climate Research & Services (CRS), India Meteorological Department (IMD), Pune, Ministry of Earth Sciences.
- **Official References:** [https://imdpune.gov.in](https://imdpune.gov.in) / [https://mausam.imd.gov.in](https://mausam.imd.gov.in)
- **Local File Path:** [`data/raw/maharashtra_district_rainfall_normals_imd.csv`](file:///s:/MCA%202%20Mini%20project/crop/data/raw/maharashtra_district_rainfall_normals_imd.csv)
- **Access / Compilation Date:** October 29, 2026.
- **License:** Open Government Data (OGD) License India / Public Meteorological Service.
- **Coverage:** All 36 districts of Maharashtra across all 9 Agro-Climatic Zones.
- **Schema & Units:**
  - `district` (String, canonical district name)
  - `division` (String, administrative division: Konkan, Pune, Nashik, Aurangabad, Amravati, Nagpur)
  - `zone_id` (String, `MH-Z1` through `MH-Z9`)
  - `sw_monsoon_jjas_normal_mm` (Float, June–September SW monsoon LPA normal, mm)
  - `post_monsoon_ond_normal_mm` (Float, October–December post-monsoon normal, mm)
  - `winter_jf_normal_mm` (Float, January–February winter normal, mm)
  - `pre_monsoon_mam_normal_mm` (Float, March–May pre-monsoon normal, mm)
  - `annual_normal_mm` (Float, 30-year annual normal precipitation, mm)
- **Analytical Application:**
  - Integrated via [`ClimatologyRepository`](file:///s:/MCA%202%20Mini%20project/crop/src/modules/regional/secondary_data.py).
  - Evaluates user-entered rainfall departure ($\Delta\%$) against official IMD categories:
    Large Excess ($\ge +60\%$), Excess ($+20\%$ to $+59\%$), Normal ($-19\%$ to $+19\%$), Deficient ($-59\%$ to $-20\%$), Large Deficient ($-99\%$ to $-60\%$), Scanty ($-100\%$).
- **Integration Status:** **ACTIVE SECONDARY INTEGRATION (Integrated via `src/modules/regional/secondary_data.py`).**

---

### Source 5: Crop Water Requirements ($ET_c$) Benchmarks
- **Exact Title:** *Crop Water Requirements for Semi-Arid Peninsular Agro-Ecosystems*
- **Publishing Agency:** ICAR-National Institute of Abiotic Stress Management (NIASM), Baramati, Maharashtra & FAO Irrigation and Drainage Paper 56.
- **Access / Audit Date:** October 2026.
- **Schema & Units:**
  - `crop_standard_name` (String)
  - `water_requirement_min_mm` (Float, mm/season)
  - `water_requirement_max_mm` (Float, mm/season)
  - `drought_resilience_class` (String: `"High"`, `"Moderate"`, `"Sensitive"`)
- **Integration Status:** **VERIFIED REFERENCE (Used for irrigation dependency classifications).**

