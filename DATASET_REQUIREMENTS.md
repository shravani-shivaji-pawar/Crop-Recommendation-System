# AgriWise: Dataset Integration Requirements & Provenance Specification

**Project Title:** AgriWise Decision Support System  
**Document Version:** 1.0 (Integration Specification)  
**Status:** Requirements Specification (Pending User Dataset Approval)  

---

## 1. Current Dataset Capability Matrix

| Analysis Dimension | Current Dataset (`Crop_recommendation.csv`) | Capability Status | Missing Data Needed for Full Vision |
|---|---|---|---|
| **7-Feature Agro-Climatic Suitability** | $N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}$ | **Fully Supported** | None (Baseline benchmark functional) |
| **Soil Texture & Organic Carbon** | Not present | **Not Supported** | State/District Soil Health Card summaries |
| **Regional / District Alignment** | No state/district columns | **Not Supported** | District boundary mapping & agro-climatic zoning |
| **Seasonal Sowing Windows** | No season/month columns | **Not Supported** | Kharif / Rabi / Zaid crop calendar tables |
| **Water Budgeting & Irrigation** | Total rainfall only | **Partially Supported** | Crop water requirement coefficients ($ET_c$, mm) & irrigation source data |
| **Economic Feasibility & Mandi Prices** | No price or cost columns | **Not Supported** | Agmarknet wholesale mandi prices & CACP cost of cultivation |
| **Historical Crop Yield Analysis** | No production/yield columns | **Not Supported** | DES / Ministry of Agriculture district crop yield statistics |

---

## 2. Specification for Future Supplementary Datasets

To ensure reliable, defensible integration without data fabrication, the following schemas define the exact required fields, granularity, join keys, and validation checks.

### 2.1 Regional Crop Calendar & Sowing Windows Dataset
- **Purpose:** Connect candidate crops to agronomic seasons and sowing/harvesting windows (focus: Maharashtra).
- **Target Source:** Department of Agriculture (Govt. of Maharashtra) / ICAR / Mahatma Phule Krishi Vidyapeeth (MPKV).
- **Granularity:** District or Agro-Climatic Zone level.
- **Required Fields:**
  | Field Name | Type | Description | Allowed Values / Examples |
  |---|---|---|---|
  | `crop_standard_name` | String | Harmonized crop identifier | `rice`, `cotton`, `chickpea`, `soybean` |
  | `state` | String | State name | `Maharashtra` |
  | `district` | String | District name | `Pune`, `Nagpur`, `Nashik`, `Ahmednagar` |
  | `agro_climatic_zone` | String | Official zone designation | `Western Maharashtra Ghat`, `Marathwada` |
  | `season` | String | Primary cropping season | `Kharif`, `Rabi`, `Summer` |
  | `sowing_start_month` | Integer | Sowing window start | `1` to `12` (e.g., `6` for June) |
  | `sowing_end_month` | Integer | Sowing window end | `1` to `12` (e.g., `7` for July) |
  | `duration_days_min` | Integer | Minimum crop maturity days | $60 \dots 365$ |
  | `duration_days_max` | Integer | Maximum crop maturity days | $60 \dots 365$ |

### 2.2 Crop Water Requirement ($ET_c$) Dataset
- **Purpose:** Evaluate water sustainability and irrigation requirements.
- **Target Source:** FAO Irrigation and Drainage Paper 56 / Central Ground Water Board (CGWB) / ICAR.
- **Granularity:** Crop-wise baseline requirements under semi-arid conditions.
- **Required Fields:**
  | Field Name | Type | Unit | Description |
  |---|---|---|---|
  | `crop_standard_name` | String | — | Harmonized crop key |
  | `water_requirement_min_mm` | Float | mm / season | Minimum physiological water needed |
  | `water_requirement_max_mm` | Float | mm / season | Maximum physiological water needed |
  | `drought_tolerance_level` | String | Low / Moderate / High | Resilience to dry spells |
  | `waterlogging_sensitivity` | String | Sensitive / Moderate / Tolerant | Resilience to excess rainfall |
  | `critical_stages` | String | — | Growth stages sensitive to moisture stress (e.g., flowering) |

### 2.3 Agricultural Market Prices (Agmarknet) Dataset
- **Purpose:** Provide historical wholesale prices for economic feasibility analysis.
- **Target Source:** Agmarknet (Directorate of Marketing & Inspection, Ministry of Agriculture).
- **Granularity:** District / APMC Mandi, Monthly or Daily aggregates.
- **Required Fields:**
  | Field Name | Type | Unit | Description |
  |---|---|---|---|
  | `crop_standard_name` | String | — | Harmonized crop key |
  | `market_name` | String | — | APMC mandi name (e.g., `Vashi`, `Gultekdi Pune`) |
  | `district` | String | — | District name |
  | `arrival_date` | Date (YYYY-MM-DD) | — | Date of observation |
  | `min_price` | Float | ₹ / quintal | Minimum recorded wholesale price |
  | `max_price` | Float | ₹ / quintal | Maximum recorded wholesale price |
  | `modal_price` | Float | ₹ / quintal | Most common wholesale price |
  | `arrivals_tonnes` | Float | Tonnes | Volume arrived at mandi |

### 2.4 District Crop Yield & Production Statistics
- **Purpose:** Analyze historical productivity trends by district and season.
- **Target Source:** Directorate of Economics and Statistics (DES), Ministry of Agriculture & Farmers Welfare (APS dataset).
- **Granularity:** District-wise, annual/seasonal.
- **Required Fields:**
  | Field Name | Type | Unit | Description |
  |---|---|---|---|
  | `state` | String | — | State name (`Maharashtra`) |
  | `district` | String | — | District name |
  | `crop_standard_name` | String | — | Harmonized crop key |
  | `crop_year` | Integer | Year | E.g., `2018`, `2019`, `2020` |
  | `season` | String | — | `Kharif`, `Rabi`, `Whole Year` |
  | `area_hectares` | Float | Hectares | Cultivated area |
  | `production_tonnes` | Float | Tonnes | Total production |
  | `yield_kg_per_ha` | Float | kg / hectare | Computed yield ($(\text{production} \times 1000) / \text{area}$) |

---

## 3. Entity Harmonization & Join Keys

Cross-dataset integration requires resolving naming and unit discrepancies:

### 3.1 Crop Nomenclature Harmonization Table
Different Indian datasets use regional or common vernacular terms. The system will use an internal canonical identifier:

| Canonical Key | Dataset Label (`Crop_recommendation.csv`) | Agmarknet Commodity Name | DES / Land Records Name |
|---|---|---|---|
| `pigeonpeas` | `pigeonpeas` | Arhar (Tur/Red Gram) | Arhar/Tur |
| `chickpea` | `chickpea` | Gram / Chana | Bengal Gram / Chana |
| `blackgram` | `blackgram` | Urad / Black Matpe | Urad |
| `mungbean` | `mungbean` | Moong (Green Gram) | Moong |
| `mothbeans` | `mothbeans` | Moth Dal | Matki |
| `kidneybeans` | `kidneybeans` | Rajma | Rajma |
| `cotton` | `cotton` | Cotton (Kapas) | Cotton |
| `maize` | `maize` | Maize (Makka) | Maize |
| `rice` | `rice` | Paddy (Dhan) / Rice | Rice / Paddy |

### 3.2 Geographic Entity Alignment
- **State Standard:** Title Case (`Maharashtra`).
- **District Standard:** Canonical spelling matched to Census of India 2011 / LGD (Local Government Directory) codes.
  - Examples of variance to handle: `Ahmadnagar` vs `Ahmednagar`, `Bid` vs `Beed`, `Buldanah` vs `Buldhana`.

### 3.3 Unit Standardization Protocol
- **Land Area:** Always standardized to **Hectares** ($1 \text{ Hectare} = 2.47105 \text{ Acres}$).
- **Yield:** Always standardized to **Kilograms per Hectare (kg/ha)**.
- **Prices:** Always standardized to **Indian Rupees per Quintal (₹/100 kg)** and displayed with per-kg conversions.
- **Rainfall / Water:** Always standardized to **Millimeters (mm)**.

---

## 4. Automated Data Quality & Integrity Validation Pipeline

Before any dataset is admitted into the processed pipeline, the ingestion script will enforce:

1. **Schema Integrity:** Verify all mandatory columns and valid data types.
2. **Missing Value Policy:**
   - No silent imputation for primary keys (`crop`, `district`, `year`, `season`).
   - Flag and log missing numerical values. Reject records with missing target values.
3. **Plausibility & Outlier Bounds:**
   - Soil pH must be between $3.0$ and $10.5$.
   - Temperature must be between $-5^\circ\text{C}$ and $55^\circ\text{C}$.
   - Rainfall must be $\ge 0$ and $\le 2000\text{ mm}$ for seasonal/monthly values.
   - Market prices must be $> 0$.
   - Yield must be $> 0$ and $\le 25,000\text{ kg/ha}$.
4. **Duplicate Record Detection:** Primary composite key check: `(district, crop, year, season)`.

---

## 5. Data Provenance & Reproducibility Tracking

Every ingested dataset will have an accompanying metadata record in `data/metadata.json` documenting:
- **Identifier:** Unique dataset code (e.g., `DS-AGRI-001`).
- **Source Organization & Portal:** (e.g., data.gov.in, ICAR, Kaggle).
- **Official URL:** Direct access link.
- **Access / Download Date:** Timestamp of snapshot.
- **License / Terms of Use:** Open Government Data (OGD) License India / CC-BY / Academic use.
- **Pre-processing Hash:** SHA-256 hash of the raw file to verify cryptographic integrity across runs.
