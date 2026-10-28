# AgriWise: Phase 4 — Regional & Seasonal Decision Layer Specification (Maharashtra)

**Document Title:** Reconciled Regional & Seasonal Decision Layer Specification, Dataset-Source Audit, 22-Crop Agronomic Feasibility Matrix, and Academic Evaluation Protocol  
**Project:** AgriWise Decision Support System  
**Author:** Shravani Pawar  
**Date:** October 2026  
**Status:** Revised Specification & Audit (Awaiting Explicit Review & Approval Before Code Implementation)  

---

## 1. Executive Summary & Objective

In Phases 2 and 3, AgriWise established a hardened machine learning inference pipeline (99.77% benchmark test accuracy) and a local TreeSHAP explainability engine. However, as documented in [`PROJECT_AUDIT.md`](file:///s:/MCA%202%20Mini%20project/crop/PROJECT_AUDIT.md), **the benchmark 2,200-sample Kaggle dataset possesses no geographic, seasonal, or regional attributes**.

A purely statistical classifier trained on this benchmark evaluates inputs strictly based on mathematical proximity to 7 nominal variables ($N, P, K$, temperature, humidity, pH, rainfall). Consequently, an unconstrained model will output recommendations that are unfeasible in Maharashtra (e.g. recommending temperate **Apple** in Solapur or deltaic **Jute** in Vidarbha, or recommending rainfed **Rice** during the dry winter Rabi season).

### Phase 4 Mandate
Design and specify a **Regional and Seasonal Decision Layer** centered on the **36 districts of Maharashtra**, reconciling all 22 baseline crops against verified agro-climatic zoning, agricultural university calendars, and seasonal sowing windows.

### Strict Governance Rules for this Document
1. **Zero Data Fabrication:** No missing values will be invented, imputed without evidence, or synthetic records merged into the training dataset.
2. **Preservation of Raw Model Probabilities:** The original ML Top-5 rankings and confidence scores will remain intact and visible. The regional layer provides an independent, parallel agronomic assessment.
3. **No Unsubstantiated Claims:** Crops are not claimed to be "biologically incapable" or "100% intercepted" without documented academic and empirical evidence. Where evidence is incomplete, crops are classified as **"Insufficient Evidence / Not Assessed"**.
4. **Approval Boundary:** No application code, database files, or UI modifications will be made until this revised document is reviewed and approved.

---

## 2. Reconciled 22-Crop Maharashtra Agronomic Feasibility Inventory

Every single one of the 22 baseline crop classes from `Crop_recommendation.csv` is reconciled below against verified agronomic research from the Indian Council of Agricultural Research (ICAR) and Maharashtra's four State Agricultural Universities (MPKV Rahuri, VNMKV Parbhani, Dr. PDKV Akola, Dr. BSKKV Dapoli):

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              RECONCILED 22-CROP INVENTORY MATRIX                                       │
├────┬─────────────┬──────────────────────────┬───────────────────┬──────────────────────────────────────┤
│ #  │ Crop Name   │ Regional Viability Status│ Primary Season(s) │ Agro-Climatic Zones & Primary Districts│
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 1  │ Rice        │ Prominently Cultivated   │ Kharif            │ Konkan (Z1, Z2), Eastern Vidarbha    │
│    │             │                          │                   │ (Z9: Bhandara, Gondia, Chandrapur),  │
│    │             │                          │                   │ Western Ghats/Sub-montane (Kolhapur) │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 2  │ Maize       │ Prominently Cultivated   │ Kharif, Rabi,     │ Western Maharashtra (Nashik, Sangli, │
│    │             │                          │ Summer            │ Kolhapur), Marathwada (Aurangabad),  │
│    │             │                          │ (Irrigated)       │ Khandesh (Jalgaon)                   │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 3  │ Chickpea    │ Prominently Cultivated   │ Rabi              │ Marathwada (Latur, Parbhani, Beed,   │
│    │ (Gram)      │ (Major Dryland Pulse)    │ (Post-monsoon)    │ Dharashiv), Western MH (Solapur,     │
│    │             │                          │                   │ Ahmednagar, Pune), Vidarbha (Akola)  │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 4  │ Pigeonpeas  │ Prominently Cultivated   │ Kharif            │ Marathwada (Latur, Parbhani, Nanded),│
│    │ (Tur/Arhar) │ (Major Kharif Pulse)     │ (Long duration)   │ Vidarbha (Amravati, Akola, Yavatmal),│
│    │             │                          │                   │ Khandesh (Jalgaon, Dhule)            │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 5  │ Cotton      │ Prominently Cultivated   │ Kharif            │ Vidarbha (Yavatmal, Akola, Amravati, │
│    │             │ (Primary Cash Crop)      │ (Monsoon onset)   │ Wardha), Marathwada (Nanded, Jalna), │
│    │             │                          │                   │ Khandesh (Jalgaon, Dhule)            │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 6  │ Blackgram   │ Prominently Cultivated   │ Kharif            │ Marathwada (Latur, Nanded, Dharashiv)│
│    │ (Urad)      │ (Short-duration Pulse)   │                   │ Vidarbha (Yavatmal, Wardha, Akola),  │
│    │             │                          │                   │ Western Maharashtra (Kolhapur)       │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 7  │ Mungbean    │ Prominently Cultivated   │ Kharif            │ Marathwada (Jalna, Parbhani, Nanded),│
│    │ (Moong)     │ (60-70 day Pulse)        │ (Early monsoon)   │ Vidarbha (Amravati, Akola, Buldhana),│
│    │             │                          │                   │ Khandesh (Jalgaon, Dhule)            │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 8  │ Mothbeans   │ Cultivated (Arid/Scarcity│ Kharif            │ Scarcity Zone (Z6: Ahmednagar,       │
│    │ (Matki)     │ Drought-tolerant Pulse)  │ (Dryland)         │ Solapur, eastern Pune, Dhule, Sangli)│
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 9  │ Pomegranate │ Prominently Cultivated   │ Perennial         │ Scarcity & Transition Zones: Solapur │
│    │             │ (Commercial Horticulture)│ (Ambe/Mrig/Hasta) │ (ICAR-NRCP), Nashik, Ahmednagar,     │
│    │             │                          │                   │ Sangli, Dharashiv, Pune              │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 10 │ Grapes      │ Prominently Cultivated   │ Perennial         │ Western Maharashtra: Nashik (ICAR-   │
│    │             │ (Export Viticulture)     │ (Two prunings/yr) │ NRCG), Sangli, Solapur, Pune,        │
│    │             │                          │                   │ Ahmednagar                           │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 11 │ Banana      │ Prominently Cultivated   │ Perennial         │ Khandesh: Jalgaon ("Banana Hub"),    │
│    │             │ (Commercial Fruit)       │ (10-12 months)    │ Marathwada (Nanded, Parbhani,        │
│    │             │                          │ (High water/Drip) │ Hingoli), Western MH (Solapur)       │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 12 │ Mango       │ Prominently Cultivated   │ Perennial         │ Konkan Coast (Alphonso: Ratnagiri,   │
│    │             │ (Major Fruit Crop)       │ (Flowering Dec-Jan│ Sindhudurg, Raigad); Marathwada      │
│    │             │                          │ Harvest Mar-May)  │ (Kesar: Aurangabad, Jalna, Beed)     │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 13 │ Orange      │ Prominently Cultivated   │ Perennial         │ Vidarbha: Nagpur (ICAR-CCRI),        │
│    │ (Mandarin)  │ (Major Citrus Crop)      │ (Ambia / Mrig)    │ Amravati (Morshi/Warud), Wardha      │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 14 │ Papaya      │ Cultivated               │ Annual / Biennial │ Khandesh (Nandurbar, Jalgaon),       │
│    │             │ (Irrigated Horticulture) │ (Monsoon / Spring)│ Western MH (Pune, Solapur, Ahmednagar│
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 15 │ Watermelon  │ Cultivated               │ Summer            │ River basins: Solapur (Bhima), Pune, │
│    │             │ (Irrigated / Riverbed)   │ (Jan-May)         │ Ahmednagar, Marathwada, Konkan       │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 16 │ Muskmelon   │ Cultivated               │ Summer            │ Western Maharashtra (Pune, Solapur,  │
│    │             │ (Irrigated Cucurbit)     │ (Jan-April)       │ Ahmednagar), Khandesh (Jalgaon)      │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 17 │ Coconut     │ Cultivated               │ Perennial         │ Coastal Konkan (Sindhudurg,          │
│    │             │ (Restricted Coastal Belt)│ (Plantation)      │ Ratnagiri, Raigad, Palghar, Thane);  │
│    │             │                          │                   │ Unsuited to dry inland plateau       │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 18 │ Lentil      │ Cultivated               │ Rabi              │ Eastern Vidarbha (Nagpur, Wardha,    │
│    │ (Masoor)    │ (Restricted Minor Pulse) │ (Post-paddy cold) │ Bhandara, Chandrapur); minor acreage │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 19 │ Coffee      │ Restricted Micro-Pocket  │ Perennial         │ Chikhaldara Hill Station only        │
│    │             │ (Non-Commercial Plateau) │ (Montane Shade)   │ (Amravati district, ~1100m MSL, <100 │
│    │             │                          │                   │ ha). Unsuited to plains/plateau      │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 20 │ Kidneybeans │ Limited / Experimental   │ Rabi              │ Pockets of Western MH (Pune, Nashik) │
│    │ (Rajma)     │ (Non-traditional Hill)   │ (Under Irrigation)│ under assured irrigation; not part   │
│    │             │                          │                   │ of standard statewide extension      │
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 21 │ Jute        │ Climatically & Culturally│ N/A (Not grown    │ Requires deltaic slow-moving standing│
│    │             │ Incompatible             │ in Maharashtra)   │ retting waters (Bengal/Assam/Bihar). │
│    │             │                          │                   │ Zero commercial DES acreage reported.│
├────┼─────────────┼──────────────────────────┼───────────────────┼──────────────────────────────────────┤
│ 22 │ Apple       │ Climatically Incompatible│ N/A (Temperate    │ Commercial pome fruit requiring      │
│    │             │ (Commercial Deficit)     │ winter chilling)  │ 800-1200 chilling hours (<7°C).      │
│    │             │                          │                   │ Not viable in tropical peninsular MH.│
└────┴─────────────┴──────────────────────────┴───────────────────┴──────────────────────────────────────┘
```

### Agronomic Sowing Windows & Irrigation Dependencies (Detailed Profile)

1. **Rice:** Sowing: June 15 – July 15 (Transplanting: July 10 – August 5); Harvest: October 15 – November 25. Water: 1100–1400 mm. Highly sensitive to moisture stress at panicle initiation.
2. **Cotton:** Sowing: June 15 – July 10 (Dryland monsoon onset) or May 15 – June 1 (Pre-monsoon irrigated). Harvest: November – January (multiple pickings). Water: 700–900 mm.
3. **Pigeonpeas (Tur):** Sowing: June 20 – July 15; Harvest: December 15 – January 31. Deep taproot system enables drought tolerance during post-monsoon terminal pod filling.
4. **Chickpea (Gram):** Sowing: October 10 – November 10; Harvest: February 1 – February 28. Residual soil moisture crop. Excessive rainfall/irrigation during flowering causes vegetative overgrowth and collar rot.
5. **Maize:** Kharif: June 20 – July 15; Rabi: October 15 – November 15; Summer: January 15 – February 15. Requires 500–700 mm. Sensitive to waterlogging at seedling stage and moisture stress at tasseling/silking.
6. **Blackgram & Mungbean:** Sowing: June 15 – July 5; Harvest: August 25 – September 15. Short duration fits well in crop rotation before Rabi chickpea or rabi sorghum.
7. **Mothbeans:** Sowing: June 25 – July 20; Harvest: October. Extreme drought resilience; thrives in sandy/gravelly light shallow soils of Scarcity Zone where other pulses fail.
8. **Pomegranate:** Ambe Bahar (Flowering Jan–Feb, Harvest July–Aug; requires assured summer irrigation); Mrig Bahar (Flowering June–July, Harvest Nov–Dec; preferred in scarcity rainfed tracts); Hasta Bahar (Flowering Sept–Oct, Harvest Feb–March; premium market value, prone to bacterial blight).
9. **Grapes:** Pruning calendar: Foundation pruning (April), Forward/Fruit pruning (October). Soil moisture deficit required during veraison/berry ripening (Jan–Feb) to enhance sugar Brix.
10. **Banana:** Planting: Mrig baug (June–July) or Kandebar (October–November). Water: 1800–2200 mm/year. Drip irrigation and fertigation mandatory for commercial tissue-culture Grand Naine.
11. **Mango:** Vegetative growth: Monsoon (June–Sept); Floral bud differentiation & flowering: December–January (requires cool, dry weather; rain during flowering causes anthracnose and fruit drop); Harvest: March–May.
12. **Orange (Mandarin):** Ambia Bahar (Flowering Jan–Feb, Harvest Nov–Dec); Mrig Bahar (Flowering June–July, Harvest Feb–March). Deep black soils (Vertisols) require good drainage to prevent Phytophthora gummosis.
13. **Papaya:** Sowing: Sept–Oct (nursery), Transplanting: Nov–Dec; or Spring nursery Feb, Transplanting April. Sensitive to frost, waterlogging, and Papaya Ring Spot Virus (PRSV).
14. **Watermelon & Muskmelon:** Sowing: January 15 – February 28; Harvest: April–May. Warm sunny days ($28\text{–}35^\circ\text{C}$) and low humidity required for fruit sweetness and to prevent powdery/downy mildew.
15. **Coconut:** Continuous flowering; harvesting every 45–60 days. Coastal belt only (relative humidity $> 60\%$, annual rainfall $> 2000\text{ mm}$ or high water-table). Unsuited to inland continental dry air.
16. **Lentil:** Sowing: October 15 – November 15; Harvest: February. Cold tolerant; cultivated in Eastern Vidarbha utilizing post-rice residual moisture.
17. **Kidneybeans (Rajma):** Irrigated Rabi sowing only: October 15 – November 10. Temperature sensitivity: flowering disrupted if temperature $> 30^\circ\text{C}$ or $< 8^\circ\text{C}$.
18. **Coffee:** Sowing/Planting: Monsoon. Limited strictly to Chikhaldara plateau (Amravati). Outside this micro-pocket, high summer heat ($> 40^\circ\text{C}$) and dry air cause floral abortion and bush desiccation.
19. **Apple:** Cultivar chilling requirements: Commercial standard cultivars (Red Delicious, Golden Delicious, Gala) require 800–1200 hours $< 7^\circ\text{C}$. In Maharashtra, winter mean minimum temperatures are $11\text{–}15^\circ\text{C}$. Low-chill varieties (HRMN-99, Anna, Dorsett Golden) can set fruit in trial gardens at higher altitudes (Mahabaleshwar), but do not have commercial production scale, yield certification, or university extension package of practices.
20. **Jute:** Retting requirement: Harvested stems require immersion in slow-moving water bodies for 15–20 days at water temperatures $> 34^\circ\text{C}$ for microbiological retting. Maharashtra lacks the deltaic hydrology, retting infrastructure, and processing mills.

---

## 3. Verified External Datasets: Specification & Audit

To satisfy the user requirement for full provenance documentation, each prospective dataset is audited below with official publishing authority, access status, schema, units, and potential join keys:

### 3.1 Dataset 1: Historical District-Wise Crop Area & Production (APS / DES)
- **Official Dataset Title:** *Area, Production and Yield of Principal Crops in India (District-wise)*
- **Publishing Agency:** Directorate of Economics and Statistics (DES), Department of Agriculture & Farmers Welfare, Ministry of Agriculture and Farmers Welfare, Government of India.
- **Official URL:** [https://aps.dac.gov.in](https://aps.dac.gov.in) (also accessible via Open Government Data portal: [https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics))
- **Access / Audit Date:** October 2026.
- **License:** Open Government Data (OGD) License India.
- **Geographic Coverage:** All 36 districts of Maharashtra (State Code: `MH`).
- **Temporal Coverage:** Annual series from 1997–98 to 2020–21 (24 years).
- **Schema & Data Types:**
  - `State_Name` (String): e.g. "Maharashtra"
  - `District_Name` (String): e.g. "PUNE", "NASHIK", "SOLAPUR", "NAGPUR"
  - `Crop_Year` (Integer): e.g. 2018, 2019
  - `Season` (String): Categorical — `"Kharif"`, `"Rabi"`, `"Summer"`, `"Whole Year"`
  - `Crop` (String): Standardized commodity title (e.g. "Rice", "Cotton(lint)", "Gram", "Maize", "Arhar/Tur", "Moong(Green Gram)", "Urad", "Moth", "Banana", "Mango", "Grapes", "Orange", "Papaya", "Pomegranate")
  - `Area` (Float, Unit: Hectares)
  - `Production` (Float, Unit: Metric Tonnes, or Bales for cotton)
  - `Yield` (Float, Unit: kg/hectare — calculated as $(\text{Production} \times 1000) / \text{Area}$)
- **Missing-Value Pattern:** Production is marked `NaN` or `0` for certain perennial horticultural crops where only area is enumerated in land revenue records. Cotton production is officially reported in Bales of 170 kg, requiring explicit unit conversion to metric tonnes.
- **Join Keys:** `[district_name, crop_standard_name, season]`.
- **Role in AgriWise:** Provides empirical historical evidence of whether a crop has documented commercial cultivation history in a given district and season.

### 3.2 Dataset 2: Official Agro-Climatic Zones & Sowing Calendars of Maharashtra
- **Official Title:** *Agro-Climatic Zones of Maharashtra State and Crop Recommendation Calendars*
- **Publishing Agency:** Department of Agriculture, Government of Maharashtra (Commissionerate of Agriculture, Pune: [krishi.maharashtra.gov.in](https://krishi.maharashtra.gov.in)) in joint collaboration with MPKV Rahuri, VNMKV Parbhani, Dr. PDKV Akola, and Dr. BSKKV Dapoli.
- **Access / Audit Date:** October 2026.
- **License:** Official Public Agricultural Extension Documentation.
- **Geographic Coverage:** Complete coverage of all 36 districts partitioned into 9 agro-climatic zones.
- **Schema & Data Types:**
  - `district` (String): District name matching Census 2011 standard.
  - `zone_id` (String): Official zone code (`MH-Z1` through `MH-Z9`).
  - `zone_name` (String): e.g. "Scarcity Zone", "Western Ghat Zone", "Central Vidarbha Zone".
  - `normal_rainfall_min_mm` (Float): Lower boundary of zone rainfall normal.
  - `normal_rainfall_max_mm` (Float): Upper boundary of zone rainfall normal.
  - `dominant_soil_types` (List[String]): Classification per National Bureau of Soil Survey and Land Use Planning (NBSS&LUP).
  - `crop_feasibility_matrix` (Dict): Per-crop seasonal sowing start/end calendar week, irrigation prerequisite, and restriction reason.
- **Role in AgriWise:** Authoritative rule base for regional viability, sowing calendars, and agronomic restriction rationales.

### 3.3 Dataset 3: District Rainfall Normals and Climatology (IMD Pune)
- **Official Title:** *Customized District-Wise Rainfall Climatology and Normal Tables (1991–2020)*
- **Publishing Agency:** India Meteorological Department (IMD), Climate Research and Services, Pune, Ministry of Earth Sciences.
- **Official URL:** [https://mausam.imd.gov.in](https://mausam.imd.gov.in) / [https://imdpune.gov.in](https://imdpune.gov.in)
- **Access / Audit Date:** October 2026.
- **License:** Open Government Data (OGD) License India.
- **Geographic Coverage:** 36 districts of Maharashtra.
- **Schema & Data Types:**
  - `district` (String)
  - `sw_monsoon_rainfall_normal_mm` (Float, June–September long-period average)
  - `post_monsoon_rainfall_normal_mm` (Float, October–December)
  - `winter_rainfall_normal_mm` (Float, January–February)
  - `pre_monsoon_rainfall_normal_mm` (Float, March–May)
  - `annual_rainfall_normal_mm` (Float)
- **Role in AgriWise:** Distinguishes normal climatological rainfall from user-provided single-season observation inputs.

### 3.4 Dataset 4: Crop Water Requirements ($ET_c$) Benchmarks
- **Official Title:** *Crop Water Requirements for Semi-Arid Agricultural Systems*
- **Publishing Agency:** ICAR-National Institute of Abiotic Stress Management (NIASM), Baramati, Maharashtra & FAO Irrigation and Drainage Paper 56.
- **Access / Audit Date:** October 2026.
- **Schema & Data Types:**
  - `crop_standard_name` (String)
  - `water_requirement_min_mm` (Float, mm/season)
  - `water_requirement_max_mm` (Float, mm/season)
  - `critical_growth_stages` (String): e.g. "Flowering, Pod Development"
  - `drought_resilience_class` (String): `"High"`, `"Moderate"`, `"Sensitive"`
- **Role in AgriWise:** Evaluates irrigation dependency when a crop is recommended in low-rainfall zones (e.g. Scarcity Zone).

---

## 4. Refined Regional Decision Layer Architecture

To fulfill user constraint #3 (*"Preserve the original ML Top-5 rankings and scores without modification; present regional and seasonal assessments separately"*), the architecture operates as follows:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │ User Inputs: N, P, K, Climate, District, Season / Month │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                      ┌────────────────────────┴────────────────────────┐
                      ▼                                                 ▼
        ┌───────────────────────────┐                     ┌───────────────────────────┐
        │       ML Predictor        │                     │  Regional Decision Layer  │
        │    (src/ml/predictor.py)  │                     │   (src/modules/regional)  │
        └─────────────┬─────────────┘                     └─────────────┬─────────────┘
                      │                                                 │
                      ▼                                                 ▼
        ┌───────────────────────────┐                     ┌───────────────────────────┐
        │ Raw Statistical Top-5     │                     │ Regional & Seasonal       │
        │ Candidates & Confidences  │                     │ Viability Assessment      │
        │ (Unmodified: e.g.         │                     │ - In-Season & Cultivable  │
        │  #1 Apple 82%, #2 Rice 8%)│                     │ - Cultivable Off-Season   │
        └─────────────┬─────────────┘                     │ - Incompatible / Flagged  │
                      │                                   │ - Insufficient Evidence   │
                      │                                   └─────────────┬─────────────┘
                      │                                                 │
                      └────────────────────────┬────────────────────────┘
                                               ▼
                              ┌──────────────────────────────────┐
                              │    Synthesized Decision View     │
                              │ - Shows raw ML rank side-by-side │
                              │ - Clear status badge per crop    │
                              │ - Explicit rationale and source  │
                              └──────────────────────────────────┘
```

### 4.1 Four Explicit Viability Classes
For every crop in the ML candidate pool, the decision layer outputs one of four mutually exclusive statuses:

1. **`Cultivable & In-Season`:**  
   Documented commercial cultivation history in the district AND the current date/season falls inside the certified university sowing window.
2. **`Cultivable but Off-Season`:**  
   Documented commercial cultivation in the district, but the current date/season is outside the recommended sowing window. (e.g. *Chickpea is highly cultivable in Solapur, but planting during June Kharif risks collar rot and fungal blight*).
3. **`Agronomically Incompatible / Climatically Unsuited`:**  
   Documented physiological limitation or absence of commercial viability in the selected agro-climatic zone (e.g. *Apple: Temperate winter chilling requirement deficit (<7°C); Jute: Absence of deltaic floodwater retting hydrology*).
4. **`Insufficient Evidence / Not Assessed`:**  
   Used when government statistics or university extension bulletins have no definitive cultivation records or clear rejection data for that crop in that specific district (e.g. *Kidneybeans in Gadchiroli*).

---

## 5. Strengthened Academic Evaluation Protocol for Phase 4

To avoid premature claims (such as "100% interception" or "improved agricultural success") before running empirical tests, the evaluation protocol is specified with mathematical rigor:

### 5.1 Independent Reference Standard
We define an independent ground-truth matrix $\mathcal{G}(d, s, c) \in \{\text{Feasible}, \text{Unseasonal}, \text{Incompatible}, \text{Unknown}\}$ compiled exclusively from MPKV/VNMKV/PDKV/BSKKV university extension packages and DES historical production records.

### 5.2 Mathematical Formulation of Evaluation Metrics

1. **Agronomic False Positive Rate (AFPR):**  
   Measures how frequently the raw statistical ML model recommends an incompatible or unseasonal crop as its Top-1 choice:
   $$\text{AFPR}_{\text{ML}} = \frac{\sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\Big(\text{Top-1}_{\text{ML}}(\mathbf{x}_i) \in \mathcal{U}(d_i, s_i)\Big)}{N_{\text{eval}} - N_{\text{unknown}}} \times 100\%$$
   Where:
   - $\mathcal{U}(d_i, s_i)$ is the set of crops verified as agronomically incompatible or unseasonal in district $d_i$ and season $s_i$.
   - $N_{\text{unknown}}$ is the count of queries where the Top-1 crop has status "Insufficient Evidence / Not Assessed" (strictly excluded from the binary denominator).

2. **Incompatible Crop Interception Rate (ICIR):**  
   Measures the proportion of out-of-region or unviable recommendations caught and flagged by the decision layer:
   $$\text{ICIR} = \frac{\sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\Big(\text{Top-1}_{\text{ML}}(\mathbf{x}_i) \in \mathcal{U}(d_i, s_i) \;\land\; \text{FlaggedByRegionalLayer}(\mathbf{x}_i)\Big)}{\sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\Big(\text{Top-1}_{\text{ML}}(\mathbf{x}_i) \in \mathcal{U}(d_i, s_i)\Big)} \times 100\%$$

3. **In-Season Retention Rate (ISRR):**  
   Measures whether legitimate in-season regional crops (e.g. Cotton in Yavatmal Kharif, Chickpea in Solapur Rabi) are correctly preserved without false rejection:
   $$\text{ISRR} = \frac{\sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\Big(\text{Top-1}_{\text{ML}}(\mathbf{x}_i) \in \mathcal{F}(d_i, s_i) \;\land\; \text{Status} = \text{'Cultivable \& In-Season'}\Big)}{\sum_{i=1}^{N_{\text{eval}}} \mathbb{I}\Big(\text{Top-1}_{\text{ML}}(\mathbf{x}_i) \in \mathcal{F}(d_i, s_i)\Big)} \times 100\%$$

### 5.3 Diagnostic Evaluation Test Suite Design
The initial test suite will comprise **72 multi-district, multi-season scenario queries** structured across all 9 Agro-Climatic Zones of Maharashtra:

| Test Group | Query Count | Purpose | Representative Test Scenarios |
|---|---|---|---|
| **Group A: Positive Controls (In-Season Valid)** | 24 Queries | Verify that prominent in-season crops are approved with High Viability | Cotton in Yavatmal (Kharif); Rice in Ratnagiri (Kharif); Chickpea in Latur (Rabi); Pomegranate in Solapur; Grapes in Nashik |
| **Group B: Negative Controls (Climatic Incompatibility)** | 16 Queries | Test whether out-of-region temperate/deltaic crops are intercepted when ML suggests them | Apple-like soil/weather inputs in Pune/Solapur; Jute-like inputs in Vidarbha |
| **Group C: Seasonal Mismatch Controls** | 16 Queries | Test whether cultivable crops are flagged as Off-Season when requested in the wrong month | Chickpea requested in July Kharif; Rice requested in December Rabi in Scarcity Zone |
| **Group D: Boundary & Insufficient Evidence Controls** | 16 Queries | Test edge cases: Coffee in Chikhaldara vs. Coffee in Solapur; Kidneybeans in Gadchiroli; extreme rainfall deviations | Coffee in Amravati (Hill) vs. Coffee in Jalna (Plains); Kidneybeans in Eastern Vidarbha |

---

## 6. Approval Gate & Next Actions

As requested by the user, **no application code or dataset integration will occur until this plan receives formal approval**.

Upon your review and authorization:
1. We will implement the clean `src/modules/regional/` package containing `maharashtra_data.py` and `regional_filter.py`.
2. We will integrate the district and season selectors into the existing Flask app while keeping raw ML predictions untouched.
3. We will write automated unit and integration tests in `tests/test_regional.py`.
4. We will execute the 72-query evaluation protocol and document the empirical before-and-after metrics in a formal report.
