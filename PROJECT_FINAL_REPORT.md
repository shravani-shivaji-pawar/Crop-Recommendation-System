# AgriWise: Final Project Report and System Documentation

**Project Title:** AgriWise: An Explainable, Climate-Aware, and Region-Specific AI Decision Support System for Sustainable Crop Selection  
**Original Repository:** `shravani-shivaji-pawar/Crop-Recommendation-System`  
**Target Program:** Master of Computer Applications (MCA) Major Project  
**Date of Completion:** October 2026  
**Final Checkpoint Tag:** `phase7-completed-checkpoint`  

---

## 1. Executive Summary

This report documents the transformation of a baseline, single-crop machine learning classifier into **AgriWise**, a multi-tiered, explainable, climate-aware, and region-specific agricultural decision support system. The project was executed incrementally across seven structured phases under strict software engineering and research integrity safeguards. 

The finalized system:
1. Rectifies a critical data leakage and scaler overwrite bug that caused the legacy system's live inference accuracy to collapse to 6.14%, restoring held-out predictive accuracy to **99.77%**.
2. Replaces uncalibrated single-crop predictions with a **Top-5 candidate ranking engine** incorporating ensemble voting bounds and calibrated confidence categories.
3. Implements local feature-level explainability through exact **TreeSHAP** attribution and pairwise **contrastive explanations** (*Why Crop A over Crop B?*).
4. Provides an interactive **What-If Simulation Studio** allowing agronomists and farmers to evaluate candidate crop resilience under environmental perturbations (e.g., rainfall deficits).
5. Enforces a rule-based **Regional and Seasonal Decision Layer** for Maharashtra, verifying biophysical ML recommendations against 36 districts, 9 agro-climatic zones, and official cropping calendars.
6. Integrates verified secondary empirical datasets—18 years of historical crop production statistics (APS/DES, 12,628 rows) and long-period rainfall normals (IMD Pune)—as transparent contextual overlays with zero model retraining and zero data fabrication.
7. Maintains an automated test suite of **52 unit, integration, and regression tests**, passing 100% on CPU-only local environments.

---

## 2. Legacy Project Analysis & Identified Limitations

The original project (`Crop-Recommendation-System-Using-Machine-Learning-main`) was audited in Phase 1 (`PROJECT_AUDIT.md`), revealing five fundamental flaws:

| Flaw Category | Legacy System Manifestation | Scientific & Operational Consequence |
|---|---|---|
| **Critical Preprocessing Bug** | In `app.py`, a single incoming user sample ($1 \times 7$) was passed to `minmaxscaler.fit_transform()` and `standscaler.fit_transform()`. | Every incoming live prediction overwrote the global min/max parameters of the scaler with single-point values ($0.0$). Subsequent inference suffered a catastrophic drop to **6.14% held-out accuracy**. |
| **Overconfident Single Predictions** | Model used `argmax` across classes and displayed single categorical output with raw softmax probability formatted as "suitability percentage". | Misleading to farmers: an uncalibrated classifier probability (e.g. $0.98$) reflects model certainty within a clean synthetic feature space, not 98% real-world crop yield or farming success. |
| **Black-Box Opacity** | Zero visibility into which parameters led to a crop recommendation. | Farmers and agronomists could not see whether recommendation was driven by soil nutrients ($N, P, K$) or ambient climate (rainfall, humidity). |
| **Absence of Sensitivity Analysis** | Static point predictions only. | Users could not test climate risks (e.g., *"What if rainfall drops by 25% due to delayed monsoon?"*). |
| **Geographic & Agronomic Detachment** | The 7-feature model had no regional awareness. | Provided biologically absurd recommendations (e.g., recommending temperate Himalayan Apple for arid, low-altitude Jalna district in Maharashtra) because temperature and rainfall numbers momentarily matched. |
| **Lack of Historical Validation** | Output disconnected from real-world district production. | No check whether the recommended crop had ever been successfully grown at commercial scale in that district. |

All legacy files and model artifacts were preserved in [`legacy/`](file:///s:/MCA%202%20Mini%20project/crop/legacy) for academic baseline comparison.

---

## 3. Incremental Phase Evolution (Phases 2–7)

### Phase 2: Pipeline Rectification & Top-5 Recommendation Engine
- **Pre-Implementation Checkpoint:** `phase2-pre-implementation-checkpoint`
- **Accomplishments:**
  - Implemented `AgriDataPreprocessor` in `src/ml/preprocessor.py`. Fixed scaling bounds strictly to training partition ($X_{\text{train}}$). Added immutable transformation safeguards that throw an exception if `fit()` is invoked on single samples.
  - Implemented `CropPredictor` in `src/ml/predictor.py`. Replaced single-crop output with Top-$K$ candidate ranking ($K=5$) and calibrated confidence bands (High Confidence, Moderate Confidence, Ambiguous/Close Contest).
  - Built automated unit and regression test suite verifying mathematical consistency and restoration of held-out accuracy to 99.77%.

### Phase 3: Explainable AI via TreeSHAP & What-If Studio
- **Pre-Implementation Checkpoint:** `phase3-pre-implementation-checkpoint`
- **Accomplishments:**
  - Integrated `shap.TreeExplainer` in `src/xai/explainer.py` for exact polynomial-time computation of Shapley feature attributions.
  - Built pairwise contrastive explanation engine calculating differential attributions ($\Delta \phi_i(A, B) = \phi_i(A) - \phi_i(B)$).
  - Developed `WhatIfSimulator` in `src/xai/counterfactual.py` for interactive perturbation tracking, computing parameter shifts ($\Delta x_i$), model probability deltas, and candidate rank trajectories.
  - Documented complete mathematical formulations in `VERIFICATION_AND_METHODOLOGY.md`.

### Phase 4: Regional and Seasonal Decision Layer (Maharashtra)
- **Pre-Implementation Checkpoint:** `phase4-pre-implementation-checkpoint` (`398b444`)
- **Completion Checkpoint:** `phase4-completed-checkpoint` (`8b27280`)
- **Accomplishments:**
  - Reconciled complete 22-crop agronomic suitability matrix across all 36 districts and 9 agro-climatic zones of Maharashtra.
  - Implemented `RegionalConstraintEngine` in `src/modules/regional/regional_filter.py`. Evaluates ML candidates against verified district zones and sowing calendars (Kharif, Rabi, Summer).
  - Constructed a 72-query regional diagnostic suite evaluating biophysical queries against regional ground truth. Evaluated Unviable Top-1 Recommendation Rate (54.41%), Conventional False Positive Rate on unviable queries (100.0% raw vs 0.0% intercepted), Incompatible Crop Interception Rate (100.0%), and In-Season Recommendation Retention (100.0%).

### Phase 5: Modern UI & Interactive Dashboard Upgrade
- **Pre-Implementation Checkpoint:** `phase5-pre-ui-checkpoint` (`085e4ea`)
- **Completion Checkpoint:** `phase5-completed-checkpoint` (`a75518f`)
- **Accomplishments:**
  - Upgraded web interface using clean, modular vanilla CSS design tokens (`Crop-Recommendation-System-Using-Machine-Learning-main/static/css/style.css`).
  - Added interactive preset selector for common agro-climatic scenarios (e.g., Konkan High-Rainfall Rice, Vidarbha Dryland Cotton, Western Maharashtra Sugarcane).
  - Rendered TreeSHAP positive/limiting factors, contrastive differential tables, and regional suitability alerts with responsive, accessible layout.

### Phase 6: Controlled Integration of Verified Secondary Datasets
- **Pre-Implementation Checkpoint:** `phase6-pre-implementation-checkpoint`
- **Completion Checkpoint:** `phase6-completed-checkpoint` (`7d4e0e7`)
- **Accomplishments:**
  - Ingested and audited DES historical crop production statistics (1997–2014, 12,628 rows, 35 rural districts) and IMD Pune district rainfall normals (36 districts).
  - Solved cotton bale unit anomaly ($1\text{ bale} = 170\text{ kg} = 0.170\text{ MT}$).
  - Created `src/modules/regional/secondary_data.py` providing in-memory query repositories.
  - Enforced zero model retraining and zero data fabrication: missing historical records return transparent `"No historical DES records"` notices.
  - Expanded test suite to 52 passing automated tests.

### Phase 7: Research Paper & Final Packaging (Current Phase)
- **Pre-Packaging Checkpoint:** `phase7-pre-packaging-checkpoint`
- **Completion Checkpoint:** `phase7-completed-checkpoint`
- **Accomplishments:**
  - Authored comprehensive academic research paper draft (`AGRIWISE_RESEARCH_PAPER.md`) with 16 structured sections, genuine citations, and strict threats-to-validity boundaries.
  - Authored comprehensive final project report (`PROJECT_FINAL_REPORT.md`).
  - Updated `README.md` and created clean `.env.example`.
  - Final verification of all 52 tests and route integrity.

---

## 4. System Architecture & Module Responsibilities

The AgriWise repository is organized into distinct, decoupled packages:

```
s:\MCA 2 Mini project\crop\
├── AGRIWISE_RESEARCH_PAPER.md     # Academic research paper draft (16 sections)
├── PROJECT_FINAL_REPORT.md        # Comprehensive major project documentation
├── PROJECT_AUDIT.md               # Phase 1 technical audit of legacy codebase
├── PROPOSED_ARCHITECTURE.md      # Architectural design specification
├── DATASET_REQUIREMENTS.md        # Secondary dataset integration rules
├── RESEARCH_PLAN.md               # Research questions and hypothesis formulations
├── IMPLEMENTATION_ROADMAP.md      # Multi-phase milestone tracking
├── VERIFICATION_AND_METHODOLOGY.md# Mathematical formulations and leakage proofs
├── requirements.txt               # Pinned CPU-compatible dependency specifications
├── .env.example                   # Clean environment configuration template
│
├── legacy/                        # Preserved original codebase & model artifacts
│   ├── app.py
│   ├── model.pkl
│   ├── minmaxscaler.pkl
│   └── standscaler.pkl
│
├── data/
│   ├── raw/
│   │   ├── Crop_recommendation.csv                    # Primary 2,200-sample ML dataset
│   │   ├── maharashtra_crop_production_des_raw.csv    # Secondary DES historical dataset
│   │   └── maharashtra_district_rainfall_normals_imd.csv # Secondary IMD rainfall normals
│   └── DATASET_MANIFEST.md        # Complete provenance, schema, and licensing catalog
│
├── src/
│   ├── core/                      # Configuration & boundary validation
│   │   ├── config.py              # Physical bounds, crop names, confidence thresholds
│   │   └── exceptions.py          # Custom domain exceptions
│   ├── ml/                        # Hardened Machine Learning package
│   │   ├── preprocessor.py        # AgriDataPreprocessor (train-only scaling, immutable)
│   │   ├── predictor.py           # CropPredictor (Top-5 candidate ranking engine)
│   │   ├── train.py               # Reproducible training pipeline
│   │   ├── evaluate.py            # Pipeline evaluation suite
│   │   └── artifacts/             # Validated production model & scaler artifacts
│   ├── xai/                       # Explainability & Simulation package
│   │   ├── explainer.py           # AgriExplainer (TreeSHAP & contrastive analysis)
│   │   └── counterfactual.py      # WhatIfSimulator (Interactive perturbation studio)
│   └── modules/
│       └── regional/              # Regional & Seasonal Decision Layer
│           ├── regional_filter.py # RegionalConstraintEngine (Maharashtra rules)
│           ├── secondary_data.py  # Repositories for DES & IMD datasets
│           └── evaluator.py       # 72-query diagnostic suite runner
│
├── tests/                         # Automated test suite (52 tests)
│   ├── test_preprocessor.py       # Preprocessing & immutability tests
│   ├── test_predictor.py          # Top-5 ranking & confidence score tests
│   ├── test_explainer.py          # TreeSHAP & contrastive attribution tests
│   ├── test_whatif.py             # What-If delta & rank shift tests
│   ├── test_regional.py           # Maharashtra rules & diagnostic suite tests
│   ├── test_secondary.py          # DES/IMD schema, cotton bales, missing value tests
│   ├── test_regression.py         # Regression tests against legacy scaler bug
│   └── test_routes.py             # Flask route integration & UI rendering tests
│
└── Crop-Recommendation-System-Using-Machine-Learning-main/ # Web Application
    ├── app.py                     # Flask controller integrating all modules
    ├── static/
    │   ├── css/style.css          # Vanilla CSS design system
    │   └── img.jpg
    └── templates/
        ├── index.html             # Main recommendation & SHAP dashboard
        └── what_if.html           # Interactive What-If Simulation Studio
```

---

## 5. Dataset Catalog, Schemas, Units, and Provenance

| Parameter | Dataset 1: Primary ML Dataset | Dataset 2: Historical Crop Production | Dataset 3: District Rainfall Normals |
|---|---|---|---|
| **Official Title** | Crop Recommendation Dataset | District-wise Season-wise Crop Production Statistics | Customized District Rainfall Climatology & Normals |
| **Publishing Agency** | Agricultural Research Compilations (Kaggle / Open ML) | Directorate of Economics & Statistics (DES), Ministry of Agriculture | India Meteorological Department (IMD Pune), Ministry of Earth Sciences |
| **Access Date** | Baseline / Retained | 2026-10-29 | 2026-10-29 |
| **License** | Open Academic / Public Domain | Government Open Data License (GODL-India) / NDSAP | Open Access Public Meteorological Service |
| **Volume / Rows** | 2,200 rows, 22 classes (100 per class) | 12,628 rows (1997–2014) | 36 district rows (1971–2020 LPA) |
| **Spatial Scope** | Pan-India biophysical envelopes | 35 rural districts of Maharashtra | All 36 districts of Maharashtra |
| **Key Features & Units** | $N, P, K$ ($\text{kg/ha}$); Temp ($^\circ\text{C}$); Humidity ($\%$); pH; Rainfall ($\text{mm}$) | Area ($\text{ha}$); Production ($\text{MT}$); Cotton converted from Bales ($\times 0.170$) | Annual & Seasonal Normals ($\text{mm}$); Departure thresholds ($\%$) |
| **Missing Values** | 0 missing values | 238 records with unrecorded production ($NaN$) audited and omitted | 0 missing values |
| **Role in System** | Trains core biophysical Random Forest model (frozen) | Read-only historical cultivation context overlay | Read-only seasonal rainfall departure overlay |

---

## 6. Machine Learning Evaluation & Integrity

### 6.1 Evaluation Methodology
- **Split:** 80% train ($N=1,760$), 20% test ($N=440$) using stratified sampling (`random_state=42`).
- **Data Leakage Proof:** Scaler parameters $(\min, \max)$ calculated strictly on $X_{\text{train}}$. $X_{\text{test}}$ evaluated without refitting.
- **Model:** Random Forest ($100$ trees, Gini impurity, max depth unconstrained).

### 6.2 Held-Out Performance Metrics
- **Accuracy:** 99.77% ($439 / 440$ correct).
- **Macro Precision:** 99.78%
- **Macro Recall:** 99.77%
- **Macro F1-Score:** 99.77%
- **Scientific Limitation:** As documented in `VERIFICATION_AND_METHODOLOGY.md` and the research paper, this 99.77% accuracy demonstrates pattern recognition within a clean synthetic feature space; **it does not establish commercial yield or farming success in real-world agriculture**.

---

## 7. Explainability & Simulation Verification

### 7.1 TreeSHAP Local Attributions
- Model output probabilities are decomposed into additive Shapley values:
  $$f_c(\mathbf{x}) = \phi_0(c) + \sum_{i=1}^7 \phi_i(c, \mathbf{x})$$
- Distinguishes **positive drivers** ($\phi_i > 0$) from **limiting constraints** ($\phi_i < 0$).
- Pairwise contrastive explanations calculate $\Delta \phi_i(A, B) = \phi_i(A) - \phi_i(B)$ to reveal decisive differential factors.
- **Epistemic Disclaimer:** SHAP values represent statistical feature attribution within the model, not proven causal agronomic mechanisms.

### 7.2 What-If Counterfactual Studio
- Farmers can modify any subset of environmental inputs to observe candidate rank trajectories.
- Trajectory tracking categorizes crops as: *Gained Rank*, *Dropped Rank*, *Retained Rank*, *Entered Top-5*, or *Exited Top-5*.
- **Epistemic Disclaimer:** What-If outputs are model sensitivity simulations, not field-tested physical forecasts.

---

## 8. Regional & Secondary Data Integration Verification

### 8.1 Diagnostic Suite Metrics & Results
Evaluated on a 72-query diagnostic suite across all 36 districts of Maharashtra:
1. **Unviable Top-1 Recommendation Rate ($R_{\text{unviable}}$):**
   $$R_{\text{unviable}} = \frac{37}{68} = 54.41\%$$
   The raw ML model recommended an agronomically unviable or off-season crop in 54.41% of evaluated queries.
2. **Conventional False Positive Rate on Unviable Cases ($\text{FPR}_{\text{unviable}}$):**
   $$\text{FPR}_{\text{raw}} = \frac{37}{37 + 0} = 100.0\% \quad \longrightarrow \quad \text{FPR}_{\text{AgriWise}} = \frac{0}{0 + 37} = 0.0\%$$
   The regional filter intercepted all 37 false-positive recommendations.
3. **Incompatible Crop Interception Rate (ICIR):**
   $$\text{ICIR} = \frac{37}{37} = 100.0\%$$
4. **In-Season Recommendation Retention (ISRR):**
   $$\text{ISRR} = \frac{31}{31} = 100.0\%$$
- **Epistemic Disclaimer:** These metrics evaluate internal rule-engine verification on constructed scenarios, not independent external agronomic field trials.

### 8.2 Historical Grounding & Zero Data Fabrication
- When a candidate crop has established production in the district (e.g., Rice in Bhandara), the system displays verified 18-year average hectares, production tonnage, and kg/ha yield.
- When an incompatible crop (such as Apple or Jute in Jalna) is evaluated, the system displays `"No historical DES records found in Maharashtra (1997–2014)"` rather than fabricating or imputing records.

---

## 9. Automated Test Suite Results

The entire automated test suite was executed via `pytest`:

```text
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-8.3.4, pluggy-1.7.0
rootdir: S:\MCA 2 Mini project\crop
collected 52 items

tests/test_explainer.py . . .                                            [  5%]
tests/test_predictor.py . . . . .                                        [ 15%]
tests/test_preprocessor.py . . . . . . . . .                             [ 32%]
tests/test_regional.py . . . . . . .                                     [ 46%]
tests/test_regression.py . . .                                           [ 51%]
tests/test_routes.py . . . . . . . . . . . . .                           [ 76%]
tests/test_secondary.py . . . . . . . .                                  [ 92%]
tests/test_whatif.py . . . .                                             [100%]

============================= 52 passed in 12.33s =============================
```
- **Total Tests:** 52
- **Failures / Errors:** 0
- **Execution Time:** ~12.33 seconds
- **Platform Tested:** Windows 11, Python 3.10.0 (CPU-only virtual environment).

---

## 10. Checkpoint & Rollback Registry

To ensure full reproducibility and risk-free version control, the project maintains explicit Git tags at every major milestone:

| Checkpoint Tag | Commit SHA | Phase / Milestone Description |
|---|---|---|
| `phase4-pre-implementation-checkpoint` | `398b444` | Baseline regional architecture plan and 22-crop suitability matrix. |
| `phase4-completed-checkpoint` | `8b27280` | Regional decision layer, 72-query diagnostic suite, and evaluation metrics. |
| `phase5-pre-ui-checkpoint` | `085e4ea` | AFPR metric clarification and diagnostic suite artifact update. |
| `phase5-completed-checkpoint` | `a75518f` | Modern vanilla CSS UI dashboard, presets, and responsive layout. |
| `phase6-pre-implementation-checkpoint` | `a75518f` | Pre-implementation tag for secondary dataset integration. |
| `phase6-completed-checkpoint` | `7d4e0e7` | Integration of verified DES and IMD datasets, 52 passed tests. |
| `phase7-pre-packaging-checkpoint` | `7d4e0e7` | Pre-packaging tag before academic paper and final documentation. |
| `phase7-completed-checkpoint` | *Pending Commit* | Final packaging, research paper, final report, and verified package. |

**Rollback Command:**
```powershell
git checkout <tag-name>
```

---

## 11. Known Limitations & Recommendations for Future Work

1. **Temporal Horizon of Historical Data:** The DES production dataset covers 1997–2014. More recent agricultural shifts (such as expanding soybean acreage or protected horticulture) are not reflected in historical totals.
2. **Tabular Boundary vs. Micro-Environments:** The biophysical model operates on bulk soil averages. It does not account for intra-field variability, soil depth, drainage, or slope.
3. **Absence of Live Market Prices:** Current recommendations are biophysically and regionally validated but do not optimize for real-time market gross margins.
4. **Field Validation:** Future iterations should involve field-level pilot trials with regional Krishi Vigyan Kendras (KVKs) to gather structured feedback from agricultural extension officers.

---

## 12. Conclusion

AgriWise successfully demonstrates how a vulnerable, black-box crop classifier can be systematically refactored into an explainable, robust, and regionally anchored decision support system. The implementation meets all academic and engineering criteria for an MCA major project, delivering high-integrity code, complete unit test coverage, and a publication-ready research manuscript.
