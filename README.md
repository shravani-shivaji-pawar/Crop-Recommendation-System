# AgroVista AI: Explainable Crop Recommendation with Regional and Seasonal Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Phase 7 Completed](https://img.shields.io/badge/Status-Phase%207%20Completed-brightgreen.svg)]()
[![Tests: 52 Passed](https://img.shields.io/badge/Tests-52%20Passed-success.svg)]()
[![Explainability: TreeSHAP](https://img.shields.io/badge/XAI-TreeSHAP-orange.svg)]()
[![License: Academic / Open](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

**AgroVista AI** is an explainable, climate-aware, and region-specific agricultural decision support system designed to upgrade conventional single-crop classifiers into a scientifically grounded, multi-criteria recommendation platform.

---

## 1. Project Overview & Evolution

AgroVista AI transforms a legacy, black-box crop classification script into a publication-grade decision support platform:

- **Phase 1: Project Audit & Baseline Evaluation**  
  Uncovered a critical data leakage and scaler overwrite bug that caused live prediction accuracy to collapse to 6.14%. Legacy artifacts safely preserved in [`legacy/`](file:///s:/MCA%202%20Mini%20project/crop/legacy).
- **Phase 2: Pipeline Rectification & Top-5 Engine**  
  Restored held-out benchmark accuracy to **99.77%** with an immutable preprocessing pipeline and multi-candidate Top-5 ranking engine with calibrated confidence bands.
- **Phase 3: Explainable AI & What-If Studio**  
  Integrated exact TreeSHAP feature attributions, pairwise contrastive explanations (*Why Crop A over Crop B?*), and an interactive counterfactual What-If simulation studio.
- **Phase 4: Regional & Seasonal Decision Layer (Maharashtra)**  
  Formulated an agro-climatic decision engine mapping 22 crops across 36 districts and 9 agro-climatic zones, intercepting unviable crops while preserving in-season candidates.
- **Phase 5: Modern UI Dashboard Upgrade**  
  Implemented a responsive, accessible vanilla CSS interface with agro-climatic presets, waterfall factor charts, and contrastive comparison cards.
- **Phase 6: Verified Secondary Dataset Integration**  
  Ingested 18 years of historical crop production statistics (APS/DES, 12,628 rows) and district rainfall normals (IMD Pune) as decoupled contextual overlays.
- **Phase 7: Research Paper & Final Packaging**  
  Authored academic manuscript draft ([`AGRIWISE_RESEARCH_PAPER.md`](file:///s:/MCA%202%20Mini%20project/crop/AGRIWISE_RESEARCH_PAPER.md)) and final project report ([`PROJECT_FINAL_REPORT.md`](file:///s:/MCA%202%20Mini%20project/crop/PROJECT_FINAL_REPORT.md)).

---

## 2. Empirical Benchmark Evaluation

| Pipeline Version | Held-Out Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
|---|---|---|---|---|---|
| **Legacy Pipeline** (`legacy/minmaxscaler.pkl` + `legacy/standscaler.pkl`) | **6.14%** | 5.09% | 6.14% | 2.50% | **FAILED** (Catastrophic Scaler Overwrite) |
| **AgriWise Rectified Pipeline** (`src/ml/artifacts/scaler.pkl`) | **99.77%** | **99.78%** | **99.77%** | **99.77%** | **PASSED** (High Integrity) |

> **Scientific Disclaimer:** The 99.77% accuracy is a held-out benchmark on the 2,200-sample dataset. It measures pattern separation in synthetic feature space and does not constitute a guarantee of real-world biological yield or farming success.

---

## 3. Directory Layout

```
s:\MCA 2 Mini project\crop\
├── AGRIWISE_RESEARCH_PAPER.md        # Academic research paper draft (16 sections)
├── PROJECT_FINAL_REPORT.md           # Comprehensive major project documentation
├── PROJECT_AUDIT.md                  # Phase 1 technical audit of legacy codebase
├── PROPOSED_ARCHITECTURE.md         # Multi-module architectural design specification
├── DATASET_REQUIREMENTS.md           # Secondary dataset integration rules
├── RESEARCH_PLAN.md                  # Research questions & experimental design
├── IMPLEMENTATION_ROADMAP.md         # Multi-phase milestone tracking
├── VERIFICATION_AND_METHODOLOGY.md   # Mathematical formulations & leakage proofs
├── requirements.txt                  # Pinned CPU-compatible dependencies
├── .env.example                      # Clean environment template (no credentials)
│
├── legacy/                           # Preserved original codebase and model artifacts
│   ├── app.py
│   ├── model.pkl
│   ├── minmaxscaler.pkl
│   ├── standscaler.pkl
│   └── Crop_recommendation.csv
│
├── data/
│   ├── raw/
│   │   ├── Crop_recommendation.csv                    # Primary 2,200-sample ML dataset
│   │   ├── maharashtra_crop_production_des_raw.csv    # Secondary DES historical dataset
│   │   └── maharashtra_district_rainfall_normals_imd.csv # Secondary IMD rainfall normals
│   └── DATASET_MANIFEST.md           # Provenance, schema, and licensing catalog
│
├── src/
│   ├── core/                         # Configuration, bounds, and domain exceptions
│   │   ├── config.py
│   │   └── exceptions.py
│   ├── ml/                           # Hardened Machine Learning package
│   │   ├── preprocessor.py           # AgriDataPreprocessor (train-only scaling)
│   │   ├── predictor.py              # CropPredictor (Top-5 candidate ranking engine)
│   │   ├── train.py                  # Reproducible training pipeline
│   │   ├── evaluate.py               # Pipeline evaluation suite
│   │   └── artifacts/                # Validated production model & scaler artifacts
│   ├── xai/                          # Explainability & Simulation package
│   │   ├── explainer.py              # AgriExplainer (TreeSHAP & contrastive analysis)
│   │   └── counterfactual.py         # WhatIfSimulator (Interactive perturbation studio)
│   └── modules/
│       └── regional/                 # Regional & Seasonal Decision Layer
│           ├── regional_filter.py    # RegionalConstraintEngine (Maharashtra rules)
│           ├── secondary_data.py     # Repositories for DES & IMD datasets
│           └── evaluator.py          # 72-query diagnostic suite runner
│
├── tests/                            # Automated unit & integration test suite (52 tests)
│   ├── test_preprocessor.py          # Preprocessing & immutability tests
│   ├── test_predictor.py             # Top-5 ranking & confidence score tests
│   ├── test_explainer.py             # TreeSHAP & contrastive attribution tests
│   ├── test_whatif.py                # What-If delta & rank shift tests
│   ├── test_regional.py              # Maharashtra rules & diagnostic suite tests
│   ├── test_secondary.py             # DES/IMD schema, cotton bales, missing value tests
│   ├── test_regression.py            # Regression tests against legacy scaler bug
│   └── test_routes.py                # Flask route integration & UI rendering tests
│
└── Crop-Recommendation-System-Using-Machine-Learning-main/ # Web Application
    ├── app.py                        # Flask controller integrating all modules
    ├── static/
    │   ├── css/style.css             # Vanilla CSS design system
    │   └── img.jpg                   # Static UI asset
    └── templates/
        ├── index.html                # Main recommendation & SHAP dashboard
        └── what_if.html              # Interactive What-If Simulation Studio
```

---

## 4. Local Installation and Execution Instructions

### Prerequisites
- **Python:** Version 3.10+ (tested on Python 3.10.0, 64-bit)
- **Operating System:** Windows, Linux, or macOS (CPU-only, no GPU required)
- **Memory:** Minimum 2 GB RAM

### Step 1: Clone and Set Up Virtual Environment

```powershell
# From project root:
cd "s:\MCA 2 Mini project\crop"

# Create virtual environment (if not already created):
python -m venv venv

# Activate virtual environment on Windows (PowerShell):
.\Crop-Recommendation-System-Using-Machine-Learning-main\venv\Scripts\Activate.ps1
# Or if using root venv:
# .\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite

```powershell
python -m pytest tests/ -v
```
*(All 52 tests should pass in approximately 12 seconds).*

### Step 4: Run the Web Application

```powershell
cd Crop-Recommendation-System-Using-Machine-Learning-main
python app.py
```

The application starts on the local development server:
- **Main Recommendation & SHAP Dashboard:** [`http://127.0.0.1:5000`](http://127.0.0.1:5000)
- **Interactive What-If Simulation Studio:** [`http://127.0.0.1:5000/what-if`](http://127.0.0.1:5000/what-if)

---

## 5. System Features & Usage Guide

1. **Biophysical Recommendation:** Input Nitrogen ($N$), Phosphorus ($P$), Potassium ($K$), Temperature, Humidity, pH, and Rainfall.
2. **Regional Metadata (Optional):** Select a Maharashtra district (e.g., Pune, Jalna, Ratnagiri) and target sowing month to trigger regional validation and historical cultivation context.
3. **TreeSHAP Feature Decomposition:** View positive drivers (factors favoring the crop) and limiting constraints (factors penalizing the crop).
4. **Contrastive Analysis:** Inspect pairwise feature differences between the Top-1 crop and runner-up candidate (*Why Crop A over Crop B?*).
5. **What-If Studio:** Adjust environmental sliders to simulate climate perturbations (e.g., monsoonal drought) and track candidate rank shifts.

---

## 6. Research Disclaimers & Epistemic Boundaries

- **Not Causal Agronomy:** TreeSHAP attributions represent statistical feature contributions within the machine learning model; they do not establish biological or causal mechanisms.
- **Simulation vs. Prediction:** What-If Studio outputs are sensitivity simulations of model behavior, not field-tested physical forecasts.
- **Historical Scope:** DES agricultural records cover 1997–2014. Absence of a crop in DES survey records indicates zero surveyed commercial cultivation in that period, not biological impossibility.
- **Diagnostic Suite vs. Real-World Trials:** The 100% Interception Rate and 100% Retention Rate reflect software verification against structured diagnostic test cases, not randomized multi-season field trials.
