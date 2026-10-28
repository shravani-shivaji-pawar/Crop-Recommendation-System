# AgriWise: Explainable, Climate-Aware & Region-Specific AI Agricultural Decision Support System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Phase 2 Hardened](https://img.shields.io/badge/Status-Phase%202%20Hardened-brightgreen.svg)]()
[![Tests: 21 Passed](https://img.shields.io/badge/Tests-21%20Passed-success.svg)]()
[![License: Academic / Open](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

**AgriWise** is an explainable, climate-aware, and region-specific agricultural decision support system designed to evolve conventional single-crop classifiers into a scientifically grounded, multi-criteria recommendation platform.

---

## 1. Phase 2 Status & Key Enhancements

- **Critical Scaler Fix:** Eliminated the legacy scaling pipeline bug that previously caused live application accuracy to collapse to 5.91%. Restored empirical held-out accuracy to **99.77%**.
- **Hardened Machine Learning Package (`src/ml`):**
  - [`AgriDataPreprocessor`](file:///s:/MCA%202%20Mini%20project/crop/src/ml/preprocessor.py): Strict data integrity, train-only scaling, and immutable transformation.
  - [`CropPredictor`](file:///s:/MCA%202%20Mini%20project/crop/src/ml/predictor.py): Top-5 candidate crop ranking with calibrated confidence bands and explicit scientific disclaimers.
  - [`train_production_model`](file:///s:/MCA%202%20Mini%20project/crop/src/ml/train.py): Deterministic, reproducible training harness (`random_state=42`).
- **Input Validation:** Agronomic range validation (e.g. soil pH between 3.0 and 11.0, non-negative rainfall) and alias matching (`P`, `Phosphorus`, `Phosporus`).
- **Automated Test Suite:** 21 unit, integration, and regression tests ensuring zero pipeline regression.
- **Legacy Artifact Preservation:** Original model and notebook files safely preserved in `legacy/` for academic baseline comparison.

---

## 2. Empirical Benchmark Evaluation

| Pipeline Version | Held-Out Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Status |
|---|---|---|---|---|---|
| **Legacy Pipeline** (`legacy/minmaxscaler.pkl` + `legacy/standscaler.pkl`) | **6.14%** | 5.09% | 6.14% | 2.50% | **FAILED** (Severe Scaler Overwrite Bug) |
| **AgriWise Rectified Pipeline** (`src/ml/artifacts/scaler.pkl`) | **99.77%** | **99.78%** | **99.77%** | **99.77%** | **PASSED** (High Integrity) |

---

## 3. Directory Layout

```
s:\MCA 2 Mini project\crop\
├── PROJECT_AUDIT.md              # Technical audit of legacy codebase & baseline benchmarks
├── PROPOSED_ARCHITECTURE.md     # Multi-module AgriWise architectural specification
├── DATASET_REQUIREMENTS.md       # Integration specifications for future secondary datasets
├── RESEARCH_PLAN.md              # Academic problem statement, research questions, ablation studies
├── IMPLEMENTATION_ROADMAP.md     # 7-phase implementation strategy
├── requirements.txt              # Pinned CPU-compatible dependencies
│
├── legacy/                       # Safely preserved original codebase and model artifacts
│   ├── app.py
│   ├── model.pkl
│   ├── minmaxscaler.pkl
│   ├── standscaler.pkl
│   └── Crop_recommendation.csv
│
├── data/raw/
│   └── Crop_recommendation.csv   # Pristine 2,200-sample benchmark dataset
│
├── src/
│   ├── core/                     # Configuration, bounds, and custom exceptions
│   │   ├── config.py
│   │   └── exceptions.py
│   └── ml/                       # Hardened Machine Learning package
│       ├── preprocessor.py       # AgriDataPreprocessor
│       ├── predictor.py          # CropPredictor (Top-5 recommendation engine)
│       ├── train.py              # Reproducible training pipeline
│       ├── evaluate.py           # Pipeline evaluation suite
│       └── artifacts/            # Serialized production models & scalers
│           ├── best_model.pkl
│           ├── scaler.pkl
│           └── metadata.json
│
├── tests/                        # Automated unit and regression test suite
│   ├── test_preprocessor.py      # Preprocessing, validation, and immutability tests
│   ├── test_predictor.py         # Top-K candidate ranking and confidence tests
│   ├── test_regression.py        # Regression tests against legacy scaling bug
│   └── test_routes.py            # Flask route integration tests
│
└── Crop-Recommendation-System-Using-Machine-Learning-main/  # Working Flask UI
    ├── app.py                    # Connected to hardened CropPredictor
    ├── static/img.jpg            # Static demonstration image
    └── templates/index.html      # Responsive UI with validation alerts & Top-5 ranking
```

---

## 4. Local Installation and Execution Instructions

### Prerequisites
- Python 3.10+ (CPU-compatible, no GPU required)
- Operating System: Windows, Linux, or macOS

### Step 1: Activate Virtual Environment
```powershell
# From project root:
.\Crop-Recommendation-System-Using-Machine-Learning-main\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
```powershell
python -m pytest tests -v
```
*(All 21 tests should pass in under 5 seconds)*

### Step 4: Run the Web Application
```powershell
cd Crop-Recommendation-System-Using-Machine-Learning-main
python app.py
```
Access the application in your browser at: **`http://localhost:5000`**
