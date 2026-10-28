# AgriWise: Explainable, Climate-Aware & Region-Specific AI Agricultural Decision Support System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status: Phase 3 Completed](https://img.shields.io/badge/Status-Phase%203%20Completed-brightgreen.svg)]()
[![Tests: 31 Passed](https://img.shields.io/badge/Tests-31%20Passed-success.svg)]()
[![Explainability: TreeSHAP](https://img.shields.io/badge/XAI-TreeSHAP-orange.svg)]()
[![License: Academic / Open](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

**AgriWise** is an explainable, climate-aware, and region-specific agricultural decision support system designed to evolve conventional single-crop classifiers into a scientifically grounded, multi-criteria recommendation platform.

---

## 1. Project Evolution & Phases

### Phase 1: Project Audit & Baseline Evaluation (Completed)
- Uncovered legacy scaler overwrite bug that caused live application accuracy to collapse to 5.91%.
- Preserved legacy files in [`legacy/`](file:///s:/MCA%202%20Mini%20project/crop/legacy) for academic before-and-after baseline comparison.
- Formulated research questions, architecture, and dataset specifications.

### Phase 2: Pipeline Rectification & Top-5 Engine (Completed)
- Rectified scaling pipeline in [`src/ml/preprocessor.py`](file:///s:/MCA%202%20Mini%20project/crop/src/ml/preprocessor.py) (train-only scaling, immutable transformations, strict input bounds).
- Restored held-out test accuracy from **6.14%** to **99.77%**.
- Implemented Top-5 candidate crop ranking with calibrated confidence bands and explicit scientific disclaimers in [`src/ml/predictor.py`](file:///s:/MCA%202%20Mini%20project/crop/src/ml/predictor.py).

### Phase 3: Explainable AI & What-If Simulation Studio (Completed)
- **TreeSHAP Feature Attribution ([`src/xai/explainer.py`](file:///s:/MCA%202%20Mini%20project/crop/src/xai/explainer.py)):**
  - Exact Shapley values decomposing model output probabilities into positive drivers and limiting constraints.
  - Pairwise contrastive explanations (e.g. *Why was Rice chosen over Maize?* identifying decisive differential features).
  - Strict distinction between statistical feature attribution and biological causality.
- **Interactive What-If Studio ([`src/xai/counterfactual.py`](file:///s:/MCA%202%20Mini%20project/crop/src/xai/counterfactual.py)):**
  - Side-by-side simulation comparing baseline vs. perturbed soil/weather scenarios.
  - Environmental parameter delta tracking ($\Delta x_i$ and $\%$ shift).
  - Top-5 candidate rank shift tracker (gained rank, dropped rank, entered/exited Top-5).
  - Explicit sensitivity guardrails preventing simulated outputs from being interpreted as guaranteed physical yields.
- **Verification & Methodology Document ([`VERIFICATION_AND_METHODOLOGY.md`](file:///s:/MCA%202%20Mini%20project/crop/VERIFICATION_AND_METHODOLOGY.md)):**
  - Exact mathematical formulas for ensemble voting, probability averaging, confidence band derivations, proof of zero data leakage, and academic limitations of the 99.77% score.

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
├── VERIFICATION_AND_METHODOLOGY.md # Mathematical formulation & evaluation proof
├── requirements.txt              # Pinned CPU-compatible dependencies (including shap)
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
│   ├── ml/                       # Hardened Machine Learning package
│   │   ├── preprocessor.py       # AgriDataPreprocessor
│   │   ├── predictor.py          # CropPredictor (Top-5 recommendation engine)
│   │   ├── train.py              # Reproducible training pipeline
│   │   ├── evaluate.py           # Pipeline evaluation suite
│   │   └── artifacts/            # Serialized production models & scalers
│   └── xai/                      # Explainable AI & Simulation package
│       ├── explainer.py          # AgriExplainer (TreeSHAP feature attributions)
│       └── counterfactual.py     # WhatIfSimulator (Interactive perturbation studio)
│
├── tests/                        # Automated unit and regression test suite (31 tests)
│   ├── test_preprocessor.py      # Preprocessing, validation, and immutability tests
│   ├── test_predictor.py         # Top-K candidate ranking and confidence tests
│   ├── test_explainer.py         # TreeSHAP feature attribution & contrastive tests
│   ├── test_whatif.py            # What-If simulation & delta calculation tests
│   ├── test_regression.py        # Regression tests against legacy scaling bug
│   └── test_routes.py            # Flask route integration tests (/predict, /what-if)
│
└── Crop-Recommendation-System-Using-Machine-Learning-main/  # Web Application
    ├── app.py                    # Connected to CropPredictor, AgriExplainer, WhatIfSimulator
    ├── static/img.jpg            # Static demonstration image
    └── templates/
        ├── index.html            # Main UI with TreeSHAP waterfall & contrastive analysis
        └── what_if.html          # Interactive What-If Simulation Studio
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
*(All 31 tests should pass in under 5 seconds)*

### Step 4: Run the Web Application
```powershell
cd Crop-Recommendation-System-Using-Machine-Learning-main
python app.py
```
- Main Recommendation & SHAP Explanations: **`http://localhost:5000`**
- Interactive What-If Simulation Studio: **`http://localhost:5000/what-if`**
