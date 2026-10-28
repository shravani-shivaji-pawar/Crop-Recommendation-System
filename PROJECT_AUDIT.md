# AgriWise: Existing Project Audit & Baseline Report

**Project Title:** Crop Recommendation System Using Machine Learning  
**Target Evolution:** AgriWise: An Explainable, Climate-Aware and Region-Specific AI Decision Support System for Sustainable Crop Selection  
**Audit Date:** October 2026  
**Auditor / Assistant:** Antigravity AI  
**Git Branch / Checkpoint:** `agriwise-architecture` (branched from `main`, commit clean)  

---

## 1. Executive Summary

This audit evaluates the existing repository `Crop-Recommendation-System` (originally cloned as `Crop-Recommendation-System-Using-Machine-Learning-main`). The application is a conventional 7-feature agronomic classifier wrapped in a Flask web application that accepts user-entered soil and environmental numbers and outputs a single recommended crop.

### Critical Audit Findings
1. **Critical Pipeline Scaler Bug (Live Accuracy Collapse to 5.91%):**  
   In the Jupyter Notebook (`Crop Classification With Recommendation System.ipynb`, Cell 31/34), the `MinMaxScaler` object was re-fitted via `ms.fit_transform()` on a single arbitrary test record (`[10, 10, 10, 15, 80, 4.5, 10]`) before being exported as `minmaxscaler.pkl`. In `app.py`, input features are transformed through this corrupted scaler followed by a dummy `StandardScaler`. Consequently, feature vectors fed to `model.predict()` are completely out of range (split thresholds in the trees are between 0.03 and 0.93, while live inputs produce values like 80.0, 32.0, 192.9). **Live application predictions collapse to an empirical test accuracy of only 5.91%**, almost always predicting "Apple" or "Grapes" regardless of input.
2. **True Baseline Model Quality (99.55% CV Accuracy):**  
   When evaluated cleanly using a pipeline that preserves the correct `MinMaxScaler` fitted on training splits, the underlying 100-tree `RandomForestClassifier` achieves **99.55% Stratified 5-Fold Cross-Validation Accuracy** and **99.54% Macro F1-score**.
3. **Missing Static Folder & Assets:**  
   `templates/index.html` references `{{ url_for('static', filename='img.jpg') }}`, but `img.jpg` resides in the root folder without a `static/` directory, resulting in an HTTP 404 image load error.
4. **Missing Dependency Manifest:**  
   `README.md` instructs users to run `pip install -r requirements.txt`, but no `requirements.txt` file exists in the repository.
5. **Academic & Practical Limitations:**  
   The current dataset is a synthetic/curated 2200-row benchmark with 22 balanced classes. It has **no geographic coordinates, no state/district identifiers, no seasonal flags (Kharif/Rabi/Zaid), no market prices, no yield figures, and no water-table or irrigation constraints**. Therefore, it functions solely as an idealized agronomic condition matcher, not a region-specific or economic decision support system.

---

## 2. Existing Directory Structure & Assets

```
s:\MCA 2 Mini project\crop\
├── .git\                                  # Git version control (branch: agriwise-architecture)
├── .gitignore                             # Ignores venv, pycache, .vscode, python/
├── python\
│   └── Python-3.10.20\                    # Python source tarball directory
└── Crop-Recommendation-System-Using-Machine-Learning-main\
    ├── .vscode\                           # VS Code workspace settings
    ├── Crop Classification With Recommendation System.ipynb  # Exploratory and training notebook (67.2 KB)
    ├── Crop_recommendation.csv             # Kaggle benchmark dataset (150 KB, 2,200 rows)
    ├── README.md                          # Basic markdown documentation (3.55 KB)
    ├── app.py                             # Flask web server (53 lines, 1.79 KB)
    ├── img.jpg                            # Demonstration image (13.5 KB, placed incorrectly in root)
    ├── minmaxscaler.pkl                   # Corrupted MinMaxScaler artifact (760 bytes)
    ├── standscaler.pkl                    # Dummy StandardScaler artifact (617 bytes)
    ├── model.pkl                          # Trained RandomForestClassifier (3.53 MB)
    ├── templates\
    │   └── index.html                     # Bootstrap 5.3 frontend template (132 lines, 4.88 KB)
    └── venv\                              # Python 3.10.0 virtual environment
```

---

## 3. Technology Stack & Runtime Environment

| Layer | Component / Tool | Version / Spec | Notes |
|---|---|---|---|
| **Runtime** | Python | 3.10.0 (in `venv`) / 3.14.5 (system) | Active virtual environment in `venv/` |
| **Backend** | Flask | 3.1.3 | Synchronous WSGI web application |
| **Data Processing** | Pandas, NumPy | 2.3.3, 2.2.6 | Data manipulation and array transformations |
| **Machine Learning**| scikit-learn, SciPy | 1.7.2, 1.15.3 | Model training, scaling, evaluation |
| **Model Serialization** | Pickle | Python stdlib | Binary serialization of scalers and models |
| **Frontend** | Jinja2, HTML5, Bootstrap | Bootstrap 5.3.0-alpha3 (via CDN) | Vanilla CSS & responsive grid |
| **Hardware Constraint**| CPU-only, 16 GB RAM | Local machine | Verified CPU-compatible, no GPU required |

---

## 4. Current Dataset Audit: `Crop_recommendation.csv`

### 4.1 Schema and Data Profiling

| Feature Name | Role | Data Type | Non-Null Count | Min | Mean | Max | Unit / Description |
|---|---|---|---|---|---|---|---|
| `N` | Input Feature | `int64` | 2,200 | 0.00 | 50.55 | 140.00 | Ratio of Nitrogen content in soil |
| `P` | Input Feature | `int64` | 2,200 | 5.00 | 53.36 | 145.00 | Ratio of Phosphorus content in soil |
| `K` | Input Feature | `int64` | 2,200 | 5.00 | 48.15 | 205.00 | Ratio of Potassium content in soil |
| `temperature` | Input Feature | `float64` | 2,200 | 8.83 | 25.62 | 43.68 | Ambient temperature in °C |
| `humidity` | Input Feature | `float64` | 2,200 | 14.26 | 71.48 | 99.98 | Relative humidity in % |
| `ph` | Input Feature | `float64` | 2,200 | 3.50 | 6.47 | 9.94 | Soil pH value (0 to 14 scale) |
| `rainfall` | Input Feature | `float64` | 2,200 | 20.21 | 103.46 | 298.56 | Precipitation / rainfall in mm |
| `label` | Target Class | `object` (string) | 2,200 | — | — | — | Crop species name (22 classes) |

### 4.2 Class Distribution
- **Total rows:** 2,200
- **Missing values:** 0 across all columns
- **Duplicate rows:** 0
- **Number of distinct classes:** 22
- **Distribution:** Exactly 100 rows per crop (perfectly balanced 4.545% per class).
- **Crops included:**
  `rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`.

### 4.3 Provenance and Limitations of the Dataset
- **Origin:** Curated Indian Council of Agricultural Research (ICAR) envelope dataset popularized via Kaggle.
- **Nature:** Highly synthetic/curated. Feature distributions for each crop form distinct, low-variance clusters.
- **Geographic Blindness:** No state, district, or sub-district fields exist.
- **Temporal Blindness:** No season (Kharif, Rabi, Zaid) or month/year stamps exist.
- **Agronomic Oversimplification:** Crops are treated as mutually exclusive discrete classes rather than co-viable crops with varying yield, water, and market feasibility.

---

## 5. Existing ML Model & Training Procedure

### 5.1 Training Workflow in Notebook
1. `Crop_recommendation.csv` loaded into Pandas DataFrame.
2. Label encoding: Crop string labels mapped to integers $1 \dots 22$ using a manual dictionary `crop_dict`.
3. Train/Test Split: 80% train ($N=1760$), 20% test ($N=440$), fixed `random_state=42`.
4. Feature Scaling: `MinMaxScaler` applied to $X_{train}$ and $X_{test}$.
5. Multi-Model Exploration in Cell 27:
   - Logistic Regression, Gaussian Naive Bayes, SVM, KNN, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, Extra Trees.
6. Model Selection: `RandomForestClassifier` was selected and trained on `X_train` ($n\_estimators=100$, default hyperparameters).

### 5.2 The Corrupted Scaler & Live Prediction Pipeline
- In Cell 31 of the notebook, the developer defined:
  ```python
  def recommendation(N,P,k,temperature,humidity,ph,rainfal):
      features = np.array([[N,P,k,temperature,humidity,ph,rainfal]])
      transformed_features = ms.fit_transform(features)  # BUG: re-fits scaler on 1 sample!
      prediction = rfc.predict(transformed_features)
      return prediction[0]
  ```
- Running Cell 34 with test inputs `[10, 10, 10, 15, 80, 4.5, 10]` permanently mutated the global `ms` instance so that:
  - `data_min_` = `[10, 10, 10, 15, 80, 4.5, 10]`
  - `data_max_` = `[10, 10, 10, 15, 80, 4.5, 10]`
  - `scale_` = `[1, 1, 1, 1, 1, 1, 1]`
- Cell 35 dumped this ruined scaler to `minmaxscaler.pkl`.
- In `app.py`:
  ```python
  scaled_features = ms.transform(single_pred)
  final_features = sc.transform(scaled_features)
  prediction = model.predict(final_features)
  ```
- Because `standscaler.pkl` has `mean_ = 0` and `scale_ = 1` (an identity transformation), `final_features` retains values between 10 and 200+.
- Since the Random Forest split points were learned on normalized features in $[0.0, 1.0]$, every decision tree is routed to extreme right leaves, resulting in deterministic misclassification:
  - Typical Rice $(N=90, P=42, K=43, \dots) \rightarrow$ **Apple** (Class 8)
  - Typical Maize $(N=71, P=54, K=16, \dots) \rightarrow$ **Grapes** (Class 11)
  - Typical Cotton $(N=118, P=46, K=20, \dots) \rightarrow$ **Apple** (Class 8)

---

## 6. Baseline Model Evaluation (Verified Empirical Benchmark)

To establish an honest, reproducible baseline without fabrication, we evaluated the dataset using a **Stratified 5-Fold Cross-Validation** pipeline ($K=5, N=2200, \text{random\_state}=42$) and compared it with the existing `app.py` pipeline.

### 6.1 Reproducible Baseline Benchmark Results (5-Fold CV)

| Model Architecture | CV Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Fit Time (ms) | Inference Time (ms) | Model Size |
|---|---|---|---|---|---|---|---|
| **Random Forest ($n=100$)** *(Current)* | **99.55 ± 0.35** | **99.57** | **99.55** | **99.54** | 277.05 | 12.90 | 3.53 MB |
| **Gaussian Naive Bayes** | **99.45 ± 0.40** | 99.50 | 99.45 | 99.45 | 3.25 | 9.45 | ~45 KB |
| **Gradient Boosting (sklearn)** | 99.09 ± 0.45 | 99.13 | 99.09 | 99.09 | 7500.08 | 14.80 | ~1.8 MB |
| **Decision Tree (CART)** | 98.77 ± 0.65 | 98.83 | 98.77 | 98.77 | 17.15 | 4.80 | ~120 KB |
| **Support Vector Machine (RBF)** | 98.27 ± 0.70 | 98.46 | 98.27 | 98.27 | 142.67 | 41.03 | ~380 KB |
| **K-Nearest Neighbors ($k=5$)** | 97.82 ± 0.85 | 98.08 | 97.82 | 97.81 | 1.60 | 28.70 | ~150 KB |
| **Logistic Regression (Softmax)** | 94.09 ± 1.10 | 94.79 | 94.09 | 93.98 | 34.01 | 9.92 | ~25 KB |

### 6.2 Existing App Live Pipeline vs. True Baseline Comparison

| Pipeline Configuration | Evaluation Set | Empirical Accuracy | Practical Outcome |
|---|---|---|---|
| **Existing `app.py` live pipeline** (`ms_corrupted` + `sc_dummy` + `model.pkl`) | 20% Held-Out Test ($N=440$) | **5.91%** | **Catastrophic Failure.** Predictions default almost exclusively to Apple or Grapes. |
| **Corrected Pipeline** (`true_MinMaxScaler` + `model.pkl`) | 20% Held-Out Test ($N=440$) | **99.32%** | **Correct Functionality.** Correctly identifies Rice, Maize, Chickpea, Cotton, etc. |

---

## 7. Existing User Interface & User Experience

- **Framework:** Flask + Jinja2 template (`templates/index.html`).
- **Styling:** Bootstrap 5.3 CDN stylesheet with custom inline styles.
- **Page Structure:** Single-page application containing:
  - Top navigation bar (Home, Contact, About [disabled], search input [decorative, non-functional]).
  - 7 numerical `<input type="number">` fields for N, P, K, Temperature, Humidity, pH, Rainfall.
  - Submit button: "Get Recommendation".
  - Conditional result card rendering crop name and `img.jpg`.
- **UX Deficiencies:**
  1. No input validation ranges (users can enter negative rainfall or pH 25).
  2. No explanation of what the features represent or recommended measurement units.
  3. Single-crop output without confidence scores or alternative candidate crops.
  4. Non-responsive card layout on mobile devices (`margin-left: 410px` hardcoded in CSS).
  5. Image link fails with HTTP 404.

---

## 8. Summary of Technical Debt & Bugs

1. **Bug #1 (High Severity):** Scaler corruption in `minmaxscaler.pkl` causes live prediction failure (5.91% accuracy).
2. **Bug #2 (Medium Severity):** `standscaler.pkl` is an unfitted dummy scaler applied redundantly in `app.py`.
3. **Bug #3 (Low Severity):** `img.jpg` is located in the root directory instead of `static/`, breaking image display.
4. **Bug #4 (Medium Severity):** Missing `requirements.txt` breaks automated deployment and reproducibility.
5. **Technical Debt #1:** Hardcoded label dictionaries in both `app.py` and notebook (class index mapping is manual and fragile).
6. **Technical Debt #2:** Hardcoded root directory relative paths (`open('model.pkl', 'rb')` fails if executed outside the directory).
7. **Technical Debt #3:** Lack of top-$K$ probabilities; only the argmax label is returned.
8. **Technical Debt #4:** No explainability, no regional awareness, no economic context, and no validation of real-world agronomic plausibility.

---

## 9. Baseline Preservation Plan

To adhere to research reproducibility standards:
- The original files `Crop-Recommendation-System-Using-Machine-Learning-main/model.pkl`, `minmaxscaler.pkl`, `standscaler.pkl`, and `Crop Classification With Recommendation System.ipynb` **will be preserved intact** as legacy artifacts.
- A new modular architecture will encapsulate the baseline model under a dedicated `baseline/` namespace, allowing honest before-and-after empirical comparison in the research paper and project evaluation.
