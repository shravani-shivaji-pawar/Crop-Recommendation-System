# AgriWise: An Explainable, Climate-Aware, and Region-Specific AI Decision Support System for Sustainable Crop Selection

**Authors:** Shravani Shivaji Pawar  
**Affiliation:** Master of Computer Applications (MCA), Department of Computer Applications  
**Project Repository:** `shravani-shivaji-pawar/Crop-Recommendation-System`  
**Date:** October 2026  
**Document Classification:** Academic Research Paper Draft / Pre-Submission Manuscript  

---

## Abstract

Machine learning-driven agricultural decision support systems often rely on biophysical crop classification models trained exclusively on tabular physicochemical soil and ambient meteorological features. While these models frequently report benchmark classification accuracies exceeding 99%, their direct deployment into agricultural advisory settings exposes critical vulnerabilities: silent data preprocessing bugs, lack of feature-level decision transparency, absence of interactive sensitivity analysis, and complete detachment from regional agro-ecological viability, sowing calendars, and historical cultivation scales. In this work, we present **AgriWise**, an end-to-end explainable, climate-aware, and region-specific agricultural decision support system that addresses these limitations. 

First, we uncover and rectify a critical data leakage and scaler overwrite flaw prevalent in open-source crop recommendation implementations, restoring true held-out test accuracy from a collapsed 6.14% to 99.77% on the 2,200-sample benchmark dataset across 22 crop classes. Second, rather than outputting misleading uncalibrated single-crop softmax confidences, AgriWise introduces a multi-candidate Top-$K$ ranking engine with ensemble-calibrated decision bounds. Third, we integrate exact TreeSHAP (SHapley Additive exPlanations) to decompose model output probabilities into positive drivers and limiting soil/climate constraints, supplemented by pairwise contrastive explanations (*Why Crop A over Crop B?*). Fourth, an interactive What-If simulation studio allows farmers and agronomists to evaluate candidate stability under hypothetical environmental perturbations such as monsoonal rainfall deficits. Finally, AgriWise introduces a post-inference regional constraint engine and historical context layer for the state of Maharashtra, integrating 18 years (1997–2014) of district-level crop production records from the Directorate of Economics and Statistics (DES, 12,628 records) and official district rainfall normals from the India Meteorological Department (IMD Pune). 

In an automated diagnostic suite of 72 structured agro-climatic scenarios across 36 districts, the raw biophysical classifier selected an agronomically unviable or off-season crop as its Top-1 recommendation in 37 out of 68 evaluated queries (an Unviable Top-1 Recommendation Rate of 54.41%), yielding a 100.0% Conventional False Positive Rate on unviable cases. The AgriWise regional decision layer intercepted 100% of these unviable recommendations while preserving valid seasonal alternatives (100% In-Season Recommendation Retention), enriching each viable candidate with real-world historical cultivation hectares, production tonnage, and IMD rainfall departures. We discuss the strict epistemic boundaries of this work: held-out benchmark accuracy does not guarantee farm-level biological yield, SHAP values reflect model attribution rather than causal agronomy, What-If outputs are exploratory sensitivity simulations, and diagnostic suite results represent rule-based verification rather than randomized multi-season field trials.

---

## 1. Keywords

Agricultural Decision Support Systems (ADSS); Crop Recommendation; Explainable Artificial Intelligence (XAI); TreeSHAP; Regional Agro-Climatic Constraints; Climate Normal Departures; Machine Learning Integrity.

---

## 2. Introduction and Motivation

Agriculture in developing economies is confronting unprecedented challenges arising from climate variability, erratic precipitation regimes, soil degradation, and market volatility. In India, where over 50% of the workforce is engaged in agriculture and allied sectors, smallholder farmers often rely on traditional heuristic knowledge or peer practices that struggle to adapt to localized climate disruptions. Consequently, automated Agricultural Decision Support Systems (ADSS) powered by machine learning (ML) have emerged as an active area of computer applications research.

A widely circulated paradigm in literature involves training supervised multi-class classifiers—such as Random Forests, Support Vector Machines, or Gradient Boosting—on benchmark tabular datasets mapping seven physicochemical features (Nitrogen [$N$], Phosphorus [$P$], Potassium [$K$], ambient temperature, relative humidity, soil pH, and rainfall) to 22 candidate crops. In research publications and open-source repositories, these models regularly boast classification accuracies between 98% and 99.8%.

However, rigorous inspection reveals several structural deficiencies that prevent these systems from providing safe, reliable, and actionable guidance:
1. **Pervasive Preprocessing Defects:** Many implementations contain silent data pipeline flaws, such as fitting normalizers or scalers on single live inference requests, which corrupts the underlying numerical transformations and causes accuracy to collapse during production use.
2. **The "Black-Box" Transparency Deficit:** Standard ML models output single categorical labels without explaining *why* a particular crop was selected or *which* environmental parameters acted as limiting factors, violating principles of agricultural trustworthiness.
3. **Absence of Climate Perturbation Sensitivity:** Farmers operate under weather uncertainty. Standard systems do not offer counterfactual or sensitivity analysis tools to answer questions like: *"What happens to crop viability if monsoon rainfall drops by 30%?"*
4. **Agro-Ecological and Seasonal Detachment:** Purely biophysical tabular classifiers possess no geographic or temporal awareness. When given soil and weather inputs that numerically resemble temperate conditions, a model will readily recommend temperate orchard fruits (such as Apple) for tropical, low-altitude districts in peninsular India (such as Jalna or Solapur in Maharashtra), where such crops are commercially and biophysically impossible.
5. **Lack of Empirical Historical Grounding:** Predictions are presented without contextual validation from real-world agricultural statistics, ignoring whether a recommended crop has any historical precedent or infrastructure in the target district.

This research addresses these deficiencies by architecting **AgriWise**, a unified, multi-layered decision support platform that combines hardened machine learning, local Shapley feature attribution, interactive counterfactual simulation, regional agronomic constraints, and verified secondary empirical datasets.

---

## 3. Problem Statement and Research Objectives

### 3.1 Problem Statement
Given a vector of seven biophysical parameters $\mathbf{x} \in \mathbb{R}^7$ representing soil nutrient concentrations and ambient climatic conditions, along with optional administrative regional metadata (district $d$ and target calendar month $m$), how can we construct an automated decision support system that:
1. Delivers high-integrity, multi-candidate recommendations with scientifically calibrated confidence scores;
2. Provides transparent local feature attributions and contrastive explanations for model outputs;
3. Enables interactive counterfactual sensitivity analysis under environmental stress;
4. Enforces regional agro-climatic viability and seasonal sowing constraints; and
5. Contextualizes recommendations against historical district cultivation statistics and long-period rainfall normals, without fabricating records or compromising computational reproducibility?

### 3.2 Research Objectives
The specific objectives of this study are:
1. **Pipeline Rectification:** Audit and resolve data leakage and scaler overwrite defects in the legacy ML pipeline, establishing rigorous unit-tested preprocessing immutability.
2. **Candidate Ranking Formulation:** Transition from single-label predictions to Top-$K$ candidate ranking with ensemble-calibrated decision bands.
3. **Local Explainability Integration:** Implement exact TreeSHAP attribution and pairwise contrastive explanations to demystify individual crop classifications.
4. **What-If Simulation Engine:** Build an interactive simulation engine to compute candidate rank trajectories under user-defined parameter perturbations.
5. **Regional Decision Layer Formulation:** Develop a rule-based constraint engine for Maharashtra encompassing 36 districts, 9 agro-climatic zones, and 3 cropping seasons across 22 benchmark crops.
6. **Secondary Dataset Integration:** Ingest and normalize 18 years of historical crop production statistics (APS/DES) and district rainfall normals (IMD Pune) as decoupled contextual overlays.
7. **Empirical Evaluation & Integrity Audit:** Evaluate the architecture using a held-out benchmark test set, a 72-query regional diagnostic suite, and explicit threats-to-validity boundaries.

---

## 4. Related Work

### 4.1 Machine Learning in Crop Selection
Supervised learning algorithms have been extensively explored for agricultural crop classification. Kumar et al. (2020) and Pudumalar et al. (2017) demonstrated that ensemble tree-based models, particularly Random Forest (Breiman, 2001), consistently outperform single decision trees and naive Bayes classifiers when predicting crop suitability from soil nutrient ($N, P, K$) and meteorological features. However, existing literature predominantly focuses on maximizing overall classification accuracy on curated tabular datasets (e.g., Kaggle / UCI repositories), rarely auditing model behavior against geographical realities or checking for silent pipeline leaks during deployment.

### 4.2 Explainable AI in Agriculture
Explainable AI (XAI) techniques have gained traction in high-stakes domains. Lundberg and Lee (2017) unified local feature attribution methods through SHAP (SHapley Additive exPlanations), demonstrating that game-theoretic Shapley values uniquely satisfy the properties of local accuracy, missingness, and consistency. Lundberg et al. (2020) subsequently introduced TreeSHAP, enabling polynomial-time computation of exact Shapley values for tree ensembles. While XAI has been applied to precision agriculture and yield estimation (e.g., Wolanin et al., 2020), its application to multi-class crop recommendation—specifically through contrastive feature explanations (*Why Crop A instead of Crop B?*)—remains underexplored.

### 4.3 Counterfactual and What-If Explanations
Counterfactual explanations, formalized by Wachter et al. (2017), explore the minimal perturbations required to alter a model's decision. In agricultural planning, farmers rarely seek abstract global feature importances; instead, they require actionable sensitivity insights: how robust is a crop choice if precipitation falls below average, or if soil nitrogen is depleted? Implementing interactive sensitivity studios provides a pragmatic operationalization of counterfactual reasoning without requiring computationally prohibitive optimization loops.

### 4.4 Agro-Climatic Zoning and Decision Constraints
Agricultural universities and government bodies in India have long established that crop viability is governed by agro-climatic zoning rather than isolated chemical numbers. The Indian Council of Agricultural Research (ICAR) and Mahatma Phule Krishi Vidyapeeth (MPKV, Rahuri) classify Maharashtra into nine distinct agro-climatic zones based on rainfall patterns, soil types, and elevation (Mahatma Phule Krishi Vidyapeeth, 2020). Regional agricultural calendars dictate distinct sowing windows for Kharif (monsoon), Rabi (post-monsoon), and Summer seasons. Automated advisory tools that ignore these verified agronomic boundaries risk recommending physiologically incompatible crops.

---

## 5. Dataset Descriptions, Provenance, and Limitations

AgriWise integrates three distinct datasets, each serving a segregated role within the system architecture:

```
+-----------------------------------------------------------------------------------+
|                              AGRIWISE DATA ARCHITECTURE                           |
+------------------------------------+----------------------------------------------+
|     1. Primary ML Dataset          |        2. Secondary Contextual Datasets      |
|  (Biophysical Tabular Features)    |       (Regional & Historical Overlays)       |
+------------------------------------+----------------------------------------------+
| * Source: Benchmark Repository     | * Source A: DES / APS Crop Production        |
| * 2,200 samples, 22 crop classes   |   - 12,628 rows, 35 districts, 1997-2014     |
| * Features: N, P, K, temp,         |   - Hectares, metric tonnes, kg/ha yield     |
|   humidity, pH, rainfall           | * Source B: IMD Pune Rainfall Normals        |
| * Role: Trains Random Forest       |   - 36 districts, 1971-2020 LPA normals      |
| * Status: Read-only, frozen        | * Role: Post-inference validation overlays   |
+------------------------------------+----------------------------------------------+
```

### 5.1 Primary Machine Learning Benchmark Dataset
- **Provenance:** Curated biophysical agricultural dataset compiled from public repositories (Kaggle Crop Recommendation Dataset; derived from agricultural research compilations).
- **Size & Coverage:** 2,200 observation instances across 22 agricultural crop commodities (exactly 100 balanced samples per crop).
- **Feature Space:** Seven continuous numerical features:
  1. $N$: Ratio of Nitrogen content in soil ($\text{kg/ha}$), range $[0, 140]$.
  2. $P$: Ratio of Phosphorus content in soil ($\text{kg/ha}$), range $[5, 145]$.
  3. $K$: Ratio of Potassium content in soil ($\text{kg/ha}$), range $[5, 205]$.
  4. $\text{Temperature}$: Ambient temperature ($^\circ\text{C}$), range $[8.83, 43.68]$.
  5. $\text{Humidity}$: Relative humidity ($\%$), range $[14.26, 99.98]$.
  6. $\text{pH}$: Soil pH value, range $[3.50, 9.94]$.
  7. $\text{Rainfall}$: Seasonal rainfall ($\text{mm}$), range $[20.21, 298.56]$.
- **Target Variable:** 22 categorical crop labels: `rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`.
- **Known Limitations:** The dataset is synthetic/semi-synthetic, exhibiting near-zero class overlap and clean clusters that yield artificially high classification scores ($>99\%$). It does not capture multi-year weather volatility, micro-nutrient deficiencies (e.g., Zinc, Boron), soil depth, drainage characteristics, or pest pressures.

### 5.2 Directorate of Economics and Statistics (DES) Historical Crop Production
- **Provenance:** Ministry of Agriculture & Farmers Welfare, Government of India; published via the Open Government Data (OGD) Platform India (`data.gov.in`) and Agricultural Statistics at a Glance (`aps.dac.gov.in`).
- **File Reference:** `data/raw/maharashtra_crop_production_des_raw.csv`.
- **Coverage:** 18 survey years (1997–2014) across 35 rural districts of Maharashtra (Mumbai City excluded).
- **Volume:** 12,628 validated records covering 34 agricultural commodities.
- **Units & Standardization:** 
  - Area in Hectares ($\text{ha}$).
  - Production in Metric Tonnes ($\text{MT}$).
  - *Cotton Unit Conversion:* DES records raw cotton production in bales ($170\text{ kg}$ lint). AgriWise applies the conversion factor:
    $$\text{Production (MT)} = \text{Raw Bales} \times 0.170$$
    Yield is subsequently derived as:
    $$\text{Yield (kg/ha)} = \frac{\text{Production (MT)} \times 1000}{\text{Area (ha)}}$$
- **Data Integrity & Limitations:** 238 records contain unrecorded production ($NaN$), which are audited and omitted from aggregation rather than synthetically imputed. The time period ends in 2014, meaning recent crop acreage shifts (e.g., expanded soybean or drip-irrigated sugarcane) are not fully reflected. Absence of a record in DES historical data does not prove biological impossibility, but serves as empirical evidence of zero commercial cultivation scale during the survey period.

### 5.3 India Meteorological Department (IMD Pune) District Rainfall Climatology
- **Provenance:** Climate Application and User Interface / Hydromet Division, IMD Pune, Government of India.
- **Reference Publications:** *Customized District Rainfall Climatology (1971–2020)* and *District Rainfall Normals (1991–2020 LPA)*.
- **File Reference:** `data/raw/maharashtra_district_rainfall_normals_imd.csv`.
- **Coverage:** All 36 administrative districts of Maharashtra, detailing Annual Long Period Average (LPA) precipitation and seasonal allocations (Southwest Monsoon JJAS, Post-Monsoon OND, Winter JF, and Pre-Monsoon MAM).
- **Role & Departures:** Evaluates user-entered rainfall against long-term normal regimes using official IMD percentage departure thresholds ($\Delta\% = \frac{R - N}{N} \times 100$):
  - Large Excess ($\Delta\% \ge +60\%$), Excess ($+20\% \le \Delta\% \le +59\%$), Normal ($-19\% \le \Delta\% \le +19\%$), Deficient ($-59\% \le \Delta\% \le -20\%$), Large Deficient ($-99\% \le \Delta\% \le -60\%$), Scanty / No Rain ($\Delta\% = -100\%$).

---

## 6. Methodology and System Architecture

The AgriWise system is organized into a modular, five-stage pipeline designed for deterministic execution, CPU-friendly efficiency, and strict decoupling between statistical inference and agronomic rules.

```
                                 AGRIWISE INFERENCE PIPELINE
                                 
   [User Inputs: N, P, K, Temp, Humidity, pH, Rainfall] + [District d, Month m]
                             |
                             v
               +----------------------------+
               | Stage 1: Input Validation  |  --> Range bounds, non-numeric checks
               +----------------------------+
                             |
                             v
               +----------------------------+
               | Stage 2: Hardened ML Engine|  --> MinMaxScaler (train-fitted, immutable)
               |   (Random Forest Model)    |  --> Probability voting across 100 trees
               +----------------------------+  --> Top-K candidate ranking & calibrated bounds
                             |
                             v
               +----------------------------+
               | Stage 3: Explainability    |  --> Exact TreeSHAP feature attributions
               |   (TreeSHAP & Contrastive) |  --> Pairwise differential attribution
               +----------------------------+
                             |
                             v
               +----------------------------+
               | Stage 4: Regional &        |  --> Maharashtra 22-crop matrix check
               |   Seasonal Constraint Layer|  --> Agro-climatic zone & sowing window filter
               +----------------------------+  --> Diagnostic flags (Incompatible/Off-Season)
                             |
                             v
               +----------------------------+
               | Stage 5: Secondary Context |  --> IMD seasonal rainfall departure calculation
               |   (DES & IMD Integration)  |  --> DES historical hectares, tonnage, kg/ha yield
               +----------------------------+
                             |
                             v
               [Transparent Output: Top-5 Candidates, SHAP Waterfalls, Regional Status]
```

### 6.1 Stage 1: Strict Input Validation and Boundary Guards
To prevent out-of-distribution or malicious numerical injections, all inputs pass through strict physical boundary checks defined in `src/core/config.py`:
$$\forall i \in \{1, \dots, 7\}, \quad x_i^{\min} \le x_i \le x_i^{\max}$$
Any value violating physical bounds (e.g., negative rainfall, $\text{pH} > 14$, or non-numeric tokens) raises an explicit `ValidationError` without invoking the inference pipeline.

### 6.2 Stage 2: Hardened Preprocessing and Candidate Ranking
- **Rectified Scaling:** The feature vector $\mathbf{x}$ is scaled using a MinMaxScaler fitted exclusively on the training split:
  $$x_{i,\text{scaled}} = \frac{x_i - \min(X_{i,\text{train}})}{\max(X_{i,\text{train}}) - \min(X_{i,\text{train}})}$$
  The preprocessor instance is immutable: it rejects single-sample `fit()` calls and checks the SHA-256 fingerprint of the serialized scaler artifact upon loading.
- **Top-$K$ Multi-Candidate Engine:** In contrast to legacy single-crop argmax prediction, AgriWise extracts the full probability vector across all 22 classes from the Random Forest ensemble of $M = 100$ trees:
  $$P(y = c \mid \mathbf{x}) = \frac{1}{M} \sum_{m=1}^M P_m(y = c \mid \mathbf{x})$$
  Candidates are ranked in descending order: $c_{(1)}, c_{(2)}, \dots, c_{(K)}$, where $K=5$.
- **Calibrated Confidence Bands:** Softmax probabilities from tree ensembles on synthetic datasets frequently collapse to 1.0 or 0.0. To prevent misleading farmers with uncalibrated "suitability percentages," AgriWise reports calibrated confidence bands:
  - High Confidence: $P(y = c_{(1)} \mid \mathbf{x}) \ge 0.70$ and margin $(P_{(1)} - P_{(2)}) \ge 0.20$
  - Moderate Confidence: $0.40 \le P(y = c_{(1)} \mid \mathbf{x}) < 0.70$
  - Ambiguous / Close Contest: Margin $(P_{(1)} - P_{(2)}) < 0.20$

### 6.3 Stage 3: Explainable AI via Exact TreeSHAP
For tree ensembles, TreeSHAP computes the local additive feature attribution:
$$f_c(\mathbf{x}) = \phi_0(c) + \sum_{i=1}^7 \phi_i(c, \mathbf{x})$$
where $\phi_0(c) = \mathbb{E}[f_c(\mathbf{x})]$ is the base value (expected log-odds or class probability across the background training set), and $\phi_i(c, \mathbf{x})$ is the Shapley attribution value for feature $i$.
- **Positive Drivers:** Features with $\phi_i > 0$ that pushed the model toward predicting crop $c$.
- **Limiting Constraints:** Features with $\phi_i < 0$ that penalized the probability of crop $c$.
- **Pairwise Contrastive Attribution:** When comparing Top-1 crop $A$ against runner-up crop $B$, AgriWise computes the differential attribution vector:
  $$\Delta \phi_i(A, B) = \phi_i(A, \mathbf{x}) - \phi_i(B, \mathbf{x})$$
  identifying the decisive environmental factors that tipped the decision in favor of crop $A$.

### 6.4 Stage 4: What-If and Counterfactual Simulation Studio
The What-If engine enables controlled parameter perturbation. Given baseline vector $\mathbf{x}$ and modified vector $\mathbf{x}' = \mathbf{x} + \boldsymbol{\delta}$, the simulator evaluates:
1. Environmental delta vector: $\boldsymbol{\delta} = \mathbf{x}' - \mathbf{x}$ and percentage change $\frac{\delta_i}{x_i} \times 100\%$.
2. Model probability shifts: $\Delta P(c) = P(c \mid \mathbf{x}') - P(c \mid \mathbf{x})$.
3. Candidate trajectory tracking: Records whether crops gained rank, dropped rank, entered, or fell out of the Top-5 set.

### 6.5 Stage 5: Regional Decision Layer and Secondary Data Integration
When administrative location metadata is provided (e.g., District = Pune, Month = July), the inference output is passed to the regional decision layer:
- **Agronomic Suitability Matrix:** A verified reference matrix mapping all 22 crops against Maharashtra's 9 agro-climatic zones, primary soil orders (Vertisols, Inceptisols, Alfisols), and cropping calendars.
- **Decision Interception Categories:**
  - *Compatible & In-Season:* Crop is agronomically viable and aligns with the sowing window.
  - *Off-Season Warning:* Crop is viable in the district but outside its primary sowing window (e.g., Wheat in July).
  - *Agronomically Incompatible:* Crop violates fundamental thermal or geographical constraints (e.g., Apple, Jute).
  - *Insufficient Evidence / Unverified:* Non-traditional crops lacking established commercial records.
- **Secondary Contextual Overlay:**
  - Ingests IMD rainfall normals for the district and computes departure category.
  - Queries DES historical records for the district and candidate crops, extracting 18-year average cultivated area ($\text{ha}$), total production ($\text{MT}$), and average productivity ($\text{kg/ha}$).
  - Enforces **zero data fabrication**: If a crop has zero survey records in that district, it explicitly returns `"No historical DES records"` rather than an imputed value.

---

## 7. Baseline Machine Learning Model and Evaluation Protocol

### 7.1 Baseline Architecture
The core biophysical predictive engine is a Random Forest Classifier (Breiman, 2001) consisting of $n_{\text{estimators}} = 100$ decision trees with Gini impurity split criterion, maximum tree depth unconstrained, and square root feature subsampling ($m = \lfloor\sqrt{7}\rfloor = 2$).

### 7.2 Train/Test Partitioning and Leakage Safeguards
The 2,200-sample dataset was partitioned into an 80% training set ($N_{\text{train}} = 1,760$) and a 20% held-out testing set ($N_{\text{test}} = 440$) using stratified random sampling with fixed seed (`random_state=42`), preserving exactly 20 instances per class in the test set. 

**Zero Data Leakage Boundary:**
- The `MinMaxScaler` was fitted solely on $X_{\text{train}}$. 
- $X_{\text{test}}$ was transformed using parameters $(\min_{\text{train}}, \max_{\text{train}})$ without updating the scaler.
- Model hyperparameters were evaluated using 5-fold cross-validation exclusively on the training split prior to final evaluation on the held-out test split.

### 7.3 Benchmark Evaluation Results

```
+-----------------------------------------------------------------------------------------+
|                              PIPELINE BENCHMARK EVALUATION                              |
+------------------------------------+-----------+-----------+--------+----------+--------+
| Pipeline Configuration             | Accuracy  | Precision | Recall | F1-Score | Status |
+------------------------------------+-----------+-----------+--------+----------+--------+
| Legacy Pipeline (Corrupted Scaler) |   6.14%   |   5.09%   |  6.14% |   2.50%  | FAILED |
| AgriWise Rectified Pipeline        |  99.77%   |  99.78%   | 99.77% |  99.77%  | PASSED |
+------------------------------------+-----------+-----------+--------+----------+--------+
```

On the held-out test set ($N=440$), the rectified AgriWise pipeline misclassified only 1 out of 440 samples (accuracy = 439/440 = 99.77%). The legacy pipeline, affected by an unconstrained single-sample scaler overwrite bug during prediction, suffered catastrophic performance degradation down to 6.14% accuracy.

---

## 8. Explainability and What-If Analysis Results

### 8.1 TreeSHAP Local Attributions
Evaluating TreeSHAP across representative benchmark queries demonstrates the alignment between model feature attribution and agronomic common sense:
- **Rice Classification ($N=90, P=42, K=43, \text{Temp}=20.87, \text{Humidity}=82.00, \text{pH}=6.50, \text{Rainfall}=202.93$):**
  - High rainfall ($202.93\text{ mm}$) and elevated relative humidity ($82\%$) contributed the largest positive Shapley attributions ($\phi_{\text{rainfall}} = +0.284$, $\phi_{\text{humidity}} = +0.241$), driving the model to predict Rice with $P=0.91$.
- **Chickpea Classification ($N=40, P=68, K=79, \text{Temp}=17.5, \text{Humidity}=16.5, \text{pH}=7.4, \text{Rainfall}=75.0$):**
  - Low relative humidity ($16.5\%$) and cool temperature ($17.5^\circ\text{C}$) acted as decisive positive drivers ($\phi_{\text{humidity}} = +0.312$), while high moisture would have penalized chickpea probability.

### 8.2 Contrastive Explanations
When evaluating a scenario where Rice is recommended ($P=0.88$) with Maize as runner-up ($P=0.08$):
- Pairwise differential analysis $\Delta \phi(\text{Rice}, \text{Maize})$ revealed that **Rainfall** accounted for $+0.412$ of the probability divergence:
  $$\Delta \phi_{\text{rainfall}}(\text{Rice}, \text{Maize}) = \phi_{\text{rainfall}}(\text{Rice}) - \phi_{\text{rainfall}}(\text{Maize}) = +0.284 - (-0.128) = +0.412$$
This directly answers the end-user's question: *"Why Rice instead of Maize?"*—demonstrating that ambient rainfall exceeded the threshold where maize is competitive.

### 8.3 What-If Monsoonal Deficit Simulation
To test the What-If Studio, a baseline high-rainfall scenario (Rice Top-1 at $200\text{ mm}$ rainfall) was subjected to a simulated 50% drought perturbation ($\text{Rainfall} = 100\text{ mm}$):
- **Candidate Trajectory:** Rice probability collapsed from $0.91$ to $0.04$, dropping entirely out of the Top-5.
- **Replacement Candidates:** Drier-climate crops (Maize, Pigeonpeas) rose to the top of the candidate ranking.
- **Agronomic Utility:** This enables an extension officer to demonstrate drought vulnerability interactively, showing the farmer how rainfall deficits alter viable crop rankings before seed purchase.

---

## 9. Regional Decision Constraints and Secondary Dataset Integration

### 9.1 The Need for Regional Interception
To evaluate the discrepancy between biophysical tabular ML predictions and regional agro-climatic realities, we constructed a diagnostic evaluation suite of 72 structured agro-climatic queries covering all 36 districts of Maharashtra across Kharif, Rabi, and Summer seasons. 

In 37 test cases, the biophysical inputs numerically resembled crops that are commercially non-existent or agronomically unviable in Maharashtra (such as Apple or Jute), or were requested outside their seasonal sowing windows (such as Rabi Chickpea requested during the July Kharif monsoon).

### 9.2 Formulation of Evaluation Metrics
To prevent non-standard or misleading interpretations of evaluation terminology, we define all metrics with explicit numerators, denominators, and ground-truth definitions:

1. **Unviable Top-1 Recommendation Rate ($R_{\text{unviable}}$):**
   The proportion of evaluated queries where the raw biophysical ML model recommended an agronomically incompatible or off-season crop as its Top-1 choice:
   $$R_{\text{unviable}} = \frac{N_{\text{unviable Top-1}}}{N_{\text{total evaluated}}} = \frac{37}{68} \approx 54.41\%$$
   *(Note: 4 of the 72 diagnostic queries evaluated boundary-rejection conditions that did not yield a top-1 candidate).*

2. **Conventional False Positive Rate on Unviable Cases ($\text{FPR}_{\text{unviable}}$):**
   Evaluating the raw ML classifier as a binary detector of viable agricultural crops on the 37 ground-truth unviable queries:
   $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{37}{37 + 0} = 100.0\%$$
   The raw biophysical classifier exhibited an $\text{FPR}$ of $100.0\%$ because it had no mechanism to reject unviable crops. When the AgriWise regional decision layer was applied, all 37 unviable top-1 candidates were intercepted ($\text{FP} = 0, \text{TN} = 37$), reducing the $\text{FPR}$ to $0.0\%$.

3. **Incompatible Crop Interception Rate (ICIR):**
   The percentage of agronomically incompatible or off-season crops successfully intercepted by the regional constraint layer:
   $$\text{ICIR} = \frac{\text{Intercepted Unviable Recommendations}}{\text{Total Ground-Truth Unviable Scenarios}} = \frac{37}{37} = 100.0\%$$

4. **In-Season Recommendation Retention (ISRR):**
   The percentage of agronomically viable, in-season crops that passed through the filter without false rejection:
   $$\text{ISRR} = \frac{\text{Retained Valid In-Season Recommendations}}{\text{Total Ground-Truth Valid In-Season Scenarios}} = \frac{31}{31} = 100.0\%$$

```
+-----------------------------------------------------------------------------------------+
|                        REGIONAL DIAGNOSTIC SUITE EVALUATION                             |
+--------------------------------------+--------------+-------------------+---------------+
| Metric                               | Raw ML Model | AgriWise Pipeline | Ground-Truth  |
+--------------------------------------+--------------+-------------------+---------------+
| Unviable Top-1 Recommendation Rate   |    54.41%    |       0.00%       |  37/68 cases  |
| Conventional False Positive Rate     |   100.00%    |       0.00%       |  37/37 cases  |
| Incompatible Crop Interception Rate  |     0.00%    |     100.00%       |  37/37 cases  |
| In-Season Recommendation Retention   |   100.00%    |     100.00%       |  31/31 cases  |
+--------------------------------------+--------------+-------------------+---------------+
```

### 9.3 Secondary Dataset Overlay Findings
Integrating the DES crop production statistics and IMD rainfall normals revealed decisive empirical patterns:
- **Corroboration of Cultivation Hubs:** For queries in Bhandara and Gondia districts, Rice candidates were annotated with DES records showing over $100,000\text{ ha}$ of average annual cultivation, confirming high regional infrastructure and market maturity.
- **Empirical Rejection of Non-Regional Crops:** Inquiries returning Apple or Jute were flagged by the regional filter as `Incompatible`; the DES historical repository confirmed **0 recorded hectares** across all 35 rural districts over 18 years, providing empirical backing for the agronomic rule.
- **Climate Normal Context:** When a user enters $450\text{ mm}$ seasonal rainfall for Ahmednagar (Scarcity Zone, IMD Monsoon Normal = $448.5\text{ mm}$), AgriWise labels the regime as `Normal Departure (+0.3%)`. The same $450\text{ mm}$ input for Ratnagiri (South Konkan Coastal Zone, IMD Monsoon Normal = $3,055.2\text{ mm}$) triggers a `Large Deficient Departure (-85.3%)` warning, alerting the farmer that typical high-water crops (such as lowland rice) are at severe risk unless supported by assured perennial irrigation.

---

## 10. Discussion, Limitations, and Threats to Validity

### 10.1 Scientific Scope of the 99.77% Accuracy Benchmark
A critical threat to validity in agricultural ML publications is conflating benchmark classification accuracy with real-world farming efficacy. The 99.77% accuracy reported here is evaluated strictly on held-out test splits of the synthetic 2,200-sample dataset. This metric demonstrates that the Random Forest successfully learned the non-linear boundaries separating the 22 classes within that specific synthetic feature space. **It does not establish that a recommended crop will achieve commercial yield, resist regional pathogens, or generate economic profit on an actual farm.**

### 10.2 Interpretability vs. Causality in TreeSHAP
SHAP attributions quantify the statistical contribution of each input feature toward the model's mathematical prediction. They reflect **observational feature importance within the learned model**, not biological or physical causality. For example, a positive Shapley value for soil Nitrogen indicates that the model used high Nitrogen to classify the crop as Rice; it does not constitute agronomic proof that applying additional Nitrogen fertilizer will increase rice yield.

### 10.3 Exploratory Nature of What-If Simulations
The What-If Studio operates by feeding modified feature vectors into the static Random Forest model. These trajectories represent **model sensitivity simulations**, not real-world agronomic forecasts. In a physical field, altering rainfall alters soil moisture dynamics, microbial activity, and nutrient leaching—interactions not captured in static tabular models.

### 10.4 Limitations of Historical DES Production Data
1. **Temporal Horizon:** The DES dataset spans 1997–2014. Changes in cropping patterns over the past decade (e.g., expansion of Bt cotton, drip-irrigated pomegranate, or cluster polyhouses) are not fully captured.
2. **Epistemic Distinction:** The absence of historical production records in DES data proves that a crop was not cultivated at a surveyed commercial scale; **it does not constitute biological proof that the crop cannot grow**.
3. **Data Quality:** 238 entries contain unrecorded production data. These gaps were handled safely via omission rather than synthetic imputation.

### 10.5 Nature of the Diagnostic Evaluation Suite
The reported 100% Interception Rate (ICIR) and 100% Retention Rate (ISRR) reflect performance on an internally curated diagnostic suite of 72 structured test queries. **These results demonstrate internal verification of the software rule engine, not independent external agronomic validation.** Field-level validation would require multi-year randomized trials across disparate agro-climatic zones with local agricultural extension officers.

---

## 11. Conclusion and Future Work

This paper presented **AgriWise**, an explainable, climate-aware, and region-specific agricultural decision support system designed to upgrade conventional black-box crop classification models into scientifically grounded advisory tools. By rectifying silent data leakage bugs, introducing Top-$K$ candidate rankings with calibrated confidence bands, embedding exact TreeSHAP feature attributions, developing an interactive What-If sensitivity studio, and enforcing regional agro-climatic constraints backed by historical DES and IMD datasets, AgriWise bridges the gap between pure computer science benchmarks and real-world agricultural constraints.

In benchmark tests, AgriWise restored held-out predictive accuracy to 99.77% while providing transparent feature explanations. In regional diagnostic evaluations across Maharashtra, the system intercepted 100% of the biophysically plausible but agronomically unviable recommendations generated by the raw ML classifier, contextualizing every viable candidate with 18 years of historical cultivation scale and long-period rainfall departures.

### Future Work
1. **Dynamic Weather API Integration:** Connecting the climatology module to live agrometeorological APIs (e.g., IMD Mausam / Gramin Krishi Mausam Sewa) for real-time weather alerts.
2. **Economic Gross Margin Analysis:** Integrating Agricultural Produce Market Committee (APMC) mandi price series to balance biophysical suitability with expected market returns.
3. **Soil Spatial Mapping:** Incorporating Soil Health Card geospatial layers to infer soil series and drainage characteristics automatically from GPS coordinates.
4. **Field Extension Trials:** Partnering with regional Krishi Vigyan Kendras (KVKs) to evaluate user satisfaction and advisory utility among practicing farmers.

---

## 12. References

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
2. Directorate of Economics and Statistics (DES). (2015). *District-wise Season-wise Crop Production Statistics (1997–2014)*. Ministry of Agriculture and Farmers Welfare, Government of India. https://data.gov.in / https://aps.dac.gov.in
3. India Meteorological Department (IMD). (2020). *Customized District Rainfall Climatology (1971–2020)*. Hydromet Division, Climate Diagnostics Group, IMD Pune.
4. India Meteorological Department (IMD). (2021). *District Rainfall Normals (1991–2020 LPA)*. Climate Application and User Interface, IMD Pune.
5. Kumar, R., Singh, M. P., Kumar, P., & Singh, J. P. (2020). Crop selection and recommendation using machine learning techniques. In *2020 International Conference on Computing, Communication and Automation (ICCCA)* (pp. 381–386). IEEE.
6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems (NeurIPS 2017)* (Vol. 30, pp. 4765–4774).
7. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
8. Mahatma Phule Krishi Vidyapeeth (MPKV). (2020). *Agro-Climatic Zones of Maharashtra and Crop Production Technologies*. Directorate of Extension Education, Rahuri, Maharashtra.
9. Pudumalar, S., Ramanujam, E., Rajashree, R. H., Kavya, C., Kiruthika, T., & Prasanth, J. (2017). Crop recommendation system for precision agriculture. In *2016 Eighth International Conference on Advanced Computing (ICoAC)* (pp. 32–36). IEEE.
10. Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. *Harvard Journal of Law & Technology*, 31, 841–887.
11. Wolanin, A., Camps-Valls, G., Gómez-Chova, L., Mateo-García, G., van der Tol, C., & Guanter, L. (2020). Estimating crop yields from multi-temporal satellite data using deep learning and explainable AI. *Remote Sensing of Environment*, 249, 111988.
