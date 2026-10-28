# AgriWise: Mathematical Formulation, Verification & Evaluation Methodology

**Document Version:** 1.0 (Academic Verification Document)  
**Author:** Shravani Pawar  
**System:** AgriWise Decision Support System  
**Date:** October 2026  

---

## 1. Top-5 Model Confidence Formulation & Scientific Justification

### 1.1 Mathematical Derivation of Model Output Probabilities
In our Random Forest ensemble architecture comprising $B = 100$ independent decision trees, each tree $t_b$ ($b \in \{1, \dots, B\}$) partitions the normalized 7-dimensional feature space into terminal leaves.

For an input vector $X = [N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}] \in \mathbb{R}^7$, normalized via the training-fitted scaler $\phi(X) = X_{\text{norm}}$, each individual tree $t_b$ outputs an empirical class distribution at its terminal leaf:
$$p_b(c \mid X_{\text{norm}}) = \frac{n_{b, c}}{\sum_{k=1}^K n_{b, k}}$$
Where:
- $n_{b, c}$ is the count of training samples of crop class $c$ in the leaf reached by $X_{\text{norm}}$ in tree $b$.
- $K = 22$ is the total number of candidate crop classes.

The ensemble probability vector $P(C = c \mid X)$ is computed as the arithmetic mean across all $B = 100$ trees:
$$P(C = c \mid X) = \frac{1}{B} \sum_{b=1}^B p_b(c \mid X_{\text{norm}})$$

The candidate crops are then sorted in descending order of their ensemble probability:
$$\text{Rank}(c_1) \ge \text{Rank}(c_2) \ge \dots \ge \text{Rank}(c_K) \iff P(c_1 \mid X) \ge P(c_2 \mid X) \ge \dots \ge P(c_K \mid X)$$
The Top-$5$ candidates are selected as the subset $\mathcal{C}_{\text{top5}} = \{c_1, c_2, c_3, c_4, c_5\}$.

### 1.2 Mathematical Derivation of Confidence Bands
The percentage score displayed to the user is:
$$\text{Score}(c) = P(C = c \mid X) \times 100\%$$

Because the target space contains $K = 22$ mutually exclusive classes, a uniform random guess has an expected probability of $\frac{1}{22} \approx 4.55\%$. Confidence bands are therefore derived based on multi-class threshold theory:

| Band | Mathematical Condition | Agronomic & Statistical Interpretation |
|---|---|---|
| **High Confidence** | $\text{Score}(c) \ge 50.0\%$ | **Dominant Envelope Match:** Over half the ensemble trees unequivocally place the input inside the tight historical cluster for crop $c$. |
| **Moderate Confidence** | $15.0\% \le \text{Score}(c) < 50.0\%$ | **Viable Alternative Match:** Significantly above random expectation ($> 3\times$ baseline). Indicates partial overlap across several soil and weather dimensions (e.g., legumes sharing similar pH and nitrogen limits). |
| **Low Confidence** | $\text{Score}(c) < 15.0\%$ | **Marginal Candidate:** Weak alignment; input sits on the periphery of the crop's historical envelope. Included in Top-5 purely as a comparative baseline. |

### 1.3 Scientific Guardrails Against Misleading Claims
To ensure academic honesty and avoid misleading agricultural practitioners:
1. **Model Confidence $\neq$ Real-World Success:** In `CropPredictor` and the web interface, the score is explicitly labeled as **"Model Confidence Score"**, not "Crop Suitability Percentage" or "Probability of Successful Harvest".
2. **Envelope Proximity vs. Biological Growth:** The score indicates how closely the 7 inputs match the statistical distribution of the training data. It does not measure biological germination rate, disease resistance, or market return.
3. **Mandatory Scientific Notice:** Every prediction output includes the disclaimer:
   > *"Notice: Model confidence scores indicate statistical proximity to historical benchmark training envelopes. They must not be conflated with real-world agricultural success, yield, or profitability. Field viability depends on seasonal sowing calendars, regional water availability, soil depth, and farm management practices."*

---

## 2. Confirmation of Evaluation Methodology & Leakage Prevention

### 2.1 Dataset Partitioning & Stratification
- **Source Dataset:** `Crop_recommendation.csv` ($N = 2,200$ total records, 22 balanced classes, exactly 100 samples per crop).
- **Split Ratio:** Strict 80% Train ($N_{\text{train}} = 1,760$), 20% Held-Out Test ($N_{\text{test}} = 440$).
- **Stratification:** Stratified sampling (`stratify=y`) guarantees exactly $20$ held-out test samples for every single one of the 22 crop classes.
- **Reproducibility Seed:** Fixed pseudo-random number generator seed (`random_state=42`) across all split, training, and cross-validation routines.

### 2.2 Proof of Zero Data Leakage
Data leakage was rigorously prevented through structural isolation:
1. **Strict Fit Boundary:**  
   $$\mu_{\text{scale}}, \sigma_{\text{scale}} = \text{Fit}(X_{\text{train}})$$
   $$\text{Scaler Parameters Learned}: \quad \text{min} \in X_{\text{train}}, \quad \text{max} \in X_{\text{train}}$$
2. **Transform Only on Test Set:**  
   $$X_{\text{test, scaled}} = \text{Transform}(X_{\text{test}})$$
   No test samples were ever observed during the computation of minimum or maximum bounds.
3. **Single-Sample Re-Fit Guard:**  
   `AgriDataPreprocessor.fit()` explicitly checks `len(X) > 1` and raises `ValidationError` if called on single samples, preventing the catastrophic overwrite bug of the legacy notebook.

### 2.3 Side-by-Side Evaluation on the Identical Held-Out Test Split
To ensure mathematical comparability, both the legacy pipeline and the rectified production pipeline were evaluated on the **exact same 440 held-out test samples**:

| Evaluation Criterion | Legacy Pipeline (`legacy/`) | Rectified Production Pipeline (`src/ml/`) |
|---|---|---|
| **Test Set Identity** | Identical 440 samples (`random_state=42`) | Identical 440 samples (`random_state=42`) |
| **Scaler Used** | Corrupted `minmaxscaler.pkl` ($\text{min}=\text{max}$) + `standscaler.pkl` | Clean `scaler.pkl` fitted on $X_{\text{train}}$ |
| **Correctly Classified Samples** | **27 / 440** | **439 / 440** |
| **Misclassified Samples** | **413 / 440** | **1 / 440** |
| **Empirical Accuracy** | **6.14%** | **99.77%** |
| **Macro F1-Score** | **2.50%** | **99.77%** |

### 2.4 Critical Academic Limitations of the 99.77% Result
In an academic thesis or research paper, reporting 99.77% accuracy without discussing limitations is considered unscientific. The limitations of this result are explicitly documented:
1. **Synthetic Nature of Kaggle Benchmark:** The benchmark dataset has artificially clean separations with zero noise and no missing values. In real field soils, overlapping nutrient profiles between crops reduce real-world accuracy.
2. **Lack of Spatio-Temporal Dimensions:** The 2,200 samples lack geographic coordinates, rainfall seasonality (e.g. onset vs. cessation of monsoon), and soil depth.
3. **Closed-Set Assumption:** The classifier assumes every farmer must grow one of the 22 crops, ignoring regionally dominant cash crops (e.g. Sugarcane, Soybean, Onion, Turmeric).
4. **Generalization Gap:** Achieving 99.77% on this benchmark does **not** imply 99.77% field success. This gap motivates the necessity of **Phase 3 (Explainability & What-If Analysis)** and **Phase 4 (Regional Calendars)**.
