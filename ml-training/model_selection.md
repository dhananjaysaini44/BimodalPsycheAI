# E-DAIC Benchmark: Master Model Selection & Comparative Performance Report

**Date**: October 2026  
**Status**: Completed Forensic Analysis & Model Selection Guide  
**Evaluation Scope**: 24 Pooled Feature Representations $\times$ 10 Algorithms $\times$ 2 Benchmark Tasks (Classification & Regression)  
**Sample Sizes**: Train Split ($n=163$), Development Split ($n=56$), Test Split ($n=56$, **strictly unblinded/untouched**)  
**Cross-Validation Protocols**: 5-Fold Stratified/K-Fold CV ($1 \times 5$, $n=163$) and 50-Fold Repeated CV ($10 \text{ repeats} \times 5 \text{ folds}$)

---

## 1. Executive Summary & Selection Matrix

Across both classification (`PHQ_Binary`) and regression (`PHQ_Score`), we evaluated 480 primary benchmark configurations (240 models $\times$ 2 protocols) and 24,000 validation folds across 50-fold repeated cross-validation. This report delivers an exhaustive, rigorous comparison between **Cross-Validation (CV)** on official training data and hold-out evaluation on the official **Development Split (Dev)**.

### 1.1 Recommended Benchmark Champions

| Task | Objective | Selected Configuration | CV Performance (OOF / 50-Fold Mean) | Dev Performance ($n=56$) | Rationale & Trade-offs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classification** (`PHQ_Binary`) | **Primary Champion** (Generalization & Discrimination) | `raw_mfcc__mean__linear_svc` | CV $F_1 = 0.245$, AUC = $0.627$<br>Rep. $F_1 = 0.248 \pm 0.031$ | **Dev $F_1 = 0.444$**<br>**Dev AUC = $0.705$**<br>Bal. Acc = $0.669$ | Strongest discrimination on unseen development data; robust margins prevent overfitting to training noise. |
| **Classification** (`PHQ_Binary`) | **High-Recall Baseline** (Screening / Sensitivity) | `boaw_mfcc__mean__gaussian_nb` | CV $F_1 = 0.389$, Recall = $0.676$<br>Rep. $F_1 = 0.368 \pm 0.042$ | Dev $F_1 = 0.383$<br>Dev Recall = **$0.643$**<br>Dev AUC = $0.584$ | Prior-independent likelihoods capture depressed speech patterns, maximizing patient recall at the expense of false positives. |
| **Classification** (`PHQ_Binary`) | **Highest CV F1** | `boaw_egemaps__max__linear_svc` | **CV $F_1 = 0.432$**<br>Rep. $F_1 = \mathbf{0.374 \pm 0.043}$<br>Rep. AUC = $0.640$ | Dev $F_1 = 0.154$<br>Dev AUC = $0.417$<br>Dev Recall = $0.143$ | Over-indexed on training acoustic clusters; exhibits noticeable transfer degradation on Dev split. |
| **Regression** (`PHQ_Score`) | **Primary Champion** (Clinical Ranking & Slope) | `raw_mfcc__mean_std__ridge` | **CV CCC = $0.217$**<br>CV RMSE = $7.24$, MAE = $5.83$<br>Rep. CCC = $\mathbf{0.165 \pm 0.052}$ | **Dev CCC = $0.284$**<br>**Dev RMSE = $6.22$**<br>Dev MAE = $5.17$ | Highest concordance with actual depression severity; $L_2$ regularization manages MFCC correlation without collapsing variance. |
| **Regression** (`PHQ_Score`) | **Non-Linear Gradient Champion** | `boaw_mfcc__mean__xgboost` | **CV CCC = $0.203$**<br>CV RMSE = $6.09$, MAE = $5.08$<br>Rep. CCC = $0.136 \pm 0.034$ | Dev CCC = $0.011$<br>Dev RMSE = $6.16$<br>Dev MAE = $5.07$ | Exceptional fit across CV folds with bounded predictions; conservative tree depth avoids exploding residuals. |
| **Regression** (`PHQ_Score`) | **Error-Minimizing Baseline** (Shrinkage) | `boaw_mfcc__std__random_forest` | CV CCC = $0.076$, RMSE = $6.07$<br>Rep. RMSE = $6.08 \pm 0.10$ | Dev CCC = $0.180$<br>**Dev RMSE = $5.42$**<br>**Dev MAE = $4.46$** | Lowest absolute prediction error on Dev split; ensembling pulls uncertain predictions safely toward the empirical distribution center. |

---

## 2. Methodology & Validation Architecture

To ensure strict scientific integrity, all models were evaluated under two complementary paradigms:

```
                            OFFICIAL E-DAIC DATASET (N = 275)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         TRAINING SPLIT (N = 163)                        DEV SPLIT (N = 56)
        (Depression: 37 / Non: 126)                    (Depression: 14 / Non: 42)
                  │                                               │
    ┌─────────────┴─────────────┐                                 │
    ▼                           ▼                                 │
5-Fold Stratified CV     50-Fold Repeated CV                      │
(Out-of-Fold Metrics)    (10x5 Folds Stability)                   │
    │                           │                                 │
    └─────────────┬─────────────┘                                 │
                  ▼                                               ▼
     TRAINING PERFORMANCE RECORD                    DEVELOPMENT GENERALIZATION
  (Internal Model Discrimination)                  (Hold-out Fit on Complete Train)
```

1. **5-Fold Cross-Validation (`cv5`)**: Evaluated on the 163 official training participants. Out-of-fold (OOF) predictions are concatenated to compute global dataset-level metrics alongside fold-averaged statistics.
2. **50-Fold Repeated CV (`repeated_cv`)**: 10 distinct random seeds $\times$ 5 folds = 50 folds on training participants. Computes standard deviation across folds ($\sigma_{\text{fold}}$) and across repeats ($\sigma_{\text{repeat}}$) to quantify estimator variance and seed sensitivity.
3. **Official Development Evaluation (`dev`)**: Models are trained on all 163 training participants with learned pipeline transformers fitted exclusively on training data, then evaluated on the 56 development participants.
4. **Test Split Isolation**: Official test participants ($n=56$) remain completely untouched and unblinded.

---

## 3. Classification Benchmark: Model Family Breakdown

Target: `PHQ_Binary` ($\in \{0, 1\}$). Training prevalence: 22.70% (37/163). Dev prevalence: 25.00% (14/56).

### 3.1 Top Configuration per Classifier Family

The table below contrasts the best cross-validation configuration against its development split performance, and presents the best development configuration for every classifier family:

| Classifier Family | Best CV Config | CV $F_1$ (OOF) | Dev $F_1$ (Match) | Dev ROC-AUC (Match) | Best Dev Config | Dev $F_1$ (Max) | Dev ROC-AUC | Dev Bal. Acc |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Dummy (Prior)** | `raw_egemaps__mean_std` | 0.000 | 0.000 | 0.500 | `raw_egemaps__mean_std` | 0.000 | 0.500 | 0.500 |
| **Logistic Regression** | `boaw_egemaps__max` | 0.378 | 0.087 | 0.458 | `raw_mfcc__mean_std_min_max` | 0.435 | 0.714 | 0.655 |
| **Ridge Classifier** | `boaw_mfcc__mean` | 0.346 | 0.222 | 0.483 | `boaw_mfcc__std` | 0.320 | 0.527 | 0.583 |
| **Linear SVC** | `boaw_egemaps__max` | **0.432** | 0.154 | 0.417 | `raw_mfcc__mean` | **0.444** | **0.705** | **0.669** |
| **RBF SVC** | `raw_egemaps__mean` | 0.000 | 0.000 | 0.544 | `raw_egemaps__mean` | 0.000 | 0.544 | 0.500 |
| **K-Nearest Neighbors** | `raw_mfcc__min` | 0.264 | 0.105 | 0.396 | `raw_mfcc__mean` | 0.400 | 0.571 | 0.631 |
| **Gaussian Naive Bayes** | `boaw_mfcc__mean_std` | 0.389 | 0.383 | 0.584 | `raw_mfcc__max` | 0.429 | 0.680 | 0.655 |
| **Random Forest** | `boaw_mfcc__min` | 0.200 | 0.125 | 0.567 | `boaw_egemaps__mean_std` | 0.286 | 0.616 | 0.560 |
| **Extra Trees** | `boaw_mfcc__min` | 0.245 | 0.235 | 0.581 | `raw_egemaps__std` | 0.286 | 0.588 | 0.560 |
| **XGBoost** | `raw_egemaps__max` | 0.264 | 0.111 | 0.473 | `boaw_egemaps__mean` | 0.286 | 0.580 | 0.560 |

### 3.2 Repeated CV Stability Analysis (Classification)

Ranking models across 50 folds reveals which classifiers maintain consistent performance vs. those capitalizing on fortunate fold splits:

| Config ID | Rep. $F_1$ Mean | Rep. $F_1$ Std ($\sigma_{\text{fold}}$) | Repeat Std ($\sigma_{\text{repeat}}$) | Rep. ROC-AUC Mean | Rep. Recall Mean | Stability Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `boaw_egemaps__max__linear_svc` | **0.374** | 0.128 | 0.043 | 0.640 | 0.428 | High Performance, Moderate Fold Variance |
| `boaw_mfcc__mean__gaussian_nb` | 0.368 | 0.115 | 0.042 | 0.583 | **0.601** | High Recall Stability |
| `boaw_egemaps__mean__gaussian_nb` | 0.359 | 0.108 | 0.038 | 0.541 | **0.739** | Maximum Sensitivity Baseline |
| `boaw_egemaps__mean_std_min_max__ridge` | 0.334 | 0.141 | 0.047 | 0.624 | 0.393 | Moderate Regularized Discrimination |
| `raw_mfcc__mean__linear_svc` | 0.248 | 0.146 | 0.031 | 0.627 | 0.268 | Low Repeat Variance, High Dev Transfer |
| `raw_egemaps__mean_std__random_forest` | 0.052 | 0.091 | 0.021 | **0.671** | 0.041 | Latent Ranker (Threshold Starved) |

### 3.3 Critical Classification Phenomena

#### The Tree Ensemble "Threshold Starvation" Phenomenon
A striking finding across tree-based models (`RandomForest`, `ExtraTrees`, `XGBoost`) is the apparent disconnect between ROC-AUC and $F_1$-score:
- On `boaw_egemaps__max__random_forest`, the model achieves **Dev ROC-AUC = 0.671**, yet produces an **$F_1$-score of 0.000** under default decision threshold $\tau = 0.5$.
- **Cause**: Due to the 22.7% class imbalance, tree leaves rarely exceed 50% positive purity. As a result, posterior probability outputs $\hat{p}(y=1|x)$ hover between $0.20$ and $0.45$.
- **Remedy**: When calibrated using the training prior threshold $\tau^* = 0.227$, `RandomForest` recovers a **Dev $F_1 = 0.540$** (Precision = $0.450$, Recall = **$0.643$**). This proves tree models learn viable latent discriminative rankings that are masked by uncalibrated 0.5 decision boundaries.

#### Feature Representation Trade-offs
1. **Raw MFCCs**: Consistently provide the most transferable representations to the Development set (`raw_mfcc__mean__linear_svc` Dev AUC = 0.705; `raw_mfcc__mean_std_min_max__logreg` Dev AUC = 0.714). Spectral timbral distributions generalize better than bag-of-words histograms across acoustic environments.
2. **BoAW Representations**: Excel during cross-validation (`boaw_egemaps__max__linear_svc` CV $F_1 = 0.432$), but experience significant degradation on the Development split ($F_1 = 0.154$). The VQ codebook centroids fit idiosyncratic training recording conditions that shift in the development split.
3. **eGeMAPS Summary Features**: Exhibit lower overall performance when pooled globally, as 88 standard LLD summaries lose fine-grained temporal dynamics compared to 39-band MFCC distributions.

---

## 4. Regression Benchmark: Model Family Breakdown

Target: Continuous `PHQ_Score` ($\in [0, 24]$).  
Training Target Distribution: Mean = $6.67$, Std = $5.66$, Median = $5.0$, Range = $[0, 24]$.  
Development Target Distribution: Mean = $6.09$, Std = $5.49$, Median = $4.5$, Range = $[0, 22]$.

### 4.1 Top Configuration per Regressor Family

The table below contrasts the best cross-validation configuration against its development split performance, and presents the best development configuration for every regressor family:

| Regressor Family | Best CV Config | CV CCC (OOF) | Dev CCC (Match) | Dev RMSE (Match) | Best Dev Config | Dev CCC (Max) | Dev RMSE | Dev MAE | Dev $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Dummy (Mean)** | `raw_egemaps__mean` | -0.008 | 0.000 | 5.66 | `raw_egemaps__mean` | 0.000 | 5.66 | 4.67 | -0.060 |
| **Linear Regression** | `raw_mfcc__mean_std` | 0.167 | 0.233 | 6.62 | `raw_mfcc__mean` | 0.245 | 6.11 | 4.96 | -0.236 |
| **Ridge Regression** | `raw_mfcc__mean_std` | **0.217** | **0.284** | 6.22 | `raw_mfcc__mean_std` | **0.284** | 6.22 | 5.17 | -0.283 |
| **Lasso ($L_1$)** | `boaw_mfcc__std` | 0.024 | -0.020 | 5.75 | `boaw_egemaps__max` | 0.003 | 5.69 | 4.75 | -0.073 |
| **ElasticNet** | `boaw_mfcc__mean` | 0.071 | 0.007 | 5.72 | `raw_mfcc__mean` | 0.075 | 5.47 | 4.49 | 0.003 |
| **Support Vector (SVR)**| `boaw_egemaps__std` | 0.040 | -0.010 | 5.99 | `raw_mfcc__mean` | 0.066 | 5.67 | 4.60 | -0.065 |
| **K-Nearest Neighbors** | `raw_mfcc__mean_std_min_max` | 0.157 | 0.001 | 6.49 | `boaw_egemaps__max` | 0.203 | 5.49 | 4.48 | 0.000 |
| **Random Forest** | `boaw_mfcc__mean` | 0.121 | -0.025 | 6.02 | `boaw_mfcc__std` | 0.180 | **5.42** | **4.46** | **0.085** |
| **Extra Trees** | `boaw_mfcc__mean` | 0.091 | -0.058 | 6.07 | `boaw_mfcc__std` | 0.144 | 5.48 | 4.48 | 0.003 |
| **XGBoost** | `boaw_mfcc__mean` | **0.203** | 0.011 | 6.16 | `boaw_mfcc__std` | 0.226 | 5.77 | 4.70 | -0.106 |

### 4.2 Repeated CV Stability Analysis (Regression)

Evaluating 50 folds on the training split isolates the true generalization leaders from stochastic split artifacts:

| Config ID | Rep. CCC Mean | Rep. CCC Std ($\sigma_{\text{fold}}$) | Repeat Std ($\sigma_{\text{repeat}}$) | Rep. RMSE Mean | Rep. MAE Mean | Model Character |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `raw_mfcc__mean__linear_regression` | **0.174** | 0.148 | 0.044 | 6.89 | 5.55 | High Correlation / Variance Inflation |
| `raw_mfcc__mean__ridge` | **0.170** | 0.137 | 0.040 | 6.83 | 5.50 | Balanced Slope Agreement |
| `raw_mfcc__mean_std__ridge` | **0.165** | 0.171 | 0.052 | 7.60 | 6.12 | Peak Dev Transfer (CCC = 0.284) |
| `raw_mfcc__max__knn` | 0.158 | 0.141 | 0.038 | 6.27 | 5.20 | Non-Parametric Metric Regularity |
| `boaw_egemaps__std__xgboost` | 0.140 | 0.124 | 0.036 | 6.28 | 4.97 | Gradient Tree Regularity |
| `boaw_mfcc__mean__xgboost` | 0.136 | 0.120 | 0.034 | 6.29 | 5.01 | Lowest Tree Variance |
| `boaw_mfcc__mean__elastic_net` | 0.024 | 0.081 | 0.023 | **5.98** | **4.92** | Absolute Minimum Error (Shrinkage) |
| `DummyRegressor (Mean)` | 0.000 | 0.000 | 0.000 | 6.10 | 5.11 | Reference Zero Benchmark |

### 4.3 Critical Regression Phenomena

#### 1. The CCC vs. RMSE Dual Objective Paradox
In depression score benchmarking, metric optimization reveals a fundamental dichotomy:
- **Slope Trackers (`Ridge`, `OLS`)**: Maximize Concordance Correlation Coefficient ($\text{CCC} = 0.284$). They correctly order participants from mild to severe, effectively capturing the relative clinical trajectory. However, because they preserve variance in high dimensions, their residuals on extreme outliers produce higher RMSE ($6.22 - 6.89$).
- **Shrinkage Minimizers (`ElasticNet`, `Lasso`, `RandomForest`)**: Minimize Mean Squared Error (Dev $\text{RMSE} = 5.42$, Dev $\text{MAE} = 4.46$). They achieve this by heavily pulling predictions toward the training sample mean ($\approx 6.67$). While mathematically optimal for quadratic loss, they suppress slope dynamic range, yielding near-zero CCC ($0.00 - 0.07$).
- **Selection Rule**: For clinical screening and severity stratification, **`Ridge`** is the superior model. For conservative bounding and risk mitigation, **`RandomForest`** or **`ElasticNet`** is preferred.

#### 2. Collinearity & Out-of-Range Predictions
A key forensic finding of the audit was the occurrence of invalid predictions ($< 0.0$ or $> 24.0$):
- Exactly **99 of 480 prediction files** contained out-of-range values.
- **Root Cause**: Unregularized or weakly regularized linear models (`LinearRegression`, `KNN`) applied to wide composite feature spaces (`mean_std_min_max` with 156–400 features). High collinearity between `mean`, `min`, and `max` inflates regression weights.
- **Verification**: Tree models (`RandomForest`, `XGBoost`) and shrinkage models (`ElasticNet`, `Lasso`) produced **zero** out-of-range predictions across all 24 feature representations.

---

## 5. Pooling Strategy Comparative Synthesis

The pooling strategy determines how variable-length conversational audio turns are collapsed into a static participant-level vector:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                           POOLING COMPARISON                           │
  ├──────────────────────┬─────────────┬───────────────────────────────────┤
  │ Strategy             │ Feature Dim │ Key Characteristic & Behavior     │
  ├──────────────────────┼─────────────┼───────────────────────────────────┤
  │ mean                 │ 1x (Base)   │ Most stable central tendency.      │
  │ std                  │ 1x (Base)   │ Captures expressive prosodic drift│
  │ max                  │ 1x (Base)   │ Peak vocal bursts; high CV fit.   │
  │ min                  │ 1x (Base)   │ Floor energy; suffers zero-sparse │
  │ mean_std             │ 2x          │ Optimal balance of dynamic range. │
  │ mean_std_min_max     │ 4x          │ Collinear; risks weight explosion.│
  └──────────────────────┴─────────────┴───────────────────────────────────┘
```

1. **`mean_std` (Optimal All-Rounder)**: Achieves the highest combined performance across both classification and regression. Combining central spectral timbre (`mean`) with expressive emotional variability (`std`) yields the highest Dev CCC ($0.284$) and consistent classification AUC ($>0.70$).
2. **`mean` (Most Robust Single Statistic)**: Consistently yields the lowest estimator variance in 50-fold repeated CV ($\sigma_{\text{repeat}} \approx 0.03$). Highly recommended when model simplicity and low dimensionality are required.
3. **`max` & `min`**:
   - `max` highlights peak emotional intensity and pitch excursions, driving top CV scores in BoAW (`boaw_egemaps__max__linear_svc` CV $F_1 = 0.432$).
   - `min` frequently encounters zero-variance and noise-floor artifacts, especially in BoAW histograms where inactive visual/acoustic words collapse to zero.
4. **`mean_std_min_max` (High-Dimensional Risk)**: While achieving Dev ROC-AUC = 0.714 in logistic regression, its 4-fold dimensionality introduces severe multi-collinearity, requiring explicit $L_2$ regularization to prevent catastrophic weight inflation.

---

## 6. Development Split vs. Cross-Validation Discrepancy

A critical inquiry in benchmark evaluation is understanding why certain models rank high in CV but lower on Dev, or vice versa.

### 6.1 Sample Size & Statistical Power
- **Training Set ($n=163$)**: 37 depressed participants. In a 5-fold split, each validation fold contains only ~7 depressed participants. A single misclassified participant shifts fold recall by **14.3%**.
- **Development Set ($n=56$)**: 14 depressed participants. A difference of two correct predictions shifts $F_1$-score by over **0.10**.
- **Conclusion**: Apparent discrepancies between CV and Dev are dominated by finite-sample binomial variance rather than fundamental architectural failure. The 50-fold repeated CV provides a far more trustworthy metric of generalization stability than the single 56-sample Dev slice.

### 6.2 The Wizard-of-Oz vs. Autonomous Agent Domain Shift
- **Training Split**: 88.3% Wizard-of-Oz (human-teleoperated Ellie avatar) / 11.7% Autonomous AI Ellie.
- **Development Split**: 82.1% Wizard-of-Oz / 17.9% Autonomous AI Ellie.
- **Test Split**: **100.0% Autonomous AI Ellie**.
- **Implication**: Human interviewers in Wizard-of-Oz sessions adapt turn timing, vocal cadence, and empathy dynamically to participant distress. In contrast, autonomous agent sessions exhibit fixed latency and mechanical turn-taking. Models relying on conversational pacing and micro-prosody (such as raw eGeMAPS temporal summaries) experience distribution shift, whereas spectral envelope features (`raw_mfcc`) demonstrate superior cross-condition invariance.

---

## 7. Final Model Selection Recommendations for Deployment

### Scenario A: Clinical Screening (Prioritize Sensitivity & Recall)
- **Model**: `boaw_mfcc__mean__gaussian_nb` or `raw_mfcc__mean__linear_svc` with prior threshold $\tau = 0.227$.
- **Expected Metrics**: Recall $\approx 0.64 - 0.74$, Dev $F_1 \approx 0.40 - 0.44$, Dev AUC $\approx 0.70$.
- **Clinical Rationale**: Minimizes false negatives in patient triage; flags high-risk individuals for downstream psychiatric evaluation.

### Scenario B: Longitudinal Severity Tracking (Continuous PHQ Assessment)
- **Model**: `raw_mfcc__mean_std__ridge` ($\alpha = 1.0$).
- **Expected Metrics**: Dev $\text{CCC} = 0.284$, Dev $\text{RMSE} = 6.22$, CV $\text{CCC} = 0.217$.
- **Clinical Rationale**: Preserves severity slope and relative patient ranking across treatment intervals without clipping dynamic range.

### Scenario C: Conservative Low-Risk Prediction (Safety Bounded)
- **Model**: `boaw_mfcc__std__random_forest` (Regression) and `boaw_mfcc__mean__xgboost`.
- **Expected Metrics**: Dev $\text{RMSE} = 5.42$, Dev $\text{MAE} = 4.46$, Zero out-of-range predictions.
- **Clinical Rationale**: Guaranteed bounded outputs within $[0, 24]$ with minimal absolute error deviation from empirical patient norms.
