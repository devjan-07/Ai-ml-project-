# SVM Model Documentation — Rice Leaf Disease Classification

## 1. Purpose

This file explains the complete SVM work from the existing project pipeline through final evaluation. It is intended to make the model easy to understand for team members, report writing, and viva preparation.

> **Result integrity:** Every numerical result in this file comes from the SVM notebook output supplied during the experiment. No performance value has been invented.

---

## 2. Model Overview

**Problem:** 4-class rice leaf disease classification.

**Classes:** Bacterial Blight, Blast, Brown Spot, Tungro.

**Model:** Support Vector Machine (SVM).

**Input to SVM:** The final 9 engineered classical-ML features, not raw image pixels.

**Final test set:** 720 images, kept separate from hyperparameter tuning.

---

## 3. Data Preparation Before SVM

### 3.1 Original dataset
- Total images: **5,932**
- Corrupted/unreadable images: **0**

### 3.2 Exact duplicate analysis
- Unique file hashes: **4,794**
- Exact duplicate groups: **1,096**
- Images involved in exact duplicates: **2,234**
- Redundant duplicate files: **1,138**
- Cross-class exact duplicate groups: **0**

After retaining one representative for each exact-hash group, the final modelling set contains **4,794 unique images**.

### 3.3 Final class distribution

| Class | Count | Percentage |
|---|---:|---:|
| Bacterial Blight | 1,326 | 27.66% |
| Blast | 960 | 20.03% |
| Brown Spot | 1,200 | 25.03% |
| Tungro | 1,308 | 27.28% |
| **Total** | **4,794** | **100%** |

---

## 4. Train / Validation / Test Split

The same stratified split is reused across the final model set so that model comparison is fair.

| Split | Samples | Features |
|---|---:|---:|
| Training | 3,355 | 9 |
| Validation | 719 | 9 |
| Testing | 720 | 9 |

Class mapping:

```text
Bacterialblight -> 0
Blast           -> 1
Brownspot       -> 2
Tungro          -> 3
```

Previously verified dataset split leakage checks:
- Train–Validation overlap = **0**
- Train–Test overlap = **0**
- Validation–Test overlap = **0**

---

## 5. Final 9 Features

The classical ML branch uses these nine features:

```python
feature_cols = [
    'contrast',
    'homogeneity',
    'energy',
    'correlation',
    'lesion_ratio_final',
    'lesion_count_final',
    'lesion_mean_h',
    'lesion_mean_s',
    'lesion_mean_v'
]

X = texture_df[feature_cols].copy()
y = texture_df['label'].copy()
```

### What the code does
- `feature_cols`: defines the exact 9 inputs.
- `X`: stores those features.
- `y`: stores the disease label.

Final matrix shape: **(4794, 9)**.

Quality checks:
- Missing values = **0**
- Infinite values = **0**

---

## 6. Feature Quality Checks

### 6.1 Correlation

| Feature pair | Correlation |
|---|---:|
| contrast ↔ correlation | -0.834 |
| homogeneity ↔ energy | 0.862 |

No feature was automatically removed.

### 6.2 IQR outliers

The IQR method was used to identify statistical outliers. Outliers were analysed but not automatically deleted.

```text
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

---

## 7. Standardization

SVM is sensitive to feature scale, so the nine features were standardized.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### What happens here?

1. `fit_transform(X_train)` learns the mean and standard deviation from **training data only**, then scales training data.
2. `transform(X_val)` scales validation data using those training statistics.
3. `transform(X_test)` scales test data using those same training statistics.

### Why?
Fitting the scaler on validation/test data could leak information from those sets into preprocessing.

Verified standardized shapes:

```text
Training   : (3355, 9)
Validation : (719, 9)
Testing    : (720, 9)
```

Sanity check:
- Training means approximately zero = **True**
- Training standard deviations approximately one = **True**

---

## 8. SVM Imports

```python
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
```

| Import | Purpose |
|---|---|
| `SVC` | SVM classifier |
| `StratifiedKFold` | Stratified k-fold cross-validation |
| `GridSearchCV` | Hyperparameter tuning |
| `accuracy_score` | Accuracy |
| `precision_score` | Precision |
| `recall_score` | Recall |
| `f1_score` | F1 score |
| `classification_report` | Per-class evaluation |
| `confusion_matrix` | Error/correct prediction matrix |

---

## 9. Baseline SVM

Before tuning, a baseline SVM was trained.

```python
baseline_svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)

baseline_svm.fit(X_train_scaled, y_train)
```

### Meaning of the parameters
- `kernel='rbf'`: uses a non-linear radial basis function decision boundary.
- `C=1.0`: controls the penalty for classification errors.
- `gamma='scale'`: uses a data-dependent RBF gamma.
- `random_state=42`: supports reproducibility.

### Baseline validation results

| Metric | Result |
|---|---:|
| Accuracy | **0.8999** |
| Precision | **0.8986** |
| Recall | **0.8999** |
| F1 Score | **0.8971** |

These values are the baseline reference before tuning.

---

## 10. Hyperparameter Tuning

### 10.1 Stratified 5-fold cross-validation

```python
cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

**Why StratifiedKFold?** It keeps the class distribution approximately balanced across folds.

### 10.2 Search space

```python
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}
```

Number of combinations:

```text
4 C values × 4 gamma values × 2 kernels = 32 combinations
```

Five folds give:

```text
32 × 5 = 160 model fits
```

Actual notebook output:

```text
Fitting 5 folds for each of 32 candidates, totalling 160 fits
```

---

## 11. GridSearchCV Code

```python
svm_grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=svm_param_grid,
    cv=cv_strategy,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

svm_grid_search.fit(X_train_scaled, y_train)
```

### Explanation
- `estimator`: the SVM being tuned.
- `param_grid`: possible hyperparameter combinations.
- `cv`: the 5-fold stratified scheme.
- `scoring='f1_weighted'`: the search selects the combination with the strongest weighted F1.
- `n_jobs=-1`: uses available CPU cores.
- `.fit(X_train_scaled, y_train)`: tuning is performed using training data only.

**The test set is not passed to GridSearchCV.**

---

## 12. Best SVM Configuration

Actual result:

```text
C      = 10
gamma  = 1
kernel = rbf
```

Best 5-fold weighted F1: **0.9952**.

Best estimator:

```text
SVC(C=10, gamma=1, random_state=42)
```

---

## 13. Tuned Validation Evaluation

```python
best_svm = svm_grid_search.best_estimator_
y_val_pred_svm = best_svm.predict(X_val_scaled)
```

Actual validation results:

| Metric | Result |
|---|---:|
| Accuracy | **1.0000** |
| Precision | **1.0000** |
| Recall | **1.0000** |
| F1 Score | **1.0000** |

---

## 14. Final Test Evaluation

After selecting the best hyperparameters, the held-out test set was evaluated.

```python
y_test_pred_svm = best_svm.predict(X_test_scaled)

svm_test_accuracy = accuracy_score(y_test, y_test_pred_svm)
svm_test_precision = precision_score(
    y_test, y_test_pred_svm, average='weighted'
)
svm_test_recall = recall_score(
    y_test, y_test_pred_svm, average='weighted'
)
svm_test_f1 = f1_score(
    y_test, y_test_pred_svm, average='weighted'
)
```

### Why this order?

```text
Training data
    ↓
Hyperparameter tuning + 5-fold CV
    ↓
Best SVM
    ↓
Validation evaluation
    ↓
Held-out test evaluation
```

This prevents the test set from being used to choose the model.

---

## 15. Final Test Results — ACTUAL EXPERIMENT

| Metric | Test Result |
|---|---:|
| Accuracy | **1.0000 (100%)** |
| Precision | **1.0000 (100%)** |
| Recall | **1.0000 (100%)** |
| F1 Score | **1.0000 (100%)** |

Test set:
- Total test samples = **720**
- Correct predictions = **720**
- Incorrect predictions = **0**

**These values were taken from the actual notebook output supplied after running the SVM experiment. They were not invented for this document.**

---

## 16. Classification Report

```python
classification_report(
    y_test,
    y_test_pred_svm,
    target_names=[
        'Bacterial Blight',
        'Blast',
        'Brown Spot',
        'Tungro'
    ]
)
```

Actual report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Bacterial Blight | 1.00 | 1.00 | 1.00 | 199 |
| Blast | 1.00 | 1.00 | 1.00 | 144 |
| Brown Spot | 1.00 | 1.00 | 1.00 | 180 |
| Tungro | 1.00 | 1.00 | 1.00 | 197 |
| **Accuracy** | | | **1.00** | **720** |
| **Macro Avg** | **1.00** | **1.00** | **1.00** | **720** |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **720** |

---

## 17. Confusion Matrix

The actual confusion matrix was:

```text
                    Predicted
                  BB  Blast  BrownSpot  Tungro
Actual BB       199     0       0         0
       Blast      0   144       0         0
       Brown      0     0     180         0
       Tungro     0     0       0       197
```

Interpretation:
- Bacterial Blight: 199/199 correct.
- Blast: 144/144 correct.
- Brown Spot: 180/180 correct.
- Tungro: 197/197 correct.
- Total = 720 correct out of 720.
- No off-diagonal errors were observed.

---

## 18. Final Results Table

Code used:

```python
svm_final_results = pd.DataFrame({
    'Model': ['SVM'],
    'Accuracy': [svm_test_accuracy],
    'Precision': [svm_test_precision],
    'Recall': [svm_test_recall],
    'F1 Score': [svm_test_f1]
})

display(svm_final_results.style.hide(axis='index'))
```

Current result:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **SVM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

---

## 19. Leakage / Sanity Checks

A final notebook sanity-check was run.

Verified:

```text
Training means approximately zero: True
Training standard deviations approximately one: True

GridSearchCV was fitted using training data only.
Test data was not supplied to GridSearchCV.

Test samples: 720
Test predictions: 720

Prediction length matches test labels: True

Test Accuracy: 1.0000
Test F1 Score: 1.0000

Sanity check completed.
```

Important: this sanity check confirms the checks implemented in the notebook. It is not, by itself, proof against every possible form of data leakage.

---

## 20. SVM Workflow — One View

```text
5,932 original images
        ↓
Exact duplicate analysis
        ↓
4,794 unique images
        ↓
Final 9 engineered features
        ↓
Stratified split
        ↓
3,355 train | 719 validation | 720 test
        ↓
StandardScaler (fit on train only)
        ↓
Baseline SVM
        ↓
Validation F1 = 0.8971
        ↓
5-fold GridSearchCV
32 combinations / 160 fits
        ↓
Best: C=10, gamma=1, RBF
        ↓
CV weighted F1 = 0.9952
        ↓
Validation F1 = 1.0000
        ↓
Final test evaluation
        ↓
Accuracy = 1.0000
Precision = 1.0000
Recall = 1.0000
F1 = 1.0000
```

---

## 21. Viva Quick Answers

### Why standardize?
Because SVM is sensitive to feature scale and the nine features have different ranges.

### Why RBF?
RBF can model non-linear class boundaries.

### What is C?
C controls the trade-off between classification errors and model flexibility.

### What is gamma?
For RBF, gamma controls how strongly individual samples influence the decision boundary.

### Why GridSearchCV?
To compare multiple hyperparameter combinations systematically.

### Why StratifiedKFold?
To maintain approximately similar class proportions in each fold.

### Why weighted F1?
The four classes are not exactly equal in size, so weighted F1 accounts for class support.

### Why keep the test set until the end?
To avoid using test information during model selection.

### What was the final SVM result?
100% accuracy, precision, recall and F1 on the 720-image held-out test set in this experiment.

### How many test images were misclassified?
Zero.

### Should we claim 100% real-world accuracy?
No. This is the result measured on this dataset and this experimental pipeline. Real-world generalization still needs discussion.

---

## 22. SVM Completion Status

```text
Dataset preparation              ✅
Duplicate analysis               ✅
Final 9 features                 ✅
Train/validation/test split      ✅
Standardization                  ✅
Baseline SVM                     ✅
5-fold Stratified CV             ✅
Hyperparameter tuning            ✅
Best SVM selection               ✅
Validation evaluation            ✅
Final test evaluation            ✅
Classification report            ✅
Confusion matrix                 ✅
Sanity check                     ✅
Final results table              ✅
```

### Current SVM result

```text
Accuracy  = 1.0000
Precision = 1.0000
Recall    = 1.0000
F1 Score  = 1.0000
```

These results are recorded for the SVM model only. The project winner cannot be selected until all six final models are evaluated on the same 720-image test set.