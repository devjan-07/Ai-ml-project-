# Rice Leaf Disease Classification Using Deep Learning

## Project Progress Log

This document records the verified project work and current implementation status. The final pipeline is kept consistent with the six Progress Review I notebooks because both Progress I and Final notebooks will be submitted.

## 1. Project Overview

**Problem:** Multi-class classification of rice leaf diseases.

**Classes:**
- Bacterial Blight
- Blast
- Brown Spot
- Tungro

**Dataset:** Rice Leaf Disease Image Dataset (Kaggle).

**Final model plan:**
1. SVM
2. Random Forest
3. Decision Tree
4. MLP/ANN
5. Custom CNN
6. MobileNetV2

## 2. Progress Review I → Final Consistency

| Member | Progress I contribution | Final pipeline use |
|---|---|---|
| Member 1 | Standardization / scaling | StandardScaler |
| Member 2 | Correlation / multicollinearity | Correlation analysis |
| Member 3 | Statistical outliers | IQR outlier analysis |
| Member 4 | Feature engineering | Final 9-feature representation |
| Member 5 | Label encoding / stratified split | Class mapping and final split |
| Member 6 | Data cleaning / duplicate analysis | Exact duplicate handling |

The final pipeline does not blindly chain all notebooks together. Each contribution is integrated where appropriate, with separate classical-ML and deep-learning branches.

## 3. Dataset Audit and Cleaning

### Original dataset
- Total images: **5,932**
- Original class counts: Bacterialblight 1,584; Blast 1,440; Brownspot 1,600; Tungro 1,308
- Corrupted/unreadable images: **0**

### Exact duplicate analysis
- Unique file hashes: **4,794**
- Exact duplicate groups: **1,096**
- Images involved in exact duplicates: **2,234**
- Redundant duplicate files: **1,138**
- Cross-class exact duplicate groups: **0**
- Duplicate groups: size 2 = 1,078; size 3 = 6; size 5 = 12

The raw ZIP remains unchanged. One representative per exact-hash group is retained for modelling.

### Final deduplicated class distribution

| Class | Count | Percentage |
|---|---:|---:|
| Bacterialblight | 1,326 | 27.66% |
| Blast | 960 | 20.03% |
| Brownspot | 1,200 | 25.03% |
| Tungro | 1,308 | 27.28% |
| **Total** | **4,794** | **100%** |

## 4. Final Split

A single stratified split is reused across all six models:

| Subset | Images |
|---|---:|
| Training | **3,355** |
| Validation | **719** |
| Testing | **720** |
| **Total** | **4,794** |

Leakage checks: Train–Validation 0; Train–Test 0; Validation–Test 0.

Class mapping:

```text
Bacterialblight → 0
Blast           → 1
Brownspot       → 2
Tungro          → 3
```

## 5. Final Image Audit

For the 4,794-image modelling dataset: valid images **4,794**, corrupted **0**, all JPG; median and most common dimension **300×300** (3,486 images); channels: 3-channel **4,650**, 4-channel **144**. Deep-learning loading converts all images to RGB.

## 6. Deep-Learning Preprocessing — COMPLETE

Common pipeline: RGB conversion → direct resize to 224×224 → training-only augmentation → model-specific normalization → batch size 32 → prefetch. Direct resize was selected after visual validation; black/reflection padding was rejected due to artificial borders/reflected patterns.

### Custom CNN
- Input 224×224×3
- Normalization [0,255] → [0,1]
- Training augmentation only
- Validated batch (32,224,224,3), pixel range 0.0–1.0, labels 0–3

### MobileNetV2
- Input 224×224×3
- Keras MobileNetV2 preprocessing, approximate range [-1,1]
- Training augmentation only
- Validated training batch (32,224,224,3)
- Pixel minimum -1.0; maximum 0.9997853
- Validation batch (32,224,224,3)
- Visual validation completed successfully

## 7. Classical ML Feature Engineering — COMPLETE

A temporary 17-feature RGB/HSV/GLCM experiment was discarded. The final representation is exactly the Progress-I Member 4 9-feature set:

```python
feature_cols = [
    "contrast",
    "homogeneity",
    "energy",
    "correlation",
    "lesion_ratio_final",
    "lesion_count_final",
    "lesion_mean_h",
    "lesion_mean_s",
    "lesion_mean_v"
]
```

Feature extraction was connected to the **4,794** exact-deduplicated images. Final feature matrix: **(4794, 9)**. Missing values: **0**. Infinite values: **0**.

### Verified feature summary

| Feature | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| contrast | 85.592 | 114.675 | 1.534 | 1189.305 |
| homogeneity | 0.358 | 0.141 | 0.080 | 0.770 |
| energy | 0.033 | 0.016 | 0.011 | 0.150 |
| correlation | 0.971 | 0.037 | 0.665 | 1.000 |
| lesion_ratio_final | 0.157 | 0.148 | 0.001 | 0.652 |
| lesion_count_final | 7.616 | 6.862 | 0 | 38 |
| lesion_mean_h | 21.601 | 3.427 | 11.654 | 29.934 |
| lesion_mean_s | 103.842 | 31.468 | 49.976 | 200.468 |
| lesion_mean_v | 148.310 | 34.417 | 59.880 | 253.690 |

## 8. Classical ML Preprocessing — COMPLETE

### Correlation / multicollinearity

Strong feature pairs using |r| ≥ 0.80:

| Feature pair | Correlation |
|---|---:|
| contrast ↔ correlation | **-0.834** |
| homogeneity ↔ energy | **0.862** |

No automatic feature deletion was performed.

### IQR outlier analysis

Method:

```text
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

| Feature | Outlier Count | Outlier Percentage |
|---|---:|---:|
| contrast | 497 | 10.37% |
| homogeneity | 0 | 0.00% |
| energy | 104 | 2.17% |
| correlation | 577 | 12.04% |
| lesion_ratio_final | 87 | 1.81% |
| lesion_count_final | 160 | 3.34% |
| lesion_mean_h | 2 | 0.04% |
| lesion_mean_s | 8 | 0.17% |
| lesion_mean_v | 207 | 4.32% |

Outliers were analysed but not automatically deleted.

### Final feature split

```text
Training   : (3355, 9)
Validation : (719, 9)
Testing    : (720, 9)
```

Verified class distributions:
- Training: Bacterialblight 928, Tungro 915, Brownspot 840, Blast 672
- Validation: Bacterialblight 199, Tungro 196, Brownspot 180, Blast 144
- Testing: Bacterialblight 199, Tungro 197, Brownspot 180, Blast 144

### Standardization

StandardScaler was fitted on **training data only** and used to transform validation and test data.

```text
Scaled training   : (3355, 9)
Scaled validation : (719, 9)
Scaled test       : (720, 9)
```

Training feature means are approximately 0 and training standard deviations approximately 1. The classical ML preprocessing pipeline is therefore **COMPLETE**.

## 9. Classical Models — IN PROGRESS

The classical model development stage is now underway using the same established 4,794-image dataset, final 9-feature representation, stratified split, and leakage-safe preprocessing.

### 9.1 SVM

**Status:** Complete.

The SVM notebook has been completed with baseline training, hyperparameter tuning, cross-validation, validation evaluation, final test evaluation, confusion matrix, classification report, and results recording.

> SVM numerical results are retained in the completed SVM notebook and should be used as the source of truth for the final six-model comparison.

### 9.2 Random Forest

**Status:** Complete.

Notebook: `02_Random_Forest.ipynb`

#### Baseline validation

The baseline Random Forest achieved:

| Metric | Validation |
|---|---:|
| Accuracy | **0.997218** |
| Precision | **0.997233** |
| Recall | **0.997218** |
| F1-score | **0.997218** |

The baseline validation confusion matrix contained two misclassified images.

#### Hyperparameter tuning

A GridSearchCV search evaluated **24 parameter combinations using 5-fold stratified cross-validation**, resulting in **120 fits**.

Best configuration:

```text
n_estimators       = 100
max_depth          = None
min_samples_split  = 5
min_samples_leaf   = 1
```

Best cross-validation weighted F1-score:

**0.996124**

#### Tuned validation performance

| Metric | Tuned Validation |
|---|---:|
| Accuracy | **0.998609** |
| Precision | **0.998616** |
| Recall | **0.998609** |
| F1-score | **0.998608** |

The tuned model improved validation accuracy from **0.997218** to **0.998609**.

The tuned validation confusion matrix contained one misclassification: one Blast image was predicted as Bacterialblight.

#### Final test evaluation

The selected Random Forest was then evaluated once on the previously reserved **720-image test set**.

| Metric | Final Test |
|---|---:|
| Accuracy | **0.998611** |
| Precision | **0.998621** |
| Recall | **0.998611** |
| F1-score | **0.998612** |

The final test confusion matrix contained **one misclassification** out of 720 test images: one Tungro image was predicted as Blast.

Therefore, **719/720 test images were correctly classified**, corresponding to **99.8611% accuracy**.

#### Random Forest final record

| Model | CV F1 | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.996124 | 0.998609 | 0.998616 | 0.998609 | 0.998608 | 0.998611 | 0.998621 | 0.998611 | 0.998612 |

The Random Forest model is therefore **complete**. Its final result will be compared with the other five models only after all six models have been evaluated.

## 10. Deep-Learning Models After Classical Branch

Then train Custom CNN and MobileNetV2 using the already validated image pipelines.

## 11. Final Six-Model Evaluation

All six models will ultimately be evaluated on the same **720-image test set** using Accuracy, Precision, Recall, F1-score, Confusion Matrix and Classification Report. No results will be invented; all final numbers must come from actual experiments.

## 12. Current Exact Status

```text
Dataset audit                         ✅
Exact duplicate analysis             ✅
Final 4,794-image dataset            ✅
Class distribution                   ✅
Stratified split                     ✅
Leakage checks                       ✅
Image integrity                      ✅
Dimension/channel audit              ✅
RGB conversion                       ✅
224×224 resize                       ✅
Custom CNN preprocessing             ✅
MobileNetV2 preprocessing            ✅
Visual validation                    ✅
Member 4 feature engineering         ✅
Final 9-feature matrix               ✅
Missing values = 0                   ✅
Infinite values = 0                  ✅
Feature summary                      ✅
Correlation analysis                 ✅
IQR outlier analysis                 ✅
Feature train/val/test split         ✅
StandardScaler                       ✅

SVM                                  ✅ COMPLETE
Random Forest                        ✅ COMPLETE
Decision Tree                        ⏳ NEXT
MLP/ANN                              ⏳
Custom CNN                           ⏳
MobileNetV2                          ⏳
Final six-model comparison           ⏳
```

## 13. Immediate Next Step

**Start Decision Tree model development.**

The Decision Tree notebook should use the same established classical-ML data pipeline and fixed train/validation/test split. Hyperparameter tuning and stratified cross-validation must be completed before the final 720-image test evaluation.
