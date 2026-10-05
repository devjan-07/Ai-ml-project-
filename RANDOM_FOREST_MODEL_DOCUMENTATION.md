# Random Forest Model Documentation — Rice Leaf Disease Classification

## 1. Purpose

This document records the complete Random Forest experiment from the established project pipeline through final test evaluation. It is intended for team members, report writing, presentation preparation, and viva preparation.

**Result integrity:** All Random Forest numerical results below come from the actual experiment outputs. No performance values have been invented.

## 2. Model Overview

- Problem: 4-class rice leaf disease classification
- Classes: Bacterial Blight, Blast, Brown Spot, Tungro
- Model: Random Forest Classifier
- Input: final 9 engineered classical-ML features
- Final test set: 720 images, reserved for final evaluation

Random Forest is an ensemble learning method that combines multiple decision trees. For classification, the trees contribute predictions that are combined to determine the final class.

## 3. Data Preparation

Original dataset:
- Total images: 5,932
- Corrupted/unreadable images: 0

Exact duplicate analysis:
- Unique file hashes: 4,794
- Exact duplicate groups: 1,096
- Images involved in exact duplicates: 2,234
- Redundant duplicate files: 1,138
- Cross-class exact duplicate groups: 0

The final modelling dataset contains 4,794 unique images.

### Final class distribution

| Class | Count | Percentage |
|---|---:|---:|
| Bacterial Blight | 1,326 | 27.66% |
| Blast | 960 | 20.03% |
| Brown Spot | 1,200 | 25.03% |
| Tungro | 1,308 | 27.28% |
| **Total** | **4,794** | **100%** |

## 4. Train / Validation / Test Split

The same stratified split is reused across the final model set.

| Split | Samples | Features |
|---|---:|---:|
| Training | 3,355 | 9 |
| Validation | 719 | 9 |
| Testing | 720 | 9 |

Class mapping:
- Bacterialblight → 0
- Blast → 1
- Brownspot → 2
- Tungro → 3

Verified overlap checks:
- Train–Validation = 0
- Train–Test = 0
- Validation–Test = 0

## 5. Final Nine Features

The Random Forest uses the same final classical-ML representation established by the project:

1. contrast
2. homogeneity
3. energy
4. correlation
5. lesion_ratio_final
6. lesion_count_final
7. lesion_mean_h
8. lesion_mean_s
9. lesion_mean_v

Final feature matrix: **(4794, 9)**.

Quality checks:
- Missing values = **0**
- Infinite values = **0**

## 6. Feature Quality Checks

### Correlation

| Feature pair | Correlation |
|---|---:|
| contrast ↔ correlation | -0.834 |
| homogeneity ↔ energy | 0.862 |

No feature was automatically removed.

### IQR outlier analysis

The IQR method was used to identify statistical outliers:

IQR = Q3 − Q1

Lower bound = Q1 − 1.5 × IQR

Upper bound = Q3 + 1.5 × IQR

Outliers were analysed but not automatically deleted because an extreme image-derived value may represent a genuine disease characteristic.

## 7. Standardization

The established classical pipeline uses StandardScaler fitted on training data only.

Training shape: **(3355, 9)**

Validation shape: **(719, 9)**

Testing shape: **(720, 9)**

The scaler learns statistics from the training data and applies those same statistics to validation and test data. This prevents preprocessing leakage.

## 8. Why Random Forest?

Random Forest was selected as one of the four classical models because it:

- Models non-linear relationships.
- Can capture interactions between engineered features.
- Combines multiple decision trees.
- Is generally more robust than relying on a single decision tree.
- Works well with tabular engineered features.
- Provides a useful comparison against SVM, Decision Tree and MLP/ANN.

## 9. Baseline Random Forest

A baseline Random Forest was first trained before tuning.

Configuration used the reproducibility seed 42 and all available CPU cores.

The baseline validation results were:

| Metric | Result |
|---|---:|
| Accuracy | **0.997218** |
| Precision | **0.997233** |
| Recall | **0.997218** |
| F1 Score | **0.997218** |

This is approximately **99.72% validation accuracy**.

### Baseline validation confusion matrix

- Bacterial Blight: 199/199 correct
- Blast: 143/144 correct
- Brown Spot: 180/180 correct
- Tungro: 195/196 correct

Therefore:
- Correct = **717/719**
- Incorrect = **2/719**

## 10. Hyperparameter Tuning

Random Forest was tuned using GridSearchCV and stratified 5-fold cross-validation.

Parameters searched:

- n_estimators: 100, 200
- max_depth: None, 10, 20
- min_samples_split: 2, 5
- min_samples_leaf: 1, 2

This produced:

24 parameter combinations × 5 folds = **120 fits**

The actual notebook output confirmed:

**Fitting 5 folds for each of 24 candidates, totalling 120 fits**

### Parameter meanings

**n_estimators:** number of trees in the forest.

**max_depth:** maximum depth of each decision tree.

**min_samples_split:** minimum number of samples required to split an internal node.

**min_samples_leaf:** minimum number of samples required in a leaf node.

## 11. Cross-Validation

Stratified 5-fold cross-validation was used so that class proportions are approximately maintained across folds.

The tuning metric was weighted F1.

Why weighted F1?

The task has four disease classes. Weighted F1 combines precision and recall while accounting for the number of samples in each class.

The test set was not passed to GridSearchCV.

## 12. Best Random Forest Configuration

Actual best parameters:

- n_estimators = **100**
- max_depth = **None**
- min_samples_split = **5**
- min_samples_leaf = **1**

Best 5-fold weighted F1:

**0.996124**

## 13. Tuned Validation Evaluation

The best estimator returned by GridSearchCV was evaluated on the fixed validation set.

| Metric | Tuned Validation |
|---|---:|
| Accuracy | **0.998609** |
| Precision | **0.998616** |
| Recall | **0.998609** |
| F1 Score | **0.998608** |

The tuned model improved validation accuracy from **0.997218** to **0.998609**.

### Tuned validation confusion matrix

- Bacterial Blight: 199/199 correct
- Blast: 143/144 correct
- Brown Spot: 180/180 correct
- Tungro: 196/196 correct

Therefore:
- Correct = **718/719**
- Incorrect = **1/719**

The single error was **Blast predicted as Bacterial Blight**.

## 14. Baseline vs Tuned

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest Baseline | 0.997218 | 0.997233 | 0.997218 | 0.997218 |
| Random Forest Tuned | **0.998609** | **0.998616** | **0.998609** | **0.998608** |

## 15. Final Test Evaluation

After hyperparameter selection and validation evaluation, the selected Random Forest was evaluated on the previously untouched 720-image test set.

The evaluation sequence was:

Training data
→ 5-fold cross-validation
→ best hyperparameters
→ validation evaluation
→ final test evaluation

This prevents the test set from being used to choose the model.

## 16. Final Test Results — Actual Experiment

| Metric | Test Result |
|---|---:|
| Accuracy | **0.998611** |
| Precision | **0.998621** |
| Recall | **0.998611** |
| F1 Score | **0.998612** |

Approximate percentages:

- Accuracy = **99.86%**
- Precision = **99.86%**
- Recall = **99.86%**
- F1 = **99.86%**

Test set:
- Total = **720**
- Correct = **719**
- Incorrect = **1**

Therefore:

**719 / 720 = 99.8611% accuracy**

### Final test confusion matrix

- Bacterial Blight: 199/199 correct
- Blast: 144/144 correct
- Brown Spot: 180/180 correct
- Tungro: 196/197 correct

The single test error was:

**Tungro → Blast**

## 17. Final Random Forest Results Record

| Model | CV F1 | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.996124 | 0.998609 | 0.998616 | 0.998609 | 0.998608 | 0.998611 | 0.998621 | 0.998611 | 0.998612 |

This table is the Random Forest source of truth for the final six-model comparison.

## 18. Interpretation

The Random Forest achieved very strong performance using only the nine engineered features.

Baseline validation accuracy was **99.7218%**.

After tuning, validation accuracy increased to **99.8609%**.

Final test accuracy was **99.8611%**.

The CV F1 of **0.996124**, validation F1 of **0.998608**, and test F1 of **0.998612** come from different evaluation procedures and should therefore be reported separately.

The very similar validation and test results do not show an obvious numerical sign of severe overfitting in this experiment.

However, high performance on this dataset does not prove identical performance on new field images. Real-world conditions may differ in lighting, camera quality, background, disease severity, rice variety, and geographic location.

## 19. Strengths

1. Very strong classification performance.
2. Handles non-linear relationships.
3. Combines many decision trees rather than relying on one.
4. Robust compared with a single decision tree.
5. Effective with the engineered tabular feature representation.
6. Does not require designing a deep neural architecture.

## 20. Limitations

1. It depends on the quality of the engineered features.
2. It does not learn visual representations directly from raw images.
3. Increasing the number or complexity of trees increases computation.
4. Performance on this dataset does not guarantee real-world generalization.
5. External validation on independent field data would be useful before deployment.

## 21. Viva Questions and Answers

### Q1. What is Random Forest?

Random Forest is an ensemble machine-learning algorithm that combines multiple decision trees to make a final prediction.

### Q2. Why did you use Random Forest?

It can model non-linear relationships, capture interactions between engineered features, and provide a strong classical-ML comparison model.

### Q3. What is an ensemble?

An ensemble combines predictions from multiple models to produce a stronger overall prediction.

### Q4. Why can Random Forest be better than one decision tree?

A single tree can overfit easily. Random Forest combines multiple randomized trees, making the overall prediction more stable.

### Q5. What does n_estimators mean?

It is the number of decision trees in the forest.

### Q6. What does max_depth mean?

It controls the maximum depth of each decision tree. Our best configuration used None.

### Q7. What does min_samples_split mean?

It is the minimum number of samples required before an internal node can be split.

### Q8. What does min_samples_leaf mean?

It is the minimum number of samples allowed in a leaf node.

### Q9. Why did you use GridSearchCV?

To systematically test predefined hyperparameter combinations using cross-validation and select the configuration with the best weighted F1.

### Q10. Why 5-fold cross-validation?

It evaluates the model across five different folds, giving a more reliable estimate than one training split.

### Q11. Why StratifiedKFold?

Because this is a four-class problem. Stratification helps preserve class proportions in each fold.

### Q12. Why weighted F1?

It combines precision and recall while accounting for the number of samples in each class.

### Q13. Did you use the test set during tuning?

No. The 720-image test set was kept for final evaluation only.

### Q14. What was the best Random Forest configuration?

n_estimators = 100, max_depth = None, min_samples_split = 5, min_samples_leaf = 1.

### Q15. What was the best cross-validation F1?

**0.996124**.

### Q16. What was the final test accuracy?

**0.998611**, approximately **99.86%**.

### Q17. How many test images were misclassified?

**1 out of 720**.

### Q18. Which test error occurred?

One **Tungro** image was predicted as **Blast**.

### Q19. Is Random Forest definitely the best project model?

Not yet. It must be compared with SVM, Decision Tree, MLP/ANN, Custom CNN and MobileNetV2.

### Q20. Why use the same test set for all six models?

Using the same held-out test set makes the final comparison more consistent and fair.

### Q21. Why were IQR outliers not automatically removed?

Because an extreme image-derived feature value may represent a genuine disease characteristic rather than an invalid observation.

### Q22. Does 99.86% test accuracy guarantee real-world performance?

No. Real-world images may differ from the dataset in lighting, camera, background, disease severity, rice variety, and location.

## 22. Final Status

- Data preparation: **Complete**
- Feature engineering: **Complete**
- Train/validation/test split: **Complete**
- Random Forest baseline: **Complete**
- Hyperparameter tuning: **Complete**
- 5-fold cross-validation: **Complete**
- Tuned validation evaluation: **Complete**
- Final test evaluation: **Complete**
- Confusion matrix: **Complete**
- Classification report: **Complete**
- Final results record: **Complete**
- Random Forest model: **COMPLETE**

## 23. Next Model

**Decision Tree**

Notebook:

**03_Decision_Tree.ipynb**

The Decision Tree will use the same established classical-ML pipeline and fixed train/validation/test split. It will undergo baseline training, hyperparameter tuning, stratified cross-validation, validation evaluation, and final test evaluation before its results are added to the six-model comparison.
