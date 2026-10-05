# Decision Tree Model Documentation — Rice Leaf Disease Classification

## 1. Purpose

This document explains the complete Decision Tree experiment from the established project pipeline through final test evaluation.

It is written so that a team member who did not build the model can understand:
- what data was used;
- what the Decision Tree does;
- why each step was performed;
- how the model was trained;
- how hyperparameters were searched;
- why the tuned configuration was not selected as the final model;
- how the final model was evaluated; and
- what the results mean.

**Result integrity:** Every numerical result below comes from the actual Decision Tree notebook outputs.

## 2. Project Context

This is a four-class rice leaf disease classification problem:

1. Bacterial Blight
2. Blast
3. Brown Spot
4. Tungro

Final six-model plan:
1. SVM
2. Random Forest
3. Decision Tree
4. MLP
5. Custom CNN
6. MobileNetV2

Decision Tree is Model 3.

## 3. What Is a Decision Tree?

A Decision Tree is a supervised learning model that makes predictions by repeatedly splitting data according to feature conditions until reaching a final prediction at a leaf.

A simple idea is:

```text
Feature condition?
       ↓
   Yes / No
    ↓     ↓
next    next
test    test
    \   /
   final class
```

The project uses a Decision Tree as an interpretable single-tree model and as a useful comparison against the ensemble Random Forest.

## 4. Dataset and Classical-ML Input

The final classical branch uses the deduplicated dataset and the nine engineered features established by the project.

Dataset:
- Original images: **5,932**
- Final unique images after exact duplicate handling: **4,794**
- Corrupted/unreadable images: **0**

The same cleaned dataset is used for the classical models to keep the model comparison fair.

## 5. Exact Duplicate Handling

The project identified exact duplicates using hashing.

Recorded results:
- Unique file hashes: **4,794**
- Exact duplicate groups: **1,096**
- Images involved in exact duplicate groups: **2,234**
- Redundant duplicate files: **1,138**
- Cross-class exact duplicate groups: **0**

One representative was retained per exact-hash group for the final modelling dataset.

This reduces the risk that copies of the same image appear in different splits and artificially inflate model performance.

## 6. Final Class Distribution

| Class | Count | Percentage |
|---|---:|---:|
| Bacterial Blight | 1,326 | 27.66% |
| Blast | 960 | 20.03% |
| Brown Spot | 1,200 | 25.03% |
| Tungro | 1,308 | 27.28% |
| **Total** | **4,794** | **100%** |

## 7. Train / Validation / Test Split

The same stratified split is reused across the classical models.

| Split | Samples | Features |
|---|---:|---:|
| Training | **3,355** | **9** |
| Validation | **719** | **9** |
| Testing | **720** | **9** |

Label mapping:
- Bacterialblight → 0
- Blast → 1
- Brownspot → 2
- Tungro → 3

Previously checked split overlaps were zero:
- Train–Validation = 0
- Train–Test = 0
- Validation–Test = 0

The 720-image test set was kept for final evaluation.

## 8. Final Nine Engineered Features

The Decision Tree receives the same nine numerical features used by the other classical models:

1. contrast
2. homogeneity
3. energy
4. correlation
5. lesion_ratio_final
6. lesion_count_final
7. lesion_mean_h
8. lesion_mean_s
9. lesion_mean_v

These represent texture, lesion characteristics, and HSV colour information.

Final feature matrix: **(4,794, 9)**

Quality checks:
- Missing values = **0**
- Infinite values = **0**

## 9. Correlation and IQR Analysis

Two strong feature relationships recorded by the project were:

| Feature pair | Correlation |
|---|---:|
| contrast ↔ correlation | -0.834 |
| homogeneity ↔ energy | 0.862 |

No feature was automatically removed.

The IQR method was used to identify outliers:

```text
IQR = Q3 - Q1
Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR
```

Outliers were analysed rather than blindly deleted because an extreme image-derived feature value can represent a genuine disease characteristic.

## 10. Standardization

The established classical pipeline uses StandardScaler fitted on training data only.

Conceptually:

```text
Training → fit scaler → transform training

Validation → transform using training scaler

Test → transform using training scaler
```

This prevents validation/test information from being used to calculate training preprocessing statistics.

Shapes after the common pipeline:
- Training: **(3,355, 9)**
- Validation: **(719, 9)**
- Testing: **(720, 9)**

## 11. Why Use a Decision Tree?

A Decision Tree was selected because it:
- models non-linear relationships;
- can capture interactions between engineered features;
- is relatively easy to explain;
- provides a strong single-tree baseline;
- gives a meaningful comparison with Random Forest.

A single tree can overfit more easily than an ensemble, so both baseline and tuned configurations are examined.

# 12. Baseline Decision Tree

A baseline model was trained before tuning:

```python
DecisionTreeClassifier(
    random_state=42
)
```

The fixed seed makes the experiment reproducible.

### Baseline validation results

| Metric | Result |
|---|---:|
| Accuracy | **0.983310** |
| Precision | **0.983369** |
| Recall | **0.983310** |
| F1-score | **0.983303** |

Approximate validation accuracy: **98.33%**

### Baseline validation confusion matrix

The 719 validation samples produced:
- Bacterial Blight: **193/199** correct
- Blast: **141/144** correct
- Brown Spot: **179/180** correct
- Tungro: **194/196** correct

Total correct = **707/719**

## 13. Why Tune the Decision Tree?

Decision Tree performance depends on its structure. A tree that is too simple can underfit, while a very complex tree can overfit.

The experiment therefore searches meaningful tree-complexity parameters instead of relying only on defaults.

## 14. Hyperparameters Tuned

### criterion
The split-quality criterion:
- gini
- entropy

### max_depth
Maximum tree depth:
- None
- 5
- 10
- 15
- 20

### min_samples_split
Minimum samples required to split an internal node:
- 2
- 5
- 10

### min_samples_leaf
Minimum samples required in a leaf:
- 1
- 2
- 4

## 15. Hyperparameter Search Size

The search contained:

```text
2 criteria
× 5 max_depth values
× 3 min_samples_split values
× 3 min_samples_leaf values
= 90 configurations
```

With 5 folds:

```text
90 × 5 = 450 fits
```

The actual notebook output confirmed:

```text
Fitting 5 folds for each of 90 candidates, totalling 450 fits
```

## 16. Stratified 5-Fold Cross-Validation

Tuning used StratifiedKFold with:
- 5 folds
- shuffling
- random_state = 42

Stratification helps maintain the disease-class proportions in each fold.

Weighted F1 was used as the GridSearchCV scoring metric because the task contains four classes and we want tuning to consider both precision and recall while weighting classes by their support.

The test set was not used during tuning.

## 17. Best Configuration Found

Actual GridSearchCV result:

```text
criterion          = entropy
max_depth          = None
min_samples_split  = 2
min_samples_leaf   = 1
```

Best cross-validation weighted F1:

**0.978881**

## 18. Important Finding: Tuning Reduced Validation Performance

The baseline validation F1 was:

**0.983303**

The selected tuned configuration achieved:

**0.975008**

Therefore, tuning did not improve the fixed validation set.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Decision Tree Baseline | **0.983310** | **0.983369** | **0.983310** | **0.983303** |
| Decision Tree Tuned | 0.974965 | 0.975389 | 0.974965 | 0.975008 |

Difference in validation F1:

```text
0.983303 - 0.975008 = 0.008295
```

or about **0.83 percentage points**.

This is a valid experimental result. Hyperparameter tuning does not guarantee an improvement on every separate held-out validation set.

## 19. Final Model Selection

The **baseline Decision Tree was selected as the final Decision Tree model** because it had the higher validation performance.

We do not repeatedly inspect the test set to choose between baseline and tuned models. The test set is reserved for the final evaluation of the selected model.

Therefore:

```text
Final Decision Tree = baseline model
```

## 20. Final Test Evaluation

The selected baseline Decision Tree was evaluated once on the previously untouched 720-image test set.

Actual final test results:

| Metric | Result |
|---|---:|
| **Accuracy** | **0.981944** |
| **Precision** | **0.981995** |
| **Recall** | **0.981944** |
| **F1-score** | **0.981966** |

Approximate percentages:
- Accuracy = **98.19%**
- Precision = **98.20%**
- Recall = **98.19%**
- F1 = **98.20%**

Correct = **707/720**

Incorrect = **13/720**

## 21. Final Test Confusion Matrix

The actual final test confusion matrix was:

| True class | Bacterial Blight | Blast | Brown Spot | Tungro |
|---|---:|---:|---:|---:|
| **Bacterial Blight** | 196 | 2 | 1 | 0 |
| **Blast** | 2 | 139 | 2 | 1 |
| **Brown Spot** | 1 | 2 | 177 | 0 |
| **Tungro** | 0 | 2 | 0 | 195 |

Class-level interpretation:
- Bacterial Blight: **196/199** correct
- Blast: **139/144** correct
- Brown Spot: **177/180** correct
- Tungro: **195/197** correct

Blast had the largest number of incorrect predictions in the final test.

## 22. Final Decision Tree Results Record

Exact results recorded by the notebook:

| Model | CV F1 | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | **0.978881** | **0.983310** | **0.983369** | **0.983310** | **0.983303** | **0.981944** | **0.981995** | **0.981944** | **0.981966** |

This is the Decision Tree source of truth for the final six-model comparison.

## 23. Interpretation

The Decision Tree achieved strong performance using the nine engineered features, with **98.19% test accuracy**.

Random Forest has already produced higher performance on the same classical feature representation, showing the potential advantage of combining multiple trees instead of relying on a single tree.

The final project conclusion should wait until all six models are evaluated.

## 24. Strengths

1. Easy to understand conceptually.
2. Can capture non-linear feature relationships.
3. Can model interactions between features.
4. More interpretable than many neural-network architectures.
5. Useful as a single-tree baseline against Random Forest.

## 25. Limitations

1. A single tree can overfit.
2. Tree structure can be sensitive to the training data.
3. It may be less stable than an ensemble such as Random Forest.
4. It depends on the engineered feature representation.
5. High dataset performance does not guarantee field performance on unseen environments.

## 26. Generalization and Overfitting

Validation F1 = **0.983303**

Test F1 = **0.981966**

These are close, so there is no obvious large validation-to-test drop in this experiment.

However, the result is still dataset-specific. Real-world images may differ in lighting, cameras, backgrounds, disease severity, rice varieties, and geographic location.

Independent external validation would be useful before deployment.

## 27. Viva Questions and Answers

### Q1. What is a Decision Tree?
A supervised learning model that repeatedly splits data using feature-based conditions until reaching a final class at a leaf.

### Q2. Why did you use a Decision Tree?
It provides an interpretable single-tree model and a useful baseline for comparison with Random Forest.

### Q3. What is the difference between Decision Tree and Random Forest?
Decision Tree uses one tree; Random Forest combines predictions from many trees.

### Q4. What is max_depth?
The maximum depth allowed for the tree.

### Q5. What is min_samples_split?
The minimum number of samples required to split an internal node.

### Q6. What is min_samples_leaf?
The minimum number of samples required in a leaf.

### Q7. What is criterion?
The measure used to evaluate the quality of a split, such as Gini impurity or entropy.

### Q8. Why use GridSearchCV?
To systematically test predefined parameter combinations using cross-validation.

### Q9. How many configurations were tested?
**90.**

### Q10. How many total fits?
**450**, because each configuration was evaluated over 5 folds.

### Q11. Why use StratifiedKFold?
To help preserve the class distribution in each fold.

### Q12. Why weighted F1?
It combines precision and recall while accounting for class support.

### Q13. What was the best CV F1?
**0.978881.**

### Q14. What were the best parameters?
criterion=entropy, max_depth=None, min_samples_split=2, min_samples_leaf=1.

### Q15. Did tuning improve validation performance?
No. The tuned model scored lower on the fixed validation set than the baseline.

### Q16. Why use the baseline for final testing?
Because the baseline had higher validation performance, and we must not use the test set to choose between models.

### Q17. What was the final test accuracy?
**0.981944**, approximately **98.19%**.

### Q18. How many test images were misclassified?
**13 out of 720.**

### Q19. Which class had the most test errors?
Blast, with five incorrect predictions.

### Q20. Is Decision Tree the best project model?
Not yet. The final decision is made only after comparing all six models.

### Q21. Why keep the same test set for all models?
It provides a consistent and fair basis for final model comparison.

### Q22. Does 98.19% guarantee real-world performance?
No. Different field conditions can reduce generalization.

## 28. Final Status

```text
Common data pipeline                 ✅
Final 9 features                     ✅
Stratified split                     ✅
Baseline Decision Tree               ✅
Baseline validation                  ✅
Hyperparameter tuning                ✅
90 configurations                    ✅
5-fold CV                            ✅
450 fits                             ✅
Best CV configuration                ✅
Tuned validation evaluation          ✅
Baseline selected                    ✅
Final test evaluation                ✅
Confusion matrix                     ✅
Classification report                ✅
Final results record                 ✅
Decision Tree model                  ✅ COMPLETE
```

## 29. Next Model

**Model 4 — Multi-Layer Perceptron (MLP)**

Notebook:

```text
04_MLP.ipynb
```

The MLP will use the same nine engineered features, fixed split, and standardized inputs used by the classical branch. It will be treated as a feature-based shallow neural network for project comparison, while Custom CNN and MobileNetV2 remain the image-based deep-learning branch.

## 30. Documentation Standard Used for This Project

Every model document should allow another team member to answer:

```text
What data did we use?
        ↓
What features did we use?
        ↓
Why this model?
        ↓
How does the model work?
        ↓
What hyperparameters were tested?
        ↓
Why cross-validation?
        ↓
What did tuning actually produce?
        ↓
Which model was selected and why?
        ↓
How did it perform on validation?
        ↓
How did it perform on the untouched test set?
        ↓
What do the results mean?
        ↓
What are the strengths and limitations?
        ↓
What should I say in the viva?
```

This same explanation-first standard will be followed for the remaining MLP, Custom CNN, and MobileNetV2 documentation.
