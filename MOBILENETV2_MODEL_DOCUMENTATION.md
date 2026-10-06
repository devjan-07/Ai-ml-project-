# MobileNetV2 Model Documentation

## Rice Leaf Disease Classification Using Deep Learning

**Model:** MobileNetV2 Transfer Learning  
**Notebook:** `06_MobileNetV2.ipynb`  
**Dataset:** Rice Leaf Disease Image Dataset (Kaggle)  
**Task:** Four-class image classification

## 1. Objective

Classify rice leaf images into Bacterial Blight, Blast, Brown Spot and Tungro. MobileNetV2 was selected as the transfer-learning deep-learning model. It reused the existing master pipeline; preprocessing and dataset splitting were not repeated.

## 2. Existing Dataset and Preprocessing

Final dataset: **4,794 unique images** from 5,932 original images. Corrupted/unreadable images: **0**. Cross-class exact duplicate groups: **0**.

Final class distribution:
- Bacterial Blight: 1,326 (27.66%)
- Blast: 960 (20.03%)
- Brown Spot: 1,200 (25.03%)
- Tungro: 1,308 (27.28%)

Fixed split:
- Training: 3,355
- Validation: 719
- Test: 720

The test set remained untouched during model selection.

Existing MobileNetV2 preprocessing:
1. RGB conversion
2. Direct resize to 224 x 224
3. Training-only augmentation
4. MobileNetV2 normalization
5. Batch size 32
6. Prefetching

Validated batch shape: **(32, 224, 224, 3)**. Validated normalized range: approximately **[-1, 1]**.

## 3. Architecture

Baseline:

```
224 x 224 x 3
     ↓
MobileNetV2 pretrained ImageNet base
     ↓
GlobalAveragePooling2D
     ↓
Dropout(0.2)
     ↓
Dense(4, softmax)
```

The original ImageNet classification head was removed using `include_top=False`. The pretrained base was initially frozen.

Class mapping:
- 0 = Bacterial Blight
- 1 = Blast
- 2 = Brown Spot
- 3 = Tungro

## 4. Baseline Configuration

| Setting | Value |
|---|---|
| Weights | ImageNet |
| Base | Frozen |
| Input | 224 x 224 x 3 |
| Pooling | GlobalAveragePooling2D |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Sparse categorical cross-entropy |
| Maximum epochs | 20 |
| Batch size | 32 |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

Baseline validation:
- Accuracy: **0.9680**
- Precision: **0.9681**
- Recall: **0.9680**
- Weighted F1: **0.9678**

Highest validation accuracy: **0.9708 at Epoch 16**. Lowest validation loss: **0.0718 at Epoch 20**.

Baseline validation confusion matrix:

```
                    Predicted
                 BB  Blast Brown Tungro
Bacterial Blight 194   5    0     0
Blast              13 129    2     0
Brown Spot           0   3  177     0
Tungro               0   0    0   196
```

The main weakness was Blast, with approximately 89.6% validation recall.

## 5. Fine-Tuning

The final 30 MobileNetV2 layers were unfrozen while earlier layers remained frozen. Batch Normalization layers remained frozen.

A smaller learning rate was used because pretrained weights already contain useful visual representations.

| Setting | Value |
|---|---|
| Unfrozen layers | Final 30 MobileNetV2 layers |
| Batch Normalization | Frozen |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.00001 |
| Maximum epochs | 10 |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

Parameters:
- Total: **2,263,108**
- Trainable: **1,515,844**
- Non-trainable: **747,264**

## 6. Fine-Tuning Results

| Metric | Frozen Baseline | Fine-Tuned |
|---|---:|---:|
| Accuracy | 0.9680 | **0.9930** |
| Precision | 0.9681 | **0.9931** |
| Recall | 0.9680 | **0.9930** |
| Weighted F1 | 0.9678 | **0.9930** |

Best validation accuracy: **0.9930**.  
Best validation loss: **0.0295 at Epoch 10**.

Fine-tuned validation confusion matrix:

```
                    Predicted
                 BB  Blast Brown Tungro
Bacterial Blight 199   0    0     0
Blast               3 140    1     0
Brown Spot           0   1  179     0
Tungro               0   0    0   196
```

Blast recall improved to approximately **97.2%**.

The fine-tuned model was selected using validation performance. The test set was not used for this selection.

## 7. Final Test Evaluation

The selected fine-tuned model was evaluated once on the untouched 720-image test set.

| Metric | Test Result |
|---|---:|
| Accuracy | **0.9931** |
| Precision | **0.9931** |
| Recall | **0.9931** |
| Weighted F1 | **0.9931** |

Correct predictions: **715/720**.

Final test confusion matrix:

```
                    Predicted
                 BB  Blast Brown Tungro
Bacterial Blight 199   0    0     0
Blast               2 142    0     0
Brown Spot           1   2  177     0
Tungro               0   0    0   197
```

Remaining errors:
- 2 Blast → Bacterial Blight
- 1 Brown Spot → Bacterial Blight
- 2 Brown Spot → Blast
- 0 Tungro errors

Per-class test performance:
- Bacterial Blight: precision 0.99, recall 1.00, F1 0.99
- Blast: precision 0.99, recall 0.99, F1 0.99
- Brown Spot: precision 1.00, recall 0.98, F1 0.99
- Tungro: precision 1.00, recall 1.00, F1 1.00

## 8. Interpretation

Fine-tuning allowed later pretrained representations to adapt to rice-leaf disease patterns. Validation accuracy increased from 96.80% to 99.30%, and the selected model achieved 99.31% accuracy on the held-out test set.

The 99.31% result is performance on this dataset's held-out test set; it is not a guarantee of real-world field accuracy.

## 9. Strengths

1. Reuses pretrained visual representations.
2. Reuses the established leakage-controlled image pipeline.
3. Fine-tuning substantially improved validation performance.
4. Strong performance across all four classes.
5. Test set remained untouched during selection.

## 10. Limitations

1. Dataset conditions may not represent all real field environments.
2. One fixed held-out test split was used.
3. High test performance may not directly translate to field deployment.
4. Lighting, backgrounds, cameras, varieties and disease severity may differ in practice.
5. Independent external validation is needed before deployment.

## 11. Viva Questions

**Q1. What is MobileNetV2?**  
A pretrained CNN architecture designed for efficient image feature extraction.

**Q2. Why use MobileNetV2?**  
To reuse pretrained visual features and adapt them to the four rice disease classes.

**Q3. What is transfer learning?**  
Using knowledge learned on one task or dataset and adapting it to another related task.

**Q4. Why use include_top=False?**  
The original ImageNet head predicts ImageNet classes, not our four disease classes.

**Q5. What does freezing mean?**  
Frozen weights are not updated during training.

**Q6. Why fine-tune?**  
The frozen model achieved 96.80% validation accuracy but had difficulty with Blast. Fine-tuning improved validation accuracy to 99.30%.

**Q7. Why use a small learning rate?**  
To make small changes to useful pretrained weights rather than destroying learned representations.

**Q8. Why freeze Batch Normalization?**  
To keep pretrained normalization behaviour stable during fine-tuning.

**Q9. Why GlobalAveragePooling2D?**  
It produces a compact feature vector without the large parameter count of Flatten followed by a large dense layer.

**Q10. Why Dropout?**  
To reduce reliance on particular neurons and help control overfitting.

**Q11. Why softmax?**  
The task has four mutually exclusive classes, so softmax produces four class probabilities.

**Q12. Why sparse categorical cross-entropy?**  
The labels are integer encoded as 0, 1, 2 and 3.

**Q13. What was the baseline result?**  
Validation accuracy 0.9680 and weighted F1 0.9678.

**Q14. What was the final test result?**  
Accuracy, precision, recall and weighted F1 were all 0.9931 on 720 test images.

**Q15. Was the test set used for tuning?**  
No. It was evaluated only after selecting the final configuration using validation performance.

**Q16. What was the baseline weakness?**  
Blast had the lowest recall, approximately 89.6%.

**Q17. Did fine-tuning improve Blast?**  
Yes. Validation Blast recall increased to approximately 97.2%.

**Q18. Difference between Custom CNN and MobileNetV2?**  
Custom CNN is trained from scratch; MobileNetV2 uses ImageNet-pretrained weights and transfer learning.

**Q19. Does 99.31% mean perfect real-world accuracy?**  
No. It is the result on this particular held-out test set.

**Q20. Main limitation?**  
Generalization to independent real-world field images.

**Q21. Why is a confusion matrix useful?**  
It shows which specific classes are being confused.

**Q22. What did the final confusion matrix show?**  
Five errors: two Blast → Bacterial Blight, one Brown Spot → Bacterial Blight, and two Brown Spot → Blast. Tungro had no errors.

**Q23. Why keep the test set untouched?**  
Repeated test-set use can influence model selection and produce an overly optimistic estimate.

**Q24. What should be done before deployment?**  
Evaluate on independent field images with different lighting, backgrounds, cameras, varieties and disease severity, with expert validation.

**Q25. What future improvements are possible?**  
Collect more diverse field data, perform external validation, investigate explainability, and evaluate robustness under real-world conditions.

## 12. Final Conclusion

The MobileNetV2 experiment showed that transfer learning was effective for this rice leaf disease classification task.

The frozen baseline achieved **96.80% validation accuracy**. Fine-tuning the final 30 layers with a learning rate of 1e-5 improved validation accuracy to **99.30%**.

The selected fine-tuned model achieved **99.31% accuracy, precision, recall and weighted F1** on the untouched 720-image test set.

MobileNetV2 is therefore a strong candidate for the final six-model comparison. The overall project winner must only be selected after the actual Custom CNN and other final test results are compared using the same held-out test set.

## 13. Experimental Record

| Stage | Configuration | Result |
|---|---|---:|
| Baseline | Frozen MobileNetV2 | Validation accuracy **96.80%** |
| Fine-tuning | Last 30 layers, LR = 1e-5 | Validation accuracy **99.30%** |
| Final model | Fine-tuned MobileNetV2 | Test accuracy **99.31%** |

All numerical results come from the actual MobileNetV2 experiment and must remain synchronized with `06_MobileNetV2.ipynb`.
