# Custom CNN Model Documentation

## Rice Leaf Disease Classification Using Deep Learning

### 1. Model Overview

The Custom CNN is Model 5 in the Rice Leaf Disease Classification project. It belongs to the deep-learning branch and uses the raw rice-leaf images directly rather than the nine engineered numerical features used by the classical machine-learning models.

The model performs four-class classification:

1. Bacterial Blight
2. Blast
3. Brown Spot
4. Tungro

The Custom CNN is a genuine CNN developed from scratch. It does not use pretrained MobileNetV2 layers.

---

## 2. Data Used

The fixed deduplicated dataset contains 4,794 unique images.

The same fixed split used throughout the project was retained:

| Dataset | Images |
|---|---:|
| Training | 3,355 |
| Validation | 719 |
| Test | 720 |
| Total | 4,794 |

The test set was kept separate and was not used during hyperparameter tuning or model selection.

Class labels were encoded as:

| Class | Label |
|---|---:|
| Bacterial Blight | 0 |
| Blast | 1 |
| Brown Spot | 2 |
| Tungro | 3 |

---

## 3. Image Preprocessing

The validated deep-learning image pipeline was used.

### Preprocessing sequence

Raw image  
→ RGB conversion  
→ resize to 224 × 224  
→ normalize pixel values from [0, 255] to [0, 1]  
→ batch size 32  
→ prefetch

The preprocessing was verified using an actual training batch.

Observed verification:

- Image batch shape: `(32, 224, 224, 3)`
- Label batch shape: `(32,)`
- Data type: `float32`
- Minimum pixel value: `0.0`
- Maximum pixel value: `1.0`

This confirmed that the images entering the CNN had the expected dimensions, channels, data type, and normalized pixel range.

---

## 4. Data Augmentation

Data augmentation was applied only during training.

The selected augmentation operations were:

- Random horizontal flip
- Random rotation with factor 0.05
- Random zoom with height and width factors of 0.05

The augmentation was visually inspected. The first stronger augmentation setting produced excessive visual distortion, so the augmentation strength was reduced. The revised version produced more realistic variations while keeping the rice-leaf structures recognizable.

Validation and test images were not augmented.

The augmentation layer was included in the CNN model so that it operates during training while evaluation remains based on the original validation/test images.

---

## 5. Custom CNN Architecture

The final Custom CNN architecture was:

```
Input: 224 × 224 × 3
        ↓
Training-only Data Augmentation
        ↓
Conv2D: 32 filters, 3 × 3, ReLU
        ↓
MaxPooling2D: 2 × 2
        ↓
Conv2D: 64 filters, 3 × 3, ReLU
        ↓
MaxPooling2D: 2 × 2
        ↓
Conv2D: 128 filters, 3 × 3, ReLU
        ↓
MaxPooling2D: 2 × 2
        ↓
GlobalAveragePooling2D
        ↓
Dense: 128 neurons, ReLU
        ↓
Dropout: 0.5
        ↓
Dense: 4 neurons, Softmax
```

### Why GlobalAveragePooling2D was used

An initial version used `Flatten()` followed by a Dense layer. That produced approximately 12.94 million trainable parameters, with most parameters concentrated in the Dense layer after flattening.

To make the custom CNN more practical and reduce unnecessary parameters, `Flatten()` was replaced with `GlobalAveragePooling2D()`.

The revised architecture contained:

- Total parameters: 110,276
- Trainable parameters: 110,276
- Non-trainable parameters: 0

This substantially reduced the model size while retaining the three convolutional feature-extraction blocks.

---

## 6. Baseline Training Configuration

The selected baseline configuration was:

| Hyperparameter | Setting |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 32 |
| Maximum epochs | 20 |
| Dropout | 0.5 |
| Number of classes | 4 |

Sparse categorical cross-entropy was appropriate because the class labels were stored as integer class IDs rather than one-hot encoded vectors.

### Callbacks

Three callbacks were used:

1. EarlyStopping
2. ReduceLROnPlateau
3. ModelCheckpoint

EarlyStopping was configured to restore the best weights.

ReduceLROnPlateau reduced the learning rate when validation loss stopped improving.

ModelCheckpoint saved the model with the best validation loss.

---

## 7. Baseline Training Results

The baseline was trained for 20 epochs.

The best validation loss occurred at Epoch 19.

| Metric | Epoch 19 |
|---|---:|
| Training accuracy | 88.79% |
| Validation accuracy | 91.38% |
| Training loss | 0.2831 |
| Validation loss | 0.2213 |
| Learning rate | 0.0005 |

The learning rate was reduced from 0.001 to 0.0005 during training by ReduceLROnPlateau.

The training and validation curves showed an overall improvement in accuracy and reduction in loss. There was some epoch-to-epoch fluctuation in validation performance, but there was no sustained severe divergence between training and validation performance.

---

## 8. Baseline Validation Evaluation

The selected baseline model achieved the following validation results:

| Metric | Validation |
|---|---:|
| Accuracy | 91.38% |
| Weighted Precision | 91.28% |
| Weighted Recall | 91.38% |
| Weighted F1 | 91.27% |
| Macro F1 | 90.76% |
| Loss | 0.2213 |

Per-class validation results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Bacterial Blight | 88.78% | 91.46% | 90.10% |
| Blast | 87.69% | 79.17% | 83.21% |
| Brown Spot | 92.22% | 92.22% | 92.22% |
| Tungro | 95.59% | 99.49% | 97.50% |

Blast had the weakest validation recall, indicating that the model missed a comparatively larger proportion of actual Blast images.

---

## 9. Hyperparameter Tuning

A small controlled tuning experiment was used rather than a large grid search because CNN training is computationally expensive.

The baseline was compared with two alternatives.

| Configuration | Learning Rate | Dropout | Validation Accuracy |
|---|---:|---:|---:|
| Baseline | 0.001 | 0.5 | **91.38%** |
| Tuning A | 0.0005 | 0.5 | 87.07% |
| Tuning B | 0.0005 | 0.3 | 86.51% |

The tuned configurations performed worse than the baseline.

The baseline was therefore selected as the final Custom CNN configuration.

This is an important experimental result: hyperparameter tuning does not necessarily improve every model. The alternatives tested in this experiment did not provide better validation performance, so changing away from the baseline was not justified.

---

## 10. Final Held-Out Test Evaluation

After model selection, the selected Custom CNN was evaluated on the untouched 720-image test set.

Final test results:

| Metric | Test Result |
|---|---:|
| Accuracy | **90.97%** |
| Precision, weighted | **90.95%** |
| Recall, weighted | **90.97%** |
| F1 Score, weighted | **90.88%** |
| Precision, macro | 90.96% |
| Recall, macro | 90.30% |
| F1 Score, macro | 90.54% |
| Test loss | 0.2386 |

The model correctly classified:

**655 out of 720 images**

and incorrectly classified:

**65 out of 720 images**.

The final test accuracy was 90.97%.

---

## 11. Final Test Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Bacterial Blight | 86.89% | 89.95% | 88.40% | 199 |
| Blast | 90.70% | 81.25% | 85.71% | 144 |
| Brown Spot | 91.53% | 90.00% | 90.76% | 180 |
| Tungro | 94.71% | 100.00% | 97.28% | 197 |

The test-set results show that Tungro was the easiest class for the model to distinguish in this experiment, while Blast remained the most difficult class, particularly in terms of recall.

---

## 12. Final Test Confusion Matrix

The confusion matrix was:

```
[[179,   4,   9,   7],
 [ 19, 117,   6,   2],
 [  8,   8, 162,   2],
 [  0,   0,   0, 197]]
```

Rows represent the true class and columns represent the predicted class.

### Interpretation

#### Bacterial Blight

179 of 199 Bacterial Blight images were correctly classified.

#### Blast

117 of 144 Blast images were correctly classified.

The largest error was:

```
19 Blast images → Bacterial Blight
```

This is the most important confusion in the final test set.

#### Brown Spot

162 of 180 Brown Spot images were correctly classified.

#### Tungro

All 197 Tungro images were correctly classified.

Therefore:

```
Tungro recall = 100%
```

---

## 13. Validation-to-Test Generalization

The validation accuracy was **91.38%**.

The final test accuracy was **90.97%**.

The difference was **0.41 percentage points**.

The relatively small difference indicates that the selected Custom CNN produced similar performance on the unseen test set to what was observed during validation.

However, this should not be interpreted as proof of real-world deployment performance. The test images originate from the same project dataset distribution, and the dataset does not necessarily represent all rice varieties, environments, lighting conditions, cameras, and field conditions.

---

## 14. Sanity Checks

The final test evaluation was verified using the confusion matrix and prediction arrays.

```
Test samples evaluated: 720
Correct predictions: 655
Incorrect predictions: 65

Confusion matrix total: 720
Expected test samples: 720
Actual confusion matrix samples: 720
```

These checks confirm that all 720 test samples were accounted for in the final evaluation.

The test set was not used during the hyperparameter tuning experiments.

---

## 15. Strengths

- Learns directly from raw RGB rice-leaf images.
- Genuine CNN developed specifically for the project.
- Training-only augmentation helps expose the model to reasonable image variations.
- GlobalAveragePooling2D greatly reduced trainable parameters compared with the initial Flatten-based architecture.
- Achieved 90.97% test accuracy.
- Validation and test accuracies were close.
- Tungro was classified particularly well.
- Provides a useful deep-learning baseline for comparison with MobileNetV2.

---

## 16. Limitations

### Blast classification

Blast had the lowest test recall at 81.25%. A significant number of Blast images were predicted as Bacterial Blight.

### Dataset limitations

The project dataset contains four diseased classes and does not provide a healthy-leaf class. Therefore, the model should not be interpreted as a general detector of whether a rice plant is healthy or diseased.

### Generalization

The test result measures performance on the held-out portion of this dataset. It does not guarantee the same performance on photographs collected from different farms, rice varieties, cameras, environmental conditions, or geographic regions.

### Model complexity

Although the final Custom CNN is relatively small, it may still benefit from further architectural experimentation. However, further tuning was not performed because the tested alternatives already performed worse than the baseline.

---

## 17. Final Model Configuration

```
Input: 224 × 224 × 3

Augmentation:
- Random horizontal flip
- Random rotation = 0.05
- Random zoom = 0.05

Conv2D:
- 32 filters
- 3 × 3 kernel
- ReLU

MaxPooling:
- 2 × 2

Conv2D:
- 64 filters
- 3 × 3 kernel
- ReLU

MaxPooling:
- 2 × 2

Conv2D:
- 128 filters
- 3 × 3 kernel
- ReLU

MaxPooling:
- 2 × 2

GlobalAveragePooling2D

Dense:
- 128 neurons
- ReLU

Dropout:
- 0.5

Output:
- 4 neurons
- Softmax

Optimizer:
- Adam

Initial learning rate:
- 0.001

Loss:
- Sparse Categorical Crossentropy

Batch size:
- 32

Final test accuracy:
- 90.97%

Final weighted F1:
- 90.88%
```

---

## 18. Viva Questions

### Q1. Why did you use a CNN?

A CNN is suitable for image classification because it can automatically learn spatial patterns such as edges, textures, shapes, and more complex visual features directly from image pixels.

### Q2. Why did you resize images to 224 × 224?

A fixed image size is required so that all images have a consistent input shape for the CNN. The project pipeline uses 224 × 224 to provide sufficient visual detail while keeping computation manageable.

### Q3. Why did you normalize pixel values?

The original pixel values range from 0 to 255. Scaling them to 0–1 gives the neural network a smaller and more consistent numerical range, which helps optimization.

### Q4. Why was augmentation applied only to training images?

Training augmentation increases variation in the training data. Validation and test images should remain representative of the original data so that evaluation measures model performance fairly.

### Q5. Why did you use Dropout?

Dropout randomly deactivates some neurons during training. This reduces dependence on particular neurons and can help reduce overfitting.

### Q6. Why did you replace Flatten with GlobalAveragePooling2D?

The initial Flatten-based architecture produced approximately 12.94 million parameters, mainly because the flattened feature map was very large. GlobalAveragePooling2D reduced the representation to one value per feature map and reduced the final model to 110,276 parameters.

### Q7. Why did you use Softmax?

The problem has four mutually exclusive disease classes. Softmax converts the output values into a probability distribution across the four classes.

### Q8. Why did you use Sparse Categorical Crossentropy?

The labels were represented as integer class IDs from 0 to 3, so sparse categorical cross-entropy was appropriate.

### Q9. Did hyperparameter tuning improve the model?

No. The tested alternatives with an initial learning rate of 0.0005 achieved lower validation accuracy than the baseline. Therefore, the baseline configuration with learning rate 0.001 and dropout 0.5 was retained.

### Q10. Which class was most difficult?

Blast was the most difficult class based on test recall, with 81.25% recall. The main confusion was between Blast and Bacterial Blight.

### Q11. Which class performed best?

Tungro achieved 100% recall and an F1 score of 97.28% on the test set.

### Q12. What was the final test accuracy?

The final Custom CNN achieved 90.97% accuracy on the 720-image held-out test set.

### Q13. Can you say the model is 90.97% accurate in real farms?

No. The 90.97% result applies to this held-out test set. Real-world performance may differ because field conditions, varieties, cameras, lighting, backgrounds, and disease appearances can vary.

### Q14. Why did you keep the test set separate?

The test set provides an unbiased final estimate after model selection. Using it during tuning could cause information leakage and make the final evaluation overly optimistic.

---

## 19. Final Conclusion for Model 5

The Custom CNN successfully learned to classify the four rice leaf disease classes using raw RGB images. The selected model achieved **90.97% test accuracy** and **90.88% weighted F1 score** on the held-out test set.

The model showed particularly strong performance for Tungro, while Blast remained the most challenging class. Hyperparameter experiments did not improve upon the baseline, so the baseline configuration was retained as the final Custom CNN.

The Custom CNN now provides a completed deep-learning baseline for comparison with the next model, **MobileNetV2 transfer learning**.
