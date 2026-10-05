# MLP Model Documentation — Rice Leaf Disease Classification

## 1. Purpose

This document explains the complete MLP experiment used in the Rice Leaf Disease Classification project. It records what the model is doing, why each step was used, how the model was tuned, and how the final model was selected.

**Notebook:** `04_MLP.ipynb`  
**Model:** Multi-Layer Perceptron (MLP)  
**Model category in this project:** Shallow, feature-based machine learning  
**Classes:** Bacterial Blight, Blast, Brown Spot, Tungro

> **Result integrity:** All numerical results below come from the completed MLP experiment. No performance values have been invented.

---

## 2. Where MLP Fits in the Project

The six-model plan is:

1. SVM
2. Random Forest
3. Decision Tree
4. MLP
5. Custom CNN
6. MobileNetV2

The first four models use the engineered classical-ML feature representation. The CNN-based models use image data directly.

For this project, the MLP is treated as a **shallow feature-based machine-learning model** because it has one hidden layer and receives nine engineered numerical features rather than raw image pixels.

An MLP does not necessarily have only one hidden layer. It can have multiple hidden layers. Our implementation specifically uses one hidden layer.

---

## 3. What Is an MLP?

MLP stands for **Multi-Layer Perceptron**. It is a feed-forward artificial neural network consisting of an input layer, one or more hidden layers, and an output layer.

Our final architecture is:

```text
9 engineered features
        ↓
1 hidden layer
64 neurons
ReLU
        ↓
4 output neurons
Softmax
        ↓
Bacterial Blight / Blast / Brown Spot / Tungro
```

Because our network has one hidden layer, it is a **shallow MLP**.

---

## 4. Why Did We Use an MLP?

The purpose was to investigate whether a neural-network classifier could learn useful non-linear relationships from the nine engineered features.

Using the same feature representation gives a fair comparison among:

```text
Nine engineered features
        ↓
SVM / Random Forest / Decision Tree / MLP
```

---

## 5. Data Used

The MLP uses the established classical-machine-learning pipeline.

- Final unique images: **4,794**
- Classes: **4**
- Engineered features: **9**

Established split:

| Split | Samples | Features |
|---|---:|---:|
| Training | 3,355 | 9 |
| Validation | 719 | 9 |
| Testing | 720 | 9 |

The test set was kept separate from model selection and hyperparameter tuning.

---

## 6. Feature Scaling

The engineered features have different numerical ranges, so `StandardScaler` was used before MLP training.

```text
Original features
       ↓
StandardScaler
       ↓
Scaled features
       ↓
MLP
```

The scaler was fitted using training data and then applied to validation and test data. This prevents information from the validation/test sets being used to learn scaling parameters.

Scaling is particularly useful for neural-network optimization because very different feature magnitudes can make optimization less efficient.

---

## 7. Understanding a Neuron

A neuron combines its inputs using learned weights and a bias.

```text
x1 ── w1 ──┐
x2 ── w2 ──┤
x3 ── w3 ──┤──► neuron
...        │
x9 ── w9 ──┘
```

The basic calculation is:

```text
z = w1x1 + w2x2 + ... + w9x9 + b
```

Where:

- `x` = input feature
- `w` = learned weight
- `b` = learned bias
- `z` = value before activation

---

## 8. Weights

A weight controls how strongly an input contributes to a neuron.

The model learns the weights during training rather than having them manually specified.

---

## 9. Bias

Bias is another learned parameter added to the weighted sum.

It gives the neuron additional flexibility when learning the relationship between input features and classes.

---

## 10. Activation Function — ReLU

Our hidden layer uses **ReLU**:

```text
ReLU(x) = max(0, x)
```

Therefore:

- negative input → 0
- positive input → positive value

ReLU introduces non-linearity so the network can learn non-linear relationships.

---

## 11. Hidden Layer

Our final network has exactly **one hidden layer**:

```text
Input layer:   9 features
       ↓
Hidden layer:  64 neurons + ReLU
       ↓
Output layer:  4 neurons + Softmax
```

The hidden layer learns combinations and relationships between the nine input features.

---

## 12. Why 64 Neurons?

The number of hidden neurons was treated as a hyperparameter.

The focused tuning experiment tested:

```text
(16,)
(32,)
(64,)
```

The best configuration was:

```text
hidden_layer_sizes = (64,)
```

Therefore, the final MLP contains one hidden layer with 64 neurons.

---

## 13. Forward Propagation

Forward propagation is the process of passing the input through the network to produce a prediction.

```text
9 input features
      ↓
Weighted sums + bias
      ↓
ReLU
      ↓
64 hidden neurons
      ↓
Output layer
      ↓
Softmax
      ↓
Class probabilities
```

In simple terms, forward propagation is how the model goes from input features to a prediction.

---

## 14. Output Layer

There are four possible disease classes, so the output layer contains four neurons:

```text
Neuron 1 → Bacterial Blight
Neuron 2 → Blast
Neuron 3 → Brown Spot
Neuron 4 → Tungro
```

Softmax converts the output values into probabilities.

---

## 15. Softmax

Softmax produces a probability distribution across the four classes.

For example:

```text
Bacterial Blight → 0.01
Blast            → 0.97
Brown Spot       → 0.01
Tungro           → 0.01
```

The probabilities approximately sum to 1, and the class with the highest probability becomes the prediction.

---

## 16. Loss Function

During training, the model needs a way to measure how different its predictions are from the true labels.

For this multi-class classification problem, the neural-network training objective uses a cross-entropy/log-loss objective.

Conceptually:

```text
Prediction
   ↓
Compare with true label
   ↓
Calculate loss
   ↓
Calculate gradients
   ↓
Update parameters
```

Lower loss generally indicates that predictions are becoming better aligned with the target labels.

---

## 17. Optimizer — Adam

Our MLP uses the **Adam** optimizer.

Simplified training cycle:

```text
Input
 ↓
Forward propagation
 ↓
Prediction
 ↓
Loss
 ↓
Gradients
 ↓
Adam updates weights
 ↓
Next iteration
```

Adam provides an adaptive method for updating neural-network parameters.

---

## 18. Learning Rate

The learning rate controls the size of parameter updates during optimization.

- Too large → training can become unstable.
- Too small → training can become unnecessarily slow.

The selected value was:

```text
learning_rate_init = 0.001
```

---

## 19. Batch Size

The batch size is the number of training samples processed for an optimization update.

Our model uses:

```text
batch_size = 32
```

Conceptually:

```text
32 samples
   ↓
update
   ↓
next 32 samples
   ↓
update
   ↓
...
```

---

## 20. Iterations and Convergence

An iteration in scikit-learn's `MLPClassifier` is associated with an optimization step based on a batch of training data. It should not automatically be treated as exactly the same thing as an epoch.

The baseline used:

```text
max_iter = 200
```

The tuned model allowed:

```text
max_iter = 500
```

The tuned model actually stopped after:

```text
397 iterations
```

so it did not simply stop because it reached the maximum iteration limit.

---

## 21. Baseline MLP

Baseline settings:

| Parameter | Value |
|---|---|
| Hidden layer | 32 neurons |
| Hidden layers | 1 |
| Activation | ReLU |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum iterations | 200 |
| Random state | 42 |

Baseline validation results:

| Metric | Result |
|---|---:|
| Accuracy | 0.973574 |
| Precision | 0.973779 |
| Recall | 0.973574 |
| F1 Score | 0.973511 |

---

## 22. Baseline Convergence Warning

The baseline produced:

```text
ConvergenceWarning:
Maximum iterations (200) reached and the optimization
hasn't converged yet.
```

This does **not** mean training failed. It means the model reached the configured maximum number of iterations before satisfying its convergence condition.

The baseline training-loss curve was still decreasing at iteration 200. Therefore, additional training was reasonable to investigate.

---

## 23. Convergence vs Overfitting

The convergence warning should not be described as evidence of overfitting.

The available evidence showed:

- training loss was decreasing;
- the model had not converged by 200 iterations;
- validation F1 was 0.973511.

A training-loss curve alone is not sufficient to prove overfitting. Overfitting should be assessed by comparing training and validation behaviour.

Therefore, the correct interpretation is:

> The baseline showed non-convergence within the 200-iteration limit.

---

## 24. Hyperparameter Tuning

An initial broad search was computationally excessive for iterative MLP training and was stopped before completion. It is **not treated as an experimental result**.

A focused grid was then used.

Tested values:

```text
hidden_layer_sizes:
(16,), (32,), (64,)

learning_rate_init:
0.001, 0.005

max_iter:
300, 500
```

Total configurations:

```text
3 × 2 × 2 = 12
```

With 5-fold cross-validation:

```text
12 × 5 = 60 fits
```

---

## 25. Cross-Validation

Five-fold stratified cross-validation was used during tuning.

Each configuration was trained and evaluated across five folds while maintaining class proportions as far as possible.

The purpose was to obtain a more reliable estimate of performance on unseen development data.

The test set was not used during hyperparameter tuning.

---

## 26. Best MLP Parameters

The focused search selected:

```text
hidden_layer_sizes = (64,)
learning_rate_init = 0.001
max_iter           = 500
```

Best cross-validation weighted F1:

```text
0.9914
```

Other main settings:

```text
activation  = ReLU
solver      = Adam
batch_size  = 32
alpha       = 0.0001
random_state = 42
```

---

## 27. Tuned Validation Results

| Metric | Tuned Validation |
|---|---:|
| Accuracy | 0.998609 |
| Precision | 0.998616 |
| Recall | 0.998609 |
| F1 Score | 0.998609 |

Comparison:

| Model | Accuracy | F1 |
|---|---:|---:|
| MLP Baseline | 0.973574 | 0.973511 |
| **MLP Tuned** | **0.998609** | **0.998609** |

The tuned model substantially improved over the baseline.

---

## 28. Tuned Model Convergence

The tuned model used:

```text
Maximum iterations = 500
Actual iterations   = 397
```

The training-loss curve showed a strong reduction in loss followed by a very small loss toward later iterations.

This supports the conclusion that the tuned model had better training/convergence behaviour than the baseline.

---

## 29. Tuned Validation Confusion Matrix

Validation matrix:

```text
                 Predicted
                BB  Blast Brown Tungro

Bacterial Blight 199  0    0     0
Blast               0 143   0     1
Brown Spot          0   0  180    0
Tungro              0   0    0   196
```

There was one validation error: a Blast sample was predicted as Tungro.

Correct predictions:

```text
199 + 143 + 180 + 196 = 718
```

Total validation samples:

```text
719
```

Therefore:

```text
718 / 719 = 0.998609
```

---

## 30. Final Model Selection

The tuned MLP was selected because it substantially outperformed the baseline on the fixed validation set.

Final configuration:

| Parameter | Value |
|---|---|
| Input features | 9 |
| Hidden layers | 1 |
| Hidden neurons | 64 |
| Hidden activation | ReLU |
| Output neurons | 4 |
| Output activation | Softmax |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum iterations | 500 |
| Actual iterations | 397 |

---

## 31. Final Test Evaluation

After model selection, the previously unseen 720-image test set was evaluated once.

Final test results:

| Metric | Test Result |
|---|---:|
| **Accuracy** | **0.9986** |
| **Precision** | **0.9986** |
| **Recall** | **0.9986** |
| **F1 Score** | **0.9986** |

Therefore, the final MLP achieved approximately **99.86% test accuracy**.

---

## 32. Final Test Confusion Matrix

```text
                 Predicted
                BB  Blast Brown Tungro

Bacterial Blight 199  0    0     0
Blast               1 143   0     0
Brown Spot          0   0  180    0
Tungro              0   0    0   197
```

There was one incorrect prediction out of 720 test samples.

The single error was:

```text
True class:      Blast
Predicted class: Bacterial Blight
```

Correct predictions:

```text
199 + 143 + 180 + 197 = 719
```

Therefore:

```text
719 / 720 = 0.998611...
```

which rounds to 0.9986.

---

## 33. Interpretation

The final MLP performed extremely well on the available test set.

Important observations:

1. The tuned model substantially improved over the baseline.
2. The tuned model converged before reaching the maximum iteration limit.
3. Only one test sample was misclassified.
4. Validation and test performance are very close.
5. Performance was strong across all four classes.

However, the high test score does not prove that the model will perform equally well on all real-world rice leaves. The dataset may not represent all possible field conditions, lighting, camera devices, backgrounds, disease stages, rice varieties, or geographic environments.

---

## 34. Strengths

### Strong predictive performance
The final MLP achieved 99.86% accuracy on the held-out test set.

### Non-linear learning
The ReLU hidden layer allows the network to learn non-linear relationships between engineered features.

### Compact representation
Only nine engineered features are required instead of the full image pixel array.

### Very few test errors
Only one of 720 test samples was misclassified.

---

## 35. Limitations

### Does not directly learn image structure
The MLP receives engineered features rather than raw images. It does not directly learn spatial patterns such as lesion shapes, spatial arrangements, edges, and image textures.

### Depends on feature engineering
Performance depends on the quality of the nine engineered features.

### Generalisation
The very high test performance may not represent performance on completely new field conditions.

### Class distribution
The classes are not perfectly equal in size. This should still be considered when discussing possible dataset bias.

### Deployment
A real agricultural diagnostic system would require testing on independent field images before practical deployment.

---

## 36. MLP vs CNN

### MLP

```text
Engineered features
       ↓
MLP
       ↓
Disease class
```

### CNN

```text
Raw image
    ↓
Convolution
    ↓
Feature maps
    ↓
Pooling
    ↓
Dense/output
    ↓
Disease class
```

The CNN learns image representations automatically. Our MLP works on a compact engineered feature representation.

---

# 37. Viva Questions and Answers

### What is an MLP?
An MLP is a feed-forward neural network containing an input layer, one or more hidden layers, and an output layer.

### Why is our MLP called multi-layer if it has only one hidden layer?
An MLP can have one or more hidden layers. Our implementation is a shallow MLP with one hidden layer.

### How many hidden layers does our final model have?
One.

### How many neurons are in the hidden layer?
64.

### Why did we use ReLU?
ReLU introduces non-linearity and is computationally simple.

### Why did we use Softmax?
Because this is a four-class classification problem with mutually exclusive classes, so softmax provides class probabilities.

### What is a weight?
A learned parameter controlling how strongly an input contributes to a neuron.

### What is bias?
A learned parameter added to the weighted sum before activation.

### What is forward propagation?
The process of passing the input through the network to calculate a prediction.

### What is the loss?
A numerical measure of the difference between predictions and true labels.

### What is an optimizer?
The method used to update model parameters to reduce the loss.

### Why Adam?
Adam provides an adaptive optimization method for updating neural-network parameters.

### What is the learning rate?
It controls the size of parameter updates during optimization.

### What is batch size?
The number of training samples processed for an optimization update.

### What is convergence?
A state where the optimization process has sufficiently stabilised according to the model's convergence criterion.

### Why did the baseline produce a convergence warning?
Because it reached the maximum of 200 iterations before satisfying the convergence condition.

### Why did we increase max_iter?
The baseline loss was still decreasing at 200 iterations, so additional training was reasonable to investigate.

### Why was 64 neurons selected?
It produced the best cross-validation weighted F1-score among the tested configurations.

### What was the best CV F1-score?
0.9914.

### What was the final test accuracy?
0.9986, approximately 99.86%.

### How many test samples were misclassified?
One out of 720.

### Did we tune using the test set?
No. The test set was kept untouched until final evaluation.

### Why is this important?
Using the test set during tuning would cause test-set leakage and make the final performance estimate less reliable.

### Is our MLP deep learning?
Our implementation is a shallow, one-hidden-layer MLP used in the feature-based machine-learning branch. CNN and MobileNetV2 are the deep-learning models in this project.

---

# 38. Final MLP Summary

```text
Final engineered features
          ↓
StandardScaler
          ↓
MLP baseline
          ↓
Validation evaluation
          ↓
Convergence issue identified
          ↓
Focused hyperparameter tuning
          ↓
5-fold stratified CV
          ↓
Best configuration selected
          ↓
Validation comparison
          ↓
Tuned MLP selected
          ↓
Final test evaluation
```

Final architecture:

```text
9 features
   ↓
64-neuron hidden layer
ReLU
   ↓
4-neuron output layer
Softmax
```

Final test performance:

```text
Accuracy  = 99.86%
Precision = 99.86%
Recall    = 99.86%
F1 Score  = 99.86%
```

---

# 39. Project Status After MLP

```text
SVM              ✅
Random Forest    ✅
Decision Tree    ✅
MLP              ✅
Custom CNN       ⏳
MobileNetV2      ⏳
```

The next stage is to develop the Custom CNN while keeping the established project decisions consistent.

---

# 40. Reproducibility Note

All numerical results in this document come from the completed MLP experiment.

They should not be replaced with estimated or assumed values.

If the notebook is rerun with different random states, library versions, hardware, or data changes, results may differ slightly. The values documented here correspond to the completed experiment used for the project.
