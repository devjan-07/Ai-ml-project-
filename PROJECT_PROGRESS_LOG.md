# Rice Leaf Disease Classification Using Deep Learning

## Project Progress Log

### Purpose
This document records the project work from the beginning of implementation through the current preprocessing stage. It explains what was done, why it was done, what the code does, and what remains to be completed.

---

## 1. Project Overview

**Problem:** Multi-class image classification of rice leaf diseases.

**Classes:**
- Bacterial Blight
- Blast
- Brown Spot
- Tungro

**Dataset:** Rice Leaf Disease Image Dataset (Kaggle).

**Project direction:** The project will evaluate six individual models, followed by a group comparison. The current model plan is:
1. SVM
2. Random Forest
3. Decision Tree
4. MLP/ANN
5. Custom CNN
6. MobileNetV2

This plan is aligned with the final implementation requirement that each group member implements at least one model and that the group compares six individual models.

---

## 2. Project Environment and Storage

### Software
The main execution environment is **Google Colab**, using Python and the project libraries required by the assignment:
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

### Google Drive structure
```text
Rice_Leaf_Disease_Project/
├── 01_Raw_Dataset/
│   └── rice leaf diseases dataset.zip
├── 02_Notebooks/
├── 03_Results/
└── 04_Report/
```

The raw ZIP is kept unchanged as the source copy. The working extraction is performed in Colab temporary storage so the raw source is not modified.

---

## 3. Previous Progress Review I Work

The group completed six individual preprocessing/EDA notebooks for Progress Review I:

| Member | Main contribution |
|---|---|
| Member 1 | Feature standardization/scaling |
| Member 2 | Correlation and multicollinearity analysis |
| Member 3 | Statistical outlier analysis |
| Member 4 | Feature engineering |
| Member 5 | Label encoding and stratified splitting |
| Member 6 | Data cleaning and duplicate detection |

These notebooks are retained as evidence of individual contributions.

**Important:** these six notebooks are **not six separate preprocessing pipelines that should be blindly chained together**. A controlled master pipeline integrates the relevant contributions while maintaining separate branches for classical ML and image-based deep learning.

---

## 4. Master Pipeline Setup

A new notebook was created:

`Rice_Leaf_Disease_Final_Pipeline.ipynb`

### Drive connection
```python
from google.colab import drive

drive.mount("/content/drive")
```

**Why:** This mounts the project Google Drive inside Colab so the notebook can read the raw dataset and save results to the shared project folders.

### Project paths
```python
from pathlib import Path

PROJECT_DIR = Path(
    "/content/drive/MyDrive/Rice_Leaf_Disease_Project"
)

RAW_DIR = PROJECT_DIR / "01_Raw_Dataset"
NOTEBOOK_DIR = PROJECT_DIR / "02_Notebooks"
RESULTS_DIR = PROJECT_DIR / "03_Results"
REPORT_DIR = PROJECT_DIR / "04_Report"
```

**Why:** Centralising paths makes the notebook easier to reproduce and reduces hard-coded path errors.

### Dataset verification
```python
DATASET_ZIP = RAW_DIR / "rice leaf diseases dataset.zip"

print("Dataset exists:", DATASET_ZIP.exists())
print("Dataset path:", DATASET_ZIP)
```

The dataset file was successfully found in the expected raw-data directory.

---

## 5. ZIP Structure Verification

Before extraction, the ZIP contents were inspected:

```python
import zipfile

with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
    files = zip_ref.namelist()

print("Total files/folders inside ZIP:", len(files))

for item in files[:30]:
    print(item)
```

### Verified result
The ZIP contains **5,932 entries**, and the image data is organised into the four expected class folders:
- `Bacterialblight`
- `Blast`
- `Brownspot`
- `Tungro`

**Why:** We verify the real structure instead of assuming the folder layout before writing preprocessing code.

---

## 6. Dataset Extraction

The ZIP was extracted into Colab temporary storage:

```python
import shutil
from pathlib import Path

EXTRACT_DIR = Path("/content/rice_leaf_dataset")

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Dataset extracted to:")
print(EXTRACT_DIR)
```

### Verified top-level folders
```text
Bacterialblight
Blast
Brownspot
Tungro
```

**Why:** Keeping extraction in `/content` avoids duplicating thousands of image files in Google Drive while preserving the raw ZIP as the project source.

---

## 7. Initial Dataset Audit

All image files were scanned into a DataFrame.

### Original dataset counts

- Total image files: **5,932**
- Brownspot: **1,600**
- Bacterialblight: **1,584**
- Blast: **1,440**
- Tungro: **1,308**

A class-distribution bar chart was also created.

**Why:** This establishes the actual dataset size and class distribution before preprocessing.

---

## 8. Initial Image Integrity and Metadata Audit

The original dataset was opened with Pillow to inspect dimensions and colour modes.

### Initial audit result

- Readable images: **5,932**
- Corrupt/unreadable images: **0**
- RGB images: **5,776**
- RGBA images: **156**
- Most common dimension: **300 × 300** for **4,624** images
- All files inspected in this stage were JPG files.

**Why:** Image mode and dimension variation must be understood before constructing the final model input pipeline.

---

# 9. Exact Duplicate Analysis

The next stage incorporated Member 6's data-cleaning contribution.

A content-based hash was calculated for each image file. Images with the same hash were treated as exact binary duplicates.

### Results

- **Total unique file hashes:** 4,794
- **Exact duplicate groups:** 1,096
- **Images involved in exact duplicates:** 2,234
- **Exact duplicate groups appearing across multiple classes:** 0
- **Redundant duplicate files:** 1,138

### Duplicate group-size distribution

| Group size | Number of groups |
|---:|---:|
| 2 | 1,078 |
| 3 | 6 |
| 5 | 12 |

### Interpretation

The raw dataset contains substantial exact duplication. Because identical images in different train/test subsets could cause data leakage, duplicate handling is performed **before the final split**.

There were **no exact duplicate groups spanning multiple disease classes**, so the duplicate analysis did not identify a direct cross-class labelling conflict.

The raw ZIP remains unchanged. The modelling dataset keeps one representative image for each exact-hash group.

**Important limitation:** SHA-256 identifies exact binary duplicates only. Visually similar images saved differently are not automatically considered duplicates.

---

# 10. Class Distribution After Exact Deduplication

After retaining one representative per exact-hash group, the working dataset contains **4,794 images**.

| Class | Count | Percentage |
|---|---:|---:|
| Bacterialblight | 1,326 | 27.66% |
| Blast | 960 | 20.03% |
| Brownspot | 1,200 | 25.03% |
| Tungro | 1,308 | 27.28% |
| **Total** | **4,794** | **100%** |

### Interpretation

The classes are not perfectly equal, but the distribution is reasonably manageable. Blast is the smallest class at approximately 20%, while Bacterialblight and Tungro are approximately 27%.

The class distribution is retained through stratified splitting so that validation and test sets represent the same class proportions.

---

# 11. Final Stratified Train / Validation / Test Split

The deduplicated dataset was split once using stratification.

### Final split

| Subset | Images |
|---|---:|
| Training | **3,355** |
| Validation | **719** |
| Testing | **720** |
| **Total** | **4,794** |

### Class distribution

**Training**

| Class | Count | Percentage |
|---|---:|---:|
| Bacterialblight | 928 | 27.66% |
| Blast | 672 | 20.03% |
| Brownspot | 840 | 25.04% |
| Tungro | 915 | 27.27% |

**Validation**

| Class | Count | Percentage |
|---|---:|---:|
| Bacterialblight | 199 | 27.68% |
| Blast | 144 | 20.03% |
| Brownspot | 180 | 25.03% |
| Tungro | 196 | 27.26% |

**Test**

| Class | Count | Percentage |
|---|---:|---:|
| Bacterialblight | 199 | 27.64% |
| Blast | 144 | 20.00% |
| Brownspot | 180 | 25.00% |
| Tungro | 197 | 27.36% |

### Leakage checks

- Train–Validation overlap: **0**
- Train–Test overlap: **0**
- Validation–Test overlap: **0**

**Why:** A single final split is created and reused across the six models so their performance can be compared fairly.

---

# 12. Post-Deduplication Image Audit

A second image audit was performed on the **4,794-image modelling dataset**.

### Integrity

- Total images inspected: **4,794**
- Valid images: **4,794**
- Invalid/corrupted images: **0**
- Image format: **4,794 JPG**

### Width statistics

| Statistic | Width (px) |
|---|---:|
| Mean | 315.150 |
| Standard deviation | 59.051 |
| Minimum | 209 |
| Median | 300 |
| Maximum | 603 |

### Height statistics

| Statistic | Height (px) |
|---|---:|
| Mean | 315.101 |
| Standard deviation | 59.083 |
| Minimum | 209 |
| Median | 300 |
| Maximum | 603 |

### Most common dimensions

**3,486 images are exactly 300 × 300.**

Other dimensions occur in smaller groups, including rectangular images.

### Aspect ratio statistics

| Statistic | Aspect ratio |
|---|---:|
| Mean | 1.019 |
| Standard deviation | 0.202 |
| Minimum | 0.664 |
| Median | 1.000 |
| Maximum | 1.507 |

Using the audit range of **0.75–1.33**, there are **1,153 images with unusual aspect ratios**.

These images are **not automatically treated as corrupted**. Aspect-ratio variation will be handled by the image preprocessing strategy.

### Channel distribution

| Channels | Images |
|---:|---:|
| 3 | 4,650 |
| 4 | 144 |

The final image-loading pipeline will convert images to a consistent **3-channel RGB** representation.

---

# 13. Final Preprocessing Structure

The six members' work will be integrated into one controlled methodology rather than applying every technique sequentially to every image.

```text
Raw Dataset
    ↓
Dataset Audit
    ↓
Data Cleaning + Exact Duplicate Handling
(Member 6)
    ↓
Label Setup + One Stratified Split
(Member 5)
    ↓
        ┌───────────────────────────────┬───────────────────────────────┐
        │ Classical ML Branch           │ Deep Learning Branch           │
        │                               │                                │
        │ Feature Engineering (M4)      │ RGB conversion                 │
        │ Correlation Analysis (M2)     │ Aspect-ratio handling          │
        │ Outlier Analysis (M3)         │ Resize + padding               │
        │ Standardisation (M1)          │ Model-specific normalisation  │
        │                               │ Training augmentation          │
        │ SVM / RF / DT / MLP           │ Custom CNN / MobileNetV2       │
        └───────────────────────────────┴───────────────────────────────┘
```

### Why the branches are separate

Standardisation, correlation analysis, outlier analysis and engineered numerical features are appropriate for the classical ML feature representation.

CNN and MobileNetV2 are image-based models and should receive image tensors rather than having tabular preprocessing techniques forced onto the raw image pixels.

This keeps the six-member contributions while maintaining technically appropriate preprocessing for each model family.

---

# 14. Leakage-Control Decisions

1. The raw ZIP remains unchanged.
2. Exact duplicates are handled before the final split.
3. The final stratified split is created once and reused.
4. Learned transformations such as scaling or feature-selection decisions are fitted using training data and then applied to validation/test data.
5. Validation and test images will not receive random training augmentation.
6. The same test set will be reserved for final comparison.
7. Perceptual-hash or feature-similarity deletion will not be performed automatically without evidence.
8. CNN/MobileNetV2 will use image-based preprocessing.
9. Classical ML models will use the appropriate engineered numerical representation.

---

# 15. Current Project Status

## Completed

- Project workspace created.
- Raw dataset stored separately from working data.
- Google Colab connected to Drive.
- Dataset ZIP verified.
- Dataset extracted successfully.
- Four class folders verified.
- Original class counts verified.
- Initial image integrity audit completed.
- Exact duplicate analysis completed.
- Deduplicated dataset established: **4,794 images**.
- Post-dedup class distribution calculated.
- Final stratified train/validation/test split created.
- Train/validation/test overlap checks completed.
- Post-dedup image integrity audit completed.
- Image dimensions and aspect ratios analysed.
- Channel distribution analysed.
- Six-member contribution structure documented.
- Six-model final structure documented.

## Remaining preprocessing work

- Implement final RGB conversion.
- Implement aspect-ratio-preserving resize/padding.
- Implement model-specific normalisation.
- Implement training-only augmentation.
- Build the TensorFlow input pipeline.
- Validate tensor shapes and pixel ranges.
- Visualise original versus preprocessed/augmented samples.
- Save reproducible preprocessing outputs/configuration.

## Remaining project work

- Implement six final models.
- Hyperparameter tuning.
- Model validation/model-selection protocol.
- Final test evaluation.
- Accuracy, precision, recall and F1-score.
- Confusion matrices.
- Classification reports.
- Six-model comparison.
- Overfitting/generalisation analysis.
- Bias, ethics and limitations.
- Final report.
- Final presentation.
- Viva preparation.
- AI usage declaration.

---

# 16. Next Step

The immediate next step is to complete the **final image preprocessing pipeline**:

```text
Deduplicated images
        ↓
Train / Validation / Test split
        ↓
RGB conversion
        ↓
Aspect-ratio-preserving resize + padding
        ↓
224 × 224 × 3
        ↓
Training augmentation
(training only)
        ↓
Model-specific normalisation
        ↓
Batching + prefetching
        ↓
Tensor / visual validation
        ↓
Model training
```

**No final model training should begin until this preprocessing pipeline has been validated.**
