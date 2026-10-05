# Rice Leaf Disease Classification Using Deep Learning

## Project Progress Log

### Purpose
This document records the project work from the beginning of implementation through the current dataset-audit stage. It explains what was done, why it was done, what the code does, and what remains to be completed.

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

These notebooks are retained as evidence of individual contributions. A separate master pipeline is being built so the final experiments use one controlled and reproducible preprocessing workflow.

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

# Remove an earlier temporary extraction so the run starts clean.
if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

# Extract the unchanged raw ZIP into the Colab workspace.
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

## 7. Dataset Audit — File Scan

All image files were scanned into a DataFrame:

```python
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

image_records = []

for class_dir in sorted(EXTRACT_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for image_path in class_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            image_records.append({
                "image_path": str(image_path),
                "class_name": class_name
            })

dataset_df = pd.DataFrame(image_records)

print("Total image files:", len(dataset_df))
print("\nClass distribution:")
print(dataset_df["class_name"].value_counts())
```

### Current observed class counts
- Brownspot: 1,600
- Bacterialblight: 1,584
- Blast: 1,440
- Tungro: 1,308
- Total: 5,932

**Why:** This establishes the actual dataset size and class distribution that will be used as the basis for later analysis.

---

## 8. Class Distribution EDA

A bar chart was created:

```python
class_counts = (
    dataset_df["class_name"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")

plt.title("Rice Leaf Disease Class Distribution")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

**Why:** The chart makes class imbalance/variation visible and will support later discussion of class-wise precision, recall and F1-score.

---

## 9. Image Integrity and Metadata Audit

The images were opened with Pillow to collect dimensions and colour modes and to identify unreadable files:

```python
dimension_records = []
corrupt_images = []

for _, row in dataset_df.iterrows():
    image_path = row["image_path"]

    try:
        with Image.open(image_path) as img:
            dimension_records.append({
                "image_path": image_path,
                "class_name": row["class_name"],
                "width": img.width,
                "height": img.height,
                "mode": img.mode
            })

    except (UnidentifiedImageError, OSError):
        corrupt_images.append(image_path)

image_info_df = pd.DataFrame(dimension_records)
```

### Current observed results
- Readable images: **5,932**
- Corrupt/unreadable images: **0**
- RGB images: **5,776**
- RGBA images: **156**
- Most common dimension: **300 × 300** for **4,624** images
- Other image dimensions are also present.

**Why:** The model input requires consistent image representation, so image mode and dimensions must be audited before resizing/normalisation.

---

## 10. Current Project Status

Completed:
- Project workspace created.
- Raw dataset stored separately from working data.
- Google Colab connected to Drive.
- Dataset ZIP verified.
- Dataset extracted successfully.
- Four class folders verified.
- Class counts verified from the actual dataset.
- No corrupt/unreadable images found in the current audit.
- RGB/RGBA variation identified.
- Image-dimension variation identified.

### Not yet completed in the final pipeline
- Exact duplicate handling
- Final train/validation/test split
- Final feature-engineering branch
- Correlation analysis on training data only
- Outlier analysis on training data only
- Standardisation where required
- Deep-learning image preprocessing
- Six model implementations
- Hyperparameter tuning
- Cross-validation/validation protocol
- Final test evaluation
- Six-model comparison
- Ethics and bias analysis
- Final report and viva preparation

---

## 11. Planned Final Preprocessing Structure

The final pipeline will integrate the useful work from all six members without forcing every technique into a single incompatible sequence.

```text
Raw Dataset
    ↓
Dataset Audit
    ↓
Data Cleaning + Exact Duplicate Handling (Member 6)
    ↓
Label Setup + Stratified Split (Member 5)
    ↓
 ┌───────────────────────────────┬───────────────────────────────┐
 │ Classical ML Branch           │ Deep Learning Branch           │
 │                               │                                │
 │ Feature Engineering (M4)      │ RGB conversion                 │
 │ Correlation Analysis (M2)     │ Resize                         │
 │ Outlier Analysis (M3)         │ Normalisation                  │
 │ Standardisation (M1)          │ Training augmentation          │
 │                               │                                │
 │ SVM / RF / DT / MLP           │ Custom CNN / MobileNetV2       │
 └───────────────────────────────┴───────────────────────────────┘
```

The final implementation will avoid data leakage: decisions that learn from feature distributions (for example scaling or feature selection) will be fitted using training data and then applied to validation/test data.

---

## 12. Important Methodology Decisions

1. The raw ZIP remains unchanged.
2. Exact duplicate detection will occur before final splitting to reduce the risk of train/test leakage.
3. Perceptual-hash and feature-vector similarity will not be used to automatically delete large numbers of potentially valid images without verification.
4. Correlation and outlier analysis will be applied to engineered numerical features, not treated as direct raw-image operations.
5. CNN/MobileNetV2 will use image-based preprocessing rather than the handcrafted feature table.
6. Final numerical results will be generated from actual runs and will not be invented.

---

## 13. Next Step

The next implementation step is **data cleaning and exact duplicate analysis**, incorporating Member 6's Progress Review I contribution into the master pipeline. After that, the clean dataset will be split once and reused consistently across the final model experiments.
