# Music Genre Classification

Machine learning pipeline for classifying music genres from structured audio features. The project compares supervised and unsupervised approaches, shows the effect of label cleaning and PCA-based feature selection, and documents the tradeoffs between model performance and training cost.

## Walkthrough

Project video: https://www.loom.com/share/21ea57275cb0444d979809301fc0c632

## Highlights

- Processed audio-feature matrices derived from approximately 15,000 tracks.
- Cleaned inconsistent genre labels into a standardized 19-genre target set.
- Compared K-Means, Gaussian Mixture Models, Random Forest, Support Vector Machine, and XGBoost.
- Used PCA contribution scores to test reduced feature sets across multiple model families.
- Produced evaluation artifacts for F1, precision, recall, clustering stability, and feature contribution analysis.

## Results Summary

The strongest supervised results came from SVM and XGBoost, with XGBoost reaching the best reported F1 score of 0.55 on the tested feature sets. The results also show that genre classification is sensitive to label quality, class balance, and feature selection. Cleaning noisy labels and reducing feature dimensionality improved performance compared with the raw feature matrix.

## Project Structure

```text
.
├── README.md
├── DEVELOPMENT.md
├── requirements.txt
├── .editorconfig
├── .gitignore
├── Project_proposal.md
├── Midterm_report.md
├── Final_report.md
├── figures/
├── scripts/
├── Kmeans/
├── GMM/
├── SVM/
├── XGBoost/
├── PCA/
├── Matrix/
├── music_matrices/
├── Simplified Folder/
├── Random_Forest_Results/
└── XGBoost_Results/
```

The three main written reports are:

- [Project_proposal.md](Project_proposal.md)
- [Midterm_report.md](Midterm_report.md)
- [Final_report.md](Final_report.md)

## Core Files

| Path | Purpose |
| --- | --- |
| `scripts/DataCleaning.py` | Standardizes raw genre labels and filters low-signal labels. |
| `scripts/Feature_Extraction.py` | Selects top audio features from PCA contribution scores. |
| `scripts/RF_classifier.py` | Random Forest training and macro F1/recall/precision evaluation. |
| `SVM/music_svm.py` | SVM classifier experiments on reduced feature matrices. |
| `XGBoost/XGBoost.py` | XGBoost classifier experiment. |
| `GMM/Classifier.py` | Gaussian Mixture Model clustering experiments. |
| `Kmeans/kmeans_FINAL.py` | K-Means clustering experiments and result generation. |
| `PCA/` | PCA outputs, contribution scores, plots, and supporting analysis files. |

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a representative model:

```bash
python3 scripts/RF_classifier.py
```

Run other model experiments:

```bash
python3 Kmeans/kmeans_FINAL.py
python3 GMM/Classifier.py
python3 SVM/music_svm.py
python3 XGBoost/XGBoost.py
```

Some experiments use precomputed `.npy` matrices and can take a long time to train, especially SVM on larger feature sets.

## Methods

1. Build a numeric feature matrix from audio metadata.
2. Standardize genre labels by grouping spelling variants, synonyms, and related subgenres.
3. Use PCA contribution scores to identify compact feature subsets.
4. Train and compare clustering and classification models.
5. Evaluate results with F1, precision, recall, and supporting visualizations.

## Repository Notes

This repository intentionally keeps the current flat layout because most scripts are experiment-oriented and reference data artifacts by relative path. A `src/` package would be useful as a future refactor once the shared cleaning, feature-selection, and model-training logic is consolidated into reusable modules. For the current portfolio version, the more important polish is clear documentation, reproducible setup, relative paths, and keeping generated or oversized artifacts out of future commits.

## Tech Stack

- Python
- NumPy
- pandas
- scikit-learn
- XGBoost
- SciPy
- Matplotlib
- R for the original PCA analysis files
