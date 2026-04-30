# Development Guide

This guide explains how to set up the project, run the main experiments, and keep the repository portfolio-ready.

## Environment

Use a virtual environment so the machine learning dependencies stay isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

The code was written as script-based experiments, so commands should be run from the repository root unless a model-specific section says otherwise.

## Dependencies

The required Python libraries are listed in [requirements.txt](requirements.txt). The main dependencies are:

- `numpy` and `pandas` for matrix and tabular data work.
- `scikit-learn` for PCA support, model training, and metrics.
- `xgboost` for gradient-boosted tree experiments.
- `scipy` for GMM/statistical support.
- `matplotlib` for result plots.
- `tqdm`, `imageio`, and `mdutils` for experiment utilities and generated reports.

## Data Artifacts

The model scripts expect NumPy matrix files in these directories:

- `music_matrices/`
- `Simplified Folder/`
- `top7_n3/`
- model-specific result folders such as `Random_Forest_Results/` and `XGBoost_Results/`

For a public portfolio repository, prefer committing a small representative dataset or clear generated outputs, then keeping large raw matrices, archives, logs, and cache files out of future commits. The `.gitignore` is set up for that direction.

## Reports

The repository includes three main reports, named with underscores for consistency:

- `Project_proposal.md`
- `Midterm_report.md`
- `Final_report.md`

These files document the project evolution, methodology, and final findings.

## Running Models

Random Forest from the repository root:

```bash
python3 scripts/RF_classifier.py
```

K-Means:

```bash
python3 Kmeans/kmeans_FINAL.py
```

Gaussian Mixture Model:

```bash
python3 GMM/Classifier.py
```

SVM:

```bash
python3 SVM/music_svm.py
```

XGBoost:

```bash
python3 XGBoost/XGBoost.py
```

The SVM experiments can take a long time on larger feature sets. Start with one reduced matrix before running every trial.

## Pipeline

```text
Audio feature matrices
    -> genre-label cleaning
    -> PCA contribution analysis
    -> reduced feature-set generation
    -> model training
    -> F1 / precision / recall / clustering evaluation
    -> plots and reports
```

## Project Organization

The current repository is organized by experiment family rather than as an installable Python package:

- `scripts/` contains shared cleaning, feature extraction, and the primary Random Forest run.
- `Kmeans/`, `GMM/`, `SVM/`, and `XGBoost/` contain model-specific experiments.
- `PCA/` contains dimensionality-reduction artifacts.
- `Random_Forest_Results/` and `XGBoost_Results/` contain historical experiment outputs.

A `src/` folder is not required for this portfolio version. The best future refactor would be:

```text
src/music_genre_classification/
├── data.py
├── features.py
├── models.py
└── metrics.py
```

That refactor would make the scripts thinner and easier to test, but it is not necessary before sharing the repository.

## Validation

Compile the main scripts after edits:

```bash
python3 -m py_compile scripts/DataCleaning.py scripts/Feature_Extraction.py scripts/RF_classifier.py scripts/test_extraction.py
```

Run a smaller model path first when possible:

```bash
python3 scripts/RF_classifier.py
```

Check that generated scores are between 0 and 1, that arrays have matching row counts, and that any generated plots appear in the expected folder.

## Portfolio Polish Checklist

- Keep `.gitignore`, `.editorconfig`, and `requirements.txt` committed.
- Use relative paths in scripts so the project runs outside one local machine.
- Keep report names consistent: `Project_proposal.md`, `Midterm_report.md`, and `Final_report.md`.
- Avoid committing generated cache files, local environment folders, logs, and one-off submission files.
- Prefer a small reproducible sample dataset over very large raw artifacts.
- Add a short "how to run" command for each model that still runs successfully.
