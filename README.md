# California Housing Price Prediction

## Project Overview

This repository contains exploratory analysis and data pipeline notes for predicting median house values in California block groups using the California Housing dataset. The work focuses on data understanding, preprocessing (normalization), and building models to predict `MedHouseValue`.

## Repository Structure

- `data_cleaning.ipynb` — Notebook for cleaning and EDA.
- `test.ipynb` — Sandbox/testing notebook.
- `dataset/california_housing.csv` — Raw dataset (20640 rows, 8 features).
- `docs/data pipeline.md` — Data pipeline notes and insights.
- `plots/` — Generated figures and visualizations.

## Dataset

The dataset includes the following numeric features:

1. `MedInc` — Median income in block group
2. `HouseAge` — Median house age in block group
3. `AveRooms` — Average number of rooms per household
4. `AveBedrms` — Average number of bedrooms per household
5. `Population` — Block group population
6. `AveOccup` — Average household size
7. `Latitude` — Block group latitude
8. `Longitude` — Block group longitude

Target: `MedHouseValue` (median house value)

## Getting Started

Prerequisites: Python 3.8+ and a virtual environment are recommended.

Install dependencies (example):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt  # if you create one
```

Open the notebooks with Jupyter or VS Code: `data_cleaning.ipynb` contains the main EDA and preprocessing steps.

## Usage

- Run the cleaning and EDA notebook to reproduce figures and preprocessing.
- Ensure normalization uses parameters computed from the training split only (see `docs/data pipeline.md`).

## Notes & Best Practices

- Always compute normalization (mean/std) on the training set and apply to validation/test sets to avoid data leakage.
- Geographic features (`Latitude`, `Longitude`) can provide signal (e.g., proximity to coast or premium areas).

## Related Documents

- Data pipeline notes: [docs/data pipeline.md](docs/data%20pipeline.md)

---
Created for intuition development, manual modeling and learning by doing.
