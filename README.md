# Bitcoin Market Regime Detection using Unsupervised Machine Learning

## Project Overview

This project investigates whether Bitcoin market regimes can be identified using unsupervised machine learning techniques.

The analysis combines market, on-chain and network-related variables and compares different feature engineering approaches, dimensionality reduction methods and clustering algorithms.

The main objective is to determine which methodology produces the most economically meaningful Bitcoin market regimes.

---

For reasons of exploration and higher comprehensibility, the project is made completely in Jupyter Notebooks. In order to run it please run notebooks 01 to 04 in that order. If you happen to run a later one first, or an earlier one after running a later one, please start again by running 01 to 04, as the results change.

## Project Structure

```text
unsupervised_crypto/

├── data/
│   ├── raw/
│   |    └── bitcoin_data_unsupervised_ml.csv
│   └── processed/
│
├── figures/
│
├── notebooks/
│   ├── 00_introduction.ipynb
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   └── 04_clustering.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow

### 0. Introduction

General Introduction to the project, topic and the goal.

### 1. Data Exploration

* Initial data inspection
* Missing value analysis
* Correlation analysis
* Variable selection

Notebook:

```text
01_exploration.ipynb
```

---

### 2. Feature Engineering

Three different feature sets were created:

**Dataset A**

* 30-day moving average
* 90-day growth rate

**Dataset B**

* 30-day moving average
* 180-day growth rate

**Dataset C**

* 365-day rolling Z-Score

Notebook:

```text
02_preprocessing_features.ipynb
```

---

### 3. Dimensionality Reduction

Principal Component Analysis (PCA) was applied to reduce dimensionality while preserving more than 90% of the original variance.

Notebook:

```text
03_dimensionality_reduction.ipynb
```

---

### 4. Clustering

The following clustering methods were evaluated:

* K-Means
* Gaussian Mixture Models (GMM)
* HDBSCAN

The final model selection was based on clustering metrics, visual inspection and economic interpretability.

Notebook:

```text
04_clustering.ipynb
```

---

## Main Findings

* Feature engineering had, with exceptions, a larger impact on results than the clustering algorithm itself.
* The rolling Z-Score feature set (Dataset C) produced the most interpretable market regimes.
* K-Means with four clusters generated the most economically meaningful regime structure.
* GMM produced comparable but generally less interpretable results.
* HDBSCAN failed to identify stable market regimes.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Author

Robert Puselja

University of Lucerne

Unsupervised Machine Learning Project

AI was used for coding, not for thinking.
