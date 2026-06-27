# Insurance Claim Settlement Bias Analysis

A Streamlit dashboard that analyses potential bias in insurance death-claim settlements
using descriptive statistics, diagnostic Chi-square tests, and four tuned classification
models.

## Features
- Cross-tabulation analysis vs POLICY_STATUS
- Zone / age / income / payment-mode bias detection (Chi-square)
- Frequency-encoded feature engineering (20 features)
- KNN, Decision Tree, Random Forest, Gradient Boosted with RandomizedSearchCV tuning
- ROC curves, confusion matrices, FP%/FN% breakdown
- Feature importance for tree-based models

## Quickstart (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `Insurance.csv` via the sidebar.

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository (public or private).
2. Go to https://share.streamlit.io → **New app**.
3. Set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs requirements automatically.
5. Once live, upload `Insurance.csv` in the sidebar.

## Project Structure

```
insurance_project/
├── app.py               # Streamlit dashboard (main entry point)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── Insurance.csv        # Source dataset (1,790 claims)
└── plots/               # Pre-generated PNGs for reference
    ├── 01_crosstab_descriptive.png
    ├── 02_team_zone_bias.png
    ├── 03_age_income_bias.png
    ├── 04_bias_heatmaps.png
    ├── 05_model_performance.png
    ├── 06_roc_curves.png
    ├── 07_confusion_matrices.png
    ├── 08_fp_fn_breakdown.png
    └── 09_feature_importance.png
```

## Key Findings

- **PENINSULAR zone** (n≈500): only 23% approval vs 68% overall — critical bias flag
- **Payment mode gap**: Single payment (90%) vs Quarterly (45%) — 44.9pp difference
- **Best model**: Gradient Boosted (AUC ≈ 0.80), train-test gap ≈ 7.6pp after tuning
- All biases statistically significant (Chi-square p < 0.0001)
