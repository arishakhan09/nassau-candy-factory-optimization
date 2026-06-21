# Submission Checklist

**Project:** Nassau Candy Distributor — Factory Reallocation & Shipping Optimization  
**Date:** June 2026

---

## Deliverables

| # | Deliverable | Location | Status |
|---|---|---|---|
| 1 | Raw source data | `data/raw/` | ✅ |
| 2 | Cleaned dataset | `data/processed/cleaned_dataset.csv` | ✅ |
| 3 | Modeling dataset | `data/processed/modeling_dataset.csv` | ✅ |
| 4 | Trained model | `models/winning_model.joblib` | ✅ |
| 5 | Label encoders | `models/label_encoders.joblib` | ✅ |
| 6 | Model features config | `models/model_features.json` | ✅ |
| 7 | Recommendations CSV | `data/outputs/recommendations/recommendations.csv` | ✅ |
| 8 | Product summary | `data/outputs/recommendations/product_reallocation_summary.csv` | ✅ |
| 9 | Simulation summary | `data/outputs/simulations/recommendation_summary.json` | ✅ |
| 10 | Scenario matrix | `data/outputs/simulations/scenario_matrix.parquet` | ✅ |
| 11 | Model comparison | `data/outputs/model_results/model_comparison.csv` | ✅ |
| 12 | Feature importance | `data/outputs/model_results/feature_importance.csv` | ✅ |
| 13 | SHAP importance | `data/outputs/model_results/shap_importance.csv` | ✅ |
| 14 | Interactive dashboard | `dashboard/app.py` | ✅ |
| 15 | Phase reports (1–5) | `reports/phase1/` through `reports/phase5/` | ✅ |
| 16 | Executive summary | `reports/final/executive_summary.md` | ✅ |
| 17 | Documentation | `docs/` | ✅ |
| 18 | Requirements file | `requirements.txt` | ✅ |
| 19 | README | `README.md` | ✅ |

## Quality Checks

| Check | Result |
|---|---|
| Dashboard launches without errors | ✅ |
| All data files load correctly | ✅ |
| No broken imports | ✅ |
| No hardcoded absolute paths | ✅ |
| All scripts use project-relative paths | ✅ |
| .gitignore present | ✅ |
| No __pycache__ in repo | ✅ |
| No credentials or secrets | ✅ |
| README has run instructions | ✅ |
| requirements.txt pinned | ✅ |

## How to Verify

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch dashboard
streamlit run dashboard/app.py

# 3. Verify all 5 tabs render correctly
#    - Executive Summary
#    - Factory Simulator
#    - Product Reallocation
#    - Risk & Impact
#    - Technical Appendix
```

---

*Prepared for portfolio submission.*
