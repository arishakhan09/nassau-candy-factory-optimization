# Submission Checklist

**Project:** Nassau Candy Distributor — Factory Reallocation & Shipping Optimization  
**Date:** June 2026

---

## Deliverables

| # | Deliverable | Location | Status |
|---|---|---|---|
| 1 | Raw source data | `data/raw/` | ✅ |
| 2 | Trained model | `models/winning_model.joblib` | ✅ |
| 3 | Label encoders | `models/label_encoders.joblib` | ✅ |
| 4 | Model features config | `models/model_features.json` | ✅ |
| 5 | Recommendations CSV | `data/outputs/recommendations/recommendations.csv` | ✅ |
| 6 | Product summary | `data/outputs/recommendations/product_reallocation_summary.csv` | ✅ |
| 7 | Simulation summary | `data/outputs/simulations/recommendation_summary.json` | ✅ |
| 8 | Model comparison | `data/outputs/model_results/model_comparison.csv` | ✅ |
| 9 | Feature importance | `data/outputs/model_results/feature_importance.csv` | ✅ |
| 10 | SHAP importance | `data/outputs/model_results/shap_importance.csv` | ✅ |
| 11 | Audit results | `data/outputs/audit/audit_results.json` | ✅ |
| 12 | Dashboard readiness | `data/outputs/audit/dashboard_readiness.json` | ✅ |
| 13 | Interactive dashboard | `dashboard/app.py` | ✅ |
| 14 | Phase 5 report | `reports/phase5/phase5_report.md` | ✅ |
| 15 | Recommendation integrity audit | `reports/final/recommendation_integrity_audit.md` | ✅ |
| 16 | Executive summary | `reports/final/executive_summary.md` | ✅ |
| 17 | Project overview | `reports/final/project_overview.md` | ✅ |
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
