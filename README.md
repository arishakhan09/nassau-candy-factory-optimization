# Nassau Candy Distributor
## Factory Reallocation & Shipping Optimization Recommendation System

A data-driven recommendation engine that analyzes **10,194 orders** across 15 products and 5 production facilities to identify factory reallocation opportunities — reducing shipping distances and logistics costs while respecting manufacturing constraints.

---

## Project Overview

Nassau Candy Distributor operates five production facilities serving four U.S. geographic regions. Historical product-to-factory assignments were established operationally rather than optimized analytically, resulting in unnecessary shipping distances for a significant share of orders.

This project delivers a complete analytical pipeline — from raw order data through predictive modeling and counterfactual simulation to an interactive decision-support dashboard — that quantifies the reallocation opportunity and provides actionable, confidence-rated recommendations.

---

## Key Results

| Metric | Value |
|---|---|
| Orders Analyzed | 10,194 |
| Recommendations Generated | 6,509 (63.9%) |
| Strong Recommendations | 5,371 (52.7%) |
| Average Distance Reduction | 47.7% (1,557 km) |
| Total Route-KM Saved | 10.1 Million |
| Average Confidence | 89.9% |
| Winning Model R² | 0.6856 |

**Key Insight:** Ship mode explains ~71.6% of lead-time variance. Factory reallocation produces negligible lead-time changes but delivers significant distance reductions. The primary value is **logistics cost reduction through proximity optimization** — not delivery speed improvement.

---

## Business Problem

- Sub-optimal product routing inflates transportation costs
- Uneven factory utilization leaves capacity underused
- No existing decision tooling for systematic reallocation analysis
- Division-level manufacturing constraints (Chocolate, Sugar, Other) limit which factories can produce each product

---

## Methodology

```
Raw Dataset (10,194 orders)
    │
    ▼
Phase 4 — Model Training
    │   • Feature engineering (25 predictive features)
    │   • Random Forest, Gradient Boosting, XGBoost comparison
    │   • Winner: Random Forest (Tuned) — R² = 0.6856
    ▼
Phase 5 — Simulation Engine
    │   • 10,234 counterfactual scenarios (batch prediction)
    │   • 5-factor composite scoring (distance, cost, LT, utilization, risk)
    │   • Confidence calculation & recommendation classification
    ▼
Validation Audit
    │   • 7 integrity checks (all passed)
    │   • Sensitivity analysis (stable under ±5% weight variation)
    │   • Per-product robustness verification
    ▼
Interactive Dashboard (Streamlit)
    │   • 5 tabs: Executive Summary, Simulator, Products, Risk, Technical
    ▼
Final Reports & Deliverables
```

---

## Results Summary

### Recommendation Distribution

| Category | Count | Share |
|---|---|---|
| Strong Recommend | 5,371 | 52.7% |
| Moderate Recommend | 1,113 | 10.9% |
| Marginal | 25 | 0.2% |
| No Change | 3,685 | 36.1% |

### Factory Workload Rebalancing

| Factory | Current Orders | Proposed Orders | Change |
|---|---|---|---|
| Lot's O' Nuts | 5,692 | 4,105 | -1,587 |
| Wicked Choccy's | 4,152 | 5,739 | +1,587 |
| Sugar Shack | 33 | 3 | -30 |
| Secret Factory | 217 | 107 | -110 |
| The Other Factory | 100 | 240 | +140 |

---

## Repository Structure

```
├── dashboard/                     # Streamlit application (5 interactive tabs)
│   ├── app.py                     # Dashboard entry point
│   ├── components/theme.py        # Design system & UI components
│   └── utils/data_loader.py       # Cached data loading utilities
├── data/
│   ├── raw/                       # Source dataset (immutable)
│   │   └── Nassau Candy Distributor.csv
│   └── outputs/
│       ├── recommendations/       # recommendations.csv, product_reallocation_summary.csv
│       ├── simulations/           # recommendation_summary.json
│       └── model_results/         # model_comparison.csv, feature/shap importance
├── models/
│   └── model_features.json        # Feature configuration (tracked)
│   # winning_model.joblib          ← excluded (>100 MB, reproducible)
│   # label_encoders.joblib         ← excluded (reproducible)
├── scripts/
│   ├── phase4_model_training.py   # Full model training pipeline
│   ├── phase5_simulation_engine.py # Vectorized simulation & recommendation engine
│   └── validation_audit.py        # Integrity audit & dashboard readiness check
├── reports/
│   └── final/                     # Executive summary, project overview, checklist
├── requirements.txt               # Python dependencies (pinned)
├── .gitignore
└── README.md
```

---

## Model Artifacts

> **The trained model file is intentionally excluded from this repository.**

**Reason:** GitHub enforces a strict **100 MB file size limit**. The trained Random Forest model (`winning_model.joblib`) exceeds this limit. The repository is designed to be **fully reproducible** — all model artifacts can be regenerated from source code and the included dataset.

### Regenerating the trained model

```bash
python scripts/phase4_model_training.py
```

This recreates:
- `models/winning_model.joblib`
- `models/label_encoders.joblib`
- `data/outputs/model_results/model_comparison.csv`
- `data/outputs/model_results/feature_importance.csv`
- `data/outputs/model_results/shap_importance.csv`
- `reports/phase4/phase4_report.md`

### Regenerating recommendation outputs

```bash
python scripts/phase5_simulation_engine.py
```

This recreates:
- `data/outputs/recommendations/recommendations.csv`
- `data/outputs/recommendations/product_reallocation_summary.csv`
- `data/outputs/simulations/recommendation_summary.json`
- `reports/phase5/phase5_report.md`

### Regenerating validation reports

```bash
python scripts/validation_audit.py
```

This recreates:
- `reports/final/recommendation_integrity_audit.md`
- `data/outputs/audit/audit_results.json`
- `data/outputs/audit/dashboard_readiness.json`

---

## Reproducibility

This project is designed for **complete reproducibility**. Every published result can be regenerated from source code and the included raw dataset.

### Full regeneration workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (Phase 4) — generates model + feature importance artifacts
python scripts/phase4_model_training.py

# 3. Run simulation engine (Phase 5) — generates recommendations
python scripts/phase5_simulation_engine.py

# 4. Run validation audit — generates integrity reports
python scripts/validation_audit.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

### What is included in the repository

| Artifact | Included | Reason |
|---|---|---|
| Raw dataset (`Nassau Candy Distributor.csv`) | ✅ | Required input for all pipelines |
| Model feature config (`model_features.json`) | ✅ | Small JSON, defines feature lists |
| Output CSVs (recommendations, model results) | ✅ | Small files, enable dashboard without re-running pipelines |
| Summary JSON | ✅ | Dashboard dependency |
| Source scripts | ✅ | Core reproducibility |
| Dashboard code | ✅ | Presentation layer |
| Reports (Markdown) | ✅ | Documentation |
| Trained model (`.joblib`) | ❌ | Exceeds 100 MB — regenerated via `phase4_model_training.py` |
| Generated `.docx` reports | ❌ | Large binaries — regenerated via `generate_report.py` |
| Figure PNGs (`_report_figures/`) | ❌ | Large — regenerated on demand |

---

## Dashboard

### Quick Start

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Dashboard opens at `http://localhost:8501`

### Tabs

| Tab | Description |
|---|---|
| Executive Summary | KPIs, insight cards, factory workload, top opportunities |
| Factory Simulator | Filter by product/region/factory, distance comparison |
| Product Reallocation | Full data table with filters and CSV export |
| Risk & Impact | Risk distribution, savings histogram, factory impact |
| Technical Appendix | Model metrics, feature importance, validation results |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Visualization | Plotly |
| ML Framework | scikit-learn, XGBoost |
| Explainability | SHAP |
| Data Processing | pandas, NumPy |
| Serialization | joblib |
| Report Generation | python-docx, matplotlib |

---

## Winning Model

**Random Forest (Tuned)** — selected via systematic comparison of 7 model variants.

| Metric | Value |
|---|---|
| Test R² | 0.6856 |
| Test RMSE | 0.9997 |
| Test MAE | 0.8002 |
| 5-Fold CV R² | 0.7034 ± 0.0130 |
| Overfit Gap | 0.2249 |
| n_estimators | 500 |
| min_samples_leaf | 2 |

Top features by SHAP importance: `ship_mode_ordinal`, `ship_mode`, `order_year`, `order_month`

---

## License / Academic Use

This project was developed as a capstone / portfolio project demonstrating end-to-end applied data science: from raw data through predictive modeling, counterfactual simulation, and interactive application delivery.

If referencing this work in academic or professional contexts, please cite this repository.

---

*Prepared by Arisha Khan — June 2026*
