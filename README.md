# Nassau Candy Distributor

## Factory Reallocation & Shipping Optimization Recommendation System

A data-driven recommendation engine that analyzes 10,194 orders across 15 products and 5 factories to identify factory reallocation opportunities that reduce shipping distances and logistics costs.

---

## Key Results

| Metric | Value |
|---|---|
| Orders Analyzed | 10,194 |
| Recommendations Generated | 6,509 (63.9%) |
| Average Distance Reduction | 47.7% (1,557 km) |
| Total Route-KM Saved | 10.1 Million |
| Average Confidence | 89.9% |
| Strong Recommendations | 5,371 (52.7%) |

## Business Problem

Nassau Candy operates 5 production facilities serving 4 geographic regions. Historical order routing was not optimized for proximity, resulting in unnecessary shipping distances. This system identifies which products should shift production to a closer factory — while respecting division-level manufacturing constraints.

## Solution Approach

1. **Data Audit** — Cleaned and validated 10,194 orders
2. **Feature Engineering** — Built 25 predictive features (geographic, temporal, financial)
3. **Lead Time Modeling** — Trained Random Forest (R² = 0.6856) to predict delivery times
4. **Counterfactual Simulation** — Evaluated 10,234 factory-swap scenarios via batch prediction
5. **Composite Scoring** — 5-factor weighted score (distance, cost, lead time, utilization, risk)
6. **Interactive Dashboard** — Streamlit app for exploration and decision-making

**Key Insight:** Ship mode explains ~71.6% of lead time variance. Factory reallocation produces negligible lead time changes but significant distance reductions. The primary value is **logistics cost reduction through proximity optimization**.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run dashboard/app.py
```

Dashboard opens at `http://localhost:8501`

---

## Project Structure

```
nassau-candy-optimization/
├── dashboard/                 # Streamlit app (5 interactive tabs)
│   ├── app.py                 # Entry point
│   ├── components/theme.py    # Design system & UI components
│   └── utils/data_loader.py   # Cached data loading
├── data/
│   ├── raw/                   # Source data (immutable)
│   ├── processed/             # Cleaned & feature-engineered datasets
│   └── outputs/
│       ├── recommendations/   # Final recommendation CSVs
│       ├── simulations/       # Scenario matrix & summary JSON
│       └── model_results/     # Feature importance, model comparison
├── models/                    # Trained model artifacts (.joblib)
├── scripts/
│   ├── phase4/                # Model training pipeline
│   └── phase5/                # Simulation engine & validation
├── reports/
│   ├── phase1–5/              # Phase documentation
│   └── final/                 # Executive summary & checklist
├── docs/                      # Architecture, methodology, business rules
├── requirements.txt
└── .gitignore
```

## Dashboard Tabs

| Tab | Description |
|---|---|
| Executive Summary | KPIs, insight cards, factory workload, top opportunities |
| Factory Simulator | Filter by product/region/factory, side-by-side comparison |
| Product Reallocation | Full data table with filters and CSV export |
| Risk & Impact | Risk distribution, savings histogram, factory impact |
| Technical Appendix | Model metrics, feature importance, validation results |

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Visualization | Plotly |
| ML Framework | scikit-learn, XGBoost |
| Explainability | SHAP |
| Data | pandas, NumPy, PyArrow |
| Serialization | joblib |

## Winning Model

**Random Forest (Tuned)**

| Metric | Value |
|---|---|
| Test R² | 0.6856 |
| Test RMSE | 0.9997 |
| Test MAE | 0.8002 |
| 5-Fold CV R² | 0.7034 ± 0.0130 |
| Overfit Gap | 0.2249 |

Top features by SHAP importance: `ship_mode`, `ship_mode_ordinal`, `factory_region_distance_km`

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — Repository structure and data flow
- [`docs/methodology.md`](docs/methodology.md) — Phase 1–5 methodology
- [`docs/business_rules.md`](docs/business_rules.md) — Scoring, constraints, risk logic
- [`docs/data_dictionary.md`](docs/data_dictionary.md) — Column definitions
- [`reports/final/executive_summary.md`](reports/final/executive_summary.md) — Executive summary

## Future Improvements

- Integrate actual shipping cost data for dollar-denominated savings estimates
- Add temporal forecasting for demand shifts by region
- Implement A/B testing framework for gradual reallocation rollout
- Connect to live order system for real-time recommendations
- Add capacity constraint modeling per factory

---

*Built as a portfolio project demonstrating end-to-end data science: from raw data through modeling to production-ready interactive application.*
