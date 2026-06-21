# Project Overview

**Nassau Candy Distributor — Factory Reallocation & Shipping Optimization**

---

## 1. Objective

Build a recommendation system that identifies factory reallocation opportunities for 15 products across 5 factories and 4 regions, optimizing for shipping distance reduction while respecting division-level manufacturing constraints.

## 2. Scope

- **Input:** 10,194 historical orders with product, factory, region, ship mode, and lead time data
- **Output:** Per-order recommendation (STRONG / MODERATE / MARGINAL / NO CHANGE) with confidence scores
- **Constraints:** Products can only be reallocated within their division's eligible factory set

## 3. Phase Breakdown

| Phase | Deliverable | Status |
|---|---|---|
| Phase 1 | Data Audit & Quality Assessment | ✅ Complete |
| Phase 2 | Feature Engineering (41 → 25 features) | ✅ Complete |
| Phase 3 | Lead Time Analysis & Validation | ✅ Complete |
| Phase 4 | Model Training (7 models, RF winner) | ✅ Complete |
| Phase 5 | Simulation Engine (10,234 scenarios) | ✅ Complete |
| Phase 6 | Interactive Dashboard (5 tabs) | ✅ Complete |

## 4. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Visualization | Plotly |
| ML Framework | scikit-learn, XGBoost |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Serialization | joblib, JSON, Parquet |

## 5. Data Flow

```
Raw CSV (Nassau Candy Distributor.csv)
    ↓
Phase 1: Audit → cleaned_dataset.csv
    ↓
Phase 2: Feature Engineering → modeling_dataset.csv
    ↓
Phase 3: Lead Time Validation
    ↓
Phase 4: Model Training → winning_model.joblib
    ↓
Phase 5: Simulation → recommendations.csv, product_reallocation_summary.csv
    ↓
Phase 6: Dashboard → app.py (Streamlit)
```

## 6. Key Design Decisions

1. **Batch prediction architecture** — Only 2 model.predict() calls for all 10,234 scenarios (not row-by-row)
2. **Division-level hard blocks** — Chocolate products can only go to chocolate-capable factories
3. **Composite scoring** — Multi-factor score prevents single-dimension optimization
4. **Confidence modeling** — Combines distance signal strength, model R², and data volume
5. **Risk flagging** — Explicit penalties for minority factories, distance increases, cross-border farther

## 7. Validation Results

All 7 integrity checks passed:
- No cross-division violations
- No invalid factories
- No missing scores or confidence values
- Row count integrity (10,194)
- Row uniqueness verified
- Batch prediction architecture confirmed

## 8. Repository Structure

See `docs/architecture.md` for full folder layout and component relationships.

---

*For methodology details, see `docs/methodology.md`. For business rules, see `docs/business_rules.md`.*
