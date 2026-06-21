# Executive Summary

**Project:** Nassau Candy Distributor — Factory Reallocation & Shipping Optimization  
**Status:** Complete  
**Date:** June 2026

---

## Business Problem

Nassau Candy operates 5 production facilities serving 4 geographic regions. Historical order routing was not optimized for shipping distance, resulting in unnecessary logistics costs. The question: *Can we systematically identify which products should be manufactured at which factories to minimize total shipping distance?*

## Solution

A data-driven recommendation engine that evaluates every product-factory-region combination and recommends reallocations where composite benefit (distance, cost, lead time, utilization, risk) exceeds defined thresholds.

## Key Results

| Metric | Value |
|---|---|
| Orders Analyzed | 10,194 |
| Counterfactual Scenarios Evaluated | 10,234 |
| Recommendations Generated | 5,616 (55.1%) |
| No Change Decisions | 4,578 (44.9%) |
| Strong Recommendations | 3,740 (36.7%) |
| Moderate Recommendations | 1,876 (18.4%) |
| Average Distance Reduction | 45.2% (1,050 km per order) |
| Total Route-KM Saved | 5.90 Million |
| Average Confidence | 88.0% |
| Model Accuracy (R²) | 0.6037 |
| Validation Score | 96 / 100 |
| Dashboard Readiness Score | 94 / 100 |

## Architecture

- **Prediction Model:** Random Forest (Tuned), 25 features, stratified train/val/test split (R² = 0.6037, RMSE = 165.07) — a predictive lead-time scoring model used for comparative scenario evaluation
- **Simulation Engine:** Vectorized batch prediction (2 predict calls total)
- **Scoring:** 5-component weighted composite (distance 40%, cost 25%, lead time 15%, utilization 10%, risk 10%)
- **Validation:** Recommendation integrity PASSED, business rule compliance PASSED
- **Dashboard:** Streamlit, 5-tab interactive interface — deployment ready

## Key Findings

1. Factory reassignment successfully reduces logistics distance — average distance reduction is 45.2%.
2. Total logistics savings exceed 5.89 million route-km across all recommended changes.
3. Recommendations satisfy all factory eligibility constraints — no cross-division violations detected.
4. Recommendation integrity validation passed. Dashboard readiness validation passed.

## Model Limitations

Lead time prediction is primarily driven by Ship Mode. SHAP analysis and Phase 5 validation both indicate that factory reassignment has limited impact on lead time compared with shipping method selection. As a result, the recommendation engine should be interpreted primarily as a **logistics-efficiency optimization system** rather than a lead-time optimization system. Distance reduction is the primary source of business value. Lead-time improvements should not be overstated.

## Key Insight

The final tuned Random Forest model achieved a Test R² of 0.6037 with an RMSE of 165.07, providing sufficient predictive accuracy for factory reallocation scenario analysis and recommendation generation.

Ship Mode is the dominant driver of lead time. Factory reallocation produces negligible lead-time changes (average −1.12 days) but significant distance reductions. The primary business value is **logistics cost reduction through proximity optimization**, not delivery speed improvement.

## Recommendation

Implement the proposed factory reallocations for products with STRONG RECOMMEND classification (3,740 orders). Expected annual savings depend on per-km logistics cost, but at conservative estimates, the 5.90M route-km reduction represents substantial operational efficiency gains. Start with the chocolate division shifts (the largest per-order savings), then stage the 1,876 moderate recommendations over one quarter before full commitment.

---

*Technical details in `reports/phase5/` and `reports/final/recommendation_integrity_audit.md`.*
