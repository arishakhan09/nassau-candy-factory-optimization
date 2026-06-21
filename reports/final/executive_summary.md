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
| Recommendations Generated | 6,509 (63.9%) |
| Average Distance Reduction | 47.7% (1,557 km per order) |
| Total Route-KM Saved | 10.1 Million |
| Average Confidence | 89.9% |
| Strong Recommendations | 5,371 (52.7%) |
| Model Accuracy (R²) | 0.6856 |

## Architecture

- **Prediction Model:** Random Forest (Tuned), 25 features, stratified train/val/test split
- **Simulation Engine:** Vectorized batch prediction (2 predict calls total)
- **Scoring:** 5-component weighted composite (distance 40%, cost 25%, lead time 15%, utilization 10%, risk 10%)
- **Validation:** 7 integrity checks, all passed
- **Dashboard:** Streamlit, 5-tab interactive interface

## Key Insight

Ship Mode explains ~71.6% of lead time variance. Factory reallocation produces negligible lead time changes but significant distance reductions. The primary business value is **logistics cost reduction through proximity optimization**, not delivery speed improvement.

## Recommendation

Implement the proposed factory reallocations for products with STRONG RECOMMEND classification (5,371 orders). Expected annual savings depend on per-km logistics cost, but at conservative estimates, the 10.1M km reduction represents substantial operational efficiency gains.

---

*Full methodology in `docs/methodology.md`. Technical details in `reports/phase4/` and `reports/phase5/`.*
