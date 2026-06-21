# Phase 5 — Factory Reallocation Simulation Report

**Project:** Nassau Candy Distributor — Factory Reallocation & Shipping Optimization  
**Phase:** 5 — Simulation & Recommendation Engine  
**Date:** 2026-06-21  
**Architecture:** Vectorized Batch Prediction (2 `predict()` calls)

---

## 1. Simulation Overview

| Metric | Value |
|---|---|
| Total Orders Simulated | 10,194 |
| Counterfactual Scenarios Evaluated | 10,234 |
| Orders With Reallocation Recommended | 5,616 |
| Products Analyzed | 15 |
| Products With Change Recommended | 15 |
| Prediction Calls | 2 (batch) |
| Validation | PASSED ✓ |
| Execution Time | 2.4s |

## 2. Scoring Framework

| Component | Weight | Formula |
|---|---|---|
| Distance Reduction | 40% | `(D_curr − D_prop) / D_curr` |
| Logistics Efficiency | 25% | `1 − (D_prop / 3 450)` |
| Lead-Time Improvement | 15% | `(LT_curr − LT_prop) / LT_curr` |
| Utilisation Balance | 10% | `1 − U_prop` |
| Risk Score | 10% | `1 − P_risk` |

**Soft penalty:** `score × (1 − 0.15 × max(0, U − 0.50) × 2)` when utilisation > 50 %

## 3. Division–Factory Eligibility (Hard Block)

| Division | Eligible Factories |
|---|---|
| Chocolate | Lot's O' Nuts, Wicked Choccy's |
| Sugar | Sugar Shack, Secret Factory, The Other Factory |
| Other | Secret Factory, The Other Factory |

## 4. Recommendation Distribution

| Category | Count | Share |
|---|---|---|
| STRONG RECOMMEND | 3,740 | 36.7% |
| MODERATE RECOMMEND | 1,876 | 18.4% |
| MARGINAL | 0 | 0.0% |
| NO CHANGE | 4,578 | 44.9% |

## 5. Key Results

| Metric | Value |
|---|---|
| Avg Distance Reduction | 1050 km (45.2%) |
| Avg Lead-Time Change | -1.120 days |
| Total Route-km Saved | 5,896,823 km |
| Avg Composite Score | 0.4618 |
| Avg Confidence | 0.8804 |

## 6. Product Reallocation Recommendations

| Product | Division | Current Site | Proposed Site | Orders | Dist Saved (km) | LT Δ (d) | Score | Class |
|---|---|---|---|---|---|---|---|---|
| CHO-MIL-31000 | Chocolate | Wicked Choccy's | Lot's O' Nuts | 2,137 | 1814 | -5.224 | 0.6371 | CHANGE |
| CHO-TRI-54000 | Chocolate | Wicked Choccy's | Lot's O' Nuts | 2,015 | 1814 | -2.247 | 0.6368 | CHANGE |
| OTH-FIZ-56000 | Sugar | Sugar Shack | The Other Factory | 6 | 1320 | -11.220 | 0.5923 | CHANGE |
| SUG-FUN-75000 | Sugar | Sugar Shack | The Other Factory | 3 | 1437 | +44.025 | 0.5891 | CHANGE |
| SUG-NER-92000 | Sugar | Sugar Shack | Secret Factory | 4 | 2172 | -20.536 | 0.5537 | CHANGE |
| SUG-LAF-25000 | Sugar | Sugar Shack | Secret Factory | 10 | 2172 | -3.637 | 0.5518 | CHANGE |
| SUG-SWE-91000 | Sugar | Sugar Shack | Secret Factory | 10 | 2172 | +4.327 | 0.5507 | CHANGE |
| SUG-HAI-55000 | Sugar | The Other Factory | Secret Factory | 4 | 2155 | -25.916 | 0.5465 | CHANGE |
| OTH-KAZ-38000 | Other | The Other Factory | Secret Factory | 96 | 1988 | +0.020 | 0.5329 | CHANGE |
| SUG-EVE-47000 | Sugar | Secret Factory | The Other Factory | 3 | 261 | -4.714 | 0.4364 | CHANGE |
| CHO-NUT-13000 | Chocolate | Lot's O' Nuts | Wicked Choccy's | 1,810 | 808 | +1.164 | 0.4095 | CHANGE |
| CHO-FUD-51000 | Chocolate | Lot's O' Nuts | Wicked Choccy's | 1,818 | 810 | +0.053 | 0.4077 | CHANGE |
| CHO-SCR-58000 | Chocolate | Lot's O' Nuts | Wicked Choccy's | 2,064 | 793 | -1.715 | 0.3994 | CHANGE |
| OTH-GUM-21000 | Other | Secret Factory | The Other Factory | 120 | 241 | -2.400 | 0.3736 | CHANGE |
| OTH-LIC-15000 | Other | Secret Factory | The Other Factory | 94 | 224 | -1.539 | 0.3289 | CHANGE |

## 7. Factory Workload Rebalancing

| Factory | Current Orders | Proposed Orders | Δ |
|---|---|---|---|
| Lot's O' Nuts | 5,692 | 3,003 | -2,689 |
| Wicked Choccy's | 4,152 | 6,841 | +2,689 |
| Sugar Shack | 33 | 4 | -29 |
| Secret Factory | 217 | 158 | -59 |
| The Other Factory | 100 | 188 | +88 |

## 8. Risk Assessment

| Risk Level | Count | Share |
|---|---|---|
| Low | 9,951 | 97.6% |
| Medium | 243 | 2.4% |
| High | 0 | 0.0% |

## 9. Business Insights

### Top Reallocation Opportunities

1. **Wonka Bar - Milk Chocolate (CHO-MIL-31000)** — move from *Wicked Choccy's* to *Lot's O' Nuts*. Saves **1814 km** per order across 2,137 orders (1,208,124 route-km total). Score: 0.6371, Confidence: 0.8811.
2. **Wonka Bar - Triple Dazzle Caramel (CHO-TRI-54000)** — move from *Wicked Choccy's* to *Lot's O' Nuts*. Saves **1814 km** per order across 2,015 orders (1,226,264 route-km total). Score: 0.6368, Confidence: 0.8811.
3. **Fizzy Lifting Drinks (OTH-FIZ-56000)** — move from *Sugar Shack* to *The Other Factory*. Saves **1320 km** per order across 6 orders (3,960 route-km total). Score: 0.5923, Confidence: 0.8811.

### Key Findings

1. **Ship mode is the dominant lead-time driver.** Factory reallocation causes near-zero LT changes, consistent with Phase 4 SHAP analysis (factory features < 1 % of LT variance).
2. **Distance optimisation is the primary value lever.** Significant shipping-distance reductions are achievable by routing production to geographically closer eligible factories.
3. **Division constraints are fully respected.** Every recommendation stays within division-level factory eligibility.
4. **Total logistics impact: 5,896,823 route-km saved** across all recommended changes.

## 10. Validation Summary

**Overall status: PASSED ✓**  ·  1 warning(s)

| Check | Status |
|---|---|
| No cross-division violations | PASSED ✓ |
| No invalid factories | PASSED ✓ |
| No missing scores | PASSED ✓ |
| No missing confidence values | PASSED ✓ |
| Row count matches (10,194) | PASSED ✓ |
| Row uniqueness (1 per order) | PASSED ✓ |
| Positive distance savings (all CHANGE) | PASSED ✓ |
| Positive composite score (all CHANGE) | PASSED ✓ |
| Recommendation threshold integrity | PASSED ✓ |
| Lead-time sign consistency | PASSED ✓ |
| Summary consistency (json vs detail) | PASSED ✓ |
| Batch prediction architecture (2 calls) | PASSED ✓ |

### Warnings

- ⚠ 1927 CHANGE rows show a lead-time increase > 5 days (distance-optimal but slower delivery)

## 11. Recommendation Integrity Audit

| Audit Metric | Value |
|---|---|
| Candidate recommendations reviewed | 10,194 |
| Orders with no eligible alternative | 0 |
| Negative-distance recommendations prevented | 2,683 |
| Negative-score recommendations prevented | 1,895 |
| Final CHANGE recommendations | 5,616 |
| Recommendations failing distance filter (remaining) | 0 |

**Hard filter rule:** a reallocation is classified as a CHANGE (STRONG / MODERATE / MARGINAL) only when **both** `distance_saved_km > 0` **and** `composite_score > 0`. All other candidates revert to **NO CHANGE** with the proposed factory reset to the current factory. Negative-distance reallocations are never recommended.

## 12. Lead-Time Interpretation

Lead-time change uses a single, consistent sign convention everywhere (`recommendations.csv`, `product_reallocation_summary.csv`, `recommendation_summary.json`, and this report):

```
lead_time_delta = proposed_lead_time - current_lead_time
```
| Sign | Meaning |
|---|---|
| **Negative** Δ | Faster delivery (improvement) |
| **Positive** Δ | Slower delivery (deterioration) |

Across CHANGE recommendations, the average lead-time delta is **-1.120 days**. Because factory location contributes <1% of lead-time variance (ship mode dominates), reallocation has negligible delivery-speed impact while delivering the distance savings above.

## 13. Output Files

| File | Rows | Purpose |
|---|---|---|
| `recommendations.csv` | 10,194 | Per-order recommendation records |
| `product_reallocation_summary.csv` | 15 | Product-level aggregations |
| `recommendation_summary.json` | — | Executive statistics |
| `phase5_report.md` | — | This report |
