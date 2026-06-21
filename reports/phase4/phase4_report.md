# Phase 4 — Model Training & Evaluation Report

**Project:** Nassau Candy Distributor — Factory Reallocation & Shipping Optimization  
**Phase:** 4 — Model Training  
**Date:** 2026-06-21

---

## 1. Dataset Summary

| Metric | Value |
|---|---|
| Total Rows | 10,194 |
| Features | 25 |
| Target | adjusted_lead_time |
| Target Mean | 1320.84 |
| Target Std | 262.44 |

## 2. Train / Test Split

| Split | Rows | Share | Target Mean |
|---|---|---|---|
| Train | 7,135 | 70% | 1317.479 |
| Validation | 1,529 | 15% | 1327.743 |
| Test | 1,530 | 15% | 1329.628 |

**Strategy:** Stratified split on Ship Mode.

## 3. Model Comparison

| Model | Test RMSE | Test MAE | Test R² | Train Time | Overfit Gap |
|---|---|---|---|---|---|
| Linear Regression | 181.9032 | 180.9452 | 0.5188 | 0.005s | -0.0014 |
| Random Forest (Default) | 167.1296 | 142.5347 | 0.5938 | 0.762s | 0.3444 |
| Random Forest (Tuned) | 165.0710 | 144.2350 | 0.6037 | 1.579s | 0.2699 |
| Gradient Boosting (Default) | 169.2313 | 152.5189 | 0.5835 | 1.864s | 0.1164 |
| Gradient Boosting (Tuned) | 169.5719 | 148.2961 | 0.5818 | 2.884s | 0.2261 |
| XGBoost (Default) | 168.6643 | 152.5856 | 0.5863 | 0.131s | 0.1102 |
| XGBoost (Tuned) | 167.8341 | 150.6353 | 0.5904 | 0.332s | 0.1624 |

## 4. Winning Model

| Attribute | Value |
|---|---|
| Model | Random Forest (Tuned) |
| Test RMSE | 165.0710 |
| Test MAE | 144.2350 |
| Test R² | 0.6037 |
| 5-Fold CV R² | 0.6116 ± 0.0109 |
| Training Time | 1.579s |

## 5. Top 10 Features (Permutation Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | order_year | 1.1006 |
| 2 | order_weekday | 0.1801 |
| 3 | state_revenue_share | 0.0460 |
| 4 | order_month | 0.0417 |
| 5 | ship_mode | 0.0105 |
| 6 | ship_mode_ordinal | 0.0094 |
| 7 | region | 0.0051 |
| 8 | order_quarter | 0.0043 |
| 9 | is_holiday_season | 0.0037 |
| 10 | region_demand_share | 0.0016 |

## 6. SHAP Insights

| Rank | Feature | Mean |SHAP| | Direction |
|---|---|---|---|
| 1 | order_year | 181.6479 | ↑ INCREASES LT |
| 2 | order_weekday | 34.1831 | ~ MIXED |
| 3 | order_month | 9.6797 | ~ MIXED |
| 4 | state_revenue_share | 9.4343 | ~ MIXED |
| 5 | units | 6.4501 | ↑ INCREASES LT |
| 6 | distance_vs_closest_ratio | 3.4413 | ~ MIXED |
| 7 | ship_mode | 3.2282 | ~ MIXED |
| 8 | ship_mode_ordinal | 2.8614 | ↓ DECREASES LT |
| 9 | region | 2.6877 | ↑ INCREASES LT |
| 10 | order_quarter | 2.5185 | ↓ DECREASES LT |

## 7. Business Validation

| Ship Mode | Predicted LT | Actual LT | n |
|---|---|---|---|
| Same Day | 1358.92d | 1340.63d | 82 |
| First Class | 1342.54d | 1351.51d | 233 |
| Second Class | 1320.06d | 1331.69d | 297 |
| Standard Class | 1324.75d | 1322.43d | 918 |

**Verdict:** Ship Mode ordering NOT preserved.

## 8. Cross-Validation Results

| Fold | R² |
|---|---|
| Fold 1 | 0.6016 |
| Fold 2 | 0.5974 |
| Fold 3 | 0.6116 |
| Fold 4 | 0.6246 |
| Fold 5 | 0.6228 |
| **Mean** | **0.6116** |
| **Std** | **0.0109** |

## 9. Recommendation Engine Readiness

| Criterion | Assessment |
|---|---|
| Model accuracy sufficient? | YES (R²=0.6037) |
| Generalisation stable? | CAUTION (gap=0.2699) |
| CV consistent? | YES (σ=0.0109) |
| Business logic validated? | CHECK |
| SHAP explainability ready? | YES (25 features) |
| Suitable for scenario simulation? | YES |
