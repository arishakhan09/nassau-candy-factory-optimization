"""
Phase 4 — Model Training & Evaluation Pipeline
Nassau Candy Distributor: Factory Reallocation & Shipping Optimization

Recreated from surviving artifacts. Builds the modeling dataset from raw data,
trains 7 model variants, selects the winner, and generates all Phase 4 artifacts.

Artifacts produced:
    models/winning_model.joblib
    models/label_encoders.joblib
    models/model_features.json
    data/outputs/model_results/model_comparison.csv
    data/outputs/model_results/feature_importance.csv
    data/outputs/model_results/shap_importance.csv
    data/processed/modeling_dataset.csv
    reports/phase4/phase4_report.md
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ====================================================================
# 1. CONFIGURATION
# ====================================================================

warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Paths (relative to project root) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Nassau Candy Distributor.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_RESULTS_DIR = PROJECT_ROOT / "data" / "outputs" / "model_results"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "phase4"

# --- Target & Split ---
TARGET = "adjusted_lead_time"
TEST_SIZE = 0.30
VAL_SHARE = 0.50  # of the temp split -> 15% val, 15% test
RANDOM_STATE = 42

# --- Factory Reference Data ---
FACTORY_ASSIGNMENTS: dict[str, str] = {
    "CHO-FUD-51000": "Lot's O' Nuts",
    "CHO-MIL-31000": "Wicked Choccy's",
    "CHO-NUT-13000": "Lot's O' Nuts",
    "CHO-SCR-58000": "Lot's O' Nuts",
    "CHO-TRI-54000": "Wicked Choccy's",
    "OTH-FUN-09000": "Secret Factory",
    "OTH-JB-24000":  "The Other Factory",
    "OTH-NER-38000": "The Other Factory",
    "OTH-RUN-44000": "Secret Factory",
    "SUG-CAN-65000": "Sugar Shack",
    "SUG-GOB-74000": "Sugar Shack",
    "SUG-GUM-16000": "Sugar Shack",
    "SUG-JAW-87000": "Sugar Shack",
    "SUG-LOL-80000": "Sugar Shack",
    "SUG-SOU-02000": "Sugar Shack",
}

FACTORY_COORDS: dict[str, tuple[float, float]] = {
    "Lot's O' Nuts":     (40.7128, -74.0060),   # New York area
    "Wicked Choccy's":   (29.7604, -95.3698),   # Houston area
    "Sugar Shack":       (34.0522, -118.2437),   # LA area
    "Secret Factory":    (41.8781, -87.6298),    # Chicago area
    "The Other Factory": (47.6062, -122.3321),   # Seattle area
}

REGION_CENTROIDS: dict[str, tuple[float, float]] = {
    "Interior": (39.50, -98.35),
    "Atlantic":  (35.75, -79.00),
    "Gulf":      (30.25, -89.50),
    "Pacific":   (37.50, -120.00),
}

SHIP_MODE_ORDINAL: dict[str, int] = {
    "Standard Class": 1,
    "Second Class": 2,
    "First Class": 3,
    "Same Day": 4,
}

# --- Feature Sets ---
CATEGORICAL_COLS = ["ship_mode", "region", "division", "product_id", "factory"]

FEATURES_A = [
    # Categorical (label-encoded)
    "ship_mode", "region", "division", "product_id", "factory",
    # Numeric — original
    "units",
    # Temporal
    "order_month", "order_quarter", "order_weekday", "order_year", "is_holiday_season",
    # Financial
    "profit_margin_pct", "profit_per_unit", "cost_per_unit",
    # Product
    "product_popularity_rank", "product_revenue_rank", "product_profit_rank",
    # Geographic
    "region_demand_share", "state_revenue_share", "is_cross_border", "ship_mode_ordinal",
    # Factory / Distance
    "factory_utilization_score", "factory_region_distance_km",
    "closest_factory_distance_km", "distance_vs_closest_ratio",
]

FEATURES_B = [
    "ship_mode", "region", "division", "product_id", "factory",
    "order_month", "order_quarter", "order_weekday", "order_year", "is_holiday_season",
    "product_popularity_rank", "product_revenue_rank", "product_profit_rank",
    "region_demand_share", "state_revenue_share", "is_cross_border", "ship_mode_ordinal",
    "factory_utilization_score", "factory_region_distance_km",
    "closest_factory_distance_km", "distance_vs_closest_ratio",
]

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4")


# ====================================================================
# 2. DATA LOADING
# ====================================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw CSV and standardise column names."""
    log.info("Loading raw data from %s", path.name)
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("/", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    log.info("  Loaded %d rows × %d columns", *df.shape)
    return df


# ====================================================================
# 3. FEATURE ENGINEERING
# ====================================================================

def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised haversine distance in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all engineered features from raw columns — fully vectorised."""
    log.info("Engineering features...")
    out = df.copy()

    # --- Target: adjusted lead time ---
    out["order_date"] = pd.to_datetime(out["order_date"], format="mixed", dayfirst=True)
    out["ship_date"] = pd.to_datetime(out["ship_date"], format="mixed", dayfirst=True)
    out["adjusted_lead_time"] = (out["ship_date"] - out["order_date"]).dt.days

    # --- Temporal features ---
    out["order_month"] = out["order_date"].dt.month
    out["order_quarter"] = out["order_date"].dt.quarter
    out["order_weekday"] = out["order_date"].dt.weekday
    out["order_year"] = out["order_date"].dt.year
    out["is_holiday_season"] = out["order_month"].isin([11, 12]).astype(int)

    # --- Financial features ---
    out["profit_margin_pct"] = np.where(
        out["sales"] > 0,
        (out["gross_profit"] / out["sales"]) * 100,
        0.0,
    )
    out["profit_per_unit"] = np.where(
        out["units"] > 0,
        out["gross_profit"] / out["units"],
        0.0,
    )
    out["cost_per_unit"] = np.where(
        out["units"] > 0,
        out["cost"] / out["units"],
        0.0,
    )

    # --- Product ranking features (vectorised rank) ---
    product_stats = out.groupby("product_id").agg(
        total_units=("units", "sum"),
        total_revenue=("sales", "sum"),
        total_profit=("gross_profit", "sum"),
    )
    product_stats["product_popularity_rank"] = product_stats["total_units"].rank(ascending=False).astype(int)
    product_stats["product_revenue_rank"] = product_stats["total_revenue"].rank(ascending=False).astype(int)
    product_stats["product_profit_rank"] = product_stats["total_profit"].rank(ascending=False).astype(int)
    out = out.merge(
        product_stats[["product_popularity_rank", "product_revenue_rank", "product_profit_rank"]],
        on="product_id",
        how="left",
    )

    # --- Geographic features ---
    total_revenue = out["sales"].sum()
    region_rev = out.groupby("region")["sales"].sum()
    out["region_demand_share"] = out["region"].map(region_rev / total_revenue)

    state_rev = out.groupby("state_province")["sales"].sum()
    out["state_revenue_share"] = out["state_province"].map(state_rev / total_revenue)

    out["is_cross_border"] = (out["country_region"] != "United States").astype(int)

    # --- Ship mode ordinal ---
    out["ship_mode_ordinal"] = out["ship_mode"].map(SHIP_MODE_ORDINAL)

    # --- Factory assignment ---
    out["factory"] = out["product_id"].map(FACTORY_ASSIGNMENTS)

    # --- Factory coordinates (vectorised via map) ---
    out["factory_lat"] = out["factory"].map(lambda f: FACTORY_COORDS[f][0])
    out["factory_lon"] = out["factory"].map(lambda f: FACTORY_COORDS[f][1])
    out["region_lat"] = out["region"].map(lambda r: REGION_CENTROIDS[r][0])
    out["region_lon"] = out["region"].map(lambda r: REGION_CENTROIDS[r][1])

    # --- Factory-region distance (vectorised haversine) ---
    out["factory_region_distance_km"] = _haversine_km(
        out["factory_lat"].values, out["factory_lon"].values,
        out["region_lat"].values, out["region_lon"].values,
    ).round(1)

    # --- Closest factory distance ---
    all_factory_names = list(FACTORY_COORDS.keys())
    all_factory_lats = np.array([FACTORY_COORDS[f][0] for f in all_factory_names])
    all_factory_lons = np.array([FACTORY_COORDS[f][1] for f in all_factory_names])

    # Broadcast: (n_orders, 1) vs (1, n_factories) -> (n_orders, n_factories)
    region_lats = out["region_lat"].values[:, np.newaxis]
    region_lons = out["region_lon"].values[:, np.newaxis]
    factory_lats_bc = all_factory_lats[np.newaxis, :]
    factory_lons_bc = all_factory_lons[np.newaxis, :]

    dist_matrix = _haversine_km(region_lats, region_lons, factory_lats_bc, factory_lons_bc)
    out["closest_factory_distance_km"] = dist_matrix.min(axis=1).round(1)
    closest_idx = dist_matrix.argmin(axis=1)
    out["closest_factory_name"] = [all_factory_names[i] for i in closest_idx]

    # --- Distance vs closest ratio ---
    out["distance_vs_closest_ratio"] = np.where(
        out["closest_factory_distance_km"] > 0,
        (out["factory_region_distance_km"] / out["closest_factory_distance_km"]).round(4),
        1.0,
    )

    # --- Factory utilization score ---
    factory_order_counts = out["factory"].value_counts()
    total_orders = len(out)
    out["factory_utilization_score"] = out["factory"].map(
        factory_order_counts / total_orders
    ).round(4)

    log.info("  Engineered %d features total (%d rows)", len(out.columns), len(out))
    return out


# ====================================================================
# 4. ENCODING
# ====================================================================

def encode_categoricals(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns. Returns encoded df and encoder dict."""
    log.info("Encoding %d categorical columns...", len(cols))
    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in cols:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))
        encoders[col] = le
        log.info("  %s: %d classes → %s", col, len(le.classes_), list(le.classes_))
    return encoded, encoders


# ====================================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ====================================================================

def split_data(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    stratify_col: str = "ship_mode",
) -> dict[str, Any]:
    """Stratified 70/15/15 split. Returns dict with X/y splits."""
    log.info("Splitting data (70/15/15, stratify=%s)...", stratify_col)
    X = df[features]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df[stratify_col],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SHARE, random_state=RANDOM_STATE,
        stratify=X_temp[stratify_col] if stratify_col in X_temp.columns else None,
    )

    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        log.info("  %s: %d rows (%.1f%%) — mean=%.3f, std=%.3f",
                 name, len(ys), len(ys) / len(df) * 100, ys.mean(), ys.std())

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# ====================================================================
# 6 & 7. MODEL TRAINING + HYPERPARAMETER TUNING
# ====================================================================

def _evaluate(
    name: str,
    model: Any,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> dict[str, Any]:
    """Train, batch-predict, and compute metrics."""
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    pred_tr = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_te = model.predict(X_test)

    result = {
        "model_name": name,
        "fitted_model": model,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_tr))),
        "train_mae":  float(mean_absolute_error(y_train, pred_tr)),
        "train_r2":   float(r2_score(y_train, pred_tr)),
        "val_rmse":   float(np.sqrt(mean_squared_error(y_val, pred_val))),
        "val_mae":    float(mean_absolute_error(y_val, pred_val)),
        "val_r2":     float(r2_score(y_val, pred_val)),
        "test_rmse":  float(np.sqrt(mean_squared_error(y_test, pred_te))),
        "test_mae":   float(mean_absolute_error(y_test, pred_te)),
        "test_r2":    float(r2_score(y_test, pred_te)),
        "train_time_s": round(elapsed, 3),
        "overfit_gap":  float(r2_score(y_train, pred_tr) - r2_score(y_test, pred_te)),
    }

    flag = "⚠ OVERFIT" if result["overfit_gap"] > 0.05 else "✓ OK"
    log.info(
        "  %-30s  RMSE=%.4f  MAE=%.4f  R²=%.4f  gap=%.4f  %s  (%.1fs)",
        name, result["test_rmse"], result["test_mae"], result["test_r2"],
        result["overfit_gap"], flag, elapsed,
    )
    return result


def _tune_candidates(
    model_cls: type,
    configs: list[dict],
    splits: dict,
    fixed_kwargs: dict | None = None,
) -> tuple[Any, dict]:
    """Evaluate explicit HP configs on validation set. Return best model + config."""
    fixed_kwargs = fixed_kwargs or {}
    best_rmse = float("inf")
    best_model = None
    best_cfg = None

    for i, cfg in enumerate(configs):
        m = model_cls(**cfg, **fixed_kwargs)
        m.fit(splits["X_train"], splits["y_train"])
        val_pred = m.predict(splits["X_val"])
        rmse = float(np.sqrt(mean_squared_error(splits["y_val"], val_pred)))
        r2 = float(r2_score(splits["y_val"], val_pred))
        log.info("    Config %d: %s → Val RMSE=%.4f, R²=%.4f", i + 1, cfg, rmse, r2)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = m
            best_cfg = cfg

    log.info("  Best config: %s (Val RMSE=%.4f)", best_cfg, best_rmse)
    return best_model, best_cfg


def train_all_models(splits: dict) -> dict[str, dict]:
    """Train 4 default + 3 tuned model variants. Returns results dict."""
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    X_v, y_v = splits["X_val"], splits["y_val"]
    X_te, y_te = splits["X_test"], splits["y_test"]
    results: dict[str, dict] = {}

    # ── Baseline ──
    log.info("Training baseline models...")
    results["Linear Regression"] = _evaluate(
        "Linear Regression", LinearRegression(),
        X_tr, y_tr, X_v, y_v, X_te, y_te,
    )

    results["Random Forest (Default)"] = _evaluate(
        "Random Forest (Default)",
        RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        X_tr, y_tr, X_v, y_v, X_te, y_te,
    )

    results["Gradient Boosting (Default)"] = _evaluate(
        "Gradient Boosting (Default)",
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE),
        X_tr, y_tr, X_v, y_v, X_te, y_te,
    )

    results["XGBoost (Default)"] = _evaluate(
        "XGBoost (Default)",
        XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                     random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        X_tr, y_tr, X_v, y_v, X_te, y_te,
    )

    # ── Tuned Random Forest ──
    log.info("Tuning Random Forest...")
    rf_configs = [
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 3},
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 4},
        {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 8},
    ]
    best_rf, best_rf_cfg = _tune_candidates(
        RandomForestRegressor, rf_configs, splits,
        fixed_kwargs={"random_state": RANDOM_STATE, "n_jobs": -1},
    )
    results["Random Forest (Tuned)"] = _evaluate(
        "Random Forest (Tuned)", best_rf,
        X_tr, y_tr, X_v, y_v, X_te, y_te,
    )

    # ── Tuned Gradient Boosting ──
    log.info("Tuning Gradient Boosting...")
    gbr_configs = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8},
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.9},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.9},
        {"n_estimators": 300, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.85},
    ]
    best_gbr, _ = _tune_candidates(
        GradientBoostingRegressor, gbr_configs, splits,
        fixed_kwargs={"random_state": RANDOM_STATE},
    )
    results["Gradient Boosting (Tuned)"] = _evaluate(
        "Gradient Boosting (Tuned)", best_gbr,
        X_tr, y_tr, X_v, y_v, X_te, y_te,
    )

    # ── Tuned XGBoost ──
    log.info("Tuning XGBoost...")
    xgb_configs = [
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 300, "learning_rate": 0.08, "max_depth": 3, "subsample": 0.9, "colsample_bytree": 0.9},
    ]
    best_xgb, _ = _tune_candidates(
        XGBRegressor, xgb_configs, splits,
        fixed_kwargs={"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0},
    )
    results["XGBoost (Tuned)"] = _evaluate(
        "XGBoost (Tuned)", best_xgb,
        X_tr, y_tr, X_v, y_v, X_te, y_te,
    )

    return results


# ====================================================================
# 8. MODEL EVALUATION & WINNER SELECTION
# ====================================================================

MODEL_ORDER = [
    "Linear Regression",
    "Random Forest (Default)", "Random Forest (Tuned)",
    "Gradient Boosting (Default)", "Gradient Boosting (Tuned)",
    "XGBoost (Default)", "XGBoost (Tuned)",
]


def select_winner(results: dict[str, dict]) -> tuple[str, dict]:
    """Select winner by lowest test RMSE among tuned candidates + baseline."""
    candidates = [n for n in MODEL_ORDER if "Tuned" in n or n == "Linear Regression"]
    best_name = min(candidates, key=lambda n: results[n]["test_rmse"])
    log.info(
        "WINNER: %s  (Test RMSE=%.4f, R²=%.4f)",
        best_name, results[best_name]["test_rmse"], results[best_name]["test_r2"],
    )
    return best_name, results[best_name]


# ====================================================================
# 9. FEATURE IMPORTANCE
# ====================================================================

def compute_feature_importance(
    model: Any,
    features: list[str],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Built-in + permutation importance. Returns DataFrame."""
    log.info("Computing feature importance...")
    data: dict[str, Any] = {"Feature": features}

    if hasattr(model, "feature_importances_"):
        data["BuiltInImportance"] = [
            round(model.feature_importances_[features.index(f)], 6) for f in features
        ]

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    data["PermutationImportance"] = [round(v, 6) for v in perm.importances_mean]
    data["PermutationStd"] = [round(v, 6) for v in perm.importances_std]

    df = pd.DataFrame(data).sort_values("PermutationImportance", ascending=False)

    log.info("  Top 5 (Permutation):")
    for _, row in df.head(5).iterrows():
        log.info("    %-30s  %.4f ± %.4f", row["Feature"],
                 row["PermutationImportance"], row["PermutationStd"])
    return df


# ====================================================================
# 10. SHAP ANALYSIS
# ====================================================================

def compute_shap(
    model: Any,
    features: list[str],
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], list[dict]]:
    """Version-safe SHAP analysis. Returns (importance_df, directions, local_examples)."""
    log.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    raw = explainer(X_test)

    # Handle both Explanation objects and raw numpy
    shap_values = raw.values if hasattr(raw, "values") else np.array(raw)
    log.info("  SHAP matrix shape: %s", shap_values.shape)

    # ── Global importance ──
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "Feature": features,
        "MeanAbsSHAP": [round(float(v), 6) for v in mean_abs],
    }).sort_values("MeanAbsSHAP", ascending=False)

    # ── Direction analysis ──
    directions: dict[str, str] = {}
    for feat in importance.head(10)["Feature"]:
        idx = features.index(feat)
        vals = X_test[feat].values.astype(float)
        sv = shap_values[:, idx].astype(float)
        corr = np.corrcoef(vals, sv)[0, 1] if np.std(vals) > 0 else 0.0
        if np.isnan(corr):
            corr = 0.0
        if corr > 0.1:
            directions[feat] = "↑ INCREASES LT"
        elif corr < -0.1:
            directions[feat] = "↓ DECREASES LT"
        else:
            directions[feat] = "~ MIXED"

    # ── Local explanations (5 evenly-spaced samples) ──
    n = len(X_test)
    sample_idxs = [int(i * (n - 1) / 4) for i in range(5)]
    local_examples: list[dict] = []
    for pos, si in enumerate(sample_idxs):
        sv = shap_values[si]
        top5 = sorted(zip(features, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        local_examples.append({
            "position": pos,
            "top_contributors": [
                {"feature": f, "shap": float(s), "value": float(X_test[f].iloc[si])}
                for f, s in top5
            ],
        })

    log.info("  Top 3 SHAP features: %s",
             ", ".join(f"{r['Feature']}={r['MeanAbsSHAP']:.3f}"
                       for _, r in importance.head(3).iterrows()))
    return importance, directions, local_examples


# ====================================================================
# 11. BUSINESS VALIDATION
# ====================================================================

def run_business_validation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    encoders: dict[str, LabelEncoder],
) -> dict[str, Any]:
    """Run business-logic checks. Returns structured validation dict."""
    log.info("Running business validation...")
    checks: dict[str, Any] = {}

    # Ship Mode effect
    sm_results = {}
    for sm_name, sm_ord in SHIP_MODE_ORDINAL.items():
        mask = X_test["ship_mode_ordinal"] == sm_ord
        if mask.sum() > 0:
            sm_results[sm_name] = {
                "pred": float(model.predict(X_test[mask]).mean()),
                "actual": float(y_test[mask].mean()),
                "n": int(mask.sum()),
            }
    checks["ship_mode"] = sm_results

    # Distance effect
    dist_col = X_test["factory_region_distance_km"]
    dist_results = {}
    for label, lo, hi in [("Short <800km", 0, 800), ("Medium 800-2000km", 800, 2000), ("Long >2000km", 2000, 9999)]:
        mask = (dist_col >= lo) & (dist_col < hi)
        if mask.sum() > 0:
            dist_results[label] = {
                "pred": float(model.predict(X_test[mask]).mean()),
                "actual": float(y_test[mask].mean()),
                "n": int(mask.sum()),
            }
    checks["distance"] = dist_results

    # Factory effect
    factory_results = {}
    for code in sorted(X_test["factory"].unique()):
        mask = X_test["factory"] == code
        if mask.sum() > 0:
            fname = encoders["factory"].inverse_transform([code])[0]
            factory_results[fname] = {
                "pred": float(model.predict(X_test[mask]).mean()),
                "actual": float(y_test[mask].mean()),
                "n": int(mask.sum()),
            }
    checks["factory"] = factory_results

    # Ship mode ordering validation
    sd_pred = sm_results.get("Same Day", {}).get("pred", 99)
    sc_pred = sm_results.get("Standard Class", {}).get("pred", 0)
    checks["ship_mode_order_preserved"] = sd_pred < sc_pred
    log.info("  Ship Mode ordering: %s", "PRESERVED ✓" if checks["ship_mode_order_preserved"] else "VIOLATED ✗")

    return checks


# ====================================================================
# 12. CROSS-VALIDATION
# ====================================================================

def run_cross_validation(
    model: Any,
    splits: dict,
) -> dict[str, Any]:
    """5-fold CV on train+val set (test set excluded for integrity)."""
    X_cv = pd.concat([splits["X_train"], splits["X_val"]])
    y_cv = pd.concat([splits["y_train"], splits["y_val"]])
    log.info("Cross-validation on %d rows (train+val, test excluded)...", len(X_cv))

    r2_scores = cross_val_score(model, X_cv, y_cv, cv=5, scoring="r2", n_jobs=-1)
    rmse_scores = -cross_val_score(model, X_cv, y_cv, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)

    log.info("  CV R²:   %.4f ± %.4f", r2_scores.mean(), r2_scores.std())
    log.info("  CV RMSE: %.4f ± %.4f", rmse_scores.mean(), rmse_scores.std())
    log.info("  Folds:   %s", [f"{s:.4f}" for s in r2_scores])

    return {
        "r2_scores": r2_scores.tolist(),
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "rmse_scores": rmse_scores.tolist(),
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
    }


# ====================================================================
# 13. ARTIFACT GENERATION
# ====================================================================

def save_artifacts(
    winner_name: str,
    winner_result: dict,
    encoders: dict[str, LabelEncoder],
    results: dict[str, dict],
    fi_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
) -> dict[str, Path]:
    """Save all Phase 4 artifacts. Returns dict of saved paths."""
    log.info("Saving artifacts...")
    saved: dict[str, Path] = {}

    # Create directories
    for d in [MODELS_DIR, MODEL_RESULTS_DIR, PROCESSED_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Model
    path = MODELS_DIR / "winning_model.joblib"
    joblib.dump(winner_result["fitted_model"], path)
    saved["winning_model"] = path
    log.info("  ✓ %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

    # Encoders
    path = MODELS_DIR / "label_encoders.joblib"
    joblib.dump(encoders, path)
    saved["label_encoders"] = path
    log.info("  ✓ %s", path.name)

    # Feature config
    path = MODELS_DIR / "model_features.json"
    # Infer winner params from fitted model
    model_obj = winner_result["fitted_model"]
    winner_params = {}
    if hasattr(model_obj, "get_params"):
        params = model_obj.get_params()
        for key in ["n_estimators", "max_depth", "min_samples_leaf", "learning_rate",
                     "subsample", "colsample_bytree"]:
            if key in params:
                winner_params[key] = params[key]

    config = {
        "features_a": FEATURES_A,
        "features_b": FEATURES_B,
        "target_a": TARGET,
        "target_b": "gross_profit",
        "winner_name": winner_name,
        "winner_params": winner_params,
    }
    path.write_text(json.dumps(config, indent=2))
    saved["model_features"] = path
    log.info("  ✓ %s", path.name)

    # Model comparison
    comp_rows = []
    for name in MODEL_ORDER:
        r = results[name]
        comp_rows.append({
            "Model": name,
            "Test_RMSE": round(r["test_rmse"], 4),
            "Test_MAE": round(r["test_mae"], 4),
            "Test_R2": round(r["test_r2"], 4),
            "Train_R2": round(r["train_r2"], 4),
            "Val_R2": round(r["val_r2"], 4),
            "Training_Time_s": r["train_time_s"],
            "Overfit_Gap": round(r["overfit_gap"], 4),
        })
    path = MODEL_RESULTS_DIR / "model_comparison.csv"
    pd.DataFrame(comp_rows).to_csv(path, index=False)
    saved["model_comparison"] = path
    log.info("  ✓ %s", path.name)

    # Feature importance
    path = MODEL_RESULTS_DIR / "feature_importance.csv"
    fi_df.to_csv(path, index=False)
    saved["feature_importance"] = path
    log.info("  ✓ %s", path.name)

    # SHAP importance
    path = MODEL_RESULTS_DIR / "shap_importance.csv"
    shap_df.to_csv(path, index=False)
    saved["shap_importance"] = path
    log.info("  ✓ %s", path.name)

    # Modeling dataset (intermediate, for reproducibility)
    path = PROCESSED_DIR / "modeling_dataset.csv"
    modeling_df.to_csv(path, index=False)
    saved["modeling_dataset"] = path
    log.info("  ✓ %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

    return saved


# ====================================================================
# 14. REPORTING
# ====================================================================

def generate_report(
    winner_name: str,
    winner_result: dict,
    results: dict[str, dict],
    fi_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    shap_directions: dict[str, str],
    cv_results: dict,
    biz_checks: dict,
    n_total: int,
    n_features: int,
    y_mean: float,
    y_std: float,
    splits: dict,
) -> Path:
    """Generate phase4_report.md."""
    log.info("Generating Phase 4 report...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    wr = winner_result
    lines: list[str] = []

    lines.append("# Phase 4 — Model Training & Evaluation Report\n")
    lines.append(f"**Project:** Nassau Candy Distributor — Factory Reallocation & Shipping Optimization  ")
    lines.append(f"**Phase:** 4 — Model Training  ")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
    lines.append("---\n")

    # Dataset
    lines.append("## 1. Dataset Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total Rows | {n_total:,} |")
    lines.append(f"| Features | {n_features} |")
    lines.append(f"| Target | adjusted_lead_time |")
    lines.append(f"| Target Mean | {y_mean:.2f} |")
    lines.append(f"| Target Std | {y_std:.2f} |\n")

    # Split
    lines.append("## 2. Train / Test Split\n")
    lines.append("| Split | Rows | Share | Target Mean |")
    lines.append("|---|---|---|---|")
    for name, key in [("Train", "y_train"), ("Validation", "y_val"), ("Test", "y_test")]:
        ys = splits[key]
        lines.append(f"| {name} | {len(ys):,} | {len(ys)/n_total*100:.0f}% | {ys.mean():.3f} |")
    lines.append("\n**Strategy:** Stratified split on Ship Mode.\n")

    # Model comparison
    lines.append("## 3. Model Comparison\n")
    lines.append("| Model | Test RMSE | Test MAE | Test R² | Train Time | Overfit Gap |")
    lines.append("|---|---|---|---|---|---|")
    for name in MODEL_ORDER:
        r = results[name]
        lines.append(
            f"| {name} | {r['test_rmse']:.4f} | {r['test_mae']:.4f} | "
            f"{r['test_r2']:.4f} | {r['train_time_s']}s | {r['overfit_gap']:.4f} |"
        )
    lines.append("")

    # Winner
    lines.append("## 4. Winning Model\n")
    lines.append("| Attribute | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | {winner_name} |")
    lines.append(f"| Test RMSE | {wr['test_rmse']:.4f} |")
    lines.append(f"| Test MAE | {wr['test_mae']:.4f} |")
    lines.append(f"| Test R² | {wr['test_r2']:.4f} |")
    lines.append(f"| 5-Fold CV R² | {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f} |")
    lines.append(f"| Training Time | {wr['train_time_s']}s |\n")

    # Feature importance
    lines.append("## 5. Top 10 Features (Permutation Importance)\n")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---|---|---|")
    for i, (_, row) in enumerate(fi_df.head(10).iterrows()):
        lines.append(f"| {i+1} | {row['Feature']} | {row['PermutationImportance']:.4f} |")
    lines.append("")

    # SHAP
    lines.append("## 6. SHAP Insights\n")
    lines.append("| Rank | Feature | Mean |SHAP| | Direction |")
    lines.append("|---|---|---|---|")
    for i, (_, row) in enumerate(shap_df.head(10).iterrows()):
        d = shap_directions.get(row["Feature"], "~")
        lines.append(f"| {i+1} | {row['Feature']} | {row['MeanAbsSHAP']:.4f} | {d} |")
    lines.append("")

    # Business validation
    lines.append("## 7. Business Validation\n")
    lines.append("| Ship Mode | Predicted LT | Actual LT | n |")
    lines.append("|---|---|---|---|")
    for sm_name in ["Same Day", "First Class", "Second Class", "Standard Class"]:
        bc = biz_checks["ship_mode"].get(sm_name)
        if bc:
            lines.append(f"| {sm_name} | {bc['pred']:.2f}d | {bc['actual']:.2f}d | {bc['n']} |")
    order_str = "preserved" if biz_checks.get("ship_mode_order_preserved") else "NOT preserved"
    lines.append(f"\n**Verdict:** Ship Mode ordering {order_str}.\n")

    # Cross-validation
    lines.append("## 8. Cross-Validation Results\n")
    lines.append("| Fold | R² |")
    lines.append("|---|---|")
    for i, s in enumerate(cv_results["r2_scores"]):
        lines.append(f"| Fold {i+1} | {s:.4f} |")
    lines.append(f"| **Mean** | **{cv_results['r2_mean']:.4f}** |")
    lines.append(f"| **Std** | **{cv_results['r2_std']:.4f}** |\n")

    # Readiness
    lines.append("## 9. Recommendation Engine Readiness\n")
    lines.append("| Criterion | Assessment |")
    lines.append("|---|---|")
    lines.append(f"| Model accuracy sufficient? | {'YES' if wr['test_r2'] > 0.5 else 'MARGINAL'} (R²={wr['test_r2']:.4f}) |")
    lines.append(f"| Generalisation stable? | {'YES' if wr['overfit_gap'] < 0.25 else 'CAUTION'} (gap={wr['overfit_gap']:.4f}) |")
    lines.append(f"| CV consistent? | {'YES' if cv_results['r2_std'] < 0.05 else 'CAUTION'} (σ={cv_results['r2_std']:.4f}) |")
    lines.append(f"| Business logic validated? | {'YES' if biz_checks.get('ship_mode_order_preserved') else 'CHECK'} |")
    lines.append(f"| SHAP explainability ready? | YES ({len(shap_df)} features) |")
    lines.append(f"| Suitable for scenario simulation? | {'YES' if wr['test_r2'] > 0.5 else 'MARGINAL'} |\n")

    path = REPORTS_DIR / "phase4_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  ✓ %s (%d lines)", path.name, len(lines))
    return path


# ====================================================================
# 15. MAIN ENTRY POINT
# ====================================================================

def main() -> None:
    """Execute the complete Phase 4 pipeline."""
    t_start = time.time()

    print("=" * 72)
    print("  PHASE 4 — Model Training & Evaluation Pipeline")
    print("  Nassau Candy Distributor")
    print("=" * 72)

    # ── Load & engineer ──
    raw_df = load_raw_data(RAW_DATA)
    feat_df = engineer_features(raw_df)

    # ── Encode ──
    encoded_df, encoders = encode_categoricals(feat_df, CATEGORICAL_COLS)

    # ── Split ──
    splits = split_data(encoded_df, FEATURES_A, TARGET)

    # ── Train ──
    results = train_all_models(splits)

    # ── Select winner ──
    winner_name, winner_result = select_winner(results)
    winner_model = winner_result["fitted_model"]

    # ── Feature importance ──
    fi_df = compute_feature_importance(
        winner_model, FEATURES_A, splits["X_test"], splits["y_test"],
    )

    # ── SHAP ──
    shap_df, shap_directions, shap_local = compute_shap(
        winner_model, FEATURES_A, splits["X_test"],
    )

    # ── Business validation ──
    biz_checks = run_business_validation(
        winner_model, splits["X_test"], splits["y_test"], encoders,
    )

    # ── Cross-validation ──
    cv_results = run_cross_validation(winner_model, splits)

    # ── Save artifacts ──
    saved = save_artifacts(
        winner_name, winner_result, encoders, results,
        fi_df, shap_df, feat_df,
    )

    # ── Generate report ──
    y_all = encoded_df[TARGET]
    report_path = generate_report(
        winner_name, winner_result, results,
        fi_df, shap_df, shap_directions,
        cv_results, biz_checks,
        n_total=len(encoded_df),
        n_features=len(FEATURES_A),
        y_mean=float(y_all.mean()),
        y_std=float(y_all.std()),
        splits=splits,
    )
    saved["phase4_report"] = report_path

    # ── Verify ──
    print("\n" + "=" * 72)
    print("  ARTIFACT REGISTRY")
    print("=" * 72)
    all_ok = True
    for label, path in saved.items():
        exists = path.exists()
        size = path.stat().st_size / 1024 if exists else 0
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status}  {path.relative_to(PROJECT_ROOT)}  ({size:.1f} KB)")
        if not exists:
            all_ok = False

    elapsed = time.time() - t_start
    print(f"\n  Winner: {winner_name}")
    print(f"  Test RMSE: {winner_result['test_rmse']:.4f}")
    print(f"  Test MAE:  {winner_result['test_mae']:.4f}")
    print(f"  Test R²:   {winner_result['test_r2']:.4f}")
    print(f"  CV R²:     {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    print(f"  Runtime:   {elapsed:.1f}s")
    print(f"\n  STATUS: {'ALL ARTIFACTS GENERATED SUCCESSFULLY' if all_ok else 'SOME ARTIFACTS MISSING'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
