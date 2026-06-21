"""
Data loading and caching utilities for the Nassau Candy Dashboard.
All data is loaded once and cached via st.cache_data.
"""
import os
import json
import pandas as pd
import streamlit as st

# Project root = parent of dashboard/
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
DATA_OUTPUTS = os.path.join(PROJECT_ROOT, "data", "outputs")
RECOMMENDATIONS_DIR = os.path.join(DATA_OUTPUTS, "recommendations")
SIMULATIONS_DIR = os.path.join(DATA_OUTPUTS, "simulations")
MODEL_RESULTS_DIR = os.path.join(DATA_OUTPUTS, "model_results")


@st.cache_data(ttl=3600)
def load_recommendations():
    return pd.read_csv(os.path.join(RECOMMENDATIONS_DIR, "recommendations.csv"))


@st.cache_data(ttl=3600)
def load_product_summary():
    return pd.read_csv(os.path.join(RECOMMENDATIONS_DIR, "product_reallocation_summary.csv"))


@st.cache_data(ttl=3600)
def load_summary_json():
    with open(os.path.join(SIMULATIONS_DIR, "recommendation_summary.json")) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_model_comparison():
    return pd.read_csv(os.path.join(MODEL_RESULTS_DIR, "model_comparison.csv"))


@st.cache_data(ttl=3600)
def load_feature_importance():
    return pd.read_csv(os.path.join(MODEL_RESULTS_DIR, "feature_importance.csv"))


@st.cache_data(ttl=3600)
def load_shap_importance():
    return pd.read_csv(os.path.join(MODEL_RESULTS_DIR, "shap_importance.csv"))


def format_number(n, decimals=0):
    """Format large numbers with commas."""
    if decimals == 0:
        return f"{int(n):,}"
    return f"{n:,.{decimals}f}"


def format_pct(n, decimals=1):
    """Format percentage."""
    return f"{n:.{decimals}f}%"


DIVISION_COLORS = {
    'Chocolate': '#8B4513',
    'Sugar': '#FF6B9D',
    'Other': '#4ECDC4',
}

FACTORY_COLORS = {
    "Lot's O' Nuts": '#2196F3',
    "Wicked Choccy's": '#FF5722',
    "Sugar Shack": '#9C27B0',
    "Secret Factory": '#4CAF50',
    "The Other Factory": '#FF9800',
}

REC_COLORS = {
    'STRONG RECOMMEND': '#2E7D32',
    'MODERATE RECOMMEND': '#F9A825',
    'MARGINAL': '#FF8F00',
    'NO CHANGE': '#757575',
}

RISK_COLORS = {
    'Low': '#4CAF50',
    'Medium': '#FF9800',
    'High': '#F44336',
}
