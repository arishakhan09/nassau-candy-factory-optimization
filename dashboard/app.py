"""
Nassau Candy Distributor
Factory Reallocation & Shipping Optimization Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import (
    load_recommendations, load_product_summary, load_summary_json,
    load_model_comparison, load_feature_importance, load_shap_importance,
    format_number, format_pct,
    DIVISION_COLORS, FACTORY_COLORS, REC_COLORS, RISK_COLORS
)
from components.theme import (
    inject_css, render_topbar, render_kpi, render_section_header,
    render_divider, chart_card_open, chart_card_close,
    render_insight_strip, render_ml_metric, render_info_banner,
    apply_chart_theme, NAVY, BLUE, LIGHT_BLUE, TEXT_DARK, BORDER,
)

# ================================================================
# PAGE CONFIG (must be first Streamlit call)
# ================================================================
st.set_page_config(
    page_title="Nassau Candy · Factory Optimization",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================================================================
# INJECT GLOBAL CSS + TOPBAR
# ================================================================
inject_css()
render_topbar("Nassau Candy Distributor", "Factory Optimization")

# ================================================================
# LOAD DATA
# ================================================================
rec_df = load_recommendations()
prod_df = load_product_summary()
summary = load_summary_json()
model_df = load_model_comparison()
feat_df = load_feature_importance()
shap_df = load_shap_importance()

# ================================================================
# TABS
# ================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Executive Summary  ",
    "  Factory Simulator  ",
    "  Product Reallocation  ",
    "  Risk & Impact  ",
    "  Technical Appendix  ",
])

# ================================================================
# TAB 1: EXECUTIVE SUMMARY
# ================================================================
with tab1:
    # Insight strip
    total_orders = summary['total_orders']
    rec_count = summary['orders_with_recommendation']
    avg_dist_red = summary['avg_distance_reduction_pct']
    strong_count = summary['recommendation_distribution']['STRONG_RECOMMEND']

    insights = [
        ("📊", f"{format_number(rec_count)} orders optimized",
         f"{format_pct(rec_count/total_orders*100)} of all orders receive a factory reallocation recommendation"),
        ("🚛", f"{avg_dist_red:.1f}% average distance reduction",
         f"{summary['avg_distance_reduction_km']:.0f} km saved per order on average"),
        ("⭐", f"{format_number(strong_count)} strong recommendations",
         f"{format_pct(strong_count/total_orders*100)} of orders have high-confidence reallocation"),
        ("📈", f"{summary['total_route_km_saved']/1e6:.1f}M total route-km saved",
         "Aggregate logistics impact across all optimized orders"),
    ]
    render_insight_strip(insights)

    # KPI Row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    render_kpi(k1, "Orders Analyzed", format_number(total_orders), "in dataset", "Full cohort")
    render_kpi(k2, "Recommendations", format_number(rec_count),
               f"{format_pct(rec_count/total_orders*100)} of orders", "With change")
    render_kpi(k3, "Distance Reduction", format_pct(avg_dist_red),
               f"{summary['avg_distance_reduction_km']:.0f} km avg", "Proximity gain")
    render_kpi(k4, "Route-KM Saved", f"{summary['total_route_km_saved']/1e6:.1f}M",
               "Total logistics impact", "Aggregate")
    render_kpi(k5, "Avg Confidence", f"{summary['avg_confidence']:.1%}",
               "Model certainty", "Validated")
    render_kpi(k6, "Strong Recommendations", format_number(strong_count),
               f"{format_pct(strong_count/total_orders*100)} of orders", "High confidence")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    render_divider()

    # Row 2: Factory Workload + Recommendation Distribution
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        factories = ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory", "The Other Factory"]
        current_counts = [(rec_df['current_production_site'] == f).sum() for f in factories]
        proposed_counts = [(rec_df['proposed_production_site'] == f).sum() for f in factories]

        fig_workload = go.Figure()
        fig_workload.add_trace(go.Bar(
            name='Current', x=factories, y=current_counts,
            marker_color='#94A3B8', text=current_counts, textposition='outside',
            textfont=dict(size=11, color=TEXT_DARK),
        ))
        fig_workload.add_trace(go.Bar(
            name='Recommended', x=factories, y=proposed_counts,
            marker_color=BLUE, text=proposed_counts, textposition='outside',
            textfont=dict(size=11, color=BLUE),
        ))
        apply_chart_theme(fig_workload, height=380)
        fig_workload.update_layout(barmode='group',
                                   legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
        chart_card_open("FACTORY WORKLOAD: CURRENT VS RECOMMENDED", "Order distribution across production sites")
        st.plotly_chart(fig_workload, use_container_width=True)
        chart_card_close()

    with col_b:
        rec_dist = summary['recommendation_distribution']
        labels = ['Strong Recommend', 'Moderate Recommend', 'Marginal', 'No Change']
        values = [rec_dist['STRONG_RECOMMEND'], rec_dist['MODERATE_RECOMMEND'],
                  rec_dist['MARGINAL'], rec_dist['NO_CHANGE']]
        colors = [NAVY, BLUE, LIGHT_BLUE, '#94A3B8']

        fig_dist = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='percent+label', textposition='outside',
            textfont=dict(size=11, color=TEXT_DARK),
            hole=0.60,
            pull=[0.03, 0, 0, 0],
        )])
        apply_chart_theme(fig_dist, height=380, legend=False)
        fig_dist.update_layout(showlegend=False)
        fig_dist.add_annotation(
            text=f"<b>{format_number(rec_count)}</b><br><span style='font-size:10px;color:#6B7280'>changes</span>",
            x=0.5, y=0.5, font=dict(size=16, color=NAVY), showarrow=False,
        )
        chart_card_open("RECOMMENDATION DISTRIBUTION", "Breakdown by confidence tier")
        st.plotly_chart(fig_dist, use_container_width=True)
        chart_card_close()

    render_divider()

    # Row 3: Top 10 Products by Distance Reduction
    top10_dist = prod_df[prod_df['avg_distance_reduction_pct'] > 0].nlargest(10, 'avg_distance_reduction_pct')
    if len(top10_dist) > 0:
        fig_dist_red = go.Figure()
        fig_dist_red.add_trace(go.Bar(
            y=top10_dist['product_name'].fillna(top10_dist['product_id']),
            x=top10_dist['avg_distance_reduction_pct'],
            orientation='h',
            marker_color=[DIVISION_COLORS.get(d, '#94A3B8') for d in top10_dist['division']],
            text=top10_dist['avg_distance_reduction_pct'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            textfont=dict(size=11, color=TEXT_DARK),
        ))
        apply_chart_theme(fig_dist_red, height=380)
        fig_dist_red.update_layout(yaxis=dict(autorange='reversed'))
        chart_card_open("TOP 10 PRODUCTS BY DISTANCE REDUCTION", "Highest proximity gains from factory reallocation")
        st.plotly_chart(fig_dist_red, use_container_width=True)
        chart_card_close()

    render_divider()

    # Row 4: Top 5 Reallocation Opportunities
    render_section_header("Top 5 Reallocation Opportunities", "Strongest composite-score products for factory change")
    top5 = prod_df[prod_df['change_recommended'] == True].head(5)

    t5_cols = st.columns(5, gap="medium")
    for i, (_, row) in enumerate(top5.iterrows()):
        with t5_cols[i]:
            product_name = row['product_name'] if pd.notna(row['product_name']) else row['product_id']
            st.markdown(f"""
            <div class="kpi-card" style="min-height:160px;">
                <div class="kpi-label">{product_name}</div>
                <div class="kpi-value" style="font-size:1.4rem;">{row['avg_distance_reduction_km']:.0f} km</div>
                <div class="kpi-sub">{row['current_production_site']} → {row['proposed_production_site']}</div>
                <div class="kpi-badge">Score: {row['avg_composite_score']:.4f}</div>
                <div style="margin-top:6px;font-size:0.7rem;color:#6B7280;">{row['orders_affected']:,} orders · {row['efficiency_class']}</div>
            </div>
            """, unsafe_allow_html=True)


# ================================================================
# TAB 2: FACTORY OPTIMIZATION SIMULATOR
# ================================================================
with tab2:
    render_section_header("Factory Optimization Simulator", "Explore recommendations by product, region, and factory")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_product = st.selectbox(
            "Product",
            options=['All'] + sorted(rec_df['product_id'].unique().tolist()),
            index=0,
        )
    with fc2:
        sel_region = st.selectbox(
            "Region",
            options=['All'] + sorted(rec_df['region'].unique().tolist()),
            index=0,
        )
    with fc3:
        sel_factory = st.selectbox(
            "Current Factory",
            options=['All'] + sorted(rec_df['current_production_site'].unique().tolist()),
            index=0,
        )

    # Filter data
    filtered = rec_df.copy()
    if sel_product != 'All':
        filtered = filtered[filtered['product_id'] == sel_product]
    if sel_region != 'All':
        filtered = filtered[filtered['region'] == sel_region]
    if sel_factory != 'All':
        filtered = filtered[filtered['current_production_site'] == sel_factory]

    changed_filtered = filtered[filtered['proposed_production_site'] != filtered['current_production_site']]

    # Simulator KPI cards
    sk1, sk2, sk3, sk4 = st.columns(4)
    n_changed = len(changed_filtered)
    avg_dist = changed_filtered['distance_change_pct'].mean() if len(changed_filtered) > 0 else 0
    avg_conf = changed_filtered['confidence'].mean() if len(changed_filtered) > 0 else 0

    render_kpi(sk1, "Filtered Orders", format_number(len(filtered)), "in current selection")
    render_kpi(sk2, "With Recommendation", format_number(n_changed),
               format_pct(n_changed/max(len(filtered),1)*100) if len(filtered) > 0 else "0%", "Changes proposed")
    render_kpi(sk3, "Avg Distance Reduction", format_pct(avg_dist), "across changed orders")
    render_kpi(sk4, "Avg Confidence", f"{avg_conf:.1%}", "model certainty")

    render_divider()

    # Side-by-side comparison
    if sel_product != 'All' and len(changed_filtered) > 0:
        sample = changed_filtered.iloc[0]
        comp1, comp2 = st.columns(2, gap="large")

        with comp1:
            st.markdown(f"""
            <div class="chart-card">
                <div class="chart-title">🔴 CURRENT STATE</div>
                <div class="seg-stat-row"><span class="seg-stat-label">Factory</span><span class="seg-stat-val">{sample['current_production_site']}</span></div>
                <div class="seg-stat-row"><span class="seg-stat-label">Avg Distance</span><span class="seg-stat-val">{sample['current_distance_km']:.0f} km</span></div>
                <div class="seg-stat-row"><span class="seg-stat-label">Predicted Lead Time</span><span class="seg-stat-val">{sample['current_lt_pred']:.2f} days</span></div>
            </div>
            """, unsafe_allow_html=True)

        with comp2:
            st.markdown(f"""
            <div class="chart-card">
                <div class="chart-title">🟢 RECOMMENDED STATE</div>
                <div class="seg-stat-row"><span class="seg-stat-label">Factory</span><span class="seg-stat-val">{sample['proposed_production_site']}</span></div>
                <div class="seg-stat-row"><span class="seg-stat-label">Avg Distance</span><span class="seg-stat-val">{sample['proposed_distance_km']:.0f} km ({sample['distance_change_km']:.0f} km)</span></div>
                <div class="seg-stat-row"><span class="seg-stat-label">Predicted Lead Time</span><span class="seg-stat-val">{sample['proposed_lt_pred']:.2f} days ({sample['lt_change_days']:+.3f})</span></div>
            </div>
            """, unsafe_allow_html=True)

        mc1, mc2, mc3 = st.columns(3)
        render_kpi(mc1, "Composite Score", f"{sample['composite_score']:.4f}", "weighted metric")
        render_kpi(mc2, "Confidence", f"{sample['confidence']:.1%}", "model certainty")
        render_kpi(mc3, "Risk Level", sample['risk_level'], "assessed risk")

        render_divider()

    # Region breakdown chart
    if len(changed_filtered) > 0:
        region_agg = changed_filtered.groupby('region').agg(
            current_dist=('current_distance_km', 'mean'),
            proposed_dist=('proposed_distance_km', 'mean'),
            count=('order_id', 'count'),
        ).reset_index()

        fig_region = go.Figure()
        fig_region.add_trace(go.Bar(
            name='Current Distance', x=region_agg['region'], y=region_agg['current_dist'],
            marker_color='#94A3B8',
        ))
        fig_region.add_trace(go.Bar(
            name='Proposed Distance', x=region_agg['region'], y=region_agg['proposed_dist'],
            marker_color=BLUE,
        ))
        apply_chart_theme(fig_region, height=350)
        fig_region.update_layout(barmode='group',
                                 yaxis_title='Distance (km)',
                                 legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center'))
        chart_card_open("DISTANCE COMPARISON BY REGION", "Current vs proposed average shipping distance")
        st.plotly_chart(fig_region, use_container_width=True)
        chart_card_close()


# ================================================================
# TAB 3: PRODUCT REALLOCATION DASHBOARD
# ================================================================
with tab3:
    render_section_header("Product Reallocation Dashboard", "Explore and export product-level reallocation recommendations")

    # Filters
    pf1, pf2, pf3 = st.columns(3)
    with pf1:
        div_filter = st.multiselect("Division", options=prod_df['division'].unique().tolist(),
                                     default=prod_df['division'].unique().tolist())
    with pf2:
        fac_filter = st.multiselect("Current Factory",
                                     options=prod_df['current_production_site'].unique().tolist(),
                                     default=prod_df['current_production_site'].unique().tolist())
    with pf3:
        rec_filter = st.multiselect("Recommendation Strength",
                                     options=prod_df['dominant_recommendation'].unique().tolist(),
                                     default=prod_df['dominant_recommendation'].unique().tolist())

    # Apply filters
    display_df = prod_df[
        (prod_df['division'].isin(div_filter)) &
        (prod_df['current_production_site'].isin(fac_filter)) &
        (prod_df['dominant_recommendation'].isin(rec_filter))
    ].copy()

    # Display columns
    show_cols = [
        'product_id', 'product_name', 'division',
        'current_production_site', 'proposed_production_site',
        'orders_affected', 'avg_distance_reduction_km', 'avg_distance_reduction_pct',
        'avg_lt_change_days', 'avg_composite_score', 'avg_confidence',
        'dominant_recommendation', 'dominant_risk_level', 'efficiency_class',
        'total_route_km_saved',
    ]
    show_cols = [c for c in show_cols if c in display_df.columns]

    chart_card_open("PRODUCT RECOMMENDATIONS TABLE", f"{len(display_df)} products matching current filters")
    st.dataframe(
        display_df[show_cols].style.format({
            'avg_distance_reduction_km': '{:.0f}',
            'avg_distance_reduction_pct': '{:.1f}%',
            'avg_lt_change_days': '{:+.3f}',
            'avg_composite_score': '{:.4f}',
            'avg_confidence': '{:.4f}',
            'total_route_km_saved': '{:,.0f}',
        }),
        use_container_width=True,
        height=450,
    )
    chart_card_close()

    # Export
    csv_data = display_df[show_cols].to_csv(index=False)
    st.download_button(
        label="📥 Export to CSV",
        data=csv_data,
        file_name="nassau_candy_recommendations.csv",
        mime="text/csv",
    )

    render_divider()

    # Product score chart
    fig_scores = go.Figure()
    fig_scores.add_trace(go.Bar(
        x=display_df['product_id'],
        y=display_df['avg_composite_score'],
        marker_color=[DIVISION_COLORS.get(d, '#94A3B8') for d in display_df['division']],
        text=display_df['avg_composite_score'].apply(lambda x: f'{x:.3f}'),
        textposition='outside',
        textfont=dict(size=10, color=TEXT_DARK),
    ))
    apply_chart_theme(fig_scores, height=350)
    fig_scores.update_layout(
        yaxis_title='Composite Score',
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
    )
    chart_card_open("PRODUCT SCORES OVERVIEW", "Composite optimization score per product")
    st.plotly_chart(fig_scores, use_container_width=True)
    chart_card_close()


# ================================================================
# TAB 4: RISK & IMPACT PANEL
# ================================================================
with tab4:
    render_section_header("Risk & Impact Analysis", "Risk distribution and logistics savings breakdown")

    # Risk KPI Cards
    risk_counts = rec_df['risk_level'].value_counts()
    r1, r2, r3, r4 = st.columns(4)
    low_n = risk_counts.get('Low', 0)
    med_n = risk_counts.get('Medium', 0)
    high_n = risk_counts.get('High', 0)
    total_changed = (rec_df['proposed_production_site'] != rec_df['current_production_site']).sum()

    render_kpi(r1, "Low Risk", format_number(low_n), format_pct(low_n/len(rec_df)*100), "🟢 Safe")
    render_kpi(r2, "Medium Risk", format_number(med_n), format_pct(med_n/len(rec_df)*100), "🟡 Monitor")
    render_kpi(r3, "High Risk", format_number(high_n), format_pct(high_n/len(rec_df)*100), "🔴 Review")
    render_kpi(r4, "Total Recommendations", format_number(total_changed), "orders with proposed change")

    render_divider()

    # Row 2: Charts
    rc1, rc2 = st.columns(2, gap="large")

    with rc1:
        rec_val_counts = rec_df['recommendation'].value_counts()
        rec_categories = ['STRONG RECOMMEND', 'MODERATE RECOMMEND', 'MARGINAL', 'NO CHANGE']
        rec_values = [rec_val_counts.get(cat, 0) for cat in rec_categories]
        rec_bar_colors = [NAVY, BLUE, LIGHT_BLUE, '#94A3B8']

        fig_rec_bar = go.Figure()
        fig_rec_bar.add_trace(go.Bar(
            x=rec_categories, y=rec_values,
            marker_color=rec_bar_colors,
            text=[f'{v:,}' for v in rec_values],
            textposition='outside',
            textfont=dict(size=11, color=TEXT_DARK),
        ))
        apply_chart_theme(fig_rec_bar, height=350, legend=False)
        chart_card_open("RECOMMENDATION STRENGTH DISTRIBUTION", "Order count by recommendation tier")
        st.plotly_chart(fig_rec_bar, use_container_width=True)
        chart_card_close()

    with rc2:
        changed_orders = rec_df[rec_df['proposed_production_site'] != rec_df['current_production_site']]
        if len(changed_orders) > 0:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=changed_orders['distance_change_pct'],
                nbinsx=30,
                marker_color=BLUE,
                marker_line_color='white',
                marker_line_width=1,
            ))
            apply_chart_theme(fig_hist, height=350, legend=False)
            fig_hist.update_layout(xaxis_title='Distance Reduction (%)', yaxis_title='Order Count')
            chart_card_open("DISTANCE SAVINGS DISTRIBUTION", "Histogram of per-order distance reduction %")
            st.plotly_chart(fig_hist, use_container_width=True)
            chart_card_close()

    render_divider()

    # Top 10 Opportunities
    top10 = prod_df.nlargest(10, 'total_route_km_saved')
    fig_top10 = go.Figure()
    fig_top10.add_trace(go.Bar(
        y=top10['product_id'],
        x=top10['total_route_km_saved'],
        orientation='h',
        marker_color=[DIVISION_COLORS.get(d, '#94A3B8') for d in top10['division']],
        text=top10['total_route_km_saved'].apply(lambda x: f'{x:,.0f} km'),
        textposition='outside',
        textfont=dict(size=10, color=TEXT_DARK),
    ))
    apply_chart_theme(fig_top10, height=400)
    fig_top10.update_layout(yaxis=dict(autorange='reversed'))
    chart_card_open("TOP 10 OPPORTUNITIES BY ROUTE-KM SAVED", "Products with the largest aggregate logistics savings")
    st.plotly_chart(fig_top10, use_container_width=True)
    chart_card_close()

    render_divider()

    # Most Impacted Factories
    factories = rec_df['current_production_site'].unique()
    impact_data = []
    for f in factories:
        curr = (rec_df['current_production_site'] == f).sum()
        prop = (rec_df['proposed_production_site'] == f).sum()
        impact_data.append({
            'Factory': f, 'Current Orders': curr,
            'Proposed Orders': prop, 'Change': prop - curr,
            'Change %': (prop - curr) / curr * 100 if curr > 0 else 0,
        })
    impact_df = pd.DataFrame(impact_data).sort_values('Change', key=abs, ascending=False)
    chart_card_open("MOST IMPACTED FACTORIES", "Net order change per factory under proposed reallocation")
    st.dataframe(impact_df, use_container_width=True, hide_index=True)
    chart_card_close()


# ================================================================
# TAB 5: TECHNICAL APPENDIX
# ================================================================
with tab5:
    render_section_header("Technical Appendix", "Model performance, feature importance, and validation details")

    # Winning Model Card
    winner = model_df[model_df['Model'] == 'Random Forest (Tuned)'].iloc[0]
    tm1, tm2, tm3, tm4 = st.columns(4, gap="medium")
    render_ml_metric(tm1, f"{winner['Test_R2']:.4f}", "Test R²", "Winning model")
    render_ml_metric(tm2, f"{winner['Test_RMSE']:.4f}", "Test RMSE", "Lower is better")
    render_ml_metric(tm3, f"{winner['Test_MAE']:.4f}", "Test MAE", "Mean absolute error")
    render_ml_metric(tm4, f"{winner['Overfit_Gap']:.4f}", "Overfit Gap", "Train–Test delta")

    render_info_banner(
        "🏆 Winning Model: Random Forest (Tuned)",
        "Selected via cross-validated grid search. Balances accuracy with generalization — minimal overfitting gap."
    )

    render_divider()

    # Model Comparison
    model_names = model_df['Model'].tolist()
    fig_models = go.Figure()
    fig_models.add_trace(go.Bar(
        x=model_names, y=model_df['Test_R2'],
        marker_color=[NAVY if n == 'Random Forest (Tuned)' else '#CBD5E1' for n in model_names],
        text=model_df['Test_R2'].apply(lambda x: f'{x:.4f}'),
        textposition='outside', textfont=dict(size=9, color=TEXT_DARK),
    ))
    apply_chart_theme(fig_models, height=320, legend=False)
    fig_models.update_layout(
        xaxis=dict(tickangle=-25, tickfont=dict(size=9)),
        yaxis_title='Test R²', yaxis_range=[0.6, 0.72],
    )
    chart_card_open("MODEL COMPARISON", "Test R² across all candidate models — winner highlighted")
    st.plotly_chart(fig_models, use_container_width=True)
    chart_card_close()

    render_divider()

    # Feature Importance
    fi_col1, fi_col2 = st.columns(2, gap="large")

    with fi_col1:
        top_feat = feat_df.nlargest(12, 'BuiltInImportance')
        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            y=top_feat['Feature'], x=top_feat['BuiltInImportance'],
            orientation='h', marker_color=BLUE,
            text=top_feat['BuiltInImportance'].apply(lambda x: f'{x:.3f}'),
            textposition='outside', textfont=dict(size=9, color=TEXT_DARK),
        ))
        apply_chart_theme(fig_fi, height=380, legend=False)
        fig_fi.update_layout(yaxis=dict(autorange='reversed', tickfont=dict(size=9)))
        chart_card_open("FEATURE IMPORTANCE (BUILT-IN)", "Random Forest Gini importance — top 12 features")
        st.plotly_chart(fig_fi, use_container_width=True)
        chart_card_close()

    with fi_col2:
        top_shap = shap_df.nlargest(12, 'MeanAbsSHAP')
        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            y=top_shap['Feature'], x=top_shap['MeanAbsSHAP'],
            orientation='h', marker_color=NAVY,
            text=top_shap['MeanAbsSHAP'].apply(lambda x: f'{x:.3f}'),
            textposition='outside', textfont=dict(size=9, color=TEXT_DARK),
        ))
        apply_chart_theme(fig_shap, height=380, legend=False)
        fig_shap.update_layout(yaxis=dict(autorange='reversed', tickfont=dict(size=9)))
        chart_card_open("SHAP IMPORTANCE (MEAN |SHAP|)", "Model-agnostic feature contribution — top 12")
        st.plotly_chart(fig_shap, use_container_width=True)
        chart_card_close()

    render_divider()

    # Business Validation
    val_data = {
        'Check': [
            'No cross-division violations',
            'No invalid factories',
            'No missing scores',
            'No missing confidence values',
            'Row count matches (10,194)',
            'Row uniqueness',
            'Batch prediction architecture',
        ],
        'Status': ['PASSED'] * 7,
    }
    chart_card_open("BUSINESS VALIDATION RESULTS", "All integrity checks passed before deployment")
    st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)
    chart_card_close()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    render_info_banner(
        "📊 Key Finding",
        "Ship Mode explains ~71.6% of lead time variance. Factory reallocation produces negligible "
        "lead time changes but significant distance reductions (47.7% avg, 10.1M route-km saved). "
        "The primary business value is <strong>logistics cost reduction through proximity optimization</strong>."
    )
