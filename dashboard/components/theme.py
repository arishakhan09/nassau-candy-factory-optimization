"""
Design system and reusable UI components.
Cloned from reference.py design system — adapted for Nassau Candy dashboard.
"""

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
NAVY = "#0F2040"
SLATE = "#1E3A5F"
BLUE = "#2563EB"
LIGHT_BLUE = "#60A5FA"
MID_GRAY = "#6B7280"
SURFACE = "#F8F9FB"
BORDER = "#E5E7EB"
TEXT_DARK = "#111827"
TEXT_MID = "#4B5563"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Mirrors reference.py styling
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Reset & Base ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp { background: #F8F9FB; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 2rem 3rem 2rem;
    max-width: 1400px;
}

/* ── Top navigation bar ────────────────────────────────── */
.topbar {
    background: #0F2040;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    margin: 0 -2rem 2rem -2rem;
    position: sticky;
    top: 0;
    z-index: 999;
}
.topbar-brand {
    color: white;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 10px;
}
.topbar-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #60A5FA;
    display: inline-block;
}
.topbar-tag {
    color: #93C5FD;
    font-size: 0.72rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── KPI Cards ─────────────────────────────────────────── */
.kpi-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 20px 22px;
    min-height: 100px;
}
.kpi-label {
    color: #6B7280;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.kpi-value {
    color: #0F2040;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-sub {
    color: #9CA3AF;
    font-size: 0.75rem;
    font-weight: 400;
}
.kpi-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 6px;
}

/* ── Section headings ──────────────────────────────────── */
.section-header {
    margin: 2rem 0 1rem 0;
}
.section-title {
    color: #0F2040;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 0 0 2px 0;
}
.section-sub {
    color: #6B7280;
    font-size: 0.8rem;
    margin: 0;
}
.divider {
    height: 1px;
    background: #E5E7EB;
    margin: 1.5rem 0;
}

/* ── Chart containers ──────────────────────────────────── */
.chart-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 20px;
}
.chart-title {
    color: #0F2040;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.chart-desc {
    color: #9CA3AF;
    font-size: 0.73rem;
    margin-bottom: 12px;
}

/* ── Executive summary panel ───────────────────────────── */
.exec-panel {
    background: #0F2040;
    border-radius: 10px;
    padding: 24px 28px;
    color: white;
}
.exec-panel-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #93C5FD;
    margin-bottom: 10px;
}
.exec-panel-text {
    font-size: 0.9rem;
    font-weight: 400;
    line-height: 1.7;
    color: #CBD5E1;
}
.exec-panel-highlight {
    color: white;
    font-weight: 600;
}

/* ── Insight cards ─────────────────────────────────────── */
.insight-strip {
    display: flex;
    gap: 12px;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
}
.insight-card {
    flex: 1;
    min-width: 200px;
    background: white;
    border: 1px solid #E5E7EB;
    border-left: 3px solid #0F2040;
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.insight-icon {
    font-size: 1.2rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.insight-headline {
    font-size: 0.82rem;
    font-weight: 600;
    color: #0F2040;
    margin-bottom: 3px;
    line-height: 1.3;
}
.insight-detail {
    font-size: 0.73rem;
    color: #6B7280;
    line-height: 1.4;
}

/* ── ML metric cards ───────────────────────────────────── */
.ml-metric {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.ml-metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #0F2040;
}
.ml-metric-label {
    font-size: 0.75rem;
    color: #6B7280;
    font-weight: 500;
}
.ml-metric-badge {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 6px;
}

/* ── Recommendation cards ──────────────────────────────── */
.rec-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 22px;
}
.rec-seg-name {
    font-size: 0.85rem;
    font-weight: 700;
    color: #0F2040;
    margin-bottom: 14px;
}
.rec-row {
    margin-bottom: 12px;
}
.rec-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 3px;
}
.rec-text {
    font-size: 0.8rem;
    color: #374151;
    line-height: 1.5;
}

/* ── Tab styling ───────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: #6B7280;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 10px 18px;
    margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #0F2040 !important;
    border-bottom: 2px solid #0F2040 !important;
    font-weight: 600 !important;
}

/* ── Streamlit multiselect / selectbox styling ─────────── */
.stMultiSelect [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] {
    border-radius: 6px;
    font-size: 0.8rem;
}

/* ── Filter bar ────────────────────────────────────────── */
.filter-bar {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

/* ── Table styling ─────────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Segment profile cards ─────────────────────────────── */
.seg-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 22px;
    height: 100%;
}
.seg-card-top {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.seg-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.seg-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #0F2040;
}
.seg-count {
    margin-left: auto;
    font-size: 0.75rem;
    color: #6B7280;
    font-weight: 500;
}
.seg-stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #4B5563;
    padding: 5px 0;
    border-bottom: 1px solid #F3F4F6;
}
.seg-stat-label { color: #9CA3AF; }
.seg-stat-val { font-weight: 600; color: #111827; }
.seg-desc {
    font-size: 0.78rem;
    color: #6B7280;
    line-height: 1.55;
    margin-top: 12px;
}

/* ── Score badges (specific to Nassau Candy) ───────────── */
.score-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.7rem;
}
.score-strong { background: #DCFCE7; color: #166534; }
.score-moderate { background: #FEF3C7; color: #92400E; }
.score-marginal { background: #FEE2E2; color: #991B1B; }
.score-nochange { background: #F3F4F6; color: #6B7280; }

/* ── Info banner ───────────────────────────────────────── */
.info-banner {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-left: 3px solid #2563EB;
    border-radius: 8px;
    padding: 12px 18px;
    margin-top: 14px;
}
.info-banner-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #1D4ED8;
    margin-bottom: 4px;
}
.info-banner-text {
    font-size: 0.78rem;
    color: #374151;
    line-height: 1.6;
}

/* ── Download button override ──────────────────────────── */
.stDownloadButton > button {
    background: #0F2040;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.8rem;
    font-weight: 600;
}
.stDownloadButton > button:hover {
    background: #1E3A5F;
    color: white;
}

/* ── Compact expander for filter ───────────────────────── */
[data-testid="stExpander"] > details > summary {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #374151;
    letter-spacing: 0.03em;
}
[data-testid="stExpander"] > details > summary:hover {
    background: #F8F9FB;
    border-color: #CBD5E1;
}
[data-testid="stExpander"] > details[open] > summary {
    border-radius: 8px 8px 0 0;
    border-bottom-color: transparent;
}
[data-testid="stExpander"] > details > div {
    background: white;
    border: 1px solid #E5E7EB;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 12px 16px 16px 16px;
}

/* ── Metric override (prevent default dark styles) ─────── */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: none;
}
div[data-testid="stMetric"] label {
    color: #6B7280 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #0F2040 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    color: #6B7280 !important;
}
</style>
"""


def inject_css():
    """Inject the global CSS. Call once at app start."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_topbar(title="Nassau Candy Distributor", tag="Factory Optimization"):
    """Render the navy top navigation bar."""
    import streamlit as st
    st.markdown(f"""
    <div class="topbar">
        <div class="topbar-brand">
            <span class="topbar-dot"></span>
            {title}
        </div>
        <div class="topbar-tag">{tag}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi(col, label, value, sub="", badge=""):
    """Render a styled KPI card in the given column."""
    badge_html = f'<div class="kpi-badge">{badge}</div>' if badge else ""
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title, subtitle=""):
    """Render a section header with optional subtitle."""
    import streamlit as st
    sub_html = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <p class="section-title">{title}</p>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_divider():
    """Render a styled horizontal divider."""
    import streamlit as st
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def chart_card_open(title, desc=""):
    """Open a chart card container — call before st.plotly_chart."""
    import streamlit as st
    desc_html = f'<div class="chart-desc">{desc}</div>' if desc else ""
    st.markdown(f'<div class="chart-card"><div class="chart-title">{title}</div>{desc_html}', unsafe_allow_html=True)


def chart_card_close():
    """Close a chart card container — call after st.plotly_chart."""
    import streamlit as st
    st.markdown('</div>', unsafe_allow_html=True)


def render_insight_strip(cards):
    """Render insight strip. cards = list of (icon, headline, detail)."""
    import streamlit as st
    if not cards:
        return
    html = "".join([
        f'<div class="insight-card">'
        f'<div class="insight-icon">{icon}</div>'
        f'<div><div class="insight-headline">{hl}</div>'
        f'<div class="insight-detail">{detail}</div></div>'
        f'</div>'
        for icon, hl, detail in cards
    ])
    st.markdown(f'<div class="insight-strip">{html}</div>', unsafe_allow_html=True)


def render_ml_metric(col, value, label, badge=""):
    """Render an ML metric card in the given column."""
    badge_html = f'<div class="ml-metric-badge">{badge}</div>' if badge else ""
    col.markdown(f"""
    <div class="ml-metric">
        <div class="ml-metric-val">{value}</div>
        <div class="ml-metric-label">{label}</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_info_banner(title, text):
    """Render a blue info banner."""
    import streamlit as st
    st.markdown(f"""
    <div class="info-banner">
        <div class="info-banner-title">{title}</div>
        <div class="info-banner-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def apply_chart_theme(fig, height=None, legend=True, margins=None):
    """Apply the reference.py chart theme to a Plotly figure."""
    m = margins or dict(l=10, r=10, t=35, b=10)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=TEXT_DARK),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=11), title=""),
        margin=m,
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=TEXT_DARK,
                        bordercolor=BORDER),
        **({"height": height} if height else {}),
    )
    fig.update_xaxes(showgrid=False, zeroline=False,
                     tickfont=dict(size=11, color=TEXT_DARK),
                     title_font=dict(size=12, color=TEXT_DARK))
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False,
                     tickfont=dict(size=11, color=TEXT_DARK),
                     title_font=dict(size=12, color=TEXT_DARK))
    return fig
