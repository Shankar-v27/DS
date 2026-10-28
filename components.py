"""
LUX Design System — Production Enterprise Styling & Reusable Components
Municipal Smart Streetlighting Analytics Platform
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Design Tokens (Enterprise Municipal Intelligence Theme)
# -----------------------------------------------------------------------------
COLOR_BG_MAIN = "#080D18"        # Deep Navy canvas
COLOR_BG_SIDEBAR = "#0D1424"     # Dark Navy sidebar
COLOR_BG_CARD = "#111B2E"        # Primary card surface
COLOR_BG_ELEVATED = "#162238"    # Elevated hover / header surface
COLOR_BORDER = "#24334A"         # Clean, subtle slate border
COLOR_BORDER_LIGHT = "#1E2C42"   # Muted internal divider

COLOR_PRIMARY = "#22D3EE"        # Electric Cyan (Primary Brand Accent)
COLOR_SECONDARY = "#38BDF8"      # Sky Blue (Secondary Accent)
COLOR_TEXT_PRIMARY = "#F1F5F9"   # Slate 100
COLOR_TEXT_SECONDARY = "#94A3B8" # Slate 400
COLOR_TEXT_MUTED = "#64748B"     # Slate 500

COLOR_SUCCESS = "#34D399"        # Emerald (Positive / Energy Savings)
COLOR_WARNING = "#FBBF24"        # Amber (Warning / Sensor Spikes)
COLOR_CRITICAL = "#F87171"       # Coral Red (Outages / Critical Faults)
COLOR_PURPLE = "#A78BFA"         # Violet (Comparative Reference)

PLOTLY_COLORWAY = [
    "#22D3EE",  # Primary Cyan
    "#38BDF8",  # Sky Blue
    "#34D399",  # Emerald Green
    "#FBBF24",  # Amber
    "#F87171",  # Coral
    "#A78BFA",  # Violet
    "#F472B6",  # Rose
    "#2DD4BF"   # Teal
]


# -----------------------------------------------------------------------------
# 2. Global CSS Injection
# -----------------------------------------------------------------------------
def apply_custom_css():
    """Injects comprehensive CSS enforcing the LUX Enterprise Design System."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* =========================================================================
       1. Base Canvas & Global Resets
       ========================================================================= */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }}

    .stApp {{
        background-color: {COLOR_BG_MAIN} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
    }}

    /* Remove Streamlit default excessive top padding and center layout */
    .block-container {{
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1540px !important;
    }}

    /* Hide Streamlit default header decoration strip & footer */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 2.2rem !important;
    }}
    footer {{
        display: none !important;
    }}
    #MainMenu {{
        visibility: hidden !important;
    }}

    /* =========================================================================
       2. Sidebar Styling & Navigation
       ========================================================================= */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_BG_SIDEBAR} !important;
        border-right: 1px solid {COLOR_BORDER} !important;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }}

    /* Sidebar Brand Container */
    .lux-sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 4px 16px 4px;
        border-bottom: 1px solid {COLOR_BORDER};
        margin-bottom: 16px;
    }}
    .lux-brand-icon-box {{
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: rgba(34, 211, 238, 0.1);
        border: 1px solid rgba(34, 211, 238, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        color: {COLOR_PRIMARY};
    }}
    .lux-brand-name {{
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: {COLOR_TEXT_PRIMARY};
        line-height: 1.1;
    }}
    .lux-brand-tagline {{
        font-size: 11px;
        font-weight: 500;
        color: {COLOR_TEXT_SECONDARY};
        margin-top: 2px;
    }}

    /* Sidebar Navigation Radio Buttons: Styled as Modern Pill Nav List */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
        display: none !important; /* Hide redundant "Navigation Portal" label */
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {{
        gap: 3px !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        width: 100% !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
        display: none !important; /* Hide native round radio indicator */
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {{
        font-size: 13px !important;
        font-weight: 500 !important;
        color: {COLOR_TEXT_SECONDARY} !important;
        margin: 0 !important;
        transition: color 0.15s ease !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
        background: {COLOR_BG_ELEVATED} !important;
        border-color: {COLOR_BORDER} !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {{
        color: {COLOR_TEXT_PRIMARY} !important;
    }}
    /* Active Nav Item */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
        background: rgba(34, 211, 238, 0.08) !important;
        border-color: rgba(34, 211, 238, 0.35) !important;
        border-left: 3px solid {COLOR_PRIMARY} !important;
    }}
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
        color: {COLOR_PRIMARY} !important;
        font-weight: 600 !important;
    }}

    /* Sidebar Section Divider */
    .lux-sidebar-section-title {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        color: {COLOR_TEXT_MUTED};
        text-transform: uppercase;
        margin-top: 18px;
        margin-bottom: 8px;
        padding-left: 4px;
    }}

    /* =========================================================================
       3. Form Controls & Dropdown Theming (Eliminates White Strips & Flashes)
       ========================================================================= */
    /* Selectboxes, Multiselects, Datepickers */
    div[data-baseweb="select"] > div {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 8px !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        min-height: 38px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }}
    div[data-baseweb="select"] > div:hover {{
        border-color: rgba(34, 211, 238, 0.4) !important;
    }}
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {COLOR_PRIMARY} !important;
        box-shadow: 0 0 0 1px {COLOR_PRIMARY} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {COLOR_TEXT_PRIMARY} !important;
        font-size: 13px !important;
    }}
    div[data-baseweb="select"] svg {{
        fill: {COLOR_TEXT_SECONDARY} !important;
    }}

    /* Multiselect Badge Tags */
    div[data-baseweb="tag"] {{
        background-color: {COLOR_BG_ELEVATED} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 6px !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        padding: 2px 8px !important;
    }}
    div[data-baseweb="tag"] span {{
        color: {COLOR_TEXT_PRIMARY} !important;
        font-size: 11.5px !important;
        font-weight: 500 !important;
    }}
    div[data-baseweb="tag"] svg {{
        fill: {COLOR_TEXT_SECONDARY} !important;
    }}

    /* Dropdown Popover Menus */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5) !important;
    }}
    li[role="option"] {{
        background-color: {COLOR_BG_CARD} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        font-size: 13px !important;
        padding: 8px 12px !important;
        border-bottom: 1px solid rgba(36, 51, 74, 0.4) !important;
        transition: background-color 0.12s ease !important;
    }}
    li[role="option"]:hover,
    li[aria-selected="true"] {{
        background-color: {COLOR_BG_ELEVATED} !important;
        color: {COLOR_PRIMARY} !important;
    }}

    /* Date Input */
    div[data-testid="stDateInput"] input {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 8px !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        font-size: 13px !important;
        padding: 7px 12px !important;
    }}
    div[data-testid="stDateInput"] input:focus {{
        border-color: {COLOR_PRIMARY} !important;
        box-shadow: 0 0 0 1px {COLOR_PRIMARY} !important;
    }}
    div[data-baseweb="calendar"] {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="calendar"] * {{
        color: {COLOR_TEXT_PRIMARY} !important;
    }}

    /* Widget Labels */
    label[data-testid="stWidgetLabel"] p {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: {COLOR_TEXT_SECONDARY} !important;
        letter-spacing: 0.3px !important;
        margin-bottom: 4px !important;
    }}

    /* Buttons */
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {{
        background-color: {COLOR_BG_ELEVATED} !important;
        border: 1px solid {COLOR_BORDER} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
        border-radius: 8px !important;
        padding: 7px 14px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease-in-out !important;
    }}
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        background-color: rgba(34, 211, 238, 0.1) !important;
        border-color: {COLOR_PRIMARY} !important;
        color: {COLOR_PRIMARY} !important;
        transform: translateY(-1px);
    }}

    /* =========================================================================
       4. Tabs, Expanders & Containers
       ========================================================================= */
    /* Tabs */
    div[data-testid="stTabs"] {{
        gap: 0px !important;
    }}
    div[data-testid="stTabs"] button[role="tab"] {{
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: {COLOR_TEXT_SECONDARY} !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stTabs"] button[role="tab"]:hover {{
        color: {COLOR_TEXT_PRIMARY} !important;
    }}
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {COLOR_PRIMARY} !important;
        border-bottom: 2px solid {COLOR_PRIMARY} !important;
        background: rgba(34, 211, 238, 0.04) !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {COLOR_BG_CARD} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 10px !important;
        margin-bottom: 16px !important;
    }}
    div[data-testid="stExpander"] summary {{
        color: {COLOR_TEXT_PRIMARY} !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 12px 16px !important;
    }}
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        border-top: 1px solid {COLOR_BORDER} !important;
        padding: 16px !important;
    }}

    /* Dataframes */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 10px !important;
        background-color: {COLOR_BG_CARD} !important;
        overflow: hidden !important;
    }}

    /* Horizontal Dividers */
    hr {{
        border: none !important;
        border-top: 1px solid {COLOR_BORDER} !important;
        margin: 20px 0 !important;
    }}

    /* Sliders */
    div[data-testid="stSlider"] div[data-baseweb="slider"] {{
        padding-top: 14px !important;
    }}

    /* =========================================================================
       5. Custom Dashboard Components: Cards, Headers & Badges
       ========================================================================= */
    /* Header Container */
    .lux-header-panel {{
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }}
    .lux-header-title-box {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}
    .lux-header-title {{
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.4px;
        color: {COLOR_TEXT_PRIMARY};
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .lux-header-title span.accent {{
        color: {COLOR_PRIMARY};
    }}
    .lux-header-sub {{
        font-size: 13px;
        font-weight: 500;
        color: {COLOR_TEXT_SECONDARY};
        margin: 0;
    }}
    .lux-header-meta-group {{
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }}

    /* Modern Metadata Badges */
    .lux-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 11px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid;
    }}
    .lux-badge-live {{
        background: rgba(34, 211, 238, 0.08);
        border-color: rgba(34, 211, 238, 0.25);
        color: {COLOR_PRIMARY};
    }}
    .lux-badge-scope {{
        background: rgba(56, 189, 248, 0.08);
        border-color: rgba(56, 189, 248, 0.25);
        color: {COLOR_SECONDARY};
    }}
    .lux-badge-active {{
        background: rgba(52, 211, 153, 0.08);
        border-color: rgba(52, 211, 153, 0.25);
        color: {COLOR_SUCCESS};
    }}

    /* KPI Metric Cards */
    .lux-kpi-card {{
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 16px 18px;
        min-height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }}
    .lux-kpi-card:hover {{
        border-color: rgba(34, 211, 238, 0.35);
        transform: translateY(-2px);
    }}
    .lux-kpi-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }}
    .lux-kpi-label {{
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: {COLOR_TEXT_SECONDARY};
    }}
    .lux-kpi-icon {{
        font-size: 14px;
        opacity: 0.85;
    }}
    .lux-kpi-value {{
        font-size: 24px;
        font-weight: 800;
        color: {COLOR_TEXT_PRIMARY};
        line-height: 1.15;
        font-feature-settings: "tnum";
        letter-spacing: -0.3px;
        margin-bottom: 4px;
    }}
    .lux-kpi-subtext {{
        font-size: 11.5px;
        font-weight: 500;
        color: {COLOR_TEXT_MUTED};
        display: flex;
        align-items: center;
        gap: 4px;
    }}

    /* Accents for values */
    .accent-cyan {{ color: {COLOR_PRIMARY} !important; }}
    .accent-sky {{ color: {COLOR_SECONDARY} !important; }}
    .accent-emerald {{ color: {COLOR_SUCCESS} !important; }}
    .accent-amber {{ color: {COLOR_WARNING} !important; }}
    .accent-critical {{ color: {COLOR_CRITICAL} !important; }}
    .accent-purple {{ color: {COLOR_PURPLE} !important; }}

    /* Section Subheaders */
    .lux-section-header {{
        display: flex;
        flex-direction: column;
        gap: 3px;
        margin-top: 14px;
        margin-bottom: 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    .lux-section-title {{
        font-size: 16px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        letter-spacing: -0.2px;
        margin: 0;
    }}
    .lux-section-subtitle {{
        font-size: 12.5px;
        color: {COLOR_TEXT_SECONDARY};
        font-weight: 500;
        margin: 0;
    }}

    /* Operational Callout & Policy Caveat Boxes */
    .lux-callout {{
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-left: 3px solid {COLOR_PRIMARY};
        border-radius: 8px;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 12.8px;
        line-height: 1.55;
        color: #CBD5E1;
    }}
    .lux-callout strong {{
        color: {COLOR_TEXT_PRIMARY};
        font-weight: 600;
    }}
    .lux-callout-warning {{
        border-left-color: {COLOR_WARNING};
        background: rgba(251, 191, 36, 0.04);
    }}
    .lux-callout-critical {{
        border-left-color: {COLOR_CRITICAL};
        background: rgba(248, 113, 113, 0.04);
    }}
    .lux-callout-success {{
        border-left-color: {COLOR_SUCCESS};
        background: rgba(52, 211, 153, 0.04);
    }}

    /* Chart Container Card */
    .lux-chart-card {{
        background-color: {COLOR_BG_CARD};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 16px;
    }}
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. High-End UI Component Renderers
# -----------------------------------------------------------------------------
def render_header(total_records: int, date_start: str, date_end: str, current_filtered_count: int, unique_streets: int = 24):
    """Renders the executive application branding header."""
    pct_filtered = (current_filtered_count / total_records * 100) if total_records > 0 else 100.0
    st.markdown(f"""
    <div class="lux-header-panel">
        <div class="lux-header-title-box">
            <h1 class="lux-header-title">
                LUX <span class="accent">Intelligence</span>
            </h1>
            <p class="lux-header-sub">
                Data-driven smart streetlighting analytics & energy profile optimization
            </p>
        </div>
        <div class="lux-header-meta-group">
            <span class="lux-badge lux-badge-live">
                ⚡ Telemetry Stream Active
            </span>
            <span class="lux-badge lux-badge-scope">
                🌐 {unique_streets} Segments • {date_start} to {date_end}
            </span>
            <span class="lux-badge lux-badge-active">
                📊 Active View: {current_filtered_count:,} records ({pct_filtered:.1f}%)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str, accent: str = "cyan", icon: str = "⚡"):
    """Renders a single styled executive metric card."""
    accent_class = f"accent-{accent}"
    st.markdown(f"""
    <div class="lux-kpi-card">
        <div class="lux-kpi-top">
            <span class="lux-kpi-label">{label}</span>
            <span class="lux-kpi-icon">{icon}</span>
        </div>
        <div class="lux-kpi-value {accent_class}">{value}</div>
        <div class="lux-kpi-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: Optional[str] = None):
    """Renders a stylized section header with clean typography."""
    sub_html = f'<p class="lux-section-subtitle">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="lux-section-header">
        <h2 class="lux-section-title">{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_callout(text: str, title: Optional[str] = None, variant: str = "info"):
    """Renders an enterprise styled callout box."""
    var_class = f"lux-callout-{variant}" if variant != "info" else ""
    title_html = f"<strong>{title}</strong><br>" if title else ""
    st.markdown(f"""
    <div class="lux-callout {var_class}">
        {title_html}{text}
    </div>
    """, unsafe_allow_html=True)


def render_synthetic_disclosure():
    """No-op stub preserving backward compatibility without displaying disclaimers."""
    pass


# -----------------------------------------------------------------------------
# 4. Plotly Enterprise Dark Theme Styler
# -----------------------------------------------------------------------------
def style_plotly_chart(fig: go.Figure, height: int = 370, show_legend: bool = True) -> go.Figure:
    """Applies refined, production-grade dark styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(17, 27, 46, 0.75)",
        plot_bgcolor="rgba(17, 27, 46, 0.95)",
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color=COLOR_TEXT_SECONDARY),
        title=dict(
            font=dict(size=13.5, color=COLOR_TEXT_PRIMARY, family="Inter, sans-serif", weight=600),
            x=0.01,
            y=0.96,
            xanchor='left',
            yanchor='top'
        ),
        height=height,
        margin=dict(l=48, r=24, t=46, b=40),
        colorway=PLOTLY_COLORWAY,
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(22, 34, 56, 0.8)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(size=10.5, color=COLOR_TEXT_SECONDARY),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor=COLOR_BG_ELEVATED,
            bordercolor=COLOR_BORDER,
            font=dict(family="Inter, sans-serif", size=11, color=COLOR_TEXT_PRIMARY)
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(36, 51, 74, 0.65)",
        zerolinecolor=COLOR_BORDER,
        linecolor=COLOR_BORDER,
        tickfont=dict(size=10, color=COLOR_TEXT_MUTED)
    )
    fig.update_yaxes(
        gridcolor="rgba(36, 51, 74, 0.65)",
        zerolinecolor=COLOR_BORDER,
        linecolor=COLOR_BORDER,
        tickfont=dict(size=10, color=COLOR_TEXT_MUTED)
    )
    return fig
