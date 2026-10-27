"""
UI/UX components, CSS design system, Plotly theme stylers,
and reusable dashboard widgets for LUX - Streetlight Intelligence.
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Color Tokens
COLOR_PRIMARY = "#00e5ff"       # Electric Cyan
COLOR_SECONDARY = "#38bdf8"     # Sky Blue
COLOR_ACCENT = "#818cf8"        # Indigo
COLOR_SUCCESS = "#10b981"       # Emerald
COLOR_WARNING = "#f59e0b"       # Amber
COLOR_DANGER = "#ef4444"        # Crimson
COLOR_BG_CARD = "#131e32"       # Dark Slate Blue Card
COLOR_TEXT_MUTED = "#94a3b8"    # Slate 400

PLOTLY_COLORWAY = [
    "#00e5ff", "#38bdf8", "#10b981", "#f59e0b", 
    "#ef4444", "#a855f7", "#ec4899", "#14b8a6"
]

def apply_custom_css():
    """Injects modern dark-surface executive dashboard styles."""
    st.markdown("""
    <style>
    /* Main Background and Typography */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Top Header Banner */
    .lux-header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .lux-brand-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00e5ff 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: inline-block;
    }
    .lux-subtitle {
        color: #94a3b8;
        font-size: 13.5px;
        margin-top: 4px;
        margin-bottom: 12px;
        font-weight: 500;
    }
    .lux-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 8px;
    }
    .badge-synthetic {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .badge-telemetry {
        background-color: rgba(0, 229, 255, 0.12);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.3);
    }
    .badge-status {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* KPI Cards */
    .kpi-card {
        background: #131e32;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
        margin-bottom: 14px;
    }
    .kpi-card:hover {
        border-color: rgba(0, 229, 255, 0.35);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 25px;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
        margin-bottom: 4px;
        font-feature-settings: "tnum";
    }
    .kpi-subtext {
        font-size: 11.5px;
        color: #64748b;
        font-weight: 500;
    }
    .kpi-accent-cyan { color: #00e5ff; }
    .kpi-accent-emerald { color: #10b981; }
    .kpi-accent-amber { color: #f59e0b; }
    .kpi-accent-crimson { color: #ef4444; }

    /* Section Subheaders */
    .section-header {
        font-size: 17px;
        font-weight: 700;
        color: #f1f5f9;
        border-bottom: 2px solid rgba(0, 229, 255, 0.25);
        padding-bottom: 6px;
        margin-top: 18px;
        margin-bottom: 16px;
        letter-spacing: -0.2px;
    }

    /* Alert / Callout Boxes */
    .callout-box {
        background: rgba(15, 23, 42, 0.7);
        border-left: 4px solid #00e5ff;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 16px 0;
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.5;
    }
    .callout-warning {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.08);
    }

    /* Streamlit Tables & DataFrames */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        background: #0f172a;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(total_records: int, date_start: str, date_end: str, current_filtered_count: int):
    """Renders the executive application branding header."""
    pct_filtered = (current_filtered_count / total_records * 100) if total_records > 0 else 100.0
    st.markdown(f"""
    <div class="lux-header-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <h1 class="lux-brand-title">LUX | Streetlight Intelligence</h1>
                <div class="lux-subtitle">
                    Data-Driven Smart Streetlighting Analytics & Energy Profile Optimization Platform
                </div>
            </div>
            <div style="margin-top: 4px;">
                <span class="lux-badge badge-synthetic">⚠️ Synthetic Dataset (Simulated)</span>
                <span class="lux-badge badge-telemetry">24 Grid Segments | {date_start} to {date_end}</span>
                <span class="lux-badge badge-status">Active View: {current_filtered_count:,} records ({pct_filtered:.1f}%)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, subtext: str, accent: str = "cyan"):
    """Renders a single styled executive metric card."""
    accent_class = f"kpi-accent-{accent}"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {accent_class}">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: Optional[str] = None):
    """Renders a stylized section title."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def style_plotly_chart(fig: go.Figure, height: int = 420, show_legend: bool = True) -> go.Figure:
    """Applies high-end dark theme styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=11, color="#cbd5e1"),
        title=dict(font=dict(size=14, color="#f8fafc", family="Inter, sans-serif")),
        height=height,
        margin=dict(l=45, r=25, t=45, b=40),
        colorway=PLOTLY_COLORWAY,
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(size=10, color="#94a3b8")
        )
    )
    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.15)"
    )
    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.07)",
        zerolinecolor="rgba(255, 255, 255, 0.12)",
        linecolor="rgba(255, 255, 255, 0.15)"
    )
    return fig


def render_synthetic_disclosure():
    """Displays the academic integrity disclosure banner."""
    st.markdown("""
    <div class="callout-box callout-warning">
        <strong>Academic Disclosure & Modeling Caveat:</strong><br>
        All observations, traffic counts, environmental conditions, and energy readings are generated from a 
        synthetic computer simulation (<code>SEED = 42</code>) for an archetype warm South Indian city. 
        Reported baseline differences and optimization scenarios are mathematical models based on component 
        power equations. They have <strong>not been calibrated against physical field telemetry or measured 
        road-surface luminance</strong>. Deployment requires engineering validation against applicable lighting 
        standards (EN 13201 / IESNA RP-8 / IS 1944).
    </div>
    """, unsafe_allow_html=True)
