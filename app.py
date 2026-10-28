"""
LUX - Smart Streetlighting Intelligence Platform
Production Enterprise Streamlit Application Entrypoint

Project Theme: Data-Driven Smart Streetlighting: Optimizing Energy Usage Profiles Through Exploratory Data Analysis.
"""

import sys
import io
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local Modules
import data_loader
import analytics
import components
import ml_energy_predictor

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LUX | Smart Streetlighting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

components.apply_custom_css()

# -----------------------------------------------------------------------------
# 2. Data Loading & Session Initialization
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset():
    return data_loader.load_dataset()

@st.cache_resource(show_spinner="Training machine learning energy surrogate...")
def get_cached_ml_model(_df):
    return ml_energy_predictor.train_energy_model(_df, sample_size=50000)


try:
    df_raw = get_cached_dataset()
    filter_opts = data_loader.get_filter_options(df_raw)
except Exception as e:
    st.error(f"Critical Dataset Loading Failure: {e}")
    st.info("Please ensure that 'smart_streetlighting_synthetic_dataset.csv' is present in the workspace.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Global Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    # Brand Header
    st.markdown("""
    <div class="lux-sidebar-brand">
        <div class="lux-brand-icon-box">⚡</div>
        <div>
            <div class="lux-brand-name">LUX</div>
            <div class="lux-brand-tagline">Smart Streetlight Intelligence</div>
        </div>
    </div>
    <div class="lux-sidebar-section-title">NAVIGATION PORTAL</div>
    """, unsafe_allow_html=True)

    nav_page = st.radio(
        "Navigation Portal",
        [
            "📊 Executive Overview",
            "⚡ Energy Consumption Analytics",
            "🛣️ Street & Infrastructure Explorer",
            "🚦 Traffic, Weather & Lighting Dynamics",
            "🛡️ Data Quality & Anomaly Monitor",
            "🚀 Energy Optimization Simulator",
            "📥 Reports & Data Export"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="lux-sidebar-section-title" style="margin-top: 24px;">GLOBAL FILTERS</div>
    """, unsafe_allow_html=True)

    # Reset Filters Button
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state['f_streets'] = []
        st.session_state['f_zones'] = []
        st.session_state['f_road_classes'] = []
        st.session_state['f_tech'] = []
        st.session_state['f_modes'] = []
        st.session_state['f_status'] = []
        st.session_state['f_quality'] = []
        st.session_state['f_date_range'] = (filter_opts['min_date'], filter_opts['max_date'])
        st.rerun()

    # Date Range Filter
    date_val = st.date_input(
        "Calendar Date Range",
        value=(filter_opts['min_date'], filter_opts['max_date']),
        min_value=filter_opts['min_date'],
        max_value=filter_opts['max_date'],
        key="f_date_range"
    )
    if isinstance(date_val, tuple) and len(date_val) == 2:
        date_filter = date_val
    else:
        date_filter = (filter_opts['min_date'], filter_opts['max_date'])

    # Multiselect filters
    sel_streets = st.multiselect("Street Segment (ID)", filter_opts['street_ids'], key="f_streets")
    sel_zones = st.multiselect("Zone Classification", filter_opts['zone_types'], key="f_zones")
    sel_roads = st.multiselect("Road Class", filter_opts['road_classes'], key="f_road_classes")
    sel_tech = st.multiselect("Lamp Technology", filter_opts['lamp_technologies'], key="f_tech")
    sel_modes = st.multiselect("Installed Control Mode", filter_opts['control_modes'], key="f_modes")
    sel_status = st.multiselect("Lamp Operating Status", filter_opts['lamp_statuses'], key="f_status")
    sel_quality = st.multiselect("Data Quality Flag", filter_opts['data_quality_flags'], key="f_quality")

    # Global filter dictionary
    active_filters = {
        'date_range': date_filter,
        'streets': sel_streets,
        'zones': sel_zones,
        'road_classes': sel_roads,
        'technologies': sel_tech,
        'modes': sel_modes,
        'lamp_statuses': sel_status,
        'quality_flags': sel_quality
    }

# Apply global filters to create active view
df_view = data_loader.apply_global_filters(df_raw, active_filters)
kpis = analytics.calculate_executive_kpis(df_view)

# Render Global Executive Branding Header
components.render_header(
    total_records=len(df_raw),
    date_start=df_raw['date_dt'].min().strftime('%Y-%m-%d'),
    date_end=df_raw['date_dt'].max().strftime('%Y-%m-%d'),
    current_filtered_count=len(df_view),
    unique_streets=kpis['unique_streets']
)

# Empty State Warning
if len(df_view) == 0:
    components.render_callout(
        "No records match the active sidebar filter criteria. Please broaden your selections or click 'Reset All Filters'.",
        title="Zero Filter Matches",
        variant="warning"
    )
    st.stop()


# =============================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# =============================================================================
if nav_page == "📊 Executive Overview":
    components.render_section_header("System Overview & Key Performance Indicators", "Executive operational telemetry across municipal lighting circuits")

    # Row 1: Primary Energy & Scope Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        components.render_kpi_card(
            "Total Observed Energy",
            f"{kpis['total_energy_kwh']:,.1f} kWh",
            f"Valid subset: {kpis['valid_energy_kwh']:,.1f} kWh",
            accent="cyan",
            icon="⚡"
        )
    with c2:
        components.render_kpi_card(
            "Full-Output Reference",
            f"{kpis['full_output_kwh']:,.1f} kWh",
            "Theoretical continuous load",
            accent="purple",
            icon="💡"
        )
    with c3:
        components.render_kpi_card(
            "Baseline Savings",
            f"{kpis['valid_savings_pct']:.2f}%",
            f"{kpis['valid_diff_kwh']:,.0f} kWh reduction vs. full",
            accent="emerald",
            icon="🌿"
        )
    with c4:
        components.render_kpi_card(
            "Active Grid Scope",
            f"{kpis['unique_streets']} Segments",
            f"{kpis['total_records']:,} observations (2025)",
            accent="sky",
            icon="🛣️"
        )

    # Row 2: Operational Quality & Reliability Metrics
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        components.render_kpi_card(
            "Mean Night Dimming",
            f"{kpis['avg_measured_dimming']:.1f}%",
            f"Command setpoint: {kpis['avg_commanded_dimming']:.1f}%",
            accent="cyan",
            icon="🌙"
        )
    with c6:
        components.render_kpi_card(
            "Cleanliness Ratio",
            f"{kpis['clean_pct']:.2f}%",
            f"{kpis['clean_records']:,} uncorrupted records",
            accent="emerald",
            icon="🛡️"
        )
    with c7:
        components.render_kpi_card(
            "Sensor Anomalies",
            f"{kpis['sensor_anomalies']:,}",
            f"{kpis['sensor_anomaly_pct']:.2f}% of active records",
            accent="amber",
            icon="⚠️"
        )
    with c8:
        components.render_kpi_card(
            "Simulated Outages",
            f"{kpis['lamp_faults']:,}",
            f"{kpis['lamp_fault_pct']:.2f}% (zero-output events)",
            accent="critical",
            icon="🚨"
        )

    components.render_section_header("Diurnal & Temporal Grid Load Patterns", "High-resolution time-series load curves and composite diurnal profiles")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Daily Grid Consumption Time Series
        daily_df = analytics.get_daily_trend(df_view)
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_df['date_dt'], y=daily_df['total_energy'],
            mode='lines', name='Daily Total (kWh)',
            line=dict(color='rgba(34, 211, 238, 0.4)', width=1.2)
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily_df['date_dt'], y=daily_df['rolling_7d'],
            mode='lines', name='7-Day Centered MA',
            line=dict(color='#22D3EE', width=2.5)
        ))
        fig_daily.update_layout(
            title="365-Day Energy Consumption Trend (2025)",
            xaxis_title="Calendar Date", yaxis_title="Total Daily Energy (kWh)"
        )
        components.style_plotly_chart(fig_daily, height=360)
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_chart2:
        # Diurnal 24-Hour Profile
        hourly_df = analytics.get_hourly_profile(df_view)
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly_df['hour'], y=hourly_df['total_energy'],
            name='Total Grid Energy (kWh)',
            marker=dict(color='#38BDF8', line=dict(color='#0284C7', width=1))
        ))
        fig_hourly.add_trace(go.Scatter(
            x=hourly_df['hour'], y=hourly_df['mean_energy'] * kpis['unique_streets'],
            mode='lines+markers', name='Expected Active Load',
            line=dict(color='#FBBF24', width=2)
        ))
        fig_hourly.update_layout(
            title="Diurnal Hourly Profile (All Grid Segments Combined)",
            xaxis_title="Hour of Day (0–23)", yaxis_title="Annual Energy (kWh)",
            xaxis=dict(tickmode='linear', tick0=0, dtick=2)
        )
        components.style_plotly_chart(fig_hourly, height=360)
        st.plotly_chart(fig_hourly, use_container_width=True)

    # Zone Breakdown and Traffic Interaction
    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        zone_df = analytics.get_zone_summary(df_view)
        fig_zone = go.Figure()
        fig_zone.add_trace(go.Bar(
            x=zone_df['zone_type'], y=zone_df['total_energy'],
            name='Observed Energy (kWh)', marker_color='#22D3EE'
        ))
        fig_zone.add_trace(go.Bar(
            x=zone_df['zone_type'], y=zone_df['full_output'],
            name='Full Output Reference (kWh)', marker_color='rgba(255, 255, 255, 0.22)'
        ))
        fig_zone.update_layout(
            title="Energy Consumption & Benchmark by Zone Classification",
            xaxis_title="Zone Classification", yaxis_title="Energy (kWh)",
            barmode='group'
        )
        components.style_plotly_chart(fig_zone, height=350)
        st.plotly_chart(fig_zone, use_container_width=True)

    with col_chart4:
        # Sample active night traffic vs dimming
        night_sample = df_view[df_view['lighting_required_flag'] == 1].sample(
            min(1500, (df_view['lighting_required_flag'] == 1).sum()), random_state=42
        )
        fig_scatter = px.scatter(
            night_sample, x='traffic_count', y='measured_dimming_pct',
            color='zone_type', opacity=0.6,
            title="Night Illumination Level (%) vs. Vehicular Traffic Demand",
            labels={'traffic_count': 'Traffic Count (Vehicles/hr)', 'measured_dimming_pct': 'Measured Dimming (%)'}
        )
        components.style_plotly_chart(fig_scatter, height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Executive Analytical Findings Callout Box
    components.render_callout("""
        <ul>
            <li><strong>Diurnal Concentration:</strong> Over 99.8% of total grid electrical energy is consumed between 18:00 and 06:00, peaking during evening vehicular rush hours (19:00–21:00).</li>
            <li><strong>Infrastructure Disparities:</strong> Arterial zones consume the highest aggregate energy due to road width and fixture density. Conversely, Campus streets exhibit the highest specific energy per fixture (557 kWh/fixture) due to legacy High-Pressure Sodium lamps.</li>
            <li><strong>Static Mode Inefficiency:</strong> The 6 Fixed-mode circuits run at constant 100% capacity all night, producing 0.00% baseline savings and presenting the primary opportunity for schedule optimization.</li>
        </ul>
    """, title="Executive Analytical Findings:", variant="info")

    components.render_synthetic_disclosure()


# =============================================================================
# PAGE 2 — ENERGY CONSUMPTION ANALYTICS
# =============================================================================
elif nav_page == "⚡ Energy Consumption Analytics":
    components.render_section_header("Temporal Dynamics & Infrastructure Energy Analytics", "Multi-dimensional examination of streetlighting consumption profiles")

    tab_temporal, tab_infra = st.tabs(["🕒 Temporal & Seasonal Patterns", "🏗️ Hardware & Infrastructure Analytics"])

    with tab_temporal:
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            # Hourly Profile by Control Mode
            mode_hourly = df_view[df_view['is_valid_energy']].groupby(['hour', 'control_mode'])['energy_consumption_kwh'].mean().reset_index()
            fig_mode_h = px.line(
                mode_hourly, x='hour', y='energy_consumption_kwh', color='control_mode',
                title="Average Hourly Consumption by Installed Control Mode",
                labels={'hour': 'Hour of Day (0–23)', 'energy_consumption_kwh': 'Avg Energy (kWh / segment-hr)'},
                markers=True,
                color_discrete_map={'Fixed': '#F87171', 'Scheduled dimming': '#FBBF24', 'Adaptive dimming': '#22D3EE'}
            )
            components.style_plotly_chart(fig_mode_h, height=380)
            st.plotly_chart(fig_mode_h, use_container_width=True)

        with col_t2:
            # Weekday vs Weekend Hourly Profile
            dow_hourly = df_view[df_view['is_valid_energy']].groupby(['hour', 'is_weekend'])['energy_consumption_kwh'].mean().reset_index()
            dow_hourly['Day Type'] = dow_hourly['is_weekend'].map({0: 'Weekday (Mon–Fri)', 1: 'Weekend (Sat–Sun)'})
            fig_dow = px.line(
                dow_hourly, x='hour', y='energy_consumption_kwh', color='Day Type',
                title="Diurnal Consumption: Weekdays vs. Weekends",
                labels={'hour': 'Hour of Day (0–23)', 'energy_consumption_kwh': 'Avg Energy (kWh / segment-hr)'},
                markers=True, color_discrete_sequence=['#22D3EE', '#FBBF24']
            )
            components.style_plotly_chart(fig_dow, height=380)
            st.plotly_chart(fig_dow, use_container_width=True)

        # Monthly Consumption with Daily Normalization
        monthly_df = analytics.get_monthly_summary(df_view)
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly_df['month_name'], y=monthly_df['total_energy'],
            name='Total Monthly Energy (kWh)', marker_color='#38BDF8', yaxis='y1'
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly_df['month_name'], y=monthly_df['energy_kwh_per_day'],
            name='Daily-Normalized Energy (kWh/day)', line=dict(color='#34D399', width=3),
            yaxis='y2', mode='lines+markers'
        ))
        fig_monthly.update_layout(
            title="Monthly Energy Consumption: Total vs. Daily-Normalized (kWh/day)",
            xaxis_title="Month of Year (2025)",
            yaxis=dict(title=dict(text="Total Monthly Energy (kWh)", font=dict(color="#38BDF8"))),
            yaxis2=dict(
                title=dict(text="Daily-Normalized Energy (kWh/day)", font=dict(color="#34D399")),
                overlaying='y', side='right'
            )
        )
        components.style_plotly_chart(fig_monthly, height=390)
        st.plotly_chart(fig_monthly, use_container_width=True)

        components.render_callout(
            "Notice how daily-normalized energy drops from ~559.5 kWh/day in Jan–May down to ~507.0 kWh/day in Jul–Nov. "
            "Earlier sunrises and later sunsets in summer shorten the dark operating window from 12.0 hours to 11.0 hours, "
            "producing an organic 9.4% seasonal reduction that must be accounted for in municipal baseline budgeting.",
            title="Seasonal Solar Dynamics:",
            variant="info"
        )

    with tab_infra:
        col_i1, col_i2 = st.columns(2)

        with col_i1:
            # Technology Comparison: LED vs HPS
            tech_agg = df_view[df_view['is_valid_energy']].groupby('lamp_technology').agg(
                total_energy=('energy_consumption_kwh', 'sum'),
                luminaires=('luminaire_count', lambda x: df_raw[df_raw['lamp_technology'] == x.name].drop_duplicates('street_id')['luminaire_count'].sum())
            ).reset_index()
            tech_agg['kwh_per_luminaire'] = tech_agg['total_energy'] / tech_agg['luminaires']

            fig_tech = px.bar(
                tech_agg, x='lamp_technology', y='kwh_per_luminaire', color='lamp_technology',
                title="Specific Energy Consumption by Lamp Technology (kWh / fixture-year)",
                labels={'lamp_technology': 'Technology', 'kwh_per_luminaire': 'kWh / Luminaire'},
                color_discrete_map={'LED': '#34D399', 'High-pressure sodium (HPS)': '#F87171'}
            )
            components.style_plotly_chart(fig_tech, height=370)
            st.plotly_chart(fig_tech, use_container_width=True)

        with col_i2:
            # Control Mode Savings vs Full Output
            mode_agg = df_view[df_view['is_valid_energy']].groupby('control_mode').agg(
                total_energy=('energy_consumption_kwh', 'sum'),
                full_output=('full_output_reference_kwh', 'sum')
            ).reset_index()
            mode_agg['savings_pct'] = (mode_agg['full_output'] - mode_agg['total_energy']) / mode_agg['full_output'] * 100

            fig_mode_sav = px.bar(
                mode_agg, x='control_mode', y='savings_pct', color='control_mode',
                title="Baseline Reference Savings % by Installed Control Mode",
                labels={'control_mode': 'Control Mode', 'savings_pct': 'Savings vs. Full Output (%)'},
                color_discrete_map={'Fixed': '#F87171', 'Scheduled dimming': '#FBBF24', 'Adaptive dimming': '#22D3EE'}
            )
            components.style_plotly_chart(fig_mode_sav, height=370)
            st.plotly_chart(fig_mode_sav, use_container_width=True)

        # Energy per Luminaire by Zone
        zone_sum = analytics.get_zone_summary(df_view)
        fig_zone_lum = px.bar(
            zone_sum, x='zone_type', y='kwh_per_luminaire', color='zone_type',
            title="Normalized Energy per Luminaire Across Zone Types (kWh / fixture)",
            labels={'zone_type': 'Zone Classification', 'kwh_per_luminaire': 'kWh / Luminaire-Year'}
        )
        components.style_plotly_chart(fig_zone_lum, height=380)
        st.plotly_chart(fig_zone_lum, use_container_width=True)

    components.render_synthetic_disclosure()


# =============================================================================
# PAGE 3 — STREET AND INFRASTRUCTURE EXPLORER
# =============================================================================
elif nav_page == "🛣️ Street & Infrastructure Explorer":
    components.render_section_header("Street Segment Inventory & Single-Street Deep Dive", "Granular performance, electrical specifications, and telemetry audit")

    street_table = analytics.get_street_ranking_table(df_view)

    # Search and Filterable Table
    st.markdown("#### Municipal Street Segment Inventory (24 Synthetic Corridors)")
    st.dataframe(
        street_table.style.format({
            'rated_watts': '{:.0f} W',
            'circuit_kw': '{:.2f} kW',
            'total_kwh': '{:,.1f}',
            'full_kwh': '{:,.1f}',
            'kwh_per_luminaire': '{:,.1f}',
            'savings_pct': '{:.2f}%',
            'avg_traffic': '{:,.1f}',
            'avg_pedestrians': '{:,.1f}'
        }),
        use_container_width=True,
        height=320
    )

    # Street Selector Deep-Dive
    components.render_section_header("Individual Segment Deep-Dive Telemetry", "Inspect corridor-specific diurnal profiles, traffic response, and hardware architecture")

    all_streets = sorted(df_view['street_id'].unique().tolist())
    selected_street = st.selectbox("Select Street Segment to Inspect:", all_streets, index=0)

    street_df = df_view[df_view['street_id'] == selected_street].sort_values('timestamp_local')
    s_meta = street_df.iloc[0]

    # Street Telemetry Cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        components.render_kpi_card("Segment Identity", f"{s_meta['street_id']}", f"{s_meta['street_name']}", accent="cyan", icon="📍")
    with sc2:
        components.render_kpi_card("Hardware Architecture", f"{s_meta['lamp_technology']}", f"{s_meta['luminaire_count']} fixtures @ {s_meta['rated_watts_per_luminaire']:.0f}W", accent="purple", icon="💡")
    with sc3:
        components.render_kpi_card("Control Strategy", f"{s_meta['control_mode']}", f"Road Class: {s_meta['road_class']}", accent="amber", icon="⚙️")
    with sc4:
        s_total = street_df[street_df['is_valid_energy']]['energy_consumption_kwh'].sum()
        s_full = street_df[street_df['is_valid_energy']]['full_output_reference_kwh'].sum()
        s_sav = (s_full - s_total) / s_full * 100 if s_full > 0 else 0
        components.render_kpi_card("Annual Energy", f"{s_total:,.1f} kWh", f"Baseline Savings: {s_sav:.1f}%", accent="emerald", icon="⚡")

    # Diurnal Profile & Timeline for Selected Street
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s_hourly = street_df[street_df['is_valid_energy']].groupby('hour')['energy_consumption_kwh'].mean().reset_index()
        fig_s_hour = px.line(
            s_hourly, x='hour', y='energy_consumption_kwh',
            title=f"Diurnal Consumption Profile: {selected_street} ({s_meta['zone_type']})",
            labels={'hour': 'Hour of Day (0–23)', 'energy_consumption_kwh': 'Average Energy (kWh)'},
            markers=True
        )
        components.style_plotly_chart(fig_s_hour, height=350)
        st.plotly_chart(fig_s_hour, use_container_width=True)

    with col_s2:
        s_night = street_df[street_df['lighting_required_flag'] == 1].sample(min(1000, len(street_df)), random_state=42)
        fig_s_dim = px.scatter(
            s_night, x='traffic_count', y='measured_dimming_pct',
            title=f"Traffic vs. Dimming Response: {selected_street}",
            labels={'traffic_count': 'Vehicles / Hour', 'measured_dimming_pct': 'Measured Dimming (%)'},
            opacity=0.6, color_discrete_sequence=['#22D3EE']
        )
        components.style_plotly_chart(fig_s_dim, height=350)
        st.plotly_chart(fig_s_dim, use_container_width=True)

    # Download Selected Street Data
    csv_street = street_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Telemetry CSV for {selected_street} ({len(street_df):,} records)",
        data=csv_street,
        file_name=f"street_telemetry_{selected_street}_2025.csv",
        mime="text/csv"
    )

    components.render_synthetic_disclosure()


# =============================================================================
# PAGE 4 — TRAFFIC, WEATHER AND LIGHTING RELATIONSHIPS
# =============================================================================
elif nav_page == "🚦 Traffic, Weather & Lighting Dynamics":
    components.render_section_header("Multivariate Dynamics & Cross-Feature Associations", "Empirical correlations between human activity, meteorological conditions, and grid draw")

    tab_corr, tab_scatter, tab_weather = st.tabs(["📊 Correlation Matrix", "🔍 Bivariate Scatter Explorer", "🌧️ Weather & Twilight Dynamics"])

    with tab_corr:
        corr_method = st.radio("Correlation Metric:", ["Spearman Rank Correlation (Non-linear)", "Pearson Linear Correlation"], horizontal=True)
        m_str = 'spearman' if 'Spearman' in corr_method else 'pearson'

        features_for_corr = [
            'energy_consumption_kwh', 'full_output_reference_kwh', 'luminaire_count',
            'rated_watts_per_luminaire', 'circuit_rated_kw', 'commanded_dimming_pct',
            'measured_dimming_pct', 'lighting_required_flag', 'ambient_light_lux',
            'traffic_count', 'pedestrian_count', 'temperature_c', 'humidity_pct',
            'rainfall_mm', 'cloud_cover_pct', 'visibility_km'
        ]
        clean_df = df_view[df_view['data_quality_flag'] == 'OK'][features_for_corr].dropna()
        corr_matrix = clean_df.corr(method=m_str)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            title=f"{corr_method} Matrix (Operational, Environmental & Infrastructure Telemetry)",
            aspect="auto"
        )
        components.style_plotly_chart(fig_corr, height=580)
        st.plotly_chart(fig_corr, use_container_width=True)
        components.render_callout(
            "Correlation does not establish causation. Observed associations reflect generative controller rules "
            "and circuit physics, not independent real-world laws. For example, the high correlation between commanded "
            "dimming and energy reflects the mathematical component power equations.",
            title="Methodological Note:",
            variant="warning"
        )

    with tab_scatter:
        col_sc1, col_sc2, col_sc3 = st.columns(3)
        with col_sc1:
            x_var = st.selectbox("X-Axis Feature:", features_for_corr, index=features_for_corr.index('traffic_count'))
        with col_sc2:
            y_var = st.selectbox("Y-Axis Feature:", features_for_corr, index=features_for_corr.index('measured_dimming_pct'))
        with col_sc3:
            color_var = st.selectbox("Color Stratification:", ['zone_type', 'lamp_technology', 'control_mode', 'traffic_period'], index=0)

        scatter_sample = df_view[df_view['data_quality_flag'] == 'OK'].sample(min(2500, len(df_view)), random_state=42)
        fig_custom_scatter = px.scatter(
            scatter_sample, x=x_var, y=y_var, color=color_var,
            title=f"Bivariate Explorer: {x_var} vs. {y_var}",
            opacity=0.6, trendline="ols" if x_var != y_var else None
        )
        components.style_plotly_chart(fig_custom_scatter, height=450)
        st.plotly_chart(fig_custom_scatter, use_container_width=True)

    with tab_weather:
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            # Twilight cliff at 50 lux
            twilight_df = df_view[df_view['hour'].isin([5, 6, 7, 17, 18, 19])].sample(min(1500, len(df_view)), random_state=42)
            fig_twilight = px.scatter(
                twilight_df, x='ambient_light_lux', y='commanded_dimming_pct',
                log_x=True, color='lighting_required_flag',
                title="Twilight Switching: Ambient Lux vs. Commanded Dimming",
                labels={'ambient_light_lux': 'Ambient Light (Lux, Log Scale)', 'commanded_dimming_pct': 'Commanded Dimming (%)'}
            )
            fig_twilight.add_vline(x=50.0, line_dash="dash", line_color="#F87171", annotation_text="50 Lux Switching Threshold")
            components.style_plotly_chart(fig_twilight, height=380)
            st.plotly_chart(fig_twilight, use_container_width=True)

        with col_w2:
            # Weather condition impact on energy
            fig_weather = px.box(
                df_view[df_view['is_valid_energy']], x='weather_condition', y='energy_consumption_kwh',
                color='weather_condition',
                title="Hourly Energy Distribution Across Weather Conditions",
                labels={'weather_condition': 'Weather State', 'energy_consumption_kwh': 'Energy (kWh)'}
            )
            components.style_plotly_chart(fig_weather, height=380, show_legend=False)
            st.plotly_chart(fig_weather, use_container_width=True)

    components.render_synthetic_disclosure()


# =============================================================================
# PAGE 5 — DATA QUALITY AND ANOMALY MONITOR
# =============================================================================
elif nav_page == "🛡️ Data Quality & Anomaly Monitor":
    components.render_section_header("Data Quality Assurance & Anomaly Diagnostics", "Auditable integrity tracking, sensor corruptions, and simulated lamp faults")

    # Quality KPIs
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        components.render_kpi_card("Validated Clean Records", f"{kpis['clean_records']:,}", f"{kpis['clean_pct']:.2f}% of active population", accent="emerald", icon="🛡️")
    with qc2:
        components.render_kpi_card("Sensor Anomaly Flags", f"{kpis['sensor_anomalies']:,}", f"{kpis['sensor_anomaly_pct']:.2f}% deliberate spikes/missingness", accent="amber", icon="⚠️")
    with qc3:
        components.render_kpi_card("Simulated Lamp Faults", f"{kpis['lamp_faults']:,}", "Operational zero-output outages", accent="critical", icon="🚨")
    with qc4:
        meter_spikes = df_view['sensor_anomaly_type'].str.contains('Energy-meter spike', na=False).sum()
        components.render_kpi_card("Meter Spikes", f"{meter_spikes:,}", "Excluded from valid energy comparisons", accent="purple", icon="⚡")

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        # Injected Anomaly Breakdown
        anom_series = df_view['sensor_anomaly_type'].dropna()
        split_items = [sub for item in anom_series for sub in item.split('; ')]
        anom_counts = pd.Series(split_items).value_counts().reset_index()
        anom_counts.columns = ['Anomaly Type', 'Count']

        fig_anom = px.bar(
            anom_counts, x='Count', y='Anomaly Type', orientation='h',
            title="Injected Anomaly Taxonomy & Frequency",
            color='Count', color_continuous_scale="Oranges"
        )
        components.style_plotly_chart(fig_anom, height=380, show_legend=False)
        st.plotly_chart(fig_anom, use_container_width=True)

    with col_q2:
        # Missing Value Profiling
        missing_s = df_view.isna().sum()
        missing_s = missing_s[missing_s > 0].drop(labels=['sensor_anomaly_type'], errors='ignore').reset_index()
        missing_s.columns = ['Feature', 'Missing Count']
        missing_s['Missing Pct'] = missing_s['Missing Count'] / len(df_view) * 100

        fig_miss = px.bar(
            missing_s, x='Feature', y='Missing Count', text_auto=True,
            title="Missing Sensor Readings (Controlled Synthetic Dropouts)",
            color_discrete_sequence=['#38BDF8']
        )
        components.style_plotly_chart(fig_miss, height=380)
        st.plotly_chart(fig_miss, use_container_width=True)

    # Flagged Records Table Inspector
    st.markdown("#### Flagged Records Inspector & Audit Log")
    flagged_df = df_view[df_view['data_quality_flag'] != 'OK'][
        ['record_id', 'timestamp_local', 'street_id', 'zone_type', 'data_quality_flag', 
         'sensor_anomaly_type', 'lamp_status', 'energy_consumption_kwh', 'traffic_count', 'ambient_light_lux']
    ]

    st.dataframe(flagged_df, use_container_width=True, height=280)

    csv_flagged = flagged_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Flagged Anomaly Audit Log CSV ({len(flagged_df):,} records)",
        data=csv_flagged,
        file_name="flagged_anomalies_audit_log_2025.csv",
        mime="text/csv"
    )

    components.render_callout(
        "Simulated lighting faults (101 records) represent service failures where luminaires were inoperative during required dark hours. "
        "In our analytical pipeline, <strong>outages are strictly segregated and never counted as successful energy savings</strong>. "
        "Similarly, energy-meter spikes are excluded from baseline benchmarking to prevent distorted metrics.",
        title="Academic Accounting Safeguard:",
        variant="warning"
    )

    components.render_synthetic_disclosure()


# =============================================================================
# PAGE 6 — ENERGY OPTIMIZATION & ML PREDICTION ENGINE
# =============================================================================
elif nav_page == "🚀 Energy Optimization Simulator":
    components.render_section_header(
        "Lighting Optimization & ML Intelligence Center",
        "Policy-driven scenario simulations and machine learning energy prediction surrogates"
    )

    tab_rules, tab_ml = st.tabs([
        "⚖️ Rule-Based Scenario Optimization",
        "🤖 ML Energy Prediction & Surrogate Engine"
    ])

    with tab_rules:
        st.markdown("""
        This simulator applies the physical luminaire power models derived in the verified EDA notebook to project 
        annual grid energy savings under configurable civil safety floors and demand-responsive setpoint rules.
        """)

        # Interactive Optimization Controls
        with st.expander("⚙️ Configure Scenario Policy Parameters", expanded=True):
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown("**Scenario 1: Schedule Tuning**")
                p_s1_floor = st.slider("Deep-Night Minimum Floor (%)", 50.0, 90.0, 75.0, 5.0)
                st.caption("Applied to Fixed/Scheduled circuits during 23:00–04:00.")

            with col_p2:
                st.markdown("**Scenario 2: Adaptive Road-Class Floors**")
                p_s2_hwy = st.slider("Highway Safety Floor (%)", 50.0, 80.0, 65.0, 5.0)
                p_s2_art = st.slider("Arterial Safety Floor (%)", 45.0, 75.0, 60.0, 5.0)
                p_s2_res = st.slider("Residential Safety Floor (%)", 30.0, 60.0, 40.0, 5.0)

            with col_p3:
                st.markdown("**Scenario 3 & Safety Overrides**")
                p_s3_deep = st.slider("Scenario 3 Eco Floor (%)", 20.0, 50.0, 30.0, 5.0)
                p_weather_boost = st.slider("Adverse Weather Boost (%)", 10.0, 30.0, 20.0, 5.0)
                st.caption("Triggered when rain > 0.3mm or visibility < 3.5km.")

        # Execute simulation
        sim_results = analytics.simulate_scenarios(
            df_view,
            s1_offpeak_floor=p_s1_floor,
            s2_hwy_floor=p_s2_hwy,
            s2_art_floor=p_s2_art,
            s2_res_floor=p_s2_res,
            s3_deep_floor=p_s3_deep,
            weather_boost=p_weather_boost
        )

        if sim_results:
            components.render_section_header("Modeled Scenario Performance Summary", "Comparative projections against observed baseline and unconstrained full-output load")

            # Scenario Table
            st.dataframe(
                sim_results['summary_table'].style.format({
                    'Annual Energy (kWh)': '{:,.1f}',
                    'Savings vs Baseline (kWh)': '{:,.1f}',
                    'Savings vs Baseline (%)': '{:+.2f}%',
                    'Total Savings vs Full Output (%)': '{:.2f}%',
                    'Mean Active Dimming (%)': '{:.2f}%'
                }),
                use_container_width=True
            )

            col_sim1, col_sim2 = st.columns(2)

            with col_sim1:
                # Bar Chart Comparison
                sum_df = sim_results['summary_table'][sim_results['summary_table']['Scenario'] != 'Full-Output Reference Comparator']
                fig_bar_scen = px.bar(
                    sum_df, x='Scenario', y='Annual Energy (kWh)', color='Scenario',
                    title="Annual Grid Energy Consumption by Scenario Profile",
                    text_auto='.0f',
                    color_discrete_sequence=['#38BDF8', '#34D399', '#22D3EE', '#F87171']
                )
                components.style_plotly_chart(fig_bar_scen, height=380, show_legend=False)
                st.plotly_chart(fig_bar_scen, use_container_width=True)

            with col_sim2:
                # Diurnal Profile Comparison
                h_sim = sim_results['hourly_df']
                fig_diurnal_scen = go.Figure()
                fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Baseline'], name='Baseline Observed', line=dict(color='#38BDF8', width=2.5)))
                fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Scenario 1'], name='Scenario 1: Conservative', line=dict(color='#34D399', width=2, dash='dash')))
                fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Scenario 2'], name='Scenario 2: Adaptive (Recommended)', line=dict(color='#22D3EE', width=2.5)))
                fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Scenario 3'], name='Scenario 3: Eco Deep', line=dict(color='#F87171', width=1.8, dash='dot')))
                fig_diurnal_scen.update_layout(
                    title="24-Hour Diurnal Energy Profile Across Scenarios",
                    xaxis_title="Hour of Day (0–23)", yaxis_title="Average Energy (kWh / segment-hr)",
                    xaxis=dict(tickmode='linear', tick0=0, dtick=2)
                )
                components.style_plotly_chart(fig_diurnal_scen, height=380)
                st.plotly_chart(fig_diurnal_scen, use_container_width=True)

            # Zone-wise Scenario Performance
            z_sim = sim_results['zone_df']
            z_melt = pd.melt(z_sim, id_vars=['Zone'], var_name='Profile', value_name='kWh')
            fig_zone_scen = px.bar(
                z_melt, x='Zone', y='kWh', color='Profile', barmode='group',
                title="Zone-Level Annual Energy Comparison Across Scenarios",
                color_discrete_sequence=['#38BDF8', '#34D399', '#22D3EE', '#F87171']
            )
            components.style_plotly_chart(fig_zone_scen, height=400)
            st.plotly_chart(fig_zone_scen, use_container_width=True)

        # Engineering Compliance Caveat
        components.render_callout(
            "Controller setpoints (commanded dimming %) are <strong>not proof of achieved road-surface illumination</strong>. "
            "Physical municipal deployment requires on-site photometric validation using luminance cameras to guarantee "
            "compliance with civil standards (EN 13201 / IESNA RP-8 / IS 1944). In event of communication failure or sensor "
            "dropouts, fixtures must automatically revert to fail-safe 100% full illumination.",
            title="Civil Engineering & Photometric Disclaimer:",
            variant="warning"
        )
        components.render_synthetic_disclosure()

    with tab_ml:
        components.render_section_header(
            "Random Forest Energy Prediction Surrogate",
            "Supervised machine learning model predicting circuit-level hourly energy consumption without target leakage"
        )

        try:
            with st.spinner("Loading & caching Random Forest surrogate model..."):
                ml_model = get_cached_ml_model(df_raw)
            model_ready = True
        except Exception as e:
            model_ready = False
            st.error(f"ML Model Training Failed: {e}")

        if model_ready:
            metrics = ml_model['metrics']

            # Model Status Banner
            st.markdown(f"""
            <div style="background: rgba(34, 211, 238, 0.08); border: 1px solid rgba(34, 211, 238, 0.25); border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-weight: 700; color: #22D3EE; font-size: 13.5px;">ESTIMATOR:</span>
                    <span style="color: #F1F5F9; font-weight: 600; margin-left: 6px;">RandomForestRegressor</span>
                    <span style="color: #34D399; font-weight: 600; margin-left: 12px;">● Trained & Validated</span>
                </div>
                <div style="font-size: 12px; color: #94A3B8;">
                    Trained in <strong style="color: #22D3EE;">{metrics['train_time_sec']:.2f}s</strong> on {metrics['train_rows']:,} records (Jan–Oct) • Tested on {metrics['test_rows']:,} holdout records (Nov–Dec)
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4 Metric KPI Cards
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                components.render_kpi_card(
                    "Holdout MAE",
                    f"{metrics['mae']:.4f} kWh",
                    "Mean Absolute Error on holdout test set",
                    accent="cyan",
                    icon="🎯"
                )
            with col_m2:
                components.render_kpi_card(
                    "Holdout RMSE",
                    f"{metrics['rmse']:.4f} kWh",
                    "Root Mean Squared Error (penalizes outliers)",
                    accent="sky",
                    icon="📐"
                )
            with col_m3:
                components.render_kpi_card(
                    "Goodness of Fit (R²)",
                    f"{metrics['r2']:.4f}",
                    "Variance explained on chronological test set",
                    accent="emerald",
                    icon="📈"
                )
            with col_m4:
                components.render_kpi_card(
                    "Holdout Observations",
                    f"{metrics['test_rows']:,} hrs",
                    "Months 11–12 (strictly unseen test window)",
                    accent="purple",
                    icon="📅"
                )

            # Synthetic Disclosure Callout
            components.render_callout(
                f"<strong>Target Leakage Safeguard:</strong> Direct mathematical duplicates of the target variable "
                f"(such as <code>full_output_reference_kwh</code>, <code>energy_savings_vs_full_output_pct</code>, "
                f"<code>specific_load_ratio</code>, and <code>energy_diff_vs_full_kwh</code>) were strictly excluded from the feature matrix. "
                f"<br><strong>Synthetic Data Disclosure:</strong> The ML model learns from synthetic telemetry generated for an archetype municipality. "
                f"While high holdout metrics (R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.4f} kWh) confirm excellent surrogate convergence against the underlying data generation physics, "
                f"these metrics <em>do not establish real-world accuracy or photometric compliance</em> without physical deployment and field calibration.",
                title="Methodological Rigor & Academic Disclosure:",
                variant="warning"
            )

            # Diagnostic Visualizations
            components.render_section_header(
                "Holdout Evaluation & Feature Importance Visualizations",
                "Model explainability and diagnostic fit on chronological holdout telemetry"
            )

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_scatter = ml_energy_predictor.plot_actual_vs_predicted_scatter(ml_model['scatter_df'])
                st.plotly_chart(fig_scatter, use_container_width=True)
            with col_c2:
                fig_fi = ml_energy_predictor.plot_feature_importance(ml_model['feature_importance_df'], top_n=10)
                st.plotly_chart(fig_fi, use_container_width=True)

            fig_time = ml_energy_predictor.plot_prediction_timeline(ml_model['sample_vis'])
            st.plotly_chart(fig_time, use_container_width=True)

            # Interactive Single-Instance Prediction Form
            components.render_section_header(
                "Interactive Circuit Load Predictor",
                "Input live operational telemetry to evaluate real-time hourly energy inference (kWh)"
            )

            street_meta = df_raw.drop_duplicates(subset=['street_id']).set_index('street_id')
            available_streets = sorted(street_meta.index.tolist())

            with st.form("ml_inference_form"):
                st.markdown("<p style='font-size: 13px; color: #94A3B8; margin-bottom: 12px;'>Configure corridor infrastructure parameters and operating telemetry conditions:</p>", unsafe_allow_html=True)

                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    sel_street_id = st.selectbox("Corridor Segment ID", available_streets, index=0)
                    cur_street = street_meta.loc[sel_street_id]

                    st.caption(f"Infrastructure: **{cur_street['zone_type']}** | **{cur_street['road_class']}**")
                    st.caption(f"Luminaires: **{int(cur_street['luminaire_count'])}** fixtures @ **{cur_street['rated_watts_per_luminaire']:.0f}W** ({cur_street['lamp_technology']})")

                    p_hour = st.slider("Hour of Day (0–23)", 0, 23, 21, help="Night lighting hours: typically 18–06")
                    p_month = st.slider("Calendar Month (1–12)", 1, 12, 11, help="Chronological month of year")
                    p_weekend = st.checkbox("Weekend Day (Saturday / Sunday)", value=False)

                with col_f2:
                    p_dimming = st.slider("Measured Dimming Level (%)", 0.0, 100.0, 70.0, 5.0, help="Commanded / measured controller output percentage")
                    p_traffic = st.slider("Vehicular Traffic (vehicles/hr)", 0, 350, 75, 5)
                    p_ped = st.slider("Pedestrian Count (peds/hr)", 0, 80, 12, 1)
                    p_lux = st.slider("Ambient Light (lux)", 0.0, 120.0, 2.0, 0.5)

                with col_f3:
                    p_weather = st.selectbox("Weather Condition", ['Clear', 'Partly cloudy', 'Cloudy', 'Rain', 'Thunderstorm'], index=0)
                    p_mode = st.selectbox("Control Mode", ['Adaptive dimming', 'Fixed', 'Scheduled dimming'], index=0)
                    p_temp = st.slider("Ambient Temperature (°C)", 15.0, 42.0, 26.5, 0.5)
                    p_rain = st.slider("Precipitation (mm)", 0.0, 12.0, 0.0, 0.5)
                    p_vis = st.slider("Atmospheric Visibility (km)", 0.5, 10.0, 9.5, 0.5)

                submit_pred = st.form_submit_button("⚡ Predict Circuit Load with ML Model", use_container_width=True)

            if submit_pred:
                cur_street = street_meta.loc[sel_street_id]
                lum_count = int(cur_street['luminaire_count'])
                rated_w = float(cur_street['rated_watts_per_luminaire'])
                circuit_kw = float(cur_street['circuit_rated_kw'])

                input_features = {
                    'hour': p_hour,
                    'month': p_month,
                    'is_weekend': 1 if p_weekend else 0,
                    'luminaire_count': lum_count,
                    'rated_watts_per_luminaire': rated_w,
                    'circuit_rated_kw': circuit_kw,
                    'traffic_count': p_traffic,
                    'pedestrian_count': p_ped,
                    'ambient_light_lux': p_lux,
                    'temperature_c': p_temp,
                    'humidity_pct': 65.0,
                    'rainfall_mm': p_rain,
                    'visibility_km': p_vis,
                    'measured_dimming_pct': p_dimming,
                    'zone_type': cur_street['zone_type'],
                    'road_class': cur_street['road_class'],
                    'lamp_technology': cur_street['lamp_technology'],
                    'control_mode': p_mode,
                    'weather_condition': p_weather
                }

                pred_kwh = ml_energy_predictor.predict_instance(ml_model, input_features)
                rated_hourly_kwh = circuit_kw * 1.0
                savings_pct = max(0.0, (1.0 - pred_kwh / rated_hourly_kwh) * 100.0) if rated_hourly_kwh > 0 else 0.0
                specific_wh = (pred_kwh * 1000.0) / lum_count if lum_count > 0 else 0.0

                st.markdown("#### ⚡ ML Prediction Output")
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    components.render_kpi_card(
                        "Predicted Energy",
                        f"{pred_kwh:.3f} kWh",
                        f"Hourly consumption for segment {sel_street_id}",
                        accent="cyan",
                        icon="⚡"
                    )
                with col_r2:
                    components.render_kpi_card(
                        "Specific Luminaire Load",
                        f"{specific_wh:.1f} Wh/fixture",
                        f"Across {lum_count} installed luminaires",
                        accent="sky",
                        icon="💡"
                    )
                with col_r3:
                    components.render_kpi_card(
                        "Energy Reduction vs Full Output",
                        f"{savings_pct:.1f}%",
                        f"Compared to rated circuit capacity ({circuit_kw:.1f} kW)",
                        accent="emerald",
                        icon="📉"
                    )

            # Scenario Comparison via ML Surrogate
            components.render_section_header(
                "Policy Scenario Cross-Validation (Rule-Based vs. ML Surrogate)",
                "Comparing annual grid energy totals projected by the physics simulation engine versus the ML regression surrogate"
            )

            st.markdown("""
            This comparative analysis evaluates policy scenarios (Conservative, Adaptive, and Eco Deep) using the trained Random Forest model 
            as an independent surrogate evaluator. **Note:** The ML model does not optimize policies—it predicts energy consumption under 
            the scenario dimming distributions to validate physics-based calculations.
            """)

            eval_target_df = df_view[df_view['is_valid_energy']] if len(df_view[df_view['is_valid_energy']]) > 0 else df_raw[df_raw['is_valid_energy']]
            with st.spinner("Evaluating scenario distributions through ML surrogate..."):
                ml_scen = ml_energy_predictor.evaluate_scenarios_with_ml(ml_model, eval_target_df)

            # Compare physics-based totals vs ML surrogate totals
            s_table = sim_results['summary_table'] if (sim_results and 'summary_table' in sim_results) else None
            if s_table is not None:
                def get_phys_kwh(scen_name, fallback):
                    matches = s_table.loc[s_table['Scenario'].str.contains(scen_name, case=False, na=False), 'Annual Energy (kWh)']
                    return float(matches.values[0]) if len(matches) > 0 else fallback

                phys_base = get_phys_kwh('Baseline', 311000.0)
                phys_s1 = get_phys_kwh('Scenario 1', 280000.0)
                phys_s2 = get_phys_kwh('Scenario 2', 254000.0)
                phys_s3 = get_phys_kwh('Scenario 3', 241000.0)
            else:
                phys_base, phys_s1, phys_s2, phys_s3 = 311000.0, 280000.0, 254000.0, 241000.0

            scen_comp_df = pd.DataFrame([
                {
                    'Policy Profile': 'Baseline Observed',
                    'Rule-Based Physics (kWh)': phys_base,
                    'ML Surrogate Prediction (kWh)': ml_scen['ml_base_kwh'],
                    'Deviation (kWh)': ml_scen['ml_base_kwh'] - phys_base,
                    'Surrogate Variance (%)': ((ml_scen['ml_base_kwh'] - phys_base) / phys_base) * 100.0
                },
                {
                    'Policy Profile': 'Scenario 1: Schedule Tuning',
                    'Rule-Based Physics (kWh)': phys_s1,
                    'ML Surrogate Prediction (kWh)': ml_scen['ml_s1_kwh'],
                    'Deviation (kWh)': ml_scen['ml_s1_kwh'] - phys_s1,
                    'Surrogate Variance (%)': ((ml_scen['ml_s1_kwh'] - phys_s1) / phys_s1) * 100.0
                },
                {
                    'Policy Profile': 'Scenario 2: Adaptive Controls',
                    'Rule-Based Physics (kWh)': phys_s2,
                    'ML Surrogate Prediction (kWh)': ml_scen['ml_s2_kwh'],
                    'Deviation (kWh)': ml_scen['ml_s2_kwh'] - phys_s2,
                    'Surrogate Variance (%)': ((ml_scen['ml_s2_kwh'] - phys_s2) / phys_s2) * 100.0
                },
                {
                    'Policy Profile': 'Scenario 3: Deep Eco Profile',
                    'Rule-Based Physics (kWh)': phys_s3,
                    'ML Surrogate Prediction (kWh)': ml_scen['ml_s3_kwh'],
                    'Deviation (kWh)': ml_scen['ml_s3_kwh'] - phys_s3,
                    'Surrogate Variance (%)': ((ml_scen['ml_s3_kwh'] - phys_s3) / phys_s3) * 100.0
                }
            ])

            st.dataframe(
                scen_comp_df.style.format({
                    'Rule-Based Physics (kWh)': '{:,.1f}',
                    'ML Surrogate Prediction (kWh)': '{:,.1f}',
                    'Deviation (kWh)': '{:+,.1f}',
                    'Surrogate Variance (%)': '{:+.2f}%'
                }),
                use_container_width=True
            )

            components.render_synthetic_disclosure()



# =============================================================================
# PAGE 7 — REPORTS AND DATA EXPORT
# =============================================================================
elif nav_page == "📥 Reports & Data Export":
    components.render_section_header("Data Export Center & Executive Reporting Hub", "Download filtered telemetry, aggregated summaries, and project audit reports")

    st.markdown("""
    Generate and download live analytical exports based on the current active filters. 
    All generated files reflect the current session state and calculation outputs.
    """)

    c_exp1, c_exp2 = st.columns(2)

    with c_exp1:
        st.markdown("#### 📄 Tabular Data Exports (CSV)")

        # 1. Filtered Raw Telemetry
        csv_raw_filtered = df_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Active Filtered Telemetry ({len(df_view):,} rows)",
            data=csv_raw_filtered,
            file_name="lux_filtered_telemetry_2025.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 2. Street Segment Summary
        street_exp = analytics.get_street_ranking_table(df_view)
        csv_streets = street_exp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Street Segments Summary CSV",
            data=csv_streets,
            file_name="lux_street_inventory_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 3. Monthly Performance Summary
        monthly_exp = analytics.get_monthly_summary(df_view)
        csv_monthly = monthly_exp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Monthly Performance Summary CSV",
            data=csv_monthly,
            file_name="lux_monthly_performance_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    with c_exp2:
        st.markdown("#### 📊 Analytical Model & Anomaly Exports")

        # 4. Zone Comparison Export
        zone_exp = analytics.get_zone_summary(df_view)
        csv_zone = zone_exp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Zone Classification Summary CSV",
            data=csv_zone,
            file_name="lux_zone_comparison_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

        # 5. Optimization Scenarios Export
        scen_results = analytics.simulate_scenarios(df_view)
        if scen_results:
            csv_scen = scen_results['summary_table'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Optimization Scenario Projections CSV",
                data=csv_scen,
                file_name="lux_optimization_scenario_projections.csv",
                mime="text/csv",
                use_container_width=True
            )

        # 6. Flagged Quality Issues Export
        flagged_exp = df_view[df_view['data_quality_flag'] != 'OK']
        csv_anom = flagged_exp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Quality Anomalies Log CSV ({len(flagged_exp):,} records)",
            data=csv_anom,
            file_name="lux_quality_anomalies_log.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    components.render_section_header("Executive Findings Documentation", "Direct access to the comprehensive project findings report")

    # Read existing findings report if available or generate snapshot
    findings_path = Path("EDA_Findings_Summary.md")
    if findings_path.exists():
        report_text = findings_path.read_text(encoding='utf-8')
    else:
        report_text = f"# LUX Executive Findings Summary\nTotal Active Records: {len(df_view)}\nTotal Energy: {kpis['total_energy_kwh']:.2f} kWh"

    st.download_button(
        label="📥 Download Complete Executive Project Findings Report (.md)",
        data=report_text.encode('utf-8'),
        file_name="LUX_Executive_Findings_Summary.md",
        mime="text/markdown",
        use_container_width=True
    )

    with st.expander("👁️ Preview Executive Findings Summary Report"):
        st.markdown(report_text[:4000] + "\n\n*(Full report available via download above)*")

    components.render_synthetic_disclosure()
