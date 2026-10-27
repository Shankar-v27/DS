"""
LUX - Smart Streetlighting Intelligence Platform
Main Streamlit Application Entrypoint

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

# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LUX | Smart Streetlighting",
    page_icon="💡",
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
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 16px;">
        <span style="font-size: 22px; font-weight: 800; background: linear-gradient(90deg, #00e5ff 0%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            LUX INTELLIGENCE
        </span>
        <div style="font-size: 11px; color: #94a3b8; font-weight: 500; margin-top: 2px;">
            Municipal Smart Lighting Analytics
        </div>
    </div>
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
        index=0
    )

    st.markdown("---")
    st.markdown("<div style='font-size: 13px; font-weight: 700; color: #e2e8f0; margin-bottom: 8px;'>GLOBAL FILTERS</div>", unsafe_allow_html=True)

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
    current_filtered_count=len(df_view)
)

# Empty State Warning
if len(df_view) == 0:
    st.warning("⚠️ No records match the active filter criteria. Please broaden your sidebar selections or click 'Reset All Filters'.")
    st.stop()


# =============================================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# =============================================================================
if nav_page == "📊 Executive Overview":
    components.render_section_header("System Overview & Key Performance Indicators", "Executive telemetry across municipal lighting circuits")

    # Row 1: Primary Energy & Scope Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        components.render_kpi_card(
            "Total Observed Energy",
            f"{kpis['total_energy_kwh']:,.1f} kWh",
            f"Valid subset: {kpis['valid_energy_kwh']:,.1f} kWh",
            accent="cyan"
        )
    with c2:
        components.render_kpi_card(
            "Full-Output Reference",
            f"{kpis['full_output_kwh']:,.1f} kWh",
            "Theoretical continuous load",
            accent="indigo"
        )
    with c3:
        components.render_kpi_card(
            "Baseline Savings %",
            f"{kpis['valid_savings_pct']:.2f}%",
            f"{kpis['valid_diff_kwh']:,.0f} kWh reduction vs. full",
            accent="emerald"
        )
    with c4:
        components.render_kpi_card(
            "Active Grid Scope",
            f"{kpis['unique_streets']} Segments",
            f"{kpis['total_records']:,} observations (2025)",
            accent="sky"
        )

    # Row 2: Operational Quality & Reliability Metrics
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        components.render_kpi_card(
            "Mean Night Dimming",
            f"{kpis['avg_measured_dimming']:.1f}%",
            f"Command setpoint: {kpis['avg_commanded_dimming']:.1f}%",
            accent="cyan"
        )
    with c6:
        components.render_kpi_card(
            "Cleanliness Ratio",
            f"{kpis['clean_pct']:.2f}%",
            f"{kpis['clean_records']:,} uncorrupted records",
            accent="emerald"
        )
    with c7:
        components.render_kpi_card(
            "Sensor Anomalies",
            f"{kpis['sensor_anomalies']:,}",
            f"{kpis['sensor_anomaly_pct']:.2f}% of active records",
            accent="amber"
        )
    with c8:
        components.render_kpi_card(
            "Simulated Lamp Outages",
            f"{kpis['lamp_faults']:,}",
            f"{kpis['lamp_fault_pct']:.2f}% (zero-output events)",
            accent="crimson"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    components.render_section_header("Diurnal & Temporal Grid Load Patterns")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Daily Grid Consumption Time Series
        daily_df = analytics.get_daily_trend(df_view)
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_df['date_dt'], y=daily_df['total_energy'],
            mode='lines', name='Daily Total (kWh)',
            line=dict(color='rgba(0, 229, 255, 0.4)', width=1.2)
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily_df['date_dt'], y=daily_df['rolling_7d'],
            mode='lines', name='7-Day Centered MA',
            line=dict(color='#00e5ff', width=2.5)
        ))
        fig_daily.update_layout(
            title="365-Day Energy Consumption Trend (2025)",
            xaxis_title="Date", yaxis_title="Total Daily Energy (kWh)"
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
            marker=dict(color='#38bdf8', line=dict(color='#0284c7', width=1))
        ))
        fig_hourly.add_trace(go.Scatter(
            x=hourly_df['hour'], y=hourly_df['mean_energy'] * kpis['unique_streets'],
            mode='lines+markers', name='Expected Active Load',
            line=dict(color='#f59e0b', width=2)
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
            name='Observed Energy (kWh)', marker_color='#00e5ff'
        ))
        fig_zone.add_trace(go.Bar(
            x=zone_df['zone_type'], y=zone_df['full_output'],
            name='Full Output Reference (kWh)', marker_color='rgba(255, 255, 255, 0.25)'
        ))
        fig_zone.update_layout(
            title="Energy Consumption & Benchmark by Zone Classification",
            xaxis_title="Zone Type", yaxis_title="Energy (kWh)",
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
            color='zone_type', opacity=0.5,
            title="Night Illumination Level (%) vs. Vehicular Traffic Demand",
            labels={'traffic_count': 'Traffic Count (Vehicles/hr)', 'measured_dimming_pct': 'Measured Dimming (%)'}
        )
        components.style_plotly_chart(fig_scatter, height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Executive Summary Callout Box
    st.markdown("""
    <div class="callout-box">
        <strong>Executive Analytical Findings:</strong>
        <ul>
            <li><strong>Diurnal Concentration:</strong> Over 99.7% of total grid electrical energy is consumed between 18:00 and 06:00, peaking at 20:00–21:00 during evening vehicular rush hours.</li>
            <li><strong>Infrastructure Disparities:</strong> Arterial zones consume the highest aggregate energy due to road width and fixture density. Conversely, Campus streets exhibit the highest specific energy per fixture (557 kWh/fixture) due to legacy High-Pressure Sodium lamps.</li>
            <li><strong>Static Mode Inefficiency:</strong> The 6 Fixed-mode circuits run at constant 100% capacity all night, producing 0.00% baseline savings and presenting the primary opportunity for schedule optimization.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

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
                markers=True
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
                markers=True, color_discrete_sequence=['#00e5ff', '#f59e0b']
            )
            components.style_plotly_chart(fig_dow, height=380)
            st.plotly_chart(fig_dow, use_container_width=True)

        # Monthly Consumption with Daily Normalization
        monthly_df = analytics.get_monthly_summary(df_view)
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly_df['month_name'], y=monthly_df['total_energy'],
            name='Total Monthly Energy (kWh)', marker_color='#38bdf8', yaxis='y1'
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly_df['month_name'], y=monthly_df['energy_kwh_per_day'],
            name='Daily-Normalized Energy (kWh/day)', line=dict(color='#10b981', width=3),
            yaxis='y2', mode='lines+markers'
        ))
        fig_monthly.update_layout(
            title="Monthly Energy Consumption: Total vs. Daily-Normalized (kWh/day)",
            xaxis_title="Month of Year (2025)",
            yaxis=dict(title=dict(text="Total Monthly Energy (kWh)", font=dict(color="#38bdf8"))),
            yaxis2=dict(
                title=dict(text="Daily-Normalized Energy (kWh/day)", font=dict(color="#10b981")),
                overlaying='y', side='right'
            )
        )
        components.style_plotly_chart(fig_monthly, height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)

        st.caption("💡 **Analytical Insight:** Notice how daily-normalized energy drops from ~559.5 kWh/day in Jan–May down to ~507.0 kWh/day in Jul–Nov. Earlier sunrises and later sunsets in summer shorten the dark operating window from 12.0 hours to 11.0 hours, producing an organic 9.4% seasonal reduction.")

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
                color_discrete_map={'LED': '#10b981', 'High-pressure sodium (HPS)': '#ef4444'}
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
                color_discrete_map={'Fixed': '#ef4444', 'Scheduled dimming': '#f59e0b', 'Adaptive dimming': '#00e5ff'}
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
    st.markdown("<br>", unsafe_allow_html=True)
    components.render_section_header("Individual Segment Deep-Dive Telemetry")

    all_streets = sorted(df_view['street_id'].unique().tolist())
    selected_street = st.selectbox("Select Street Segment to Inspect:", all_streets, index=0)

    street_df = df_view[df_view['street_id'] == selected_street].sort_values('timestamp_local')
    s_meta = street_df.iloc[0]

    # Street Telemetry Cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        components.render_kpi_card("Segment Identity", f"{s_meta['street_id']}", f"{s_meta['street_name']}", accent="cyan")
    with sc2:
        components.render_kpi_card("Hardware Architecture", f"{s_meta['lamp_technology']}", f"{s_meta['luminaire_count']} fixtures @ {s_meta['rated_watts_per_luminaire']:.0f}W", accent="indigo")
    with sc3:
        components.render_kpi_card("Control Strategy", f"{s_meta['control_mode']}", f"Road Class: {s_meta['road_class']}", accent="amber")
    with sc4:
        s_total = street_df[street_df['is_valid_energy']]['energy_consumption_kwh'].sum()
        s_full = street_df[street_df['is_valid_energy']]['full_output_reference_kwh'].sum()
        s_sav = (s_full - s_total) / s_full * 100 if s_full > 0 else 0
        components.render_kpi_card("Annual Energy", f"{s_total:,.1f} kWh", f"Baseline Savings: {s_sav:.1f}%", accent="emerald")

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
            opacity=0.6, color_discrete_sequence=['#00e5ff']
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
        components.style_plotly_chart(fig_corr, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("⚠️ **Methodological Note:** Correlation does not establish causation. Observed associations reflect generative controller rules and circuit physics, not independent real-world laws.")

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
            fig_twilight.add_vline(x=50.0, line_dash="dash", line_color="#ef4444", annotation_text="50 Lux Switching Threshold")
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
        components.render_kpi_card("Validated Clean Records", f"{kpis['clean_records']:,}", f"{kpis['clean_pct']:.2f}% of active population", accent="emerald")
    with qc2:
        components.render_kpi_card("Sensor Anomaly Flags", f"{kpis['sensor_anomalies']:,}", f"{kpis['sensor_anomaly_pct']:.2f}% deliberate spikes/missingness", accent="amber")
    with qc3:
        components.render_kpi_card("Simulated Lamp Faults", f"{kpis['lamp_faults']:,}", "Operational zero-output outages", accent="crimson")
    with qc4:
        meter_spikes = df_view['sensor_anomaly_type'].str.contains('Energy-meter spike', na=False).sum()
        components.render_kpi_card("Meter Spikes", f"{meter_spikes:,}", "Excluded from valid energy comparisons", accent="indigo")

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
            color_discrete_sequence=['#38bdf8']
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

    st.markdown("""
    <div class="callout-box">
        <strong>Academic Accounting Safeguard:</strong><br>
        Simulated lighting faults (101 records) represent service failures where luminaires were inoperative during required hours. 
        In our analytical pipeline, <strong>outages are strictly segregated and never counted as successful energy savings</strong>.
    </div>
    """, unsafe_allow_html=True)

    components.render_synthetic_disclosure()


# =============================================================================
# PAGE 6 — ENERGY OPTIMIZATION SIMULATOR
# =============================================================================
elif nav_page == "🚀 Energy Optimization Simulator":
    components.render_section_header("Rule-Based Lighting Optimization Engine", "Simulating demand-responsive dimming profiles and sensitivity bounds")

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
        st.markdown("<br>", unsafe_allow_html=True)
        components.render_section_header("Modeled Scenario Performance Summary")

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
                color_discrete_sequence=['#38bdf8', '#10b981', '#00e5ff', '#ef4444']
            )
            components.style_plotly_chart(fig_bar_scen, height=380, show_legend=False)
            st.plotly_chart(fig_bar_scen, use_container_width=True)

        with col_sim2:
            # Diurnal Profile Comparison
            h_sim = sim_results['hourly_df']
            fig_diurnal_scen = go.Figure()
            fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Baseline'], name='Baseline Observed', line=dict(color='#38bdf8', width=2.5)))
            fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Scenario 1'], name='Scenario 1: Conservative', line=dict(color='#10b981', width=2, dash='dash')))
            fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Scenario 2'], name='Scenario 2: Adaptive (Recommended)', line=dict(color='#00e5ff', width=2.5)))
            fig_diurnal_scen.add_trace(go.Scatter(x=h_sim['Hour'], y=h_sim['Scenario 3'], name='Scenario 3: Eco Deep', line=dict(color='#ef4444', width=1.8, dash='dot')))
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
            color_discrete_sequence=['#38bdf8', '#10b981', '#00e5ff', '#ef4444']
        )
        components.style_plotly_chart(fig_zone_scen, height=400)
        st.plotly_chart(fig_zone_scen, use_container_width=True)

    # Engineering Compliance Caveat
    st.markdown("""
    <div class="callout-box">
        <strong>Civil Engineering & Photometric Disclaimer:</strong><br>
        Controller setpoints (commanded dimming %) are <strong>not proof of achieved road-surface illumination</strong>. 
        Physical municipal deployment requires on-site photometric validation using luminance cameras to guarantee 
        compliance with civil standards (EN 13201 / IESNA RP-8 / IS 1944). In event of communication failure or sensor 
        dropouts, fixtures must automatically revert to fail-safe 100% full illumination.
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown("#### 📑 Executive Findings Report (Markdown/Text)")

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
