# LUX — Smart Streetlighting Intelligence Platform

**Web Application Phase | Undergraduate Data Science Capstone Project**  
*Theme:* Data-Driven Smart Streetlighting: Optimizing Energy Usage Profiles Through Exploratory Data Analysis  
*Product Name:* **LUX | Streetlight Intelligence**

---

## 1. Executive Overview

**LUX** is an interactive, production-quality municipal smart streetlighting analytics platform built with **Python, Streamlit, and Plotly**. The application showcases an end-to-end data science workflow on a full-scale municipal grid dataset comprising **210,240 hourly records across 24 synthetic street segments for the entire calendar year 2025**.

The platform translates the exploratory data analysis and mathematical modeling established in [`Smart_Streetlighting_EDA_Optimization.ipynb`](Smart_Streetlighting_EDA_Optimization.ipynb) and [`EDA_Findings_Summary.md`](EDA_Findings_Summary.md) into an interactive executive tool for municipal electrical engineers, urban planners, and academic faculty evaluators.

### Academic Integrity Notice
> **Synthetic Dataset Disclosure:**  
> All records, traffic counts, weather conditions, and energy consumption figures are simulated (`SEED = 42`) for an archetype warm South Indian city. All baseline comparisons and optimization projections are mathematical simulations based on component power equations. They have **not been calibrated against physical field telemetry or measured road-surface illumination**. Field deployment requires engineering audits and compliance with civil road-lighting standards (**EN 13201 / IESNA RP-8 / IS 1944**).

---

## 2. Application Architecture & Features

The application is structured into **seven dedicated navigation portals**, accessible via a collapsible sidebar with dynamic global filters:

### Portal 1 — 📊 Executive Overview
- **Dynamic KPI Cards:** Total Observed Energy (195,138.5 kWh), Full-Output Reference (234,655.1 kWh), Baseline Savings % (16.84%), Scope (24 segments, 210,240 records), Mean Night Dimming Level (81.3%), Cleanliness Ratio (99.45%), Flagged Sensor Anomalies (1,062), and Simulated Outages (101).
- **Time-Series Telemetry:** 365-day continuous grid load curve with a 7-day centered moving average.
- **Diurnal Load Curve:** 24-hour composite curve identifying the 20:00–21:00 peak demand window.
- **Zone Energy Comparisons:** Observed vs. full-output reference comparisons across all 6 zone classifications.
- **Demand Response Scatter:** Real-time relationship between vehicular traffic and luminaire dimming levels.

### Portal 2 — ⚡ Energy Consumption Analytics
- **Temporal Patterns:** Diurnal profiles stratified by installed control mode (Fixed, Scheduled, Adaptive).
- **Weekday vs. Weekend Dynamics:** Disparities across commercial, industrial, and campus sectors.
- **Daily-Normalized Monthly Analysis:** Dual-axis visualization demonstrating the 9.4% organic summer solar reduction (from ~559.5 kWh/day in Jan–May to ~507.0 kWh/day in Jul–Nov) caused by earlier sunrises and later sunsets.
- **Hardware Evaluation:** High-efficacy solid-state LEDs vs. legacy High-Pressure Sodium (HPS) luminaires, detailing fixture wattages, dimming linearity, and specific load ($kWh/\text{luminaire-year}$).

### Portal 3 — 🛣️ Street & Infrastructure Explorer
- **Searchable Inventory Table:** Interactive table covering all 24 synthetic corridors with physical specifications, fixture counts, rated wattages, circuit capacities, annual consumption, and baseline savings.
- **Single-Segment Deep Dive:** Granular corridor-level inspection including custom diurnal profiles, traffic-response scatters, and hardware metadata.
- **Data Export:** Instant CSV export of filtered street telemetry.

### Portal 4 — 🚦 Traffic, Weather & Lighting Dynamics
- **Interactive Correlation Matrix:** Selectable Spearman rank-order ($\rho$) and Pearson linear ($r$) correlation heatmaps across 16 operational and environmental variables.
- **Custom Bivariate Scatter Explorer:** User-selectable X and Y axes with optional Ordinary Least Squares (OLS) trendlines and zone/technology color stratification.
- **Twilight Switching Cliff:** Logarithmic ambient lux analysis demonstrating the steep 50-lux controller switching threshold.
- **Meteorological Boxplots:** Consumption distributions across Clear, Partly cloudy, Cloudy, Rain, and Thunderstorm conditions.

### Portal 5 — 🛡️ Data Quality & Anomaly Monitor
- **Quality Assurance KPIs:** Cleanliness percentages, injected sensor spikes, packet dropouts, and circuit outages.
- **Anomaly Taxonomy:** Detailed breakdown of ambient light spikes (up to 1,800 lux), traffic counter spikes (up to 2,500 veh/hr), and energy-meter spikes (up to 13.68 kWh).
- **Missing Value Matrix:** Visualizing controlled sensor dropouts across environmental and dimming fields.
- **Flagged Record Inspector:** Interactive table and audit log download for all 1,163 flagged quality events.
- **Outage Safeguard:** Explicit accounting separating physical outages from energy savings.

### Portal 6 — 🚀 Energy Optimization Simulator
- **Exact Physics-Based Engine:** Reuses the verified component power equations:
  - $\text{LED: } P_{\text{frac}} = 0.025 + 0.975 \times (\text{dimming}/100)$
  - $\text{HPS: } P_{\text{frac}} = 0.180 + 0.820 \times (\text{dimming}/100)$
  - $\text{Active: } \text{circuit\_kw} \times P_{\text{frac}} + 0.008\text{ kWh} \quad | \quad \text{Standby: } 0.002\text{ kWh}$
- **Three Modeled Scenarios:**
  1. *Scenario 1 (Conservative Schedule Tuning):* 182,336.6 kWh (+6.56% savings vs. baseline).
  2. *Scenario 2 (Adaptive Multi-Tier — Recommended):* 163,519.5 kWh (+16.20% savings vs. baseline | 30.31% total vs. full output).
  3. *Scenario 3 (Aggressive Deep-Night Eco):* 158,973.3 kWh (+18.53% savings vs. baseline).
- **Interactive Slider Controls:** Real-time parameter tuning for minimum dimming floors, deep-night hours, and adverse weather safety overrides (+20% intensity boost during rain/fog).
- **Civil Engineering Safeguards:** Clear labels identifying setpoints as controller parameters, not certified road luminance.

### Portal 7 — 📥 Reports & Data Export
- **One-Click CSV Exports:** Filtered raw telemetry, street inventory summary, monthly performance summary, zone comparison summary, optimization scenario projections, and quality anomaly audit logs.
- **Executive Findings Download:** Direct download of the complete Markdown project findings report ([`EDA_Findings_Summary.md`](EDA_Findings_Summary.md)).

---

## 3. Installation & Launch Instructions

### Prerequisites
- Python 3.10+
- Pip package manager

### 1. Install Dependencies
Install all required packages using the application requirements file:
```bash
pip install -r requirements_app.txt
```

### 2. Run Automated Verification Tests
Run the mathematical parity test suite to verify that calculation engines agree with the verified EDA notebook:
```bash
python test_calculations.py
```
*(All 5 tests must pass with `OK`)*

### 3. Launch the Application
Start the Streamlit local web server:
```bash
streamlit run app.py
```

The application will open automatically in your default web browser at:
```
http://localhost:8501
```

---

## 4. File Structure & Project Inventory

```
v:/Project/DS SAT/
├── app.py                                   # Main Streamlit web application & routing
├── data_loader.py                           # Caching, feature engineering, and filtering pipeline
├── analytics.py                             # Statistical modeling, KPIs, and optimization engine
├── components.py                            # Custom dark-theme CSS, Plotly stylers, and widgets
├── test_calculations.py                     # Automated unit test suite verifying notebook parity
├── requirements_app.txt                     # Application dependencies (Streamlit, Plotly, etc.)
├── README_app.md                            # Application setup, architecture, and feature documentation
├── Smart_Streetlighting_EDA_Optimization.ipynb # Fully executed 59-cell EDA and optimization notebook
├── EDA_Findings_Summary.md                  # Comprehensive academic project findings report
├── notebook_requirements.txt                # Notebook execution dependencies
└── smart_streetlighting_synthetic_dataset.csv # Primary 210,240-row synthetic municipal grid dataset
```

---

## 5. UI/UX Design System

- **Surfaces & Atmosphere:** Executive dark slate surfaces (`#0b1120`, `#0f172a`, `#131e32`) avoiding default Streamlit demo aesthetics.
- **Accents:** Electric Cyan (`#00e5ff`) and Sky Blue (`#38bdf8`) with Emerald Green (`#10b981`) for energy savings, Amber (`#f59e0b`) for warnings/spikes, and Crimson (`#ef4444`) for circuit outages.
- **Plotly Styling:** Customized dark charts (`template='plotly_dark'`) with responsive gridlines, readable legends, and clear physical units ($kWh$, $lux$, $veh/hr$, $\text{fixture-year}$, $\%$) on every axis and hover tooltip.
- **Responsive Navigation:** Collapsible sidebar housing active multidimensional filters with instant reset capabilities.
