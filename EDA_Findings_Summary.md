# Exploratory Data Analysis & Optimization Findings Summary

## Project Title
**Data-Driven Smart Streetlighting: Optimizing Energy Usage Profiles Through Exploratory Data Analysis**

**Academic Level:** Undergraduate Data Science Capstone Project  
**Author / Data Science Lead:** Senior Data Science Mentor & Student Project Team  
**Primary Artifact:** [`Smart_Streetlighting_EDA_Optimization.ipynb`](file:///v:/Project/DS%20SAT/Smart_Streetlighting_EDA_Optimization.ipynb)  
**Dependencies:** [`notebook_requirements.txt`](file:///v:/Project/DS%20SAT/notebook_requirements.txt)  
**Source Dataset:** `smart_streetlighting_synthetic_dataset.csv` (210,240 records, 36 features, 24 street segments, hourly throughout 2025)

---

## 1. Executive Summary & Academic Disclosure

### 1.1 Executive Summary
Streetlighting is a fundamental municipal infrastructure service, often accounting for **15% to 40% of municipal electrical expenditures**. This investigation conducts an end-to-end exploratory data analysis and rule-based optimization study on a year-long hourly dataset of 24 street segments in a simulated South Indian urban archetype.

The study establishes that current operations consume **195,138.48 kWh** annually across the 24 segments, achieving **16.84% baseline savings** against a theoretical full-output comparator (234,655.06 kWh). However, profound inefficiencies persist due to **static Fixed-mode circuits** (which draw 100% rated power all night with 0.00% savings) and **legacy High-Pressure Sodium (HPS) fixtures** (which consume up to $557.44\text{ kWh/luminaire-year}$). 

By introducing a **Multi-Tier Demand-Responsive Adaptive Lighting Profile (Scenario 2)** with road-class safety floors, vehicular/pedestrian demand scaling, and automated adverse weather overrides, the municipal grid can safely reduce consumption to **163,519.50 kWh annually**, delivering an **additional 31,618.98 kWh (16.20%) in energy reduction relative to baseline**, pushing overall grid efficiency to **30.31% savings** against full output.

### 1.2 Mandatory Academic Disclosure & Synthetic Nature
> **CRITICAL ACADEMIC INTEGRITY NOTICE:**  
> The dataset evaluated in this project is **entirely synthetic and computer-generated** (`SEED = 42`). It simulates environmental and operational conditions for a warm South Indian city archetype.  
> - **No Field Telemetry:** No observation represents measured physical meter data or traffic sensor counts from Coimbatore, Chennai, or any real municipal grid.
> - **Absence of Road-Surface Photometric Measurements:** The dataset lacks physical measurements of pavement luminance ($cd/m^2$) or ground illuminance ($lux$). Features such as `target_illuminance_lux` and `commanded_dimming_pct` are planning parameters and controller setpoints, not certified photometric field performance.
> - **Simulated Energy Savings:** All energy savings (baseline and scenario models) are mathematical projections derived from component power equations. They must **never be cited as real-world validated energy savings or field performance**.

---

## 2. Answers to Core Analytical Inquiries

### Central Research Question:
> *"How can exploratory data analysis identify patterns in streetlight energy consumption and support the development of more energy-efficient lighting schedules without compromising required illumination?"*

EDA resolves this question by decoupling the temporal, spatial, meteorological, and infrastructural drivers of energy consumption:
1. **Temporal Auditing:** Pinpoints the precise 18:00–06:00 operational window, isolating non-negotiable peak traffic hours from deep-night low-demand windows.
2. **Infrastructure Auditing:** Identifies that high total consumption on Arterials reflects high luminaire counts and safety requirements, whereas high specific consumption on Campus streets reflects inefficient HPS fixtures.
3. **Data Quality Scrubbing:** Excludes sensor spikes and physical lamp outages from energy baselines, preventing corrupted or misleading efficiency estimates.
4. **Parametric Simulation:** Evaluates rule-based adaptive controller setpoints against physical luminaire power curves, establishing achievable savings boundaries under strict civil safety floors.

---

### Specific Analytical Inquiries & Computed Findings:

| # | Analytical Question | Computed Findings from Validated Dataset | Practical Engineering Significance |
| :-: | :--- | :--- | :--- |
| **Q1** | **During which hours is energy consumption highest?** | Grid consumption peaks at **20:00 (20,534.64 kWh/yr)** and **21:00 (20,545.92 kWh/yr)**. Daylight standby (07:00–17:00) draws only **528.58 kWh** (0.27% of annual grid total). | Evening rush hours demand 100% illumination. Deep night (01:00–04:00) offers immediate curtailment potential. |
| **Q2** | **How do consumption patterns differ across street types?** | **Arterial Zone** consumes the highest total energy (**51,185.30 kWh**), while **Campus Zone** consumes the highest *energy per luminaire* (**557.44 kWh/fixture**). **Residential Zone** achieves the highest baseline savings (**32.96%**). | High total energy $\neq$ inefficiency. Fixture density and lamp technology govern specific energy consumption. |
| **Q3** | **What relationships exist between traffic, pedestrians, lux, dimming, and energy?** | Ambient lux dictates circuit activation ($\rho = -0.89$ with energy) at a steep 50-lux cliff. During active night hours, traffic and pedestrians correlate moderately with dimming ($\rho = +0.38$ to $+0.52$). | Natural daylight provides binary switching; adaptive controllers modulate intensity with human activity. |
| **Q4** | **How do weather and seasonal patterns relate to lighting?** | Shorter summer nights reduce daily grid load from **559.5 kWh/day (Jan–May)** down to **507.0 kWh/day (Jul–Nov)**, an organic **9.4% seasonal drop**. Rainy hours (5.6% of intervals) trigger weather safety overrides. | Calendar comparisons must be daily-normalized. Wet pavements require higher illumination setpoints. |
| **Q5** | **Which operational periods offer energy reduction opportunities?** | **Deep Off-Peak Hours (00:00–05:00)** across segments currently running on Fixed or rigid Scheduled modes. | Fixed circuits consume full power while traffic is near zero, representing direct energy waste. |
| **Q6** | **How much modeled energy reduction could alternative profiles produce?** | **Scenario 1 (Conservative):** Saves **12,801.87 kWh (6.56%)**.<br>**Scenario 2 (Adaptive Standard):** Saves **31,618.98 kWh (16.20%)**.<br>**Scenario 3 (Eco Deep-Night):** Saves **36,165.19 kWh (18.53%)**. | Transitioning static circuits to demand-responsive profiles safely saves up to 16.2% additional grid energy. |

---

## 3. Data Quality & Anomaly Assessment Summary

A comprehensive automated audit verified the 210,240 records across 15 validation rules:

### 3.1 Validation Suite Results
- **Row Duplication:** Zero duplicate rows; `record_id` is 100% unique across all 210,240 records (**PASS**).
- **Spatio-Temporal Key:** Unique combination of `street_id` + `timestamp_local` with zero duplicates (**PASS**).
- **Temporal Completeness:** Exactly 8,760 hourly observations for each of the 24 street segments (**PASS**).
- **Physical Bounding:** All traffic counts $\ge 0$, rainfall $\ge 0$, energy $> 0$, and dimming setpoints strictly bounded within $[0, 100]\%$ (**PASS**).
- **Operational Consistency:** When `lamp_status == 'Off'`, commanded and measured dimming are strictly $0.0\%$ (**PASS**). When `lamp_status == 'Fault'`, measured output is strictly $0.0\%$ (**PASS**).

### 3.2 Injected Anomalies vs. Operational Outages
The synthetic generator intentionally injected specific corruptions to evaluate robust preprocessing:
1. **Missing Sensor Readings (1,050 total):** `ambient_light_lux` (252 missing), `traffic_count` (210 missing), `humidity_pct` (168 missing), `visibility_km` (168 missing), `measured_dimming_pct` (126 missing).
2. **Sensor Spikes (144 total):** 73 ambient light spikes (up to 1,800 lux at midnight), 46 traffic counter spikes (up to 2,500 vehicles/hr), and 25 energy-meter spikes (up to 13.68 kWh).
3. **Simulated Lamp Outages (101 records):** Circuit-level physical faults where lamps fail during dark hours, drawing only $0.012\text{ kWh}$ standby power.

### 3.3 Auditable Analysis Subsets
Flagged records were **not deleted**, preserving data integrity across traceable subsets:
- **`df_raw` (210,240 rows):** Complete, immutable source dataset.
- **`df_valid_energy` (210,114 rows):** Excludes the 101 lighting fault rows and 25 energy-meter spikes. Serves as the primary baseline for all energy efficiency and scenario comparisons.
- **`df_clean_complete` (209,077 rows):** Excludes all 1,062 sensor anomaly records and faults (`data_quality_flag == 'OK'`). Used for statistical correlations and sensor distributions.
- **`df_active_night` (100,968 rows):** Clean records during required lighting hours (`lighting_required_flag == 1`).

---

## 4. Infrastructure & Zone Comparative Analysis

The 24 street segments were evaluated across zone types, road classifications, lamp technologies, and control modes.

### 4.1 Zone-Level Performance Breakdown (Valid Population)

| Zone Type | Segments | Total Luminaires | Annual Energy (kWh) | Full Output Ref (kWh) | Energy / Luminaire (kWh/yr) | Baseline Savings (%) | Mean Traffic (veh/hr) | Mean Pedestrians (pers/hr) |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Arterial** | 5 | 97 | 51,185.30 | 56,373.54 | 527.68 | 9.20% | 132.7 | 9.7 |
| **Highway** | 4 | 71 | 38,317.83 | 45,332.40 | 539.69 | 15.47% | 172.2 | 1.1 |
| **Campus** | 3 | 59 | 32,889.06 | 37,280.85 | **557.44** | 11.78% | 25.0 | 18.3 |
| **Residential** | 5 | 95 | 27,765.83 | 41,415.81 | **292.27** | **32.96%** | 17.1 | 7.5 |
| **Commercial** | 4 | 72 | 24,647.12 | 29,595.82 | 342.32 | 16.72% | 62.7 | 30.1 |
| **Industrial** | 3 | 53 | 20,333.34 | 24,656.65 | 383.65 | 17.53% | 29.6 | 3.6 |
| **Grid Total** | **24** | **447** | **195,138.48** | **234,655.06** | **436.55** | **16.84%** | **77.1** | **11.2** |

### 4.2 Lamp Technology: Solid-State LED vs. Legacy HPS
- **LED Circuits (18 streets, 351 luminaires):** Consumed **137,801.32 kWh** (specific load: $392.60\text{ kWh/fixture}$). LED luminaires offer linear dimming flexibility ($P_{\text{fraction}} = 0.025 + 0.975 \times \text{dimming}/100$), allowing deep power cuts down to 2.5% standby.
- **HPS Circuits (6 streets, 96 luminaires):** Consumed **57,337.16 kWh** (specific load: **$597.26\text{ kWh/fixture}$**). HPS ballasts suffer from severe dimming non-linearity ($P_{\text{fraction}} = 0.180 + 0.820 \times \text{dimming}/100$), drawing at least 18% power even when deeply dimmed, and are restricted to static Fixed or Scheduled modes.

### 4.3 Installed Control Modes: Fixed vs. Scheduled vs. Adaptive
- **Fixed Mode (6 streets):** Consumed **58,079.13 kWh** at constant 100% wattage throughout the night. Achieved **0.00% savings** vs. full output.
- **Scheduled Dimming (9 streets):** Consumed **76,145.24 kWh**, achieving **20.91% savings** via time-clock setpoint reductions.
- **Adaptive Dimming (9 streets):** Consumed **60,914.11 kWh**, achieving **23.83% savings** by dynamically tracking local traffic sensors.

---

## 5. Modeled Lighting Optimization Scenarios

To quantify the energy reduction potential of smart lighting, three rule-based scenarios were implemented using explicit component power equations matching circuit hardware.

### 5.1 Scenario Architecture & Rules
1. **Scenario 1 — Conservative Schedule Tuning:**
   - Targets only Fixed and rigid Scheduled circuits during deep off-peak hours (23:00 to 04:00).
   - Enforces conservative dimming floors: Arterial/Highway $\ge 75\%$, Commercial/Industrial $\ge 65\%$, Residential/Campus $\ge 55\%$.
   - Reverts to 100% if rain $> 0.4\text{ mm}$ or visibility $< 4.0\text{ km}$.
   - Retains existing adaptive controllers unchanged.
2. **Scenario 2 — Demand-Responsive Adaptive Multi-Tier (Recommended):**
   - Applies a unified dynamic policy across all 24 street segments.
   - Enforces road-class safety floors: Highway $65\%$, Arterial $60\%$, Commercial/Industrial $50\%$, Campus $45\%$, Residential $40\%$.
   - Dynamically scales intensity $+0\%$ to $+30\%$ based on normalized traffic density, and $+0\%$ to $+10\%$ based on pedestrian load.
   - Adds $+5\%$ during evening transition (18:00–22:00) and $+20\%$ adverse weather boost (rain $> 0.3\text{ mm}$ or visibility $< 3.5\text{ km}$).
3. **Scenario 3 — Aggressive Deep-Night Eco-Profile (Stress-Test):**
   - Implements deeper curtailment ($30\%-50\%$ floor) during the dead-of-night window (01:00 to 04:00) when traffic drops below 30 vehicles/hr.
   - Evaluated as an exploratory boundary test to identify the theoretical upper limit of energy savings.

### 5.2 Scenario Performance Comparison Table

| Profile Architecture | Annual Energy (kWh) | Reduction vs. Baseline (kWh) | Savings vs. Baseline (%) | Total Savings vs. Full Output (%) | Mean Night Dimming Level (%) |
| :--- | :-: | :-: | :-: | :-: | :-: |
| **Full-Output Reference Comparator** | 234,655.06 | -39,516.58 | -20.25% | 0.00% | 100.0% |
| **Observed Baseline Grid Telemetry** | 195,138.48 | 0.00 | 0.00% | 16.84% | 81.33% |
| **Scenario 1: Conservative Schedule Tuning** | 182,336.61 | **12,801.87** | **+6.56%** | **22.30%** | 76.51% |
| **Scenario 2: Adaptive Standard (Recommended)** | 163,519.50 | **31,618.98** | **+16.20%** | **30.31%** | 68.64% |
| **Scenario 3: Aggressive Deep-Night Eco** | 158,973.29 | **36,165.19** | **+18.53%** | **32.25%** | 66.86% |

```
Annual Grid Energy Consumption Comparison:
========================================================================================
Full Output Ref : [##################################################] 234,655 kWh (0.0% sav)
Baseline Grid   : [##########################################........] 195,138 kWh (16.8% sav)
Scenario 1      : [#######################################...........] 182,337 kWh (22.3% sav)
Scenario 2 (Rec): [#################################.................] 163,520 kWh (30.3% sav)
Scenario 3 (Eco): [################################..................] 158,973 kWh (32.3% sav)
========================================================================================
```

### 5.3 Sensitivity Analysis of Minimum Dimming Floor
Varying the base minimum dimming floor in Scenario 2 from $30\%$ to $75\%$ revealed:
- At **$75\%$ floor:** Annual grid energy is $176,170\text{ kWh}$ ($9.72\%$ savings vs baseline).
- At **$50\%$ floor (midpoint):** Annual grid energy is $163,130\text{ kWh}$ ($16.40\%$ savings vs baseline).
- At **$30\%$ floor:** Annual grid energy reaches $156,086\text{ kWh}$ ($20.01\%$ savings vs baseline), but plateauing occurs because dynamic traffic and weather boosts prevent further dimming during active hours.

---

## 6. Practical Recommendations & Municipal Implementation Roadmap

For municipal electrical engineers and urban planners seeking to translate these findings into a real-world smart streetlighting rollout, we recommend the following phased roadmap:

### Phase 1: Zero-Capital Firmware Optimization (Immediate)
- **Reprogram Fixed Circuits:** Transition the 6 Fixed-mode street segments to off-peak scheduled dimming (Scenario 1). This delivers an immediate **12,801.87 kWh (6.56%) annual energy reduction** without requiring new hardware.
- **Implement Astronomical Clocks:** Replace raw ambient light photocell triggers with astronomical clock algorithms that calculate solar zenith angles, eliminating false switching caused by dirty or shaded optical sensors.

### Phase 2: Targeted Hardware Modernization (Short-Term: 6–12 Months)
- **Prioritize HPS-to-LED Retrofits:** Replace the 96 High-Pressure Sodium fixtures across Campus (ST018–ST020), Highway (ST022–ST023), and Arterial (ST009) segments with high-efficacy LED luminaires ($130-150\text{ lm/W}$). This will reduce specific energy consumption on those circuits by over $40\%$.
- **Deploy Radar/Video Traffic Counters:** Install dual-technology vehicular and pedestrian sensors on arterial and commercial corridors to enable real-time demand-responsive dimming.

### Phase 3: Demand-Responsive Grid Orchestration (Medium-Term: 12–24 Months)
- **Commission Multi-Tier Adaptive Control (Scenario 2):** Implement dynamic setpoint modulation with road-class safety floors, yielding an estimated **16.20% additional energy reduction**.
- **Integrate Automated Weather Overrides:** Connect Central Management Software (CMS) to local Doppler radar or road-weather sensors to automatically ramp illumination to 100% during heavy rain ($>0.3\text{ mm/hr}$) or fog ($<3.5\text{ km}$ visibility).

### Phase 4: Compliance & Civil Standards Auditing
- **Photometric Road-Surface Field Audits:** Validate that dimmed setpoints satisfy civil lighting standards (EN 13201 / IESNA RP-8 / IS 1944). A controller setpoint of $60\%$ must be verified using physical luminance meters to guarantee average road luminance ($L_{\text{avg}} \ge 1.0-2.0\text{ cd/m}^2$) and overall uniformity ($U_0 \ge 0.40$).
- **Fail-Safe Controller Architecture:** Ensure that if local IoT communications or traffic sensors fail, fixtures automatically default to safe 100% full illumination.

---

## 7. Deliverables & File Index

All deliverables for this undergraduate Data Science capstone project have been constructed, validated, executed, and saved in the project root directory:

| Deliverable Artifact | File Path | Status | Verification & Content Summary |
| :--- | :--- | :---: | :--- |
| **Jupyter Notebook** | [`Smart_Streetlighting_EDA_Optimization.ipynb`](file:///v:/Project/DS%20SAT/Smart_Streetlighting_EDA_Optimization.ipynb) | **EXECUTED** | 59 cells (29 code cells, 30 markdown cells). Fully executed from fresh kernel; contains all embedded tables, stdout logs, and 24 high-resolution matplotlib/seaborn figures. |
| **Requirements Specification** | [`notebook_requirements.txt`](file:///v:/Project/DS%20SAT/notebook_requirements.txt) | **VERIFIED** | Pinned dependency constraints: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `nbformat`, `nbclient`, `ipykernel`. |
| **Findings Report** | [`EDA_Findings_Summary.md`](file:///v:/Project/DS%20SAT/EDA_Findings_Summary.md) | **COMPLETE** | Comprehensive findings report detailing computed metrics, data quality taxonomy, scenario tables, and civil engineering recommendations. |
| **Automated Build Script** | [`build_and_run_notebook.py`](file:///v:/Project/DS%20SAT/build_and_run_notebook.py) | **VERIFIED** | Programmatic builder that compiles the notebook schema, runs the kernel via `nbclient`, and asserts reproducibility. |

*Report finalized and certified for undergraduate academic evaluation.*
