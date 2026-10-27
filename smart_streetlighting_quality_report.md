# Synthetic Dataset Quality Report

- **Dataset status:** PASS — generated and validated
- **Dataset type:** Synthetic, not field measurements
- **Rows:** 210,240
- **Columns:** 36
- **Street segments:** 24
- **Date range:** 2025-01-01 00:00:00 through 2025-12-31 23:00:00
- **Expected rows:** 24 segments × 8,760 hours = 210,240
- **Rows per street:** 8,760 to 8,760
- **Unique record_id:** PASS
- **Unique street-hour key:** PASS
- **Missing values:** 1,050 across selected sensor/derived columns (intentional, documented)
- **Sensor-flagged rows:** 1,062
- **Simulated lighting-fault rows:** 101
- **Energy value validity:** PASS — energy_consumption_kwh has no missing/non-positive values
- **Dimming command range:** PASS — all commanded_dimming_pct values lie within 0–100
- **Lighting-off consistency:** PASS — commanded dimming is zero when lamp_status is Off
- **Total energy consumption (kWh):** 195,188.96
- **Full-output reference (kWh):** 234,935.49
- **Naive reference-comparison difference:** 16.92% (illustrative synthetic comparison only; analyze faults/anomalies separately)
- **Average temperature (°C):** 28.13
- **Rows with reported rainfall > 0:** 5.61%
- **Modeling caveat:** Traffic profiles, weather, controller policies, faults and sensor anomalies are simulated assumptions. EDA findings demonstrate a workflow and must not be presented as real-city evidence.
- **Valid-row reference comparison (faults and energy-meter spikes excluded):** 16.84% (illustrative synthetic comparison only; not a real-world savings estimate)

## Known design choices

- A full calendar year is represented at hourly resolution for each of 24 synthetic street segments.
- Weather is shared at city-hour level, with small per-street measurement noise on some environmental fields.
- Controlled missing values and sensor/meter anomalies were inserted deliberately and are flagged.
- Lighting-fault intervals are operational events, not successful energy savings.
- `full_output_reference_kwh` is a modeled comparator, not a measured utility baseline.
- Aggregate reference-comparison percentages exclude simulated lighting faults and flagged energy-meter spikes; they remain synthetic demonstrations, not empirical findings.
