# Data-Driven Smart Streetlighting — Synthetic Dataset

## What this is
This package contains a reproducible synthetic dataset for the college Data Science project **“Data-Driven Smart Streetlighting: Optimizing Energy Usage Profiles Through Exploratory Data Analysis.”**

The full dataset has **210,240 rows and 36 columns**: 24 fictional street segments observed once per hour from 2025-01-01 00:00 through 2025-12-31 23:00. Each street therefore has 8,760 hourly records.

**Important:** These are simulated observations, not measurements from Coimbatore or any other real city. The warm South Indian seasonal/weather pattern is an illustrative design assumption and is not calibrated against a weather station, traffic survey, lighting audit, or utility bill.

## Files
- `smart_streetlighting_synthetic_dataset.csv` — the complete dataset.
- `smart_streetlighting_data_dictionary.csv` — column definitions, units, and caveats.
- `smart_streetlighting_quality_report.md` — generated validation checks and data-quality counts.
- `smart_streetlighting_monthly_summary.csv` — monthly totals and per-day-normalized comparison metrics.
- `smart_streetlighting_zone_summary.csv` — aggregate view by street/zone type.
- `generate_smart_streetlighting_dataset.py` — reproducible generator (fixed random seed: 42).
- `smart_streetlighting_eda_companion.xlsx` — optional workbook with an inspectable sample and summary sheets (the full dataset is in the CSV, not in the sample tab).

The monthly summary includes both monthly totals and average-per-day values, so month-to-month comparisons are not distorted by different month lengths.

## Suggested project workflow
1. Load the full CSV in Pandas.
2. Parse `timestamp_local` and inspect schema, missingness, duplicates, and ranges.
3. Conduct EDA by hour, weekday/weekend, month, `zone_type`, `lamp_technology`, and `control_mode`.
4. Analyze the relationship between traffic/pedestrians, dimming intensity, ambient light, and consumption.
5. Examine `lamp_status == "Fault"` and `sensor_anomaly_flag == 1` separately; do not interpret outages as savings.
6. Compare observed energy with `full_output_reference_kwh`, stating that this comparator is synthetic.
7. Propose and evaluate an optimization profile with clear safety constraints and assumptions.

## Important interpretation notes
- `timestamp_local` is the start of the hourly interval. Weather and lighting features approximate conditions during that interval; energy fields represent simulated hourly kWh.
- `lighting_required_flag` is generated from the latent ambient-light value using a 50-lux threshold. A deliberately corrupted sensor reading can disagree with this flag.
- `commanded_dimming_pct` is a controller setpoint; `measured_dimming_pct` includes small simulated measurement noise.
- `energy_consumption_kwh` is simulated circuit consumption. It depends on the number and rated wattage of luminaires, commanded/measured dimming, technology, and simulated fault state.
- `full_output_reference_kwh` represents the theoretical full-output circuit load while lighting is required, plus small controller standby when off.
- `energy_savings_vs_full_output_pct` is missing for simulated lighting faults and injected energy-meter spikes, because those records are unsuitable for comparing normal energy use. The monthly/zone summary comparison also excludes these records; it remains synthetic.
- Some selected sensor fields intentionally contain missing values. Use the data dictionary and quality report when preprocessing.
- The generator uses NumPy and Pandas. Run `python generate_smart_streetlighting_dataset.py` to regenerate the CSV in the current `/mnt/data`-style output directory (or update `OUT` in the script for a different destination).

## Reproducibility
The generator uses `SEED = 42`. The output is deterministic for the package's pinned code and compatible NumPy/Pandas random-number behavior.

## Academic integrity
Describe the dataset explicitly as synthetic in your report/presentation. You can use it to demonstrate EDA, data cleaning, anomaly detection, and a modeled optimization scenario. Do not claim its patterns, savings, or statistical estimates are measured real-world results.
