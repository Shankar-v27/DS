import os, math, json, zipfile, textwrap, sys, statistics
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

SEED = 42
YEAR = 2025
N_STREETS = 24
rng = np.random.default_rng(SEED)

timestamps = pd.date_range(f"{YEAR}-01-01 00:00:00", f"{YEAR}-12-31 23:00:00", freq="h")
n_hours = len(timestamps)
assert n_hours == 8760

hour = timestamps.hour.to_numpy()
doy = timestamps.dayofyear.to_numpy()
dow = timestamps.dayofweek.to_numpy()
is_weekend = (dow >= 5)

# --- City-level hourly environment: plausible synthetic warm South Indian climate ---
monthly_temp_means = {1:25.0, 2:26.5, 3:29.0, 4:31.0, 5:31.5, 6:29.5, 7:28.5, 8:28.2, 9:28.2, 10:27.5, 11:26.5, 12:25.5}
monthly_cloud_means = {1:28, 2:30, 3:32, 4:40, 5:47, 6:58, 7:65, 8:65, 9:58, 10:61, 11:63, 12:45}
rain_start_prob = {1:.004, 2:.004, 3:.006, 4:.012, 5:.020, 6:.040, 7:.052, 8:.052, 9:.042, 10:.050, 11:.045, 12:.018}
rain_continue_prob = {m: (0.42 if m in [1,2,3,4,5,12] else 0.56) for m in range(1,13)}

months = timestamps.month.to_numpy()
temp_means = np.array([monthly_temp_means[int(m)] for m in months])
cloud_means = np.array([monthly_cloud_means[int(m)] for m in months])

# Cloud cover as a temporally autocorrelated process around seasonal means.
cloud = np.zeros(n_hours, dtype=float)
cloud[0] = np.clip(cloud_means[0] + rng.normal(0, 10), 0, 100)
for i in range(1, n_hours):
    cloud[i] = np.clip(cloud_means[i] + 0.82 * (cloud[i-1] - cloud_means[i]) + rng.normal(0, 7), 0, 100)

# Rainfall occurrence uses a Markov-like process, producing multi-hour rainy spells.
wet = np.zeros(n_hours, dtype=bool)
rainfall = np.zeros(n_hours, dtype=float)
wet_state = False
for i, m in enumerate(months):
    if wet_state:
        wet_state = rng.random() < rain_continue_prob[int(m)]
    else:
        wet_state = rng.random() < rain_start_prob[int(m)]
    wet[i] = wet_state
    if wet_state:
        seasonal_multiplier = 1.25 if int(m) in [6,7,8,9,10,11] else 0.8
        rainfall[i] = min(24.0, rng.gamma(shape=1.7, scale=1.25) * seasonal_multiplier)

# Adjust cloud cover during rain spells.
cloud = np.clip(cloud + wet.astype(float) * rng.uniform(10, 26, n_hours), 0, 100)

# Diurnal temperature cycle, daily weather variability, and modest sensor noise.
daily_key = pd.Series(timestamps.date).astype(str)
unique_dates, date_codes = np.unique(daily_key.to_numpy(), return_inverse=True)
daily_temp_jitter = rng.normal(0, 0.8, len(unique_dates))
temp = temp_means + 3.4 * np.sin(2 * np.pi * (hour - 8) / 24) + daily_temp_jitter[date_codes] + rng.normal(0, 0.25, n_hours)
temp = np.clip(temp, 16, 43)

# Humidity is typically higher before dawn and during rainy/cloudy periods.
humidity = (
    66
    + 13 * np.cos(2 * np.pi * (hour - 5) / 24)
    + (cloud - 45) * 0.13
    + wet.astype(float) * 5
    + rng.normal(0, 4, n_hours)
)
humidity = np.clip(humidity, 24, 100)

visibility = 10.8 - cloud * 0.018 - np.maximum(humidity - 80, 0) * 0.035 + rng.normal(0, 0.45, n_hours)
visibility -= np.minimum(rainfall, 15) * 0.38
visibility = np.clip(visibility, 0.4, 12.0)

# Approximate day length is fairly stable near 11 degrees north, with gentle seasonal shifts.
season_phase = 2 * np.pi * (doy - 80) / 365.0
sunrise = 6.03 + 0.13 * np.sin(season_phase)
sunset = 18.22 + 0.15 * np.sin(2 * np.pi * (doy - 170) / 365.0)
mid_hour = hour + 0.5

ambient_lux = np.zeros(n_hours, dtype=float)
is_daylight = (mid_hour >= sunrise) & (mid_hour <= sunset)
day_progress = np.zeros(n_hours, dtype=float)
day_progress[is_daylight] = (mid_hour[is_daylight] - sunrise[is_daylight]) / (sunset[is_daylight] - sunrise[is_daylight])
solar_shape = np.sin(np.pi * np.clip(day_progress, 0, 1))
cloud_transmission = 1 - 0.72 * cloud / 100
ambient_lux[is_daylight] = 76000 * np.power(np.maximum(solar_shape[is_daylight], 0), 1.35) * cloud_transmission[is_daylight]
# Twilight before sunrise and after sunset.
before_sunrise = mid_hour < sunrise
after_sunset = mid_hour > sunset
twilight_dist = np.zeros(n_hours, dtype=float)
twilight_dist[before_sunrise] = sunrise[before_sunrise] - mid_hour[before_sunrise]
twilight_dist[after_sunset] = mid_hour[after_sunset] - sunset[after_sunset]
twilight = (160 * np.exp(-3.7 * twilight_dist)) * (1 - 0.30 * cloud / 100)
ambient_lux[~is_daylight] = twilight[~is_daylight] + rng.lognormal(mean=-1.1, sigma=0.5, size=(~is_daylight).sum())
ambient_lux *= rng.lognormal(mean=0.0, sigma=0.055, size=n_hours)
ambient_lux = np.clip(ambient_lux, 0.05, 100000)
lighting_required = ambient_lux < 50.0

weather = np.empty(n_hours, dtype=object)
for i in range(n_hours):
    if rainfall[i] >= 6 and cloud[i] >= 78:
        weather[i] = "Thunderstorm"
    elif rainfall[i] > 0.1:
        weather[i] = "Rain"
    elif cloud[i] >= 76:
        weather[i] = "Cloudy"
    elif cloud[i] >= 42:
        weather[i] = "Partly cloudy"
    else:
        weather[i] = "Clear"

# Generic synthetic road segments. No real street-level observations are represented.
zone_specs = [
    ("Residential", 5, "Residential collector", 8.0, 45, 5, 15, 22, 12),
    ("Arterial", 5, "Urban arterial", 22.0, 62, 55, 110, 140, 50),
    ("Commercial", 4, "Commercial main road", 18.0, 55, 8, 25, 80, 95),
    ("Industrial", 3, "Industrial access road", 18.0, 55, 3, 60, 25, 70),
    ("Campus", 3, "Campus road", 12.0, 48, 1, 65, 45, 16),
    ("Highway", 4, "Highway", 24.0, 65, 120, 50, 35, 80),
]
street_rows = []
sid_num = 1
for zone, count, road_class, target_lux, min_adaptive, base_v, morning, midday, evening in zone_specs:
    for j in range(count):
        street_rows.append({
            "street_id": f"ST{sid_num:03d}",
            "street_name": f"Sector {sid_num:02d} {zone} Link",
            "zone_type": zone,
            "road_class": road_class,
            "target_illuminance_lux": round(target_lux + rng.normal(0, 1.1), 1),
            "minimum_adaptive_intensity": min_adaptive,
            "traffic_base": base_v,
            "traffic_morning": morning,
            "traffic_midday": midday,
            "traffic_evening": evening,
            "traffic_scale": float(rng.uniform(0.78, 1.28)),
            "luminaire_count": int(rng.integers(10, 31)),
        })
        sid_num += 1
streets = pd.DataFrame(street_rows)

# Assign hardware. HPS circuits are non-adaptive to keep dimming behavior plausible.
techs = np.array(["LED"] * 18 + ["High-pressure sodium (HPS)"] * 6, dtype=object)
rng.shuffle(techs)
street_modes = []
remaining = {"Fixed": 6, "Scheduled dimming": 9, "Adaptive dimming": 9}
# Allocate mode quota while ensuring HPS is Fixed or Scheduled.
for i, tech in enumerate(techs):
    allowed = ["Fixed", "Scheduled dimming"] if tech.startswith("High-pressure") else list(remaining.keys())
    candidates = [m for m in allowed if remaining[m] > 0]
    # Weighted by remaining quota, with preference to exact proportional allocation.
    weights = np.array([remaining[m] for m in candidates], dtype=float)
    weights /= weights.sum()
    m = rng.choice(candidates, p=weights)
    street_modes.append(m)
    remaining[m] -= 1
assert sum(remaining.values()) == 0, remaining

watts = []
for tech in techs:
    if tech == "LED":
        watts.append(int(rng.choice([80, 100, 120, 150], p=[0.12, 0.38, 0.32, 0.18])))
    else:
        watts.append(int(rng.choice([150, 250], p=[0.4, 0.6])))
streets["lamp_technology"] = techs
streets["control_mode"] = street_modes
streets["rated_watts_per_luminaire"] = watts

# For a more realistic city pattern, campus and industrial traffic is lower on weekends.
hour_float = hour.astype(float)
def gauss_peak(h, center, sigma):
    return np.exp(-0.5 * ((h - center) / sigma) ** 2)

all_rows = []
quality_events = []

for s_idx, s in streets.iterrows():
    zone = s["zone_type"]
    # Traffic expectation profile. Profiles are hourly counts per street segment, not total city traffic.
    lam = (
        s["traffic_base"]
        + s["traffic_morning"] * gauss_peak(hour_float, 8.0, 1.8)
        + s["traffic_midday"] * gauss_peak(hour_float, 13.2, 3.0)
        + s["traffic_evening"] * gauss_peak(hour_float, 18.5, 2.0)
    )
    # Zone-specific weekend behavior.
    weekend_mult = {
        "Residential": np.where(is_weekend, 1.06, 1.0),
        "Arterial": np.where(is_weekend, 0.78, 1.0),
        "Commercial": np.where(is_weekend, 1.22, 1.0),
        "Industrial": np.where(is_weekend, 0.38, 1.0),
        "Campus": np.where(is_weekend, 0.16, 1.0),
        "Highway": np.where(is_weekend, 0.92, 1.0),
    }[zone]
    rain_mult = np.where(rainfall > 0.4, 0.84, 1.0)
    lam = np.maximum(0.2, lam * s["traffic_scale"] * weekend_mult * rain_mult)
    traffic = rng.poisson(lam).astype(int)

    # Pedestrian counts depend on zone, time of day, and traffic.
    if zone == "Residential":
        ped_lam = 1.5 + 0.16 * traffic + 8 * gauss_peak(hour_float, 7.0, 1.4) + 10 * gauss_peak(hour_float, 19.0, 2.0)
    elif zone == "Arterial":
        ped_lam = 1 + 0.035 * traffic + 10 * gauss_peak(hour_float, 8.0, 1.5) + 12 * gauss_peak(hour_float, 18.0, 2.0)
    elif zone == "Commercial":
        ped_lam = 2 + 0.22 * traffic + 22 * gauss_peak(hour_float, 13.0, 3.0) + 28 * gauss_peak(hour_float, 19.0, 2.2)
    elif zone == "Industrial":
        ped_lam = 0.5 + 0.035 * traffic + 8 * gauss_peak(hour_float, 8.0, 1.3) + 6 * gauss_peak(hour_float, 17.0, 1.6)
    elif zone == "Campus":
        campus_activity = np.where((hour >= 7) & (hour <= 19), 24.0, 1.0)
        campus_activity *= np.where(is_weekend, 0.18, 1.0)
        ped_lam = 0.5 + 0.30 * traffic + campus_activity
    else:  # Highway
        ped_lam = 0.1 + 0.006 * traffic
    ped_lam *= np.where(is_weekend & (zone == "Commercial"), 1.12, 1.0)
    pedestrians = rng.poisson(np.maximum(0.05, ped_lam)).astype(int)

    # Night-time dimming command according to the installed control mode.
    command = np.zeros(n_hours, dtype=float)
    if s["control_mode"] == "Fixed":
        command[lighting_required] = 100.0
    elif s["control_mode"] == "Scheduled dimming":
        h = hour
        schedule = np.select(
            [
                (h >= 18) & (h <= 21),
                (h >= 22) | (h <= 1),
                (h >= 2) & (h <= 4),
                h == 5,
                h == 6,
            ],
            [100, 78, 58, 68, 82],
            default=90,
        )
        # Some road categories operate at lower overnight levels within a safe, synthetic envelope.
        if zone == "Residential":
            schedule = np.where((h >= 22) | (h <= 4), schedule - 5, schedule)
        elif zone in ("Arterial", "Highway"):
            schedule = np.where((h >= 22) | (h <= 4), schedule + 4, schedule)
        command[lighting_required] = schedule[lighting_required]
    else:
        peak_lam = max(float(np.max(lam)), 1.0)
        traffic_load = np.clip((traffic / peak_lam - 0.05) / 0.72, 0, 1)
        ped_load = np.clip(pedestrians / (45 if zone == "Commercial" else 20), 0, 1)
        bad_weather = ((rainfall > 0.3) | (visibility < 3.5)).astype(float)
        base = float(s["minimum_adaptive_intensity"])
        level = base + 27 * traffic_load + 6 * ped_load + 7 * bad_weather
        # Add a small evening-demand uplift, while maintaining the configured minimum.
        level += 4 * (((hour >= 18) & (hour <= 22)).astype(float))
        level += rng.normal(0, 1.8, n_hours)
        level = np.clip(level, base, 100)
        command[lighting_required] = np.round(level[lighting_required])
    command = np.clip(command, 0, 100)

    measured = np.clip(command + rng.normal(0, 1.35, n_hours), 0, 100)
    measured[~lighting_required] = 0.0
    measured = np.round(measured, 1)

    # Create short circuit-level outage episodes, concentrated in dark hours.
    fault = np.zeros(n_hours, dtype=bool)
    eligible = np.flatnonzero(lighting_required)
    event_count = int(rng.poisson(2.4))
    for _ in range(event_count):
        if len(eligible) == 0:
            break
        start = int(rng.choice(eligible))
        duration = int(np.clip(rng.geometric(0.38), 1, 8))
        end = min(n_hours, start + duration)
        fault[start:end] = True
    fault &= lighting_required
    measured[fault] = 0.0

    # Power model: hourly kWh for the circuit, with dimming curve and metering noise.
    count = int(s["luminaire_count"])
    rated = float(s["rated_watts_per_luminaire"])
    full_kw = count * rated / 1000.0
    active = lighting_required & (~fault)
    energy = np.full(n_hours, 0.002, dtype=float)  # small controller standby draw while lamps are off
    if s["lamp_technology"] == "LED":
        power_fraction = 0.025 + 0.975 * (measured / 100.0)
    else:
        # HPS dimming is deliberately less flexible than LED dimming.
        power_fraction = 0.18 + 0.82 * (measured / 100.0)
    energy[active] = (full_kw * power_fraction[active] + 0.008) * rng.normal(1.0, 0.012, active.sum())
    energy[fault] = 0.012
    energy = np.maximum(0.001, energy)

    # Store row-level features.
    frame = pd.DataFrame({
        "timestamp_local": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "date": timestamps.strftime("%Y-%m-%d"),
        "hour": hour.astype(int),
        "day_of_week": timestamps.day_name().to_numpy(),
        "is_weekend": is_weekend.astype(int),
        "month": months.astype(int),
        "season": np.select(
            [np.isin(months, [12,1,2]), np.isin(months, [3,4,5]), np.isin(months, [6,7,8,9]), np.isin(months, [10,11])],
            ["Cool/Dry", "Hot/Pre-monsoon", "Southwest monsoon", "Northeast monsoon"],
            default="Transition"
        ),
        "street_id": s["street_id"],
        "street_name": s["street_name"],
        "zone_type": zone,
        "road_class": s["road_class"],
        "lamp_technology": s["lamp_technology"],
        "control_mode": s["control_mode"],
        "luminaire_count": count,
        "rated_watts_per_luminaire": rated,
        "traffic_count": traffic,
        "pedestrian_count": pedestrians,
        "weather_condition": weather,
        "temperature_c": np.round(temp + rng.normal(0, 0.18, n_hours), 1),
        "humidity_pct": np.round(np.clip(humidity + rng.normal(0, 1.4, n_hours), 20, 100), 1),
        "rainfall_mm": np.round(rainfall, 2),
        "cloud_cover_pct": np.round(cloud, 1),
        "visibility_km": np.round(visibility + rng.normal(0, 0.12, n_hours), 2),
        "ambient_light_lux": np.round(ambient_lux, 2),
        "lighting_required_flag": lighting_required.astype(int),
        "target_illuminance_lux": float(s["target_illuminance_lux"]),
        "commanded_dimming_pct": np.round(command, 1),
        "measured_dimming_pct": measured,
        "lamp_status": np.where(fault, "Fault", np.where(~lighting_required, "Off", np.where(measured >= 99.5, "On", "Dimmed"))),
        "energy_consumption_kwh": np.round(energy, 4),
        "full_output_reference_kwh": np.round(np.where(lighting_required, full_kw + 0.008, 0.002), 4),
        "_fault": fault,
        "_street_order": s_idx
    })
    all_rows.append(frame)

df = pd.concat(all_rows, ignore_index=True)
# Deterministic time-major order: each time step contains all street segments.
df.sort_values(["timestamp_local", "street_id"], inplace=True, kind="mergesort", ignore_index=True)
df.insert(0, "record_id", [f"SL{idx:07d}" for idx in range(1, len(df)+1)])

# Create row-level reporting anomalies and limited missing sensor values.
n = len(df)
df["sensor_anomaly_flag"] = 0
anomaly_lists = [[] for _ in range(n)]

# Isolated ambient light sensor spikes mostly during darkness.
dark_candidates = np.flatnonzero(df["lighting_required_flag"].to_numpy() == 1)
ambient_n = max(25, int(0.00035 * n))
ambient_idx = rng.choice(dark_candidates, size=min(ambient_n, len(dark_candidates)), replace=False)
df.loc[ambient_idx, "ambient_light_lux"] = np.round(rng.uniform(250, 1800, len(ambient_idx)), 2)
for ix in ambient_idx:
    anomaly_lists[ix].append("Ambient-light spike")

# Traffic counter spikes to support anomaly detection in count variables.
traffic_n = max(20, int(0.00022 * n))
traffic_idx = rng.choice(np.arange(n), size=traffic_n, replace=False)
traffic_vals = df.loc[traffic_idx, "traffic_count"].to_numpy()
multipliers = rng.uniform(5, 14, len(traffic_idx))
df.loc[traffic_idx, "traffic_count"] = np.minimum(2500, np.maximum(traffic_vals * multipliers, traffic_vals + 250)).astype(int)
for ix in traffic_idx:
    anomaly_lists[ix].append("Traffic-counter spike")

# A small number of energy meter spikes; affected records are explicitly flagged.
energy_n = max(15, int(0.00012 * n))
energy_idx = rng.choice(np.arange(n), size=energy_n, replace=False)
df.loc[energy_idx, "energy_consumption_kwh"] = np.round(df.loc[energy_idx, "energy_consumption_kwh"].to_numpy() * rng.uniform(1.8, 2.8, len(energy_idx)), 4)
for ix in energy_idx:
    anomaly_lists[ix].append("Energy-meter spike")

# Small, documented missingness in selected sensor features.
missing_specs = {
    "ambient_light_lux": 0.0012,
    "traffic_count": 0.0010,
    "humidity_pct": 0.0008,
    "visibility_km": 0.0008,
    "measured_dimming_pct": 0.0006,
}
missing_masks = {}
for col, frac in missing_specs.items():
    count_missing = int(frac * n)
    miss_idx = rng.choice(np.arange(n), size=count_missing, replace=False)
    missing_masks[col] = miss_idx
    df.loc[miss_idx, col] = np.nan
    for ix in miss_idx:
        anomaly_lists[ix].append(f"Missing {col}")

for ix, items in enumerate(anomaly_lists):
    if items:
        df.at[ix, "sensor_anomaly_flag"] = 1

df["sensor_anomaly_type"] = ["; ".join(items) if items else "None" for items in anomaly_lists]
df["data_quality_flag"] = np.where(df["sensor_anomaly_flag"].eq(1), "Sensor issue",
                                   np.where(df["lamp_status"].eq("Fault"), "Lighting fault", "OK"))

# Derive comparison metric. Faults and energy-meter spikes are excluded (not valid "savings").
df["energy_savings_vs_full_output_pct"] = (
    (df["full_output_reference_kwh"] - df["energy_consumption_kwh"])
    / df["full_output_reference_kwh"].replace(0, np.nan) * 100
)
invalid_comparison = df["lamp_status"].eq("Fault") | df["sensor_anomaly_type"].str.contains("Energy-meter spike")
df.loc[invalid_comparison, "energy_savings_vs_full_output_pct"] = np.nan
df["energy_savings_vs_full_output_pct"] = df["energy_savings_vs_full_output_pct"].round(2)

# Clean column order and remove internals.
columns = [
    "record_id", "timestamp_local", "date", "hour", "day_of_week", "is_weekend", "month", "season",
    "street_id", "street_name", "zone_type", "road_class", "lamp_technology", "control_mode",
    "luminaire_count", "rated_watts_per_luminaire", "traffic_count", "pedestrian_count", "weather_condition",
    "temperature_c", "humidity_pct", "rainfall_mm", "cloud_cover_pct", "visibility_km", "ambient_light_lux",
    "lighting_required_flag", "target_illuminance_lux", "commanded_dimming_pct", "measured_dimming_pct",
    "lamp_status", "energy_consumption_kwh", "full_output_reference_kwh", "energy_savings_vs_full_output_pct",
    "sensor_anomaly_flag", "sensor_anomaly_type", "data_quality_flag"
]
df = df[columns]

assert len(df) == 24 * 8760
assert df["record_id"].is_unique
assert not df.duplicated(["timestamp_local", "street_id"]).any()
assert df["energy_consumption_kwh"].notna().all()
assert df["commanded_dimming_pct"].between(0, 100).all()
assert df.loc[df["lamp_status"].eq("Off"), "commanded_dimming_pct"].eq(0).all()

csv_path = OUT / "smart_streetlighting_synthetic_dataset.csv"
df.to_csv(csv_path, index=False, na_rep="")

len(df), df.shape, csv_path.stat().st_size, df.head(3).to_dict("records")