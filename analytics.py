"""
Analytical computations, KPI derivations, time-series aggregations,
statistical modeling, and rule-based optimization simulations for LUX.
"""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np

# Physical luminaire power curve functions (Matching generator and notebook physics)
def calculate_circuit_kwh(dimming_pct: np.ndarray, lighting_req: np.ndarray, 
                          circuit_kw: np.ndarray, is_led: np.ndarray) -> np.ndarray:
    """
    Computes hourly kWh per circuit according to fixture technology dimming physics:
    - LED:  P_fraction = 0.025 + 0.975 * (dimming / 100)
    - HPS:  P_fraction = 0.180 + 0.820 * (dimming / 100)
    - Standby (Daylight): 0.002 kWh
    - Parasitic Active Loss: 0.008 kWh
    """
    dim = np.clip(dimming_pct, 0.0, 100.0)
    pf = np.where(is_led, 0.025 + 0.975 * (dim / 100.0), 0.180 + 0.820 * (dim / 100.0))
    active_load = circuit_kw * pf + 0.008
    return np.where(lighting_req == 1, active_load, 0.002)


def calculate_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates top-level executive KPIs and data telemetry."""
    total_records = len(df)
    if total_records == 0:
        return {}

    # All records totals
    total_energy_kwh = df['energy_consumption_kwh'].sum()
    full_output_kwh = df['full_output_reference_kwh'].sum()
    diff_kwh = full_output_kwh - total_energy_kwh
    savings_pct = (diff_kwh / full_output_kwh * 100) if full_output_kwh > 0 else 0.0

    # Valid population totals (excluding lighting faults & meter spikes)
    valid_mask = df['is_valid_energy']
    valid_df = df[valid_mask]
    valid_energy_kwh = valid_df['energy_consumption_kwh'].sum()
    valid_full_kwh = valid_df['full_output_reference_kwh'].sum()
    valid_diff_kwh = valid_full_kwh - valid_energy_kwh
    valid_savings_pct = (valid_diff_kwh / valid_full_kwh * 100) if valid_full_kwh > 0 else 0.0

    # Operational dimming
    night_records = df[df['lighting_required_flag'] == 1]
    avg_measured_dimming = night_records['measured_dimming_pct'].mean()
    avg_commanded_dimming = night_records['commanded_dimming_pct'].mean()

    # Data Quality & Anomaly counts
    sensor_anomalies = (df['sensor_anomaly_flag'] == 1).sum()
    lamp_faults = (df['lamp_status'] == 'Fault').sum()
    clean_records = (df['data_quality_flag'] == 'OK').sum()

    return {
        'total_records': total_records,
        'unique_streets': df['street_id'].nunique(),
        'date_min': df['date_dt'].min(),
        'date_max': df['date_dt'].max(),
        'total_energy_kwh': total_energy_kwh,
        'full_output_kwh': full_output_kwh,
        'diff_kwh': diff_kwh,
        'savings_pct': savings_pct,
        'valid_records': len(valid_df),
        'valid_energy_kwh': valid_energy_kwh,
        'valid_full_kwh': valid_full_kwh,
        'valid_diff_kwh': valid_diff_kwh,
        'valid_savings_pct': valid_savings_pct,
        'avg_measured_dimming': avg_measured_dimming,
        'avg_commanded_dimming': avg_commanded_dimming,
        'sensor_anomalies': sensor_anomalies,
        'sensor_anomaly_pct': (sensor_anomalies / total_records * 100),
        'lamp_faults': lamp_faults,
        'lamp_fault_pct': (lamp_faults / total_records * 100),
        'clean_records': clean_records,
        'clean_pct': (clean_records / total_records * 100),
    }


def get_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Computes mean and total energy, traffic, and dimming by hour of day (0–23)."""
    valid_df = df[df['is_valid_energy']]
    hourly = valid_df.groupby('hour').agg(
        mean_energy=('energy_consumption_kwh', 'mean'),
        total_energy=('energy_consumption_kwh', 'sum'),
        mean_full_output=('full_output_reference_kwh', 'mean'),
        total_full_output=('full_output_reference_kwh', 'sum'),
        mean_dimming=('measured_dimming_pct', 'mean'),
        mean_traffic=('traffic_count', 'mean'),
        mean_pedestrians=('pedestrian_count', 'mean')
    ).reset_index()
    return hourly


def get_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily total energy with 7-day rolling centered average."""
    valid_df = df[df['is_valid_energy']]
    daily = valid_df.groupby('date_dt').agg(
        total_energy=('energy_consumption_kwh', 'sum'),
        full_output=('full_output_reference_kwh', 'sum'),
        avg_dimming=('measured_dimming_pct', 'mean')
    ).reset_index()
    daily['date_dt'] = pd.to_datetime(daily['date_dt'])
    daily['rolling_7d'] = daily['total_energy'].rolling(7, min_periods=1, center=True).mean()
    return daily


def get_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes monthly energy metrics with daily normalization
    to prevent distortions from varying month lengths (e.g. Feb vs Jan).
    """
    valid_df = df[df['is_valid_energy']]
    month_days = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

    monthly = valid_df.groupby('month').agg(
        records=('record_id', 'count'),
        total_energy=('energy_consumption_kwh', 'sum'),
        full_output=('full_output_reference_kwh', 'sum'),
        avg_traffic=('traffic_count', 'mean'),
        lighting_req_pct=('lighting_required_flag', lambda x: x.mean() * 100)
    ).reset_index()

    monthly['month_name'] = monthly['month'].map(month_names)
    monthly['days_in_month'] = monthly['month'].map(month_days)
    monthly['energy_kwh_per_day'] = monthly['total_energy'] / monthly['days_in_month']
    monthly['full_output_kwh_per_day'] = monthly['full_output'] / monthly['days_in_month']
    monthly['savings_pct'] = (
        (monthly['full_output'] - monthly['total_energy']) / monthly['full_output'] * 100
    )
    return monthly


def get_zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes total, normalized per-luminaire, and savings metrics across zone types."""
    valid_df = df[df['is_valid_energy']]
    # Extract unique street definitions to get exact fixture counts
    street_meta = df.drop_duplicates('street_id').set_index('street_id')

    zone_agg = valid_df.groupby('zone_type').agg(
        street_count=('street_id', 'nunique'),
        total_energy=('energy_consumption_kwh', 'sum'),
        full_output=('full_output_reference_kwh', 'sum'),
        avg_traffic=('traffic_count', 'mean'),
        avg_pedestrians=('pedestrian_count', 'mean')
    ).reset_index()

    # Sum luminaires by zone
    lum_by_zone = street_meta.groupby('zone_type')['luminaire_count'].sum().to_dict()
    zone_agg['total_luminaires'] = zone_agg['zone_type'].map(lum_by_zone)
    zone_agg['kwh_per_luminaire'] = zone_agg['total_energy'] / zone_agg['total_luminaires']
    zone_agg['savings_pct'] = (
        (zone_agg['full_output'] - zone_agg['total_energy']) / zone_agg['full_output'] * 100
    )
    return zone_agg.sort_values('total_energy', ascending=False)


def get_street_ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    """Computes detailed rankings across all street segments in the dataset."""
    valid_df = df[df['is_valid_energy']]
    summary = valid_df.groupby(['street_id', 'street_name', 'zone_type', 'road_class', 
                                'lamp_technology', 'control_mode']).agg(
        luminaire_count=('luminaire_count', 'first'),
        rated_watts=('rated_watts_per_luminaire', 'first'),
        total_kwh=('energy_consumption_kwh', 'sum'),
        full_kwh=('full_output_reference_kwh', 'sum'),
        avg_traffic=('traffic_count', 'mean'),
        avg_pedestrians=('pedestrian_count', 'mean')
    ).reset_index()

    summary['circuit_kw'] = (summary['luminaire_count'] * summary['rated_watts']) / 1000.0
    summary['kwh_per_luminaire'] = summary['total_kwh'] / summary['luminaire_count']
    summary['savings_pct'] = (
        (summary['full_kwh'] - summary['total_kwh']) / summary['full_kwh'] * 100
    )
    return summary.sort_values('total_kwh', ascending=False).reset_index(drop=True)


def simulate_scenarios(df: pd.DataFrame, 
                       s1_offpeak_floor: float = 75.0,
                       s2_hwy_floor: float = 65.0,
                       s2_art_floor: float = 60.0,
                       s2_com_floor: float = 50.0,
                       s2_res_floor: float = 40.0,
                       s3_deep_floor: float = 30.0,
                       weather_boost: float = 20.0) -> Dict[str, Any]:
    """
    Executes rule-based energy optimization simulations.
    Reuses the exact physical power equations and assumptions from the validated notebook.
    """
    valid_df = df[df['is_valid_energy']].copy()
    if len(valid_df) == 0:
        return {}

    req_arr = valid_df['lighting_required_flag'].to_numpy()
    h_arr = valid_df['hour'].to_numpy()
    zone_arr = valid_df['zone_type'].to_numpy()
    mode_arr = valid_df['control_mode'].to_numpy()
    rain_arr = valid_df['rainfall_mm'].fillna(0).to_numpy()
    vis_arr = valid_df['visibility_km'].fillna(10.0).to_numpy()
    traf_arr = valid_df['traffic_count'].fillna(0).to_numpy()
    ped_arr = valid_df['pedestrian_count'].fillna(0).to_numpy()

    is_led = (valid_df['lamp_technology'] == 'LED').to_numpy()
    circuit_kw = valid_df['circuit_rated_kw'].to_numpy()
    base_dimming = valid_df['measured_dimming_pct'].fillna(valid_df['commanded_dimming_pct']).to_numpy()

    # Full Output Reference & Observed Baseline
    base_kwh_arr = valid_df['energy_consumption_kwh'].to_numpy()
    full_kwh_arr = valid_df['full_output_reference_kwh'].to_numpy()
    base_total = base_kwh_arr.sum()
    full_total = full_kwh_arr.sum()

    # -------------------------------------------------------------
    # Scenario 1: Conservative Schedule Tuning
    # Targets Fixed and Scheduled circuits during deep night (23:00 to 04:00)
    # -------------------------------------------------------------
    s1_dim = base_dimming.copy()
    is_deep_night = ((h_arr >= 23) | (h_arr <= 4)) & (req_arr == 1)
    bad_weather_s1 = (rain_arr > 0.4) | (vis_arr < 4.0)

    for i in range(len(valid_df)):
        if req_arr[i] == 1 and not bad_weather_s1[i] and is_deep_night[i]:
            z = zone_arr[i]
            floor = s1_offpeak_floor if z in ['Arterial', 'Highway'] else (s1_offpeak_floor - 10.0 if z in ['Commercial', 'Industrial'] else s1_offpeak_floor - 20.0)
            if mode_arr[i] in ['Fixed', 'Scheduled dimming']:
                s1_dim[i] = min(s1_dim[i], floor)

    s1_kwh = calculate_circuit_kwh(s1_dim, req_arr, circuit_kw, is_led)

    # -------------------------------------------------------------
    # Scenario 2: Demand-Responsive Adaptive Multi-Tier (Recommended)
    # Unified adaptive policy scaling with traffic, pedestrians, and weather
    # -------------------------------------------------------------
    base_floors = np.where(np.isin(zone_arr, ['Highway']), s2_hwy_floor,
                  np.where(np.isin(zone_arr, ['Arterial']), s2_art_floor,
                  np.where(np.isin(zone_arr, ['Commercial', 'Industrial']), s2_com_floor,
                  np.where(np.isin(zone_arr, ['Campus']), s2_com_floor - 5.0, s2_res_floor))))

    traf_capacity = np.where(np.isin(zone_arr, ['Highway']), 250.0,
                    np.where(np.isin(zone_arr, ['Arterial']), 200.0,
                    np.where(np.isin(zone_arr, ['Commercial']), 120.0,
                    np.where(np.isin(zone_arr, ['Industrial']), 100.0, 80.0))))

    traf_boost = np.clip(traf_arr / traf_capacity, 0.0, 1.0) * 30.0
    ped_boost = np.clip(ped_arr / 30.0, 0.0, 1.0) * 10.0
    weath_boost = np.where((rain_arr > 0.3) | (vis_arr < 3.5), weather_boost, 0.0)
    eve_boost = np.where((h_arr >= 18) & (h_arr <= 22), 5.0, 0.0)

    s2_calc = base_floors + traf_boost + ped_boost + weath_boost + eve_boost
    s2_dim = np.where(req_arr == 1, np.clip(s2_calc, base_floors, 100.0), 0.0)
    s2_kwh = calculate_circuit_kwh(s2_dim, req_arr, circuit_kw, is_led)

    # -------------------------------------------------------------
    # Scenario 3: Aggressive Deep-Night Eco-Profile (Stress-Test)
    # Deeper curtailment between 01:00 and 04:00 when traffic is near zero
    # -------------------------------------------------------------
    s3_dim = s2_dim.copy()
    eco_floors = np.where(np.isin(zone_arr, ['Highway', 'Arterial']), s3_deep_floor + 20.0,
                 np.where(np.isin(zone_arr, ['Commercial', 'Industrial']), s3_deep_floor + 10.0, s3_deep_floor))
    is_eco_hours = ((h_arr >= 1) & (h_arr <= 4)) & (req_arr == 1) & (~bad_weather_s1) & (traf_arr < 30) & (ped_arr < 5)
    s3_dim = np.where(is_eco_hours, np.minimum(s3_dim, eco_floors), s3_dim)
    s3_kwh = calculate_circuit_kwh(s3_dim, req_arr, circuit_kw, is_led)

    # Compile Scenario Comparison Summary
    scen_summary = [
        {
            'Scenario': 'Full-Output Reference Comparator',
            'Annual Energy (kWh)': full_total,
            'Savings vs Baseline (kWh)': -(full_total - base_total),
            'Savings vs Baseline (%)': -(full_total - base_total) / base_total * 100,
            'Total Savings vs Full Output (%)': 0.00,
            'Mean Active Dimming (%)': 100.0
        },
        {
            'Scenario': 'Observed Baseline Grid Telemetry',
            'Annual Energy (kWh)': base_total,
            'Savings vs Baseline (kWh)': 0.00,
            'Savings vs Baseline (%)': 0.00,
            'Total Savings vs Full Output (%)': (full_total - base_total) / full_total * 100,
            'Mean Active Dimming (%)': base_dimming[req_arr == 1].mean()
        },
        {
            'Scenario': 'Scenario 1: Conservative Schedule Tuning',
            'Annual Energy (kWh)': s1_kwh.sum(),
            'Savings vs Baseline (kWh)': base_total - s1_kwh.sum(),
            'Savings vs Baseline (%)': (base_total - s1_kwh.sum()) / base_total * 100,
            'Total Savings vs Full Output (%)': (full_total - s1_kwh.sum()) / full_total * 100,
            'Mean Active Dimming (%)': s1_dim[req_arr == 1].mean()
        },
        {
            'Scenario': 'Scenario 2: Demand-Responsive Adaptive (Recommended)',
            'Annual Energy (kWh)': s2_kwh.sum(),
            'Savings vs Baseline (kWh)': base_total - s2_kwh.sum(),
            'Savings vs Baseline (%)': (base_total - s2_kwh.sum()) / base_total * 100,
            'Total Savings vs Full Output (%)': (full_total - s2_kwh.sum()) / full_total * 100,
            'Mean Active Dimming (%)': s2_dim[req_arr == 1].mean()
        },
        {
            'Scenario': 'Scenario 3: Aggressive Deep-Night Eco-Profile',
            'Annual Energy (kWh)': s3_kwh.sum(),
            'Savings vs Baseline (kWh)': base_total - s3_kwh.sum(),
            'Savings vs Baseline (%)': (base_total - s3_kwh.sum()) / base_total * 100,
            'Total Savings vs Full Output (%)': (full_total - s3_kwh.sum()) / full_total * 100,
            'Mean Active Dimming (%)': s3_dim[req_arr == 1].mean()
        }
    ]

    # Hourly Profile Breakdown for Charts
    hourly_df = pd.DataFrame({
        'Hour': h_arr,
        'Baseline': base_kwh_arr,
        'Scenario 1': s1_kwh,
        'Scenario 2': s2_kwh,
        'Scenario 3': s3_kwh
    }).groupby('Hour').mean().reset_index()

    # Zone Breakdown for Charts
    zone_df = pd.DataFrame({
        'Zone': zone_arr,
        'Baseline': base_kwh_arr,
        'Scenario 1': s1_kwh,
        'Scenario 2': s2_kwh,
        'Scenario 3': s3_kwh
    }).groupby('Zone').sum().reset_index()

    return {
        'summary_table': pd.DataFrame(scen_summary),
        'hourly_df': hourly_df,
        'zone_df': zone_df,
        's1_kwh_total': s1_kwh.sum(),
        's2_kwh_total': s2_kwh.sum(),
        's3_kwh_total': s3_kwh.sum(),
        'base_total': base_total,
        'full_total': full_total
    }
