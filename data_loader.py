"""
Data loading, caching, feature engineering, and filtering pipeline
for LUX - Smart Streetlighting Intelligence Platform.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st

def find_dataset_file(filename: str) -> Path:
    """Locate project files dynamically across local and package directories."""
    candidates = [
        Path(filename),
        Path(".") / filename,
        Path("smart_streetlighting_project_dataset_package") / filename,
        Path("..") / filename,
        Path(__file__).parent / filename if "__file__" in globals() else None
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Critical Error: File '{filename}' was not found. Please verify that the dataset is present."
    )

@st.cache_data(show_spinner="Loading smart streetlighting dataset (210,240 records)...")
def load_dataset(custom_file=None) -> pd.DataFrame:
    """
    Loads, validates, and feature-engineers the 2025 synthetic streetlighting dataset.
    Cached with st.cache_data to guarantee lightning-fast page transitions.
    """
    if custom_file is not None:
        df = pd.read_csv(custom_file, parse_dates=['timestamp_local'], low_memory=False)
    else:
        path = find_dataset_file("smart_streetlighting_synthetic_dataset.csv")
        df = pd.read_csv(path, parse_dates=['timestamp_local'], low_memory=False)

    # 1. Electrical Engineering Metrics
    df['circuit_rated_kw'] = (df['luminaire_count'] * df['rated_watts_per_luminaire']) / 1000.0
    df['energy_kwh_per_luminaire'] = df['energy_consumption_kwh'] / df['luminaire_count']
    df['specific_load_ratio'] = df['energy_consumption_kwh'] / df['circuit_rated_kw']
    df['energy_diff_vs_full_kwh'] = df['full_output_reference_kwh'] - df['energy_consumption_kwh']

    # 2. Traffic Demand Classification
    def get_traffic_period(hour_val: int) -> str:
        if 7 <= hour_val <= 10:
            return 'Morning Peak'
        elif 17 <= hour_val <= 21:
            return 'Evening Peak'
        elif 0 <= hour_val <= 5:
            return 'Deep Off-Peak'
        else:
            return 'Shoulder'

    df['traffic_period'] = df['hour'].apply(get_traffic_period)
    df['traffic_period'] = pd.Categorical(
        df['traffic_period'],
        categories=['Morning Peak', 'Shoulder', 'Evening Peak', 'Deep Off-Peak'],
        ordered=True
    )

    # 3. Data Integrity & Valid Population Flags
    is_fault = df['lamp_status'] == 'Fault'
    is_meter_spike = df['sensor_anomaly_type'].str.contains('Energy-meter spike', na=False)
    df['is_valid_energy'] = ~(is_fault | is_meter_spike)
    df['is_night_operating'] = df['lighting_required_flag'].astype(bool)

    # 4. Extract pure date for quick filtering
    df['date_dt'] = df['timestamp_local'].dt.date

    return df

@st.cache_data
def load_data_dictionary() -> Optional[pd.DataFrame]:
    """Loads the dataset feature definitions table."""
    try:
        dict_path = find_dataset_file("smart_streetlighting_data_dictionary.csv")
        return pd.read_csv(dict_path)
    except Exception:
        return None

def get_filter_options(df: pd.DataFrame) -> Dict[str, Any]:
    """Extracts unique values and bounds for sidebar filters."""
    return {
        'min_date': df['date_dt'].min(),
        'max_date': df['date_dt'].max(),
        'street_ids': sorted(df['street_id'].unique().tolist()),
        'street_names': sorted(df['street_name'].unique().tolist()),
        'zone_types': sorted(df['zone_type'].unique().tolist()),
        'road_classes': sorted(df['road_class'].unique().tolist()),
        'lamp_technologies': sorted(df['lamp_technology'].unique().tolist()),
        'control_modes': sorted(df['control_mode'].unique().tolist()),
        'lamp_statuses': sorted(df['lamp_status'].unique().tolist()),
        'data_quality_flags': sorted(df['data_quality_flag'].unique().tolist()),
    }

def apply_global_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies user-selected sidebar filters to the DataFrame.
    Returns filtered DataFrame view without modifying the base cached dataset.
    """
    filtered = df

    # Date range filter
    if 'date_range' in filters and filters['date_range']:
        start_d, end_d = filters['date_range']
        filtered = filtered[(filtered['date_dt'] >= start_d) & (filtered['date_dt'] <= end_d)]

    # Categorical multi-select filters (only filter if non-empty selection)
    if filters.get('streets'):
        filtered = filtered[filtered['street_id'].isin(filters['streets'])]

    if filters.get('zones'):
        filtered = filtered[filtered['zone_type'].isin(filters['zones'])]

    if filters.get('road_classes'):
        filtered = filtered[filtered['road_class'].isin(filters['road_classes'])]

    if filters.get('technologies'):
        filtered = filtered[filtered['lamp_technology'].isin(filters['technologies'])]

    if filters.get('modes'):
        filtered = filtered[filtered['control_mode'].isin(filters['modes'])]

    if filters.get('lamp_statuses'):
        filtered = filtered[filtered['lamp_status'].isin(filters['lamp_statuses'])]

    if filters.get('quality_flags'):
        filtered = filtered[filtered['data_quality_flag'].isin(filters['quality_flags'])]

    return filtered
