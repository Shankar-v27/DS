"""
LUX - Machine Learning Energy Prediction Module
Enterprise Random Forest Regression Surrogate for Municipal Streetlighting Telemetry.
"""

from typing import Dict, Any, Tuple, Optional
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

import components

# -----------------------------------------------------------------------------
# 1. Feature Specifications & Target Leakage Safeguards
# -----------------------------------------------------------------------------
# Strictly exclude direct mathematical duplicates or leakage:
# (e.g. full_output_reference_kwh, energy_savings_vs_full_output_pct, specific_load_ratio, etc.)
NUMERICAL_FEATURES = [
    'hour',
    'month',
    'is_weekend',
    'luminaire_count',
    'rated_watts_per_luminaire',
    'circuit_rated_kw',
    'traffic_count',
    'pedestrian_count',
    'ambient_light_lux',
    'temperature_c',
    'humidity_pct',
    'rainfall_mm',
    'visibility_km',
    'measured_dimming_pct'
]

CATEGORICAL_FEATURES = [
    'zone_type',
    'road_class',
    'lamp_technology',
    'control_mode',
    'weather_condition'
]

TARGET_FEATURE = 'energy_consumption_kwh'


# -----------------------------------------------------------------------------
# 2. Model Training & Evaluation Pipeline
# -----------------------------------------------------------------------------
def train_energy_model(df: pd.DataFrame, sample_size: int = 50000, random_state: int = 42) -> Dict[str, Any]:
    """
    Trains a production RandomForestRegressor on chronological historical data.
    - Chronological Train/Test split: Jan–Oct (Train), Nov–Dec (Test holdout)
    - Leakage-free preprocessing with ColumnTransformer, SimpleImputer, and OneHotEncoder
    - Returns trained pipeline, holdout metrics, feature importances, and sample predictions.
    """
    valid_df = df[df['is_valid_energy']].copy()
    if len(valid_df) == 0:
        raise ValueError("No valid records available for model training.")

    # Chronological Split: Train on months 1–10 (Jan–Oct), Test on months 11–12 (Nov–Dec)
    train_mask = valid_df['month'] <= 10
    test_mask = valid_df['month'] > 10

    df_train_full = valid_df[train_mask]
    df_test = valid_df[test_mask]

    # Subsample training data to ensure rapid training (< 3 sec) and low memory usage
    if len(df_train_full) > sample_size:
        df_train = df_train_full.sample(n=sample_size, random_state=random_state)
    else:
        df_train = df_train_full

    # Feature matrix & target vectors
    X_train = df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = df_train[TARGET_FEATURE].to_numpy()

    X_test = df_test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_test = df_test[TARGET_FEATURE].to_numpy()

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )

    # Estimator: Fast, regularized Random Forest
    rf = RandomForestRegressor(
        n_estimators=60,
        max_depth=16,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', rf)
    ])

    # Fit Model
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    t_duration = time.time() - t_start

    # Evaluate on chronological holdout
    y_pred_test = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = root_mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # Extract Feature Importances
    prep = pipeline.named_steps['prep']
    raw_feature_names = prep.get_feature_names_out()
    clean_feature_names = [
        f.replace('num__', '').replace('cat__', '')
        for f in raw_feature_names
    ]
    importances = pipeline.named_steps['model'].feature_importances_

    fi_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    # Clean holdout test sample for plotting (first 120 hours of test set for line chart)
    sample_vis = df_test.head(120).copy()
    sample_vis['Actual_kWh'] = y_test[:120]
    sample_vis['Predicted_kWh'] = y_pred_test[:120]

    # Test scatter sample (random 1,000 points)
    scatter_idx = np.random.default_rng(random_state).choice(len(y_test), size=min(1200, len(y_test)), replace=False)
    scatter_df = pd.DataFrame({
        'Actual_kWh': y_test[scatter_idx],
        'Predicted_kWh': y_pred_test[scatter_idx],
        'Zone': df_test['zone_type'].iloc[scatter_idx].to_numpy()
    })

    return {
        'pipeline': pipeline,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'train_rows': len(df_train),
            'test_rows': len(df_test),
            'train_time_sec': t_duration
        },
        'feature_importance_df': fi_df,
        'sample_vis': sample_vis,
        'scatter_df': scatter_df,
        'model_name': 'Random Forest Regressor (Pipeline: Imputer + OHE + RF)'
    }


# -----------------------------------------------------------------------------
# 3. Interactive Single-Instance Inference
# -----------------------------------------------------------------------------
def predict_instance(model_dict: Dict[str, Any], feature_values: Dict[str, Any]) -> float:
    """Predicts hourly energy (kWh) for a single user-configured state."""
    pipeline = model_dict['pipeline']
    row_df = pd.DataFrame([feature_values])
    pred = pipeline.predict(row_df)[0]
    return max(0.001, float(pred))


# -----------------------------------------------------------------------------
# 4. Scenario Evaluation via ML Surrogate
# -----------------------------------------------------------------------------
def evaluate_scenarios_with_ml(model_dict: Dict[str, Any], valid_df: pd.DataFrame) -> Dict[str, float]:
    """
    Uses the trained ML model to independently project grid energy under
    Scenario 1, Scenario 2, and Scenario 3 dimming profiles.
    """
    pipeline = model_dict['pipeline']
    
    # Sample 12,000 observations across valid telemetry to evaluate ML inference on scenarios
    eval_sample = valid_df.sample(n=min(12000, len(valid_df)), random_state=42).reset_index(drop=True)
    
    # Baseline active dimming
    base_dim = eval_sample['measured_dimming_pct'].fillna(eval_sample['commanded_dimming_pct']).to_numpy()
    h_arr = eval_sample['hour'].to_numpy()
    req_arr = eval_sample['lighting_required_flag'].to_numpy()
    zone_arr = eval_sample['zone_type'].to_numpy()
    mode_arr = eval_sample['control_mode'].to_numpy()
    rain_arr = eval_sample['rainfall_mm'].fillna(0).to_numpy()
    vis_arr = eval_sample['visibility_km'].fillna(10.0).to_numpy()
    traf_arr = eval_sample['traffic_count'].fillna(0).to_numpy()
    ped_arr = eval_sample['pedestrian_count'].fillna(0).to_numpy()

    # Scenario 1 Dimming
    is_deep_night = ((h_arr >= 23) | (h_arr <= 4)) & (req_arr == 1)
    bad_weather_s1 = (rain_arr > 0.4) | (vis_arr < 4.0)
    s1_floors = np.where(np.isin(zone_arr, ['Arterial', 'Highway']), 75.0,
                np.where(np.isin(zone_arr, ['Commercial', 'Industrial']), 65.0, 55.0))
    s1_dim = np.where(is_deep_night & (~bad_weather_s1) & np.isin(mode_arr, ['Fixed', 'Scheduled dimming']),
                      np.minimum(base_dim, s1_floors), base_dim)

    # Scenario 2 Dimming
    base_floors = np.where(np.isin(zone_arr, ['Highway']), 65.0,
                  np.where(np.isin(zone_arr, ['Arterial']), 60.0,
                  np.where(np.isin(zone_arr, ['Commercial', 'Industrial']), 50.0,
                  np.where(np.isin(zone_arr, ['Campus']), 45.0, 40.0))))
    traf_cap = np.where(np.isin(zone_arr, ['Highway']), 250.0,
               np.where(np.isin(zone_arr, ['Arterial']), 200.0,
               np.where(np.isin(zone_arr, ['Commercial']), 120.0,
               np.where(np.isin(zone_arr, ['Industrial']), 100.0, 80.0))))
    traf_boost = np.clip(traf_arr / traf_cap, 0.0, 1.0) * 30.0
    ped_boost = np.clip(ped_arr / 30.0, 0.0, 1.0) * 10.0
    weath_boost = np.where((rain_arr > 0.3) | (vis_arr < 3.5), 20.0, 0.0)
    eve_boost = np.where((h_arr >= 18) & (h_arr <= 22), 5.0, 0.0)
    s2_dim = np.where(req_arr == 1, np.clip(base_floors + traf_boost + ped_boost + weath_boost + eve_boost, base_floors, 100.0), 0.0)

    # Scenario 3 Dimming
    eco_floors = np.where(np.isin(zone_arr, ['Highway', 'Arterial']), 50.0,
                 np.where(np.isin(zone_arr, ['Commercial', 'Industrial']), 40.0, 30.0))
    is_eco_hours = (h_arr >= 1) & (h_arr <= 4) & (req_arr == 1) & (~bad_weather_s1) & (traf_arr < 30) & (ped_arr < 5)
    s3_dim = np.where(is_eco_hours, np.minimum(s2_dim, eco_floors), s2_dim)

    # Predict Baseline
    df_eval_base = eval_sample[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    pred_base = pipeline.predict(df_eval_base)

    # Predict S1
    df_eval_s1 = df_eval_base.copy()
    df_eval_s1['measured_dimming_pct'] = s1_dim
    pred_s1 = pipeline.predict(df_eval_s1)

    # Predict S2
    df_eval_s2 = df_eval_base.copy()
    df_eval_s2['measured_dimming_pct'] = s2_dim
    pred_s2 = pipeline.predict(df_eval_s2)

    # Predict S3
    df_eval_s3 = df_eval_base.copy()
    df_eval_s3['measured_dimming_pct'] = s3_dim
    pred_s3 = pipeline.predict(df_eval_s3)

    scale_factor = len(valid_df) / len(eval_sample)
    return {
        'ml_base_kwh': float(pred_base.sum() * scale_factor),
        'ml_s1_kwh': float(pred_s1.sum() * scale_factor),
        'ml_s2_kwh': float(pred_s2.sum() * scale_factor),
        'ml_s3_kwh': float(pred_s3.sum() * scale_factor)
    }


# -----------------------------------------------------------------------------
# 5. Visualizations
# -----------------------------------------------------------------------------
def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Renders a styled horizontal bar chart of Random Forest feature importances."""
    top_fi = fi_df.head(top_n).sort_values('Importance', ascending=True)
    fig = px.bar(
        top_fi,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {top_n} Predictive Features (Random Forest Gini Importance)",
        color='Importance',
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Relative Feature Importance",
        yaxis_title=""
    )
    return components.style_plotly_chart(fig, height=350, show_legend=False)


def plot_actual_vs_predicted_scatter(scatter_df: pd.DataFrame) -> go.Figure:
    """Renders an actual vs. predicted regression scatter with unity line."""
    fig = px.scatter(
        scatter_df,
        x='Actual_kWh',
        y='Predicted_kWh',
        color='Zone',
        opacity=0.6,
        title="Actual vs. Predicted Circuit Load (Holdout Test Set)",
        labels={'Actual_kWh': 'Actual Measured Load (kWh)', 'Predicted_kWh': 'RF Predicted Load (kWh)'}
    )
    # Add 45-degree perfect prediction line
    max_val = max(scatter_df['Actual_kWh'].max(), scatter_df['Predicted_kWh'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', name='Ideal Fit (y = x)',
        line=dict(color='#F87171', width=1.5, dash='dash')
    ))
    return components.style_plotly_chart(fig, height=350)


def plot_prediction_timeline(sample_vis: pd.DataFrame) -> go.Figure:
    """Renders a 120-hour timeline comparing actual vs predicted load."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(sample_vis)),
        y=sample_vis['Actual_kWh'],
        mode='lines',
        name='Actual Energy (kWh)',
        line=dict(color='#38BDF8', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(sample_vis)),
        y=sample_vis['Predicted_kWh'],
        mode='lines',
        name='RF Predicted (kWh)',
        line=dict(color='#22D3EE', width=2, dash='dot')
    ))
    fig.update_layout(
        title="Chronological Holdout Telemetry: Actual vs. RF Prediction (120-Hour Window)",
        xaxis_title="Sequential Hourly Intervals (Nov 2025)",
        yaxis_title="Energy Consumption (kWh)"
    )
    return components.style_plotly_chart(fig, height=350)
