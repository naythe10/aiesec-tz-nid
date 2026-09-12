"""
Model service for loading, fitting, and inferencing the Holt-Winters Additive model.
"""
from pathlib import Path
from typing import Tuple, List, Optional, Any
import pandas as pd
import numpy as np
import joblib

try:
    from statsforecast import StatsForecast
    from statsforecast.models import HoltWinters
    STATS_FORECAST_AVAILABLE = True
except ImportError:
    STATS_FORECAST_AVAILABLE = False

from src.data_service import load_historical_approved, prepare_statsforecast_df


def load_or_fit_model(
    model_path: str = "models/model.pkl",
    data_path: str = "data/monthly_data.csv"
) -> Tuple[Any, str]:
    """
    Attempts to load the serialized StatsForecast model from disk.
    If loading fails or environment mismatch occurs, refits on data_path as fallback.
    Returns (model, status_message).
    """
    path = Path(model_path)
    if path.exists():
        try:
            model = joblib.load(path)
            # Smoke test prediction
            _ = model.predict(h=1)
            return model, "Loaded pre-trained model from disk (models/model.pkl)"
        except Exception as e:
            # Fallback to refit if deserialization fails
            pass

    # Refit fallback
    if not STATS_FORECAST_AVAILABLE:
        raise RuntimeError(
            "statsforecast is required to fit the model. Please install statsforecast."
        )

    hist_df = load_historical_approved(data_path)
    train_df = prepare_statsforecast_df(hist_df)

    model = StatsForecast(
        models=[HoltWinters(season_length=12, error_type="A", alias="HoltWinters_Additive")],
        freq="MS",
        n_jobs=1
    )
    model.fit(train_df)
    return model, "Fitted Holt-Winters Additive model in-memory (fallback)"


def predict_approvals(
    model: Any,
    horizon: int,
    peak_multiplier: float = 1.0,
    overall_multiplier: float = 1.0,
    peak_months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Generates forecasts for the specified horizon starting from 2026-01-01.
    Applies non-negative clipping, overall multiplier, and peak season multipliers.
    """
    if peak_months is None:
        peak_months = [5, 6, 7]  # May, June, July summer peak

    # Generate raw forecast
    pred_df = model.predict(h=horizon).copy()

    # Identify the forecast column
    value_cols = [c for c in pred_df.columns if c not in ["unique_id", "ds"]]
    if not value_cols:
        raise ValueError(f"No prediction column found in model output: {pred_df.columns}")
    pred_col = value_cols[0]

    # Ensure datetime formatting
    pred_df["ds"] = pd.to_datetime(pred_df["ds"])
    pred_df = pred_df.sort_values("ds").reset_index(drop=True)

    # Post-processing: non-negative clip
    pred_df["baseline_forecast"] = pred_df[pred_col].clip(lower=0.0)
    pred_df["baseline_rounded"] = pred_df["baseline_forecast"].round().astype(int)

    # Scenario adjustments
    pred_df["adjusted_forecast"] = pred_df["baseline_forecast"] * overall_multiplier

    # High-season multiplier
    month_series = pred_df["ds"].dt.month
    is_peak = month_series.isin(peak_months)
    pred_df.loc[is_peak, "adjusted_forecast"] = (
        pred_df.loc[is_peak, "adjusted_forecast"] * peak_multiplier
    )

    pred_df["forecast_rounded"] = pred_df["adjusted_forecast"].round().astype(int)

    # Seasonal Phase Categorization
    def categorize_season(month: int) -> str:
        if month in [5, 6, 7]:
            return "Summer Peak (May–Jul)"
        elif month in [12, 1]:
            return "Winter Cycle (Dec–Jan)"
        elif month in [3, 4]:
            return "Spring Build-up (Mar–Apr)"
        else:
            return "Off-Peak / Delivery"

    pred_df["seasonal_phase"] = month_series.apply(categorize_season)

    # Month-over-Month change
    pred_df["mom_change"] = pred_df["adjusted_forecast"].diff()
    pred_df["mom_change_pct"] = (
        pred_df["adjusted_forecast"].pct_change() * 100.0
    ).fillna(0.0)

    # Date formatting for display
    pred_df["month_label"] = pred_df["ds"].dt.strftime("%b %Y")
    pred_df["year"] = pred_df["ds"].dt.year
    pred_df["month_num"] = month_series

    return pred_df


def check_baseline_parity(
    pred_df: pd.DataFrame,
    parity_path: str = "data/pred_2026.csv",
    tolerance: float = 1e-4
) -> Tuple[bool, str]:
    """
    Checks parity between the baseline 12-month prediction and data/pred_2026.csv.
    """
    path = Path(parity_path)
    if not path.exists():
        return False, f"Parity file not found at {parity_path}"

    ref_df = pd.read_csv(path)
    ref_df["ds"] = pd.to_datetime(ref_df["ds"])

    # Merge on ds
    merged = pd.merge(pred_df[["ds", "baseline_forecast"]], ref_df, on="ds", how="inner")
    if len(merged) < 12:
        return False, f"Matched only {len(merged)} of 12 reference months"

    diff = np.abs(merged["baseline_forecast"] - merged["HoltWinters_Additive"]).max()
    is_match = diff <= tolerance
    msg = (
        "Baseline 2026 forecast matches reference data/pred_2026.csv perfectly"
        if is_match else f"Discrepancy detected: max difference is {diff:.6f}"
    )
    return is_match, msg
