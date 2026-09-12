"""
Data service for loading and processing historical AIESEC monthly approved data.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd


def load_historical_approved(data_path: str = "data/monthly_data.csv") -> pd.DataFrame:
    """
    Loads monthly data and extracts the 'approved' series.
    Returns DataFrame with columns ['ds', 'approved', 'year', 'month_num', 'month_name'].
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(path)
    if "month" not in df.columns or "approved" not in df.columns:
        raise ValueError("Dataset must contain 'month' and 'approved' columns.")

    df["ds"] = pd.to_datetime(df["month"])
    df = df.sort_values("ds").reset_index(drop=True)

    df["approved"] = df["approved"].astype(float)
    df["year"] = df["ds"].dt.year
    df["month_num"] = df["ds"].dt.month
    df["month_name"] = df["ds"].dt.strftime("%b")

    return df[["ds", "approved", "year", "month_num", "month_name"]]


def prepare_statsforecast_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts historical approved data into Nixtla StatsForecast format:
    ['unique_id', 'ds', 'y']
    """
    sf_df = pd.DataFrame({
        "unique_id": 1,
        "ds": pd.to_datetime(df["ds"]),
        "y": df["approved"].astype(float)
    })
    return sf_df.sort_values("ds").reset_index(drop=True)


def get_historical_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary metrics from historical data.
    """
    total_approved = df["approved"].sum()
    monthly_avg = df["approved"].mean()
    
    # Peak month in history
    peak_row = df.loc[df["approved"].idxmax()]
    
    # 2025 total
    df_2025 = df[df["year"] == 2025]
    total_2025 = df_2025["approved"].sum() if not df_2025.empty else 0.0

    # 2024 total
    df_2024 = df[df["year"] == 2024]
    total_2024 = df_2024["approved"].sum() if not df_2024.empty else 0.0

    return {
        "total_approved": total_approved,
        "monthly_avg": monthly_avg,
        "peak_month": peak_row["ds"].strftime("%B %Y"),
        "peak_value": peak_row["approved"],
        "total_2025": total_2025,
        "total_2024": total_2024,
        "last_date": df["ds"].max(),
        "start_date": df["ds"].min(),
        "total_months": len(df),
    }
