"""
AIESEC Approvals Forecaster - Streamlit Web Application
Dedicated interactive web interface for forecasting approved exchange participants.
"""
import streamlit as st
import pandas as pd
import numpy as np

from src.data_service import load_historical_approved, get_historical_metrics
from src.model_service import load_or_fit_model, predict_approvals, check_baseline_parity
from src.plots import plot_forecast_projection, plot_seasonal_comparison

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AIESEC Approvals Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for AIESEC Brand Identity
st.markdown(
    """
    <style>
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.25rem;
        border-radius: 0.6rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        color: #5a6e85 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #0a2540 !important;
        font-weight: 700;
        font-size: 1.8rem;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .main-header {
        margin-bottom: 1.5rem;
    }
    .main-title {
        color: #0a2540;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #5a6e85;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2. Cached Resource & Data Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_or_fit_model(
        model_path="models/model.pkl",
        data_path="data/monthly_data.csv"
    )

@st.cache_data
def get_data():
    return load_historical_approved(data_path="data/monthly_data.csv")

try:
    hist_df = get_data()
    hist_metrics = get_historical_metrics(hist_df)
    model, model_status = get_model()
except Exception as e:
    st.error(f"Error initializing data or model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Sidebar: User Inputs for Seasonal Predictions
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌐 **AIESEC in Tanzania**")
    
    st.caption("Entity ID: `office_id = 567` | Metric: **`approved`**")
    st.markdown("---")

    st.subheader("🗓️ Seasonal Prediction Period")

    preset_options = [
        "Full Year 2026 (12 Months)",
        "Peak Summer Season Focus (May – August 2026)",
        "2-Year Cycle (2026 – 2027 / 24 Months)",
        "Extended Outlook (36 Months)",
        "Custom Horizon"
    ]
    selected_preset = st.selectbox("Select Forecast Horizon Preset", preset_options, index=0)

    if selected_preset == "Full Year 2026 (12 Months)":
        horizon = 12
    elif selected_preset == "Peak Summer Season Focus (May – August 2026)":
        horizon = 8
    elif selected_preset == "2-Year Cycle (2026 – 2027 / 24 Months)":
        horizon = 24
    elif selected_preset == "Extended Outlook (36 Months)":
        horizon = 36
    else:
        horizon = st.slider("Forecast Horizon (Months)", min_value=1, max_value=36, value=12, step=1)

    # Calculate horizon date range
    start_date = pd.Timestamp("2026-01-01")
    end_date = start_date + pd.DateOffset(months=horizon - 1)
    st.info(f"**Period:** {start_date.strftime('%b %Y')} → **{end_date.strftime('%b %Y')}** ({horizon} months)")

    st.markdown("---")
    st.subheader("🎯 Scenario & Peak Adjustments")
    st.markdown("<small style='color:#5a6e85;'>Simulate growth targets and intensified campaign surges.</small>", unsafe_allow_html=True)

    growth_pct = st.slider(
        "Overall Growth Adjustment (%)",
        min_value=-30,
        max_value=50,
        value=0,
        step=5,
        help="Applies a global scenario scaling multiplier across all forecasted months."
    )
    overall_multiplier = 1.0 + (growth_pct / 100.0)

    peak_multiplier = st.slider(
        "High-Season Peak Multiplier (May–July)",
        min_value=0.8,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Amplifies approvals during historical peak exchange months (May, June, July)."
    )

    st.markdown("---")
    st.subheader("⚙️ Display Preferences")
    show_historical = st.checkbox("Show Historical Baseline (2022–2025)", value=True)
    highlight_peaks = st.checkbox("Highlight Seasonal Peak Windows", value=True)
    use_rounded = st.checkbox("Round to Nearest Integer Counts", value=False)

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#8c9ba5;'>Model: Holt-Winters Additive<br>Seasonality: 12-Month Annual Cycle</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Generate Predictions & Compute Metrics
# -----------------------------------------------------------------------------
pred_df = predict_approvals(
    model=model,
    horizon=horizon,
    peak_multiplier=peak_multiplier,
    overall_multiplier=overall_multiplier,
    peak_months=[5, 6, 7]
)

# Valuation column based on rounding
val_col = "forecast_rounded" if use_rounded else "adjusted_forecast"
base_col = "baseline_rounded" if use_rounded else "baseline_forecast"

total_forecasted = pred_df[val_col].sum()
monthly_avg_forecast = pred_df[val_col].mean()

peak_idx = pred_df[val_col].idxmax()
peak_row = pred_df.loc[peak_idx]
peak_month_str = peak_row["month_label"]
peak_val = peak_row[val_col]

# 2025 Baseline comparison
# If horizon >= 12, compare first 12 months with full year 2025 actual total
first_12_months = pred_df.iloc[:min(12, len(pred_df))]
sum_12m = first_12_months[val_col].sum()
total_2025 = hist_metrics["total_2025"]
yoy_growth = ((sum_12m - total_2025) / total_2025 * 100.0) if total_2025 > 0 else 0.0

# -----------------------------------------------------------------------------
# 5. Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="main-header">
        <div class="main-title">📈 AIESEC Tanzania: Approvals Forecaster</div>
        <div class="main-subtitle">
            Interactive seasonal forecasting for <b>approved</b> exchange participants (signed contracts & paid fees) 
            powered by Nixtla's <code>StatsForecast</code> Holt-Winters model.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 6. Executive KPI Metrics Row
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Forecasted Approvals",
        value=f"{int(total_forecasted)}" if use_rounded else f"{total_forecasted:.1f}",
        delta=f"{growth_pct:+d}% vs Base" if growth_pct != 0 else None
    )

with col2:
    st.metric(
        label="Peak Projected Month",
        value=f"{peak_val:.0f} eps" if use_rounded else f"{peak_val:.1f} eps",
        delta=peak_month_str
    )

with col3:
    st.metric(
        label="Average Monthly Run-Rate",
        value=f"{monthly_avg_forecast:.1f} / mo",
        delta=f"2025 Avg: {hist_metrics['monthly_avg']:.1f}"
    )

with col4:
    st.metric(
        label="2026 vs. 2025 Baseline",
        value=f"{sum_12m:.1f} vs {total_2025:.0f}",
        delta=f"{yoy_growth:+.1f}% YoY"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. Main Time-Series Forecast Projection Visual
# -----------------------------------------------------------------------------
st.subheader("📊 Projected Predictions & Historical Trajectory")
fig_proj = plot_forecast_projection(
    hist_df=hist_df,
    pred_df=pred_df,
    show_historical=show_historical,
    highlight_peaks=highlight_peaks,
    use_rounded=use_rounded,
    show_scenario=True
)
st.plotly_chart(fig_proj, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. Seasonal Pattern Comparison Visual
# -----------------------------------------------------------------------------
st.subheader("🔄 Seasonal Cycle Overlay (Jan – Dec)")
st.markdown(
    "<p style='color:#5a6e85; font-size:0.95rem;'>"
    "Compare month-by-month seasonality across historical years (2022–2025) and verify alignment with "
    "projected peak exchange windows (summer surge in May–June and year-end cycle in December)."
    "</p>",
    unsafe_allow_html=True
)
fig_seasonal = plot_seasonal_comparison(
    hist_df=hist_df,
    pred_df=pred_df,
    use_rounded=use_rounded
)
st.plotly_chart(fig_seasonal, use_container_width=True)

# -----------------------------------------------------------------------------
# 9. Tabular Prediction Breakdown & Export
# -----------------------------------------------------------------------------
st.subheader("📋 Forecast Breakdown & Operational Export")

# Build clean export dataframe
export_df = pred_df[[
    "month_label",
    "ds",
    "baseline_forecast",
    "adjusted_forecast",
    "forecast_rounded",
    "seasonal_phase",
    "mom_change_pct"
]].copy()

export_df.columns = [
    "Month",
    "Date",
    "Baseline Forecast",
    "Adjusted Forecast",
    "Rounded Count",
    "Seasonal Phase",
    "MoM Change (%)"
]

export_df["Baseline Forecast"] = export_df["Baseline Forecast"].round(2)
export_df["Adjusted Forecast"] = export_df["Adjusted Forecast"].round(2)
export_df["MoM Change (%)"] = export_df["MoM Change (%)"].round(1)

col_table, col_export = st.columns([3, 1])

with col_table:
    st.dataframe(
        export_df[["Month", "Baseline Forecast", "Adjusted Forecast", "Rounded Count", "Seasonal Phase", "MoM Change (%)"]],
        use_container_width=True,
        hide_index=True
    )

with col_export:
    st.markdown("#### 📥 Export Data")
    st.markdown(
        "<p style='color:#5a6e85; font-size:0.9rem;'>"
        "Download this forecast configuration for entity goal-setting, budget planning, and campaign milestones."
        "</p>",
        unsafe_allow_html=True
    )

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"💾 Download CSV ({horizon} Months)",
        data=csv_data,
        file_name=f"aiesec_approved_forecast_{horizon}_months.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Parity Validation Check
    is_parity, parity_msg = check_baseline_parity(pred_df)
    if is_parity and growth_pct == 0 and peak_multiplier == 1.0:
        st.success(f"✓ {parity_msg}")
    elif growth_pct != 0 or peak_multiplier != 1.0:
        st.info("ℹ️ Custom scenario active: values differ from baseline reference by design.")

# -----------------------------------------------------------------------------
# 10. Model Details & Methodology Expander
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Model Details, Assumptions & Metric Definition"):
    st.markdown(
        """
        ### About the Model
        * **Framework:** Nixtla's `statsforecast`
        * **Model:** Additive Holt-Winters Exponential Smoothing (`HoltWinters_Additive`)
        * **Season Length:** 12 months (capturing mid-year summer peaks and year-end exchange cycles)
        * **Training Set:** Monthly historical `approved` series from **January 2022 to December 2025** (48 observations)
        * **Forecast Horizon:** Starts **2026-01-01**

        ### Operational Target Metric: `approved`
        * An applicant officially signs their exchange contract, pays the required exchange fees, and completes pre-departure screening.
        * **Why this metric matters:** `approved` is the primary revenue and conversion KPI for AIESEC. Realizations (`realized`) follow `approved` with a 1–3 month operational lag for visa processing and travel.
        """
    )
