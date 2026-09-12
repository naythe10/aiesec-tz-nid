# AI Agent Instructions: Streamlit Approvals Forecasting Web Application

## 1. Objective
Build an interactive, streamlined **Streamlit web application** focused exclusively on **`approved`** exchange participant forecasting for AIESEC in Tanzania (`office_id = 567`).

The application will:
1. Load and utilize the pre-trained time-series model (`models/model.pkl`) fitted on historical monthly `approved` data.
2. Provide interactive user inputs for **seasonal prediction horizons** and **seasonal scenario adjustments** starting from January 2026.
3. Deliver rich, interactive **visualizations of projected predictions** alongside historical `approved` activity.
4. Export customized prediction tables and seasonal summaries as CSVs for operational planning and target-setting.

*(Note: The application does not track or analyze other funnel stages like `applied`, `matched`, `accepted`, `realized`, or `completed`. It is strictly dedicated to `approved` applicant forecasting and seasonality).*

---

## 2. Model & Data Assets

### 2.1 Target Metric & Scope
* **Target Metric (`y`):** Monthly count of **`approved`** candidates (signed contracts & paid exchange fees).
* **Historical Baseline:** Monthly `approved` counts from **January 2022 to December 2025** (48 observations in `data/monthly_data.csv`).
* **Forecast Horizon Start:** `2026-01-01` (immediately following the historical baseline).

### 2.2 Pre-Trained Model Specs (`models/model.pkl`)
* **Framework:** Nixtla's `statsforecast` (serialized with `joblib`).
* **Model Class:** `HoltWinters(season_length=12, error_type="A", alias="HoltWinters_Additive")`.
* **Season Length:** 12 months (annual seasonality capturing the mid-year summer surge and year-end cycle).
* **Output Post-Processing:** All forecasted values must be non-negative (`clip(lower=0.0)`).
* **Parity Reference:** 12-month baseline forecast (`h=12`) must match values in `data/pred_2026.csv`.

---

## 3. UI Architecture & Functional Requirements

### 3.1 Layout & Theme
* **Page Config:** Wide layout (`st.set_page_config(layout="wide", page_title="AIESEC Approvals Forecaster", page_icon="📈")`).
* **Theme Styling:** AIESEC brand identity (Primary Blue: `#037ef3`, Navy: `#0a2540`, Slate: `#5a6e85`, Light background).
* **Header / Banner:** Concise explanation of the model, the target metric (`approved`), and the entity context (AIESEC in Tanzania).

---

### 3.2 Sidebar: User Inputs for Seasonal Predictions

The sidebar provides dedicated controls to configure prediction periods, seasonal windows, and target simulations:

1. **Prediction Period & Horizon:**
   * **Preset Seasonal Buttons / Selectbox:**
     * *"Full Year 2026 (12 Months)"* (Default)
     * *"Peak Summer Season Focus (May – August 2026)"*
     * *"2-Year Cycle (2026 – 2027 / 24 Months)"*
     * *"Extended Outlook (36 Months)"*
   * **Custom Horizon Slider:** 1 to 36 months (dynamically shifts the end date).
   * **Start Date Indicator:** Display fixed forecast start (`2026-01-01`).

2. **Seasonal Scenario & Target Multipliers (Optional Simulation):**
   * **Overall Growth Multiplier:** Slider from `-30%` to `+50%` (baseline: 0% / 1.0x) to simulate conservative vs. aggressive growth scenarios.
   * **High-Season Peak Multiplier:** Dedicated slider allowing users to test higher seasonal surges during peak months (May–July) reflecting intensified marketing campaigns.

3. **Display Preferences:**
   * Toggle: Show/hide historical baseline (2022–2025).
   * Toggle: Highlight seasonal peak bands (May–July, Dec–Jan).
   * Toggle: Rounded counts (integers) vs. exact decimal estimates.

---

### 3.3 Main Dashboard: Visualizations & Projections

#### A. Executive KPI Metrics Row
Instant high-level summaries responsive to user inputs:
* **Total Forecasted Approvals:** Sum of predicted approvals over the selected horizon/season.
* **Peak Forecast Month & Volume:** Highest anticipated month (e.g., *June 2026: ~8 approvals*).
* **Average Monthly Run-Rate:** Mean approvals expected per month.
* **Comparison vs. 2025 Baseline:** Percentage growth vs. 2025 actual total (`48` approvals in 2025).

#### B. Main Time-Series Projection Visual (Plotly Interactive)
* **Historical Curve (2022–2025):** Solid line with data points showing actual historical monthly approvals.
* **Projected Forecast Curve (2026+):** Distinct highlighted dashed/colored line, smoothly connecting from the last actual observation (`2025-12-01`).
* **Seasonal Scenario Curve:** If user adjusts the seasonal multiplier, render a secondary dashed line showing the adjusted scenario vs. baseline model forecast.
* **Seasonal Shading:** Vertical shaded regions marking high-exchange summer seasons (May–July) and winter cycle periods.
* **Interactive Features:** Hover tooltips displaying exact month, type (Actual vs. Forecast vs. Adjusted), approval count, and seasonal period.

#### C. Seasonal Pattern Comparison Visual
* **Month-by-Month Seasonality Overlay (Plotly):**
  * A 12-month cyclic chart (Jan to Dec) overlaying historical year curves (2022, 2023, 2024, 2025) with the projected 2026 seasonal curve.
  * Clearly illustrates whether the projected season aligns with historical peak months (May–June peaks and late-year uptick).

#### D. Tabular Prediction Breakdown & Export
* **Detailed Forecast Table:**
  * Columns: `Month`, `Baseline Forecast`, `Adjusted Forecast` (if modified), `Seasonal Phase` (e.g., Peak / Off-Peak), `MoM Change (%)`.
* **Download Button (`st.download_button`):**
  * Export the active prediction table as a clean CSV (`aiesec_approved_forecast_[horizon]_months.csv`).

---

## 4. Technical Implementation Guidelines

### 4.1 Recommended Project Structure
```text
mc-activity/
├── app.py                     # Main Streamlit application entrypoint
├── src/
│   ├── __init__.py
│   ├── model_service.py       # Loading models/model.pkl, inference, fallback refitting
│   ├── data_service.py        # Loading data/monthly_data.csv (extracting 'approved' column)
│   └── plots.py               # Plotly figure builders for projections and seasonality
├── models/
│   └── model.pkl              # Pre-trained StatsForecast Holt-Winters model
├── data/
│   ├── monthly_data.csv       # Source historical data
│   └── pred_2026.csv          # 2026 reference predictions for unit test / validation
├── docs/
│   └── field_meanings.md
└── requirements.txt
```

### 4.2 Caching & Model Inference Pattern
1. **Caching Resources:**
   * Load `models/model.pkl` with `@st.cache_resource`.
   * Load historical `approved` series with `@st.cache_data`.
2. **Inference Function:**
   ```python
   import pandas as pd

   def predict_approvals(model, horizon: int, peak_multiplier: float = 1.0, overall_multiplier: float = 1.0) -> pd.DataFrame:
       # Generate baseline prediction from StatsForecast model
       pred_df = model.predict(h=horizon)
       
       # Extract forecast column
       pred_col = [col for col in pred_df.columns if col not in ['unique_id', 'ds']][0]
       
       # Clip non-negative
       pred_df['baseline_forecast'] = pred_df[pred_col].clip(lower=0.0)
       
       # Apply overall scenario multiplier
       pred_df['adjusted_forecast'] = pred_df['baseline_forecast'] * overall_multiplier
       
       # Apply peak season multiplier for May (month 5), June (month 6), July (month 7)
       months = pd.to_datetime(pred_df['ds']).dt.month
       is_peak = months.isin([5, 6, 7])
       pred_df.loc[is_peak, 'adjusted_forecast'] *= peak_multiplier
       
       pred_df['forecast_rounded'] = pred_df['adjusted_forecast'].round().astype(int)
       return pred_df
   ```
3. **Graceful Fallback:**
   * If unpickling fails due to environment differences, provide a fallback function in `model_service.py` to fit `StatsForecast(models=[HoltWinters(season_length=12, error_type="A", alias="HoltWinters_Additive")], freq="MS")` on the `approved` column of `data/monthly_data.csv`.

---

## 5. Verification & Acceptance Checklist
- [ ] **Model-Only Focus:** UI strictly focuses on `approved` data and `models/model.pkl`, omitting other funnel stages.
- [ ] **Seasonal User Inputs:** Sliders and presets for forecast horizon (1–36 months) and seasonal adjustments update all visuals immediately.
- [ ] **Seamless Plotly Visuals:** Visuals cleanly connect historical data (`2022-01-01` to `2025-12-01`) with the forecast curve starting `2026-01-01`.
- [ ] **Parity Check:** 12-month baseline forecast without adjustments matches `data/pred_2026.csv`.
- [ ] **CSV Export:** Generated CSV exports cleanly with non-negative, valid numbers.
