# MC Tanzania: 2026 High-Activity Prediction Platform

An interactive platform built to **predict and visualize high-activity exchange months for MC Tanzania in 2026**, powered by 4 years of historical EXPA data (2022–2025).

---

## The Assignment Objective

> *"Develop a platform to predict and visualize high activity months for MC Tanzania in 2026. Get all exchange data for MC Tanzania from 01-01-2022 to 31-12-2025 from EXPA and consultations, and using the data, prepare a model that can predict high activity months for the MC in 2026. Any other trends extracted from the data are a bonus."*

---

## Key Findings: High-Activity Months for 2026

The predictive model identifies two distinct high-activity surges for MC Tanzania in 2026:

1. **The Primary Peak (May – July 2026):**
   * **Highest Activity Month:** **June 2026** (~8 approvals), followed closely by **May 2026** (~7 approvals).
   * **Why:** Aligns with the peak global exchange window where university students finalize contracts for summer projects.
2. **The Secondary Peak (December 2026):**
   * **Second Surge:** **December 2026** (~4 approvals) driving winter exchange cycles.

---

## How We Solved It: The Pipeline vs. The Platform

| Stage | What Was Done | Outcome |
| :--- | :--- | :--- |
| **Phase 1: Data & Machine Learning Pipeline**<br>*(Notebooks 01–05)* | • Extracted 48 months of monthly exchange data (Jan 1, 2022 – Dec 31, 2025) from EXPA API (`office_id = 567`).<br>• Focused on **`approved`** participants (signed contracts & paid fees) as the core revenue/activity indicator.<br>• Trained a seasonal time-series model (Holt-Winters Additive) to capture annual seasonality. | • High-accuracy predictive brain saved to `models/model.pkl`.<br>• Clean historical baseline in `data/monthly_data.csv`. |
| **Phase 2: Interactive Web Platform**<br>*(Streamlit `app.py`)* | • Built a fast, visual dashboard to fulfill the platform requirement.<br>• Added interactive timeline charts showing 2026 projections connected with 2022–2025 history.<br>• Added scenario sliders to simulate marketing surges (+% targets).<br>• Added 1-click CSV target export. | • User-friendly platform for leadership and non-technical stakeholders to explore 2026 forecasts. |

---

## Bonus Trends Extracted from the Data

1. **The 1–3 Month Delivery Lag (`approved` → `realized`):**
   * Candidate approvals surge in **May & June**, while physical arrivals (`realized`) consistently peak **1 to 3 months later in July & August** due to visa and flight lead times.
2. **Top-of-Funnel Conversion Gate:**
   * Applications (`applied`) spike heavily in March–May, but the primary drop-off occurs before `matched`. Once an applicant reaches `accepted`, conversion to `approved` is significantly higher.

---

## Quick Start: Launch the Platform

Run these commands in your python or conda environment:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.
Or click the link on the about to open it live.

---

## Project Structure

* **`app.py`** — Streamlit web dashboard for interactive visualization & simulation.
* **`models/model.pkl`** — Pre-trained Holt-Winters predictive model.
* **`data/monthly_data.csv`** — Clean historical dataset from EXPA (2022–2025).
* **`notebooks/`** — Complete data pipeline:
  * `01-get-data.ipynb` (EXPA API extraction)
  * `02-wrangling.ipynb` (data structuring)
  * `03-eda.ipynb` (exploratory trend analysis)
  * `04-forecasting.ipynb` (time-series statistical analysis)
  * `05-prepare-model.ipynb` (model training, evaluation & export)
* **`docs/field_meanings.md`** — Funnel stage glossary and operational context.
