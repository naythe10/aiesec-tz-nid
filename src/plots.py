"""
Plotly visualization builders for AIESEC approvals forecasting and seasonality analysis.
"""
from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def plot_forecast_projection(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    show_historical: bool = True,
    highlight_peaks: bool = True,
    use_rounded: bool = False,
    show_scenario: bool = True
) -> go.Figure:
    """
    Builds the primary time-series interactive chart connecting historical
    data (2022-2025) with projected predictions (2026+).
    """
    fig = go.Figure()

    # Column selection based on rounded setting
    val_col_base = "baseline_rounded" if use_rounded else "baseline_forecast"
    val_col_adj = "forecast_rounded" if use_rounded else "adjusted_forecast"

    # 1. Historical Baseline Line
    if show_historical and not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["ds"],
            y=hist_df["approved"],
            mode="lines+markers",
            name="Historical Actuals (2022–2025)",
            line=dict(color="#0a2540", width=2.5),
            marker=dict(size=5, color="#0a2540"),
            hovertemplate=(
                "<b>%{x|%B %Y}</b><br>"
                "Type: Actual Approvals<br>"
                "Count: %{y}<br>"
                "<extra></extra>"
            )
        ))

    # 2. Smooth Connection from Last Historical Point to Forecast Start
    if not hist_df.empty and not pred_df.empty:
        last_hist_date = hist_df["ds"].iloc[-1]
        last_hist_val = hist_df["approved"].iloc[-1]

        bridge_dates = [last_hist_date] + list(pred_df["ds"])
        bridge_base_vals = [last_hist_val] + list(pred_df[val_col_base])
        bridge_adj_vals = [last_hist_val] + list(pred_df[val_col_adj])
    else:
        bridge_dates = list(pred_df["ds"])
        bridge_base_vals = list(pred_df[val_col_base])
        bridge_adj_vals = list(pred_df[val_col_adj])

    # 3. Baseline Model Forecast Trace
    fig.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_base_vals,
        mode="lines+markers",
        name="Baseline Model Forecast (Holt-Winters)",
        line=dict(color="#037ef3", width=2.5, dash="dash"),
        marker=dict(size=6, color="#037ef3", symbol="circle"),
        hovertemplate=(
            "<b>%{x|%B %Y}</b><br>"
            "Type: Baseline Forecast<br>"
            "Approvals: %{y:.1f}<br>"
            "<extra></extra>"
        )
    ))

    # 4. Adjusted Scenario Forecast Trace (if user modified scenario or enabled)
    scenario_differs = not np.allclose(pred_df[val_col_base], pred_df[val_col_adj], atol=1e-2)
    if show_scenario and scenario_differs:
        fig.add_trace(go.Scatter(
            x=bridge_dates,
            y=bridge_adj_vals,
            mode="lines+markers",
            name="Adjusted Scenario Forecast",
            line=dict(color="#f85a40", width=2.8, dash="dot"),
            marker=dict(size=7, color="#f85a40", symbol="diamond"),
            hovertemplate=(
                "<b>%{x|%B %Y}</b><br>"
                "Type: Adjusted Scenario<br>"
                "Approvals: %{y:.1f}<br>"
                "<extra></extra>"
            )
        ))

    # 5. Seasonal Shading (High Summer Season: May 1 to July 31)
    if highlight_peaks:
        # Determine all years covered by the chart
        all_dates = list(hist_df["ds"]) + list(pred_df["ds"])
        min_year = min(all_dates).year
        max_year = max(all_dates).year

        for yr in range(min_year, max_year + 1):
            # Summer Peak: May 1 - July 31
            fig.add_vrect(
                x0=f"{yr}-05-01",
                x1=f"{yr}-07-31",
                fillcolor="#ffc845",
                opacity=0.12,
                layer="below",
                line_width=0,
                annotation_text="Summer Peak" if yr == 2026 else None,
                annotation_position="top left",
                annotation_font=dict(size=10, color="#8b6b00")
            )
            # Winter Cycle: Dec 1 - Jan 31
            fig.add_vrect(
                x0=f"{yr}-12-01",
                x1=f"{yr+1}-01-31",
                fillcolor="#037ef3",
                opacity=0.07,
                layer="below",
                line_width=0,
            )

    # Layout styling
    fig.update_layout(
        title=dict(
            text="<b>Monthly Exchange Approvals: Historical Baseline & Projected Horizon</b>",
            font=dict(size=18, color="#0a2540")
        ),
        xaxis=dict(
            title="<b>Month</b>",
            type="date",
            showgrid=True,
            gridcolor="#eef2f6",
            dtick="M3",
            tickformat="%b\n%Y"
        ),
        yaxis=dict(
            title="<b>Approved Participants</b>",
            showgrid=True,
            gridcolor="#eef2f6",
            rangemode="tozero"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=520
    )

    return fig


def plot_seasonal_comparison(
    hist_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    use_rounded: bool = False
) -> go.Figure:
    """
    Creates a 12-month cyclic seasonality chart (Jan to Dec) overlaying
    historical years (2022-2025) with projected forecast year(s).
    """
    fig = go.Figure()

    months_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    val_col_adj = "forecast_rounded" if use_rounded else "adjusted_forecast"

    # Historical years palette (subtle slates and grays)
    hist_colors = {
        2022: "#94a3b8",
        2023: "#64748b",
        2024: "#475569",
        2025: "#1e293b",
    }

    # Add historical curves
    for yr in sorted(hist_df["year"].unique()):
        sub_df = hist_df[hist_df["year"] == yr].sort_values("month_num")
        fig.add_trace(go.Scatter(
            x=sub_df["month_name"],
            y=sub_df["approved"],
            mode="lines+markers",
            name=f"Actual {yr}",
            line=dict(color=hist_colors.get(yr, "#94a3b8"), width=1.8),
            marker=dict(size=4),
            opacity=0.7,
            hovertemplate=f"<b>{yr} - %{{x}}</b>: %{{y}} approvals<extra></extra>"
        ))

    # Add forecast years
    pred_years = sorted(pred_df["year"].unique())
    pred_palette = ["#037ef3", "#00c16e", "#f85a40"]

    for i, yr in enumerate(pred_years):
        sub_pred = pred_df[pred_df["year"] == yr].sort_values("month_num")
        color = pred_palette[i % len(pred_palette)]

        fig.add_trace(go.Scatter(
            x=sub_pred["ds"].dt.strftime("%b"),
            y=sub_pred[val_col_adj],
            mode="lines+markers",
            name=f"Forecast {yr} (Projected)",
            line=dict(color=color, width=3.2),
            marker=dict(size=8, symbol="diamond", color=color),
            hovertemplate=f"<b>Forecast {yr} - %{{x}}</b>: %{{y:.1f}} approvals<extra></extra>"
        ))

    # Highlight summer peak window (May-July)
    fig.add_vrect(
        x0=3.5,  # between Apr and May
        x1=6.5,  # between Jul and Aug
        fillcolor="#ffc845",
        opacity=0.15,
        layer="below",
        line_width=0,
        annotation_text="Peak Season (May–Jul)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#8b6b00")
    )

    fig.update_layout(
        title=dict(
            text="<b>Seasonal Cycle Overlay: Historical Years vs. Projected Seasonality (Jan–Dec)</b>",
            font=dict(size=18, color="#0a2540")
        ),
        xaxis=dict(
            title="<b>Calendar Month</b>",
            categoryorder="array",
            categoryarray=months_labels,
            showgrid=True,
            gridcolor="#eef2f6"
        ),
        yaxis=dict(
            title="<b>Approved Participants</b>",
            showgrid=True,
            gridcolor="#eef2f6",
            rangemode="tozero"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=480
    )

    return fig
