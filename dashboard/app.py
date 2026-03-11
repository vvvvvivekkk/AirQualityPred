"""
Streamlit Dashboard for Air Quality Prediction System.
Interactive visualization of air quality predictions with charts and comparisons.
"""

import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import requests

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import AirQualityPredictor, AQI_COLORS
from src.data_preprocessing import get_aqi_category, AQI_BREAKPOINTS

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        color: #a0aec0;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .aqi-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(58, 123, 213, 0.3);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63);
    }

    div[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────

@st.cache_resource
def load_predictor():
    """Load the air quality predictor (cached)."""
    try:
        return AirQualityPredictor()
    except Exception as e:
        st.error(f"Failed to load predictor: {e}")
        return None


def get_predictions_from_api(endpoint: str, payload: dict):
    """Fetch predictions from the FastAPI service."""
    try:
        response = requests.post(f"http://localhost:8000{endpoint}", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()["predictions"]
    except requests.exceptions.ConnectionError:
        return None
    except Exception:
        return None
    return None


def get_predictions(predictor, mode: str, **kwargs):
    """Get predictions either from API or directly from predictor."""
    # Try API first
    if mode == "hour":
        api_result = get_predictions_from_api("/predict-hour", {"hours": kwargs.get("hours", 6)})
        if api_result:
            return api_result
        return predictor.predict_hours(kwargs.get("hours", 6)) if predictor else []

    elif mode == "day":
        api_result = get_predictions_from_api("/predict-day", {})
        if api_result:
            return api_result
        return predictor.predict_day() if predictor else []

    elif mode == "range":
        api_result = get_predictions_from_api("/predict-range", {
            "start_date": kwargs.get("start_date"),
            "end_date": kwargs.get("end_date"),
        })
        if api_result:
            return api_result
        return predictor.predict_range(kwargs["start_date"], kwargs["end_date"]) if predictor else []

    return []


def create_pm25_gauge(value: float):
    """Create a PM2.5 gauge chart."""
    category = get_aqi_category(value)
    color = AQI_COLORS.get(category, "#808080")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": "PM2.5 (µg/m³)", "font": {"size": 16, "color": "#a0aec0"}},
        number={"font": {"size": 40, "color": "white"}},
        gauge={
            "axis": {"range": [0, 300], "tickcolor": "#a0aec0"},
            "bar": {"color": color},
            "bgcolor": "#1a1a2e",
            "steps": [
                {"range": [0, 12], "color": "rgba(0, 228, 0, 0.15)"},
                {"range": [12, 35.4], "color": "rgba(255, 255, 0, 0.15)"},
                {"range": [35.4, 55.4], "color": "rgba(255, 126, 0, 0.15)"},
                {"range": [55.4, 150.4], "color": "rgba(255, 0, 0, 0.15)"},
                {"range": [150.4, 250.4], "color": "rgba(143, 63, 151, 0.15)"},
                {"range": [250.4, 300], "color": "rgba(126, 0, 35, 0.15)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=250,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def create_predictions_chart(predictions: list, title: str = "Air Quality Predictions"):
    """Create an interactive line chart of predictions."""
    if not predictions:
        return go.Figure()

    df = pd.DataFrame(predictions)
    df["datetime"] = pd.to_datetime(df["datetime"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Particulate Matter", "Gaseous Pollutants"),
        row_heights=[0.55, 0.45],
    )

    # PM2.5 and PM10
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["PM2.5"],
            name="PM2.5", mode="lines+markers",
            line=dict(color="#00d2ff", width=2.5),
            marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(0, 210, 255, 0.1)",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["PM10"],
            name="PM10", mode="lines+markers",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=3),
        ), row=1, col=1,
    )

    # AQI threshold lines
    for _, threshold, name in AQI_BREAKPOINTS[:4]:
        fig.add_hline(
            y=threshold, line_dash="dot", line_color="rgba(255,255,255,0.2)",
            annotation_text=name, annotation_font_color="rgba(255,255,255,0.4)",
            annotation_font_size=9,
            row=1, col=1,
        )

    # NO2, CO, SO2
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["NO2"],
            name="NO2", mode="lines",
            line=dict(color="#f59e0b", width=2),
        ), row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["CO"],
            name="CO", mode="lines",
            line=dict(color="#10b981", width=2),
        ), row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["datetime"], y=df["SO2"],
            name="SO2", mode="lines",
            line=dict(color="#ef4444", width=2),
        ), row=2, col=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 26, 46, 0.8)",
        font=dict(color="#a0aec0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            font=dict(color="white"),
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        height=550,
        margin=dict(l=50, r=30, t=80, b=50),
        hovermode="x unified",
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", showgrid=True)

    return fig


def create_aqi_distribution_chart(predictions: list):
    """Create AQI category distribution pie chart."""
    if not predictions:
        return go.Figure()

    df = pd.DataFrame(predictions)
    counts = df["aqi_category"].value_counts()

    colors = [AQI_COLORS.get(cat, "#808080") for cat in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors, line=dict(color="#1a1a2e", width=2)),
        textinfo="percent+label",
        textfont=dict(size=12, color="white"),
        hole=0.45,
    ))

    fig.update_layout(
        title=dict(text="AQI Category Distribution", font=dict(size=16, color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="white")),
    )
    return fig


def create_comparison_chart(historical: pd.DataFrame, predictions: list):
    """Create actual vs predicted comparison chart."""
    if historical is None or historical.empty or not predictions:
        return go.Figure()

    pred_df = pd.DataFrame(predictions)
    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])

    fig = go.Figure()

    # Historical (actual)
    if "datetime" in historical.columns:
        historical["datetime"] = pd.to_datetime(historical["datetime"])
        fig.add_trace(go.Scatter(
            x=historical["datetime"], y=historical["PM2.5"],
            name="Actual PM2.5", mode="lines",
            line=dict(color="#a78bfa", width=2),
            fill="tozeroy", fillcolor="rgba(167, 139, 250, 0.08)",
        ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=pred_df["datetime"], y=pred_df["PM2.5"],
        name="Predicted PM2.5", mode="lines+markers",
        line=dict(color="#00d2ff", width=2.5, dash="dot"),
        marker=dict(size=5),
    ))

    # Dividing line
    if "datetime" in historical.columns and len(historical) > 0:
        last_actual = historical["datetime"].iloc[-1]
        fig.add_vline(
            x=last_actual, line_dash="dash", line_color="rgba(255,255,255,0.4)",
            annotation_text="Now", annotation_font_color="white",
        )

    fig.update_layout(
        title=dict(text="Actual vs Predicted PM2.5", font=dict(size=18, color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 26, 46, 0.8)",
        font=dict(color="#a0aec0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)", font=dict(color="white"),
            orientation="h", yanchor="bottom", y=1.02,
        ),
        height=400,
        margin=dict(l=50, r=30, t=80, b=50),
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="PM2.5 (µg/m³)",
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    return fig


def create_heatmap(predictions: list):
    """Create hourly heatmap of PM2.5 predictions."""
    if not predictions or len(predictions) < 24:
        return go.Figure()

    df = pd.DataFrame(predictions)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date.astype(str)

    pivot = df.pivot_table(values="PM2.5", index="date", columns="hour", aggfunc="mean")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(pivot.shape[1])],
        y=pivot.index,
        colorscale=[
            [0, "#00E400"], [0.2, "#FFFF00"], [0.4, "#FF7E00"],
            [0.6, "#FF0000"], [0.8, "#8F3F97"], [1.0, "#7E0023"],
        ],
        colorbar=dict(title="PM2.5", tickfont=dict(color="white")),
        hovertemplate="Date: %{y}<br>Hour: %{x}<br>PM2.5: %{z:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="PM2.5 Hourly Heatmap", font=dict(size=16, color="white")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 26, 46, 0.8)",
        font=dict(color="#a0aec0"),
        height=300,
        margin=dict(l=80, r=30, t=60, b=50),
        xaxis_title="Hour of Day",
        yaxis_title="Date",
    )
    return fig


# ─── Main Dashboard ──────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Air Quality Prediction Dashboard</h1>
        <p>Real-time air quality forecasting powered by Temporal Fusion Transformer</p>
    </div>
    """, unsafe_allow_html=True)

    # Load predictor
    predictor = load_predictor()

    # ─── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        prediction_mode = st.selectbox(
            "📊 Prediction Mode",
            ["Next Few Hours", "Next Day (24h)", "Custom Range"],
            index=0,
        )

        if prediction_mode == "Next Few Hours":
            hours = st.slider("Hours to predict", 1, 72, 12)
        elif prediction_mode == "Custom Range":
            start_date = st.date_input("Start Date", datetime.now().date())
            end_date = st.date_input("End Date", (datetime.now() + timedelta(days=3)).date())

        st.markdown("---")
        st.markdown("### 📋 AQI Legend")
        for low, high, name in AQI_BREAKPOINTS:
            color = AQI_COLORS.get(name, "#808080")
            st.markdown(
                f'<div style="display:flex;align-items:center;margin:4px 0;">'
                f'<div style="width:14px;height:14px;border-radius:50%;background:{color};margin-right:8px;"></div>'
                f'<span style="font-size:0.85rem;">{name} ({low}-{high})</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.75rem;color:#718096;">Powered by TFT Model • v1.0</p>',
            unsafe_allow_html=True,
        )

    # ─── Generate Predictions ─────────────────────────────────
    with st.spinner("🔄 Generating predictions..."):
        if prediction_mode == "Next Few Hours":
            predictions = get_predictions(predictor, "hour", hours=hours)
        elif prediction_mode == "Next Day (24h)":
            predictions = get_predictions(predictor, "day")
        else:
            predictions = get_predictions(
                predictor, "range",
                start_date=str(start_date), end_date=str(end_date),
            )

    if not predictions:
        st.error("❌ No predictions available. Make sure data is generated and the model is trained.")
        st.info("Run: `python data/generate_sample_data.py` to generate sample data first.")
        return

    pred_df = pd.DataFrame(predictions)

    # ─── Key Metrics Row ──────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    avg_pm25 = pred_df["PM2.5"].mean()
    max_pm25 = pred_df["PM2.5"].max()
    avg_category = get_aqi_category(avg_pm25)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg PM2.5</div>
            <div class="metric-value" style="color: {AQI_COLORS.get(avg_category, '#fff')}">{avg_pm25:.1f}</div>
            <span class="aqi-badge" style="background:{AQI_COLORS.get(avg_category, '#808080')}; color:#000">
                {avg_category}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max PM2.5</div>
            <div class="metric-value" style="color: #ef4444">{max_pm25:.1f}</div>
            <span class="aqi-badge" style="background:{AQI_COLORS.get(get_aqi_category(max_pm25), '#808080')}; color:#000">
                {get_aqi_category(max_pm25)}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_no2 = pred_df["NO2"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg NO₂</div>
            <div class="metric-value" style="color: #f59e0b">{avg_no2:.1f}</div>
            <div class="metric-label">µg/m³</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_co = pred_df["CO"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg CO</div>
            <div class="metric-value" style="color: #10b981">{avg_co:.2f}</div>
            <div class="metric-label">mg/m³</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predictions</div>
            <div class="metric-value" style="color: #00d2ff">{len(predictions)}</div>
            <div class="metric-label">hours forecast</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Gauge + Distribution ─────────────────────────────────
    gauge_col, dist_col = st.columns([1, 1])

    with gauge_col:
        st.plotly_chart(create_pm25_gauge(avg_pm25), use_container_width=True)

    with dist_col:
        st.plotly_chart(create_aqi_distribution_chart(predictions), use_container_width=True)

    # ─── Main Prediction Chart ────────────────────────────────
    st.markdown('<div class="section-header">📈 Pollutant Forecast</div>', unsafe_allow_html=True)
    st.plotly_chart(
        create_predictions_chart(predictions, f"Air Quality Predictions — {prediction_mode}"),
        use_container_width=True,
    )

    # ─── Comparison Chart ─────────────────────────────────────
    st.markdown('<div class="section-header">🔄 Actual vs Predicted</div>', unsafe_allow_html=True)
    historical = predictor.get_historical_data(72) if predictor else pd.DataFrame()
    st.plotly_chart(create_comparison_chart(historical, predictions), use_container_width=True)

    # ─── Heatmap ──────────────────────────────────────────────
    if len(predictions) >= 24:
        st.markdown('<div class="section-header">🗺️ PM2.5 Hourly Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(create_heatmap(predictions), use_container_width=True)

    # ─── Data Table ───────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Prediction Details</div>', unsafe_allow_html=True)

    display_df = pred_df[["datetime", "PM2.5", "PM10", "NO2", "CO", "SO2", "aqi_category"]].copy()
    display_df.columns = ["DateTime", "PM2.5", "PM10", "NO₂", "CO", "SO₂", "AQI Category"]

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True,
    )

    # ─── Download ─────────────────────────────────────────────
    csv = display_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Predictions (CSV)",
        csv,
        "air_quality_predictions.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
