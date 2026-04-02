from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import TOX21_TASKS, HIGH_RISK, MEDIUM_RISK


                                                                                

def _risk_color(prob: float) -> str:
    if prob >= HIGH_RISK:
        return "#E24B4A"
    elif prob >= MEDIUM_RISK:
        return "#EF9F27"
    else:
        return "#1D9E75"


def _risk_label(prob: float) -> str:
    if prob >= HIGH_RISK:
        return "High"
    elif prob >= MEDIUM_RISK:
        return "Medium"
    else:
        return "Low"


                                                                                

def plot_toxicity_bars(probs: np.ndarray, title: str = "Tox21 Endpoint Predictions") -> go.Figure:
    """Horizontal bar chart for all 12 Tox21 endpoints."""
    colors = [_risk_color(p) for p in probs]
    labels = [_risk_label(p) for p in probs]

    fig = go.Figure(go.Bar(
        x=probs,
        y=TOX21_TASKS,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=HIGH_RISK,   line_dash="dash", line_color="#E24B4A", annotation_text="High risk")
    fig.add_vline(x=MEDIUM_RISK, line_dash="dash", line_color="#EF9F27", annotation_text="Medium risk")

    fig.update_layout(
        title=title,
        xaxis=dict(title="Toxicity Probability", range=[0, 1.15]),
        yaxis=dict(title=""),
        height=480,
        margin=dict(l=120, r=60, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_toxicity_radar(probs: np.ndarray) -> go.Figure:
    """Radar/spider chart of all 12 endpoints."""
    theta = TOX21_TASKS + [TOX21_TASKS[0]]
    r     = list(probs) + [probs[0]]

    fig = go.Figure(go.Scatterpolar(
        r=r, theta=theta,
        fill="toself",
        fillcolor="rgba(226,75,74,0.2)",
        line_color="#E24B4A",
        name="Toxicity",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[HIGH_RISK] * len(theta), theta=theta,
        mode="lines", line=dict(color="#E24B4A", dash="dash", width=1),
        name="High risk threshold",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=420,
        title="Toxicity Radar",
        showlegend=True,
    )
    return fig


                                                                                

def plot_model_comparison(
    gnn_probs: np.ndarray | None,
    xgb_probs: np.ndarray | None,
    ensemble: np.ndarray,
) -> go.Figure:
    """Grouped bar chart comparing GNN, XGBoost, and ensemble predictions."""
    fig = go.Figure()

    if gnn_probs is not None:
        fig.add_trace(go.Bar(
            name="GNN (GAT)", x=TOX21_TASKS, y=gnn_probs,
            marker_color="#534AB7", opacity=0.85,
        ))
    if xgb_probs is not None:
        fig.add_trace(go.Bar(
            name="XGBoost", x=TOX21_TASKS, y=xgb_probs,
            marker_color="#0F6E56", opacity=0.85,
        ))
    fig.add_trace(go.Bar(
        name="Ensemble", x=TOX21_TASKS, y=ensemble,
        marker_color="#E24B4A", opacity=0.95,
    ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison: GNN vs XGBoost vs Ensemble",
        yaxis=dict(title="Toxicity Probability", range=[0, 1.1]),
        xaxis=dict(tickangle=35),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


                                                                                

def plot_shap_values(shap_data: dict) -> go.Figure:
    """Horizontal waterfall-style SHAP bar chart."""
    names  = shap_data["feature_names"]
    values = shap_data["shap_values"]
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        title=f"SHAP Feature Importance — {shap_data['task']}",
        xaxis_title="SHAP value (impact on toxicity score)",
        height=400,
        margin=dict(l=130, r=20, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


                                                                                

def plot_counterfactual_comparison(
    original_probs: np.ndarray,
    counterfactuals: list,
    max_show: int = 3,
) -> go.Figure:
    """Side-by-side bar chart: original vs top counterfactuals per endpoint."""
    fig = go.Figure()

              
    fig.add_trace(go.Bar(
        name="Original",
        x=TOX21_TASKS,
        y=original_probs,
        marker_color="#E24B4A",
        opacity=0.9,
    ))

    palette = ["#534AB7", "#1D9E75", "#BA7517"]
    for i, cf in enumerate(counterfactuals[:max_show]):
        if cf.modified_probs is not None:
            fig.add_trace(go.Bar(
                name=cf.description[:30],
                x=TOX21_TASKS,
                y=cf.modified_probs,
                marker_color=palette[i % len(palette)],
                opacity=0.8,
            ))

    fig.update_layout(
        barmode="group",
        title="Original vs Safer Counterfactual Molecules",
        yaxis=dict(title="Toxicity Probability", range=[0, 1.1]),
        xaxis=dict(tickangle=35),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


                                                                                

def plot_risk_gauge(overall_risk: float) -> go.Figure:
    """Speedometer gauge for overall toxicity risk."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_risk * 100,
        number={"suffix": "%", "font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": _risk_color(overall_risk)},
            "steps": [
                {"range": [0, 40],  "color": "rgba(29,158,117,0.15)"},
                {"range": [40, 70], "color": "rgba(239,159,39,0.15)"},
                {"range": [70, 100],"color": "rgba(226,75,74,0.15)"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": overall_risk * 100,
            },
        },
        title={"text": "Overall Toxicity Risk"},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig


                                                                                

def plot_properties_comparison(props_orig: dict, props_mod: dict) -> go.Figure:
    """Side-by-side bar chart for key physicochemical properties."""
    numeric_keys = [
        k for k in props_orig
        if isinstance(props_orig[k], (int, float)) and k != "QED (Drug-likeness)"
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original",
        x=numeric_keys,
        y=[props_orig[k] for k in numeric_keys],
        marker_color="#E24B4A",
    ))
    fig.add_trace(go.Bar(
        name="Modified",
        x=numeric_keys,
        y=[props_mod.get(k, 0) for k in numeric_keys],
        marker_color="#534AB7",
    ))
    fig.update_layout(
        barmode="group",
        title="Physicochemical Properties: Original vs Modified",
        height=360,
        xaxis=dict(tickangle=25),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
