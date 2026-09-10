"""
src/charts.py  |  STEP 5 — Visualisation
-----------------------------------------
Turns numbers from Step 3 into visual charts for the UI.
Contains:
  - build_donut_chart()         : matched vs missing skills (circular chart)
  - build_score_gauge()         : overall match score as a speedometer
  - build_comparison_bar_chart(): side-by-side bar chart for multi-job comparison
"""

import plotly.graph_objects as go


def build_donut_chart(matched_count: int, missing_count: int) -> go.Figure:
    """Hollow donut chart showing matched vs missing skill counts."""
    labels = ["Matched", "Missing"]
    values = [matched_count, missing_count]
    colors = ["#22c55e", "#ef4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
        textinfo="label+value",
        textfont=dict(size=13, color="white"),
        hoverinfo="label+percent",
    )])

    total = matched_count + missing_count
    coverage_pct = int((matched_count / total) * 100) if total > 0 else 0

    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{coverage_pct}%</b><br><span style='font-size:11px'>coverage</span>",
            x=0.5, y=0.5, font=dict(size=22, color="white"), showarrow=False,
        )],
    )
    return fig


def build_score_gauge(score: float, label: str = "Match Score") -> go.Figure:
    """Gauge chart for overall match score."""
    pct = int(score * 100)

    if pct >= 70:
        color = "#22c55e"
    elif pct >= 50:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "white"}},
        title={"text": label, "font": {"size": 14, "color": "#9ca3af"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7280", "tickfont": {"color": "#9ca3af"}},
            "bar": {"color": color},
            "bgcolor": "#1f2937",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#1f2937"},
                {"range": [50, 70], "color": "#1f2937"},
                {"range": [70, 100], "color": "#1f2937"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def build_comparison_bar_chart(job_results: list[dict]) -> go.Figure:
    """
    Horizontal bar chart for multi-job comparison.
    Each dict in job_results should have: {"label": str, "final_score": float,
                                            "jaccard": float, "semantic": float}
    """
    labels = [r["label"] for r in job_results]
    finals = [int(r["final_score"] * 100) for r in job_results]
    keywords = [int(r["jaccard"] * 100) for r in job_results]
    semantics = [int(r["semantic"] * 100) for r in job_results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Overall Match",
        y=labels,
        x=finals,
        orientation="h",
        marker_color="#6366f1",
        text=[f"{v}%" for v in finals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Keyword Overlap",
        y=labels,
        x=keywords,
        orientation="h",
        marker_color="#22c55e",
        visible="legendonly",
    ))
    fig.add_trace(go.Bar(
        name="Semantic Similarity",
        y=labels,
        x=semantics,
        orientation="h",
        marker_color="#f59e0b",
        visible="legendonly",
    ))

    fig.update_layout(
        barmode="group",
        height=max(250, len(labels) * 60 + 100),
        margin=dict(t=20, b=20, l=10, r=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        xaxis=dict(range=[0, 115], showgrid=False, ticksuffix="%", color="#9ca3af"),
        yaxis=dict(color="#9ca3af"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    )
    return fig
