"""Streamlit dashboard for continual learning experiments."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Neuromorphic Synapse Memory", layout="wide")
st.title("Neuromorphic Synapse Memory — Continual Learning")

p = Path("viz/results.json")
if not p.exists():
    st.info("Run `python run.py` first to generate results.")
    st.stop()

data = json.loads(p.read_text())
methods = list(data.keys())

forgetting = {m: data[m]["forgetting"] for m in methods}
final_avg = {m: data[m]["final_avg"] for m in methods}

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(x=list(forgetting.keys()), y=list(forgetting.values()),
                 title="Average forgetting (lower is better)", color=list(forgetting.keys()))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.bar(x=list(final_avg.keys()), y=list(final_avg.values()),
                  title="Final average accuracy (higher is better)", color=list(final_avg.keys()))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Accuracy on Task 1 over time (forgetting)")
fig3 = go.Figure()
for m in methods:
    mat = data[m]["acc_matrix"]
    curve = [row[0] for row in mat]
    fig3.add_trace(go.Scatter(x=list(range(1, len(curve)+1)), y=curve, mode="lines+markers", name=m))
fig3.update_layout(xaxis_title="Task index", yaxis_title="Accuracy on Task 1", yaxis=dict(range=[0, 1.05]))
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Accuracy matrix (tasks x time)")
sel = st.selectbox("Method", methods)
mat = data[sel]["acc_matrix"]
fig4 = go.Figure(data=go.Heatmap(z=mat, x=[f"after T{i+1}" for i in range(len(mat))],
                                 y=[f"Task {i+1}" for i in range(len(mat[0]))],
                                 colorscale="RdYlGn", zmin=0, zmax=1))
fig4.update_layout(title=f"{sel} accuracy matrix")
st.plotly_chart(fig4, use_container_width=True)

df = pd.DataFrame([{"method": m, "forgetting": data[m]["forgetting"], "final_avg": data[m]["final_avg"]} for m in methods])
st.dataframe(df, use_container_width=True)
