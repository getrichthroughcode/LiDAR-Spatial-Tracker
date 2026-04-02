"""
dashboard.py — Application Streamlit pour la visualisation des KPIs.

Lancement : streamlit run viz/dashboard.py

L'app charge les données de simulation depuis data/simulation_log.npy
(généré par main.py --save-data) et affiche :
  - Heatmap d'occupation
  - Courbes temporelles des KPIs
  - Flux entrées/sorties par porte
  - Statistiques de tracking
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------------

st.set_page_config(
    page_title="LiDAR Spatial Tracker",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔵 LiDAR Spatial Tracker — Dashboard Analytics")
st.caption("Simulation CPU-only · DBSCAN + Kalman + Hungarian · Outsight-stack")

# ------------------------------------------------------------------
# Sidebar : paramètres de simulation
# ------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Paramètres simulation")
    n_agents   = st.slider("Nombre d'agents", 2, 20, 8)
    n_frames   = st.slider("Frames", 50, 500, 200)
    dt         = st.slider("Pas de temps (s)", 0.05, 0.5, 0.1, 0.05)
    noise_std  = st.slider("Bruit capteur (m)", 0.01, 0.2, 0.05, 0.01)
    seed       = st.number_input("Seed aléatoire", value=42, step=1)
    run_btn    = st.button("▶ Lancer la simulation", type="primary")

# ------------------------------------------------------------------
# Simulation à la demande
# ------------------------------------------------------------------

@st.cache_data(show_spinner="Simulation en cours…")
def run_simulation(n_agents, n_frames, dt, noise_std, seed):
    """Lance la simulation et retourne les logs."""
    from simulator.scene import Scene
    from simulator.agent import spawn_agents
    from tracker.tracker import MultiObjectTracker
    from analytics.kpis import KPIEngine

    rng   = np.random.default_rng(seed)
    scene = Scene.default_airport_hall()
    agents = spawn_agents(n_agents, scene, rng)
    tracker = MultiObjectTracker(dt=dt)
    kpi_engine = KPIEngine(scene, dt=dt)

    logs = []
    for frame_idx in range(n_frames):
        # Step agents
        all_points = []
        for agent in agents:
            agent.step(scene, dt, rng)
            pts = agent.generate_point_cloud(rng, noise_std=noise_std)
            all_points.append(pts)

        points = np.vstack(all_points) if all_points else np.empty((0, 3))

        # Track
        confirmed_tracks = tracker.update(points)

        # KPIs
        kpis = kpi_engine.update(confirmed_tracks, frame_idx)
        logs.append({
            "frame": frame_idx,
            "n_tracks": len(confirmed_tracks),
            "avg_speed": kpis.avg_speed,
            **{f"count_{z}": kpis.count_per_zone.get(z, 0)
               for z in kpis.count_per_zone},
            **{f"flux_in_{d}": kpis.flux_in.get(d, 0)
               for d in kpis.flux_in},
            **{f"flux_out_{d}": kpis.flux_out.get(d, 0)
               for d in kpis.flux_out},
            **{f"passage_{z}": kpis.avg_passage_time.get(z, 0.0)
               for z in kpis.avg_passage_time},
        })

    heatmap = kpi_engine.get_heatmap_normalized()
    return pd.DataFrame(logs), heatmap, scene

# ------------------------------------------------------------------
# Affichage
# ------------------------------------------------------------------

if run_btn or "sim_df" not in st.session_state:
    if run_btn:
        st.session_state.pop("sim_df", None)
        st.session_state.pop("sim_heatmap", None)
        st.session_state.pop("sim_scene", None)

    df, heatmap, scene = run_simulation(n_agents, n_frames, dt, noise_std, seed)
    st.session_state["sim_df"] = df
    st.session_state["sim_heatmap"] = heatmap
    st.session_state["sim_scene"] = scene

df: pd.DataFrame = st.session_state.get("sim_df", pd.DataFrame())
heatmap: np.ndarray = st.session_state.get("sim_heatmap", np.zeros((1, 1)))
scene = st.session_state.get("sim_scene", None)

if df.empty:
    st.info("Lance une simulation via le panneau latéral.")
    st.stop()

# ------------------------------------------------------------------
# Métriques globales
# ------------------------------------------------------------------

st.subheader("📊 Métriques globales")
col1, col2, col3, col4 = st.columns(4)

count_cols = [c for c in df.columns if c.startswith("count_")]
flux_in_cols  = [c for c in df.columns if c.startswith("flux_in_")]
flux_out_cols = [c for c in df.columns if c.startswith("flux_out_")]

col1.metric("Tracks max simultanés", int(df["n_tracks"].max()))
col2.metric("Vitesse moy (m/s)", f"{df['avg_speed'].mean():.2f}")
if flux_in_cols:
    total_in = int(df[flux_in_cols].iloc[-1].sum())
    col3.metric("Total entrées (cumulatif)", total_in)
if flux_out_cols:
    total_out = int(df[flux_out_cols].iloc[-1].sum())
    col4.metric("Total sorties (cumulatif)", total_out)

st.divider()

# ------------------------------------------------------------------
# Graphiques KPIs
# ------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("👥 Occupation par zone (tracks/frame)")
    if count_cols:
        fig = px.line(
            df, x="frame", y=count_cols,
            labels={"value": "Nombre de personnes", "variable": "Zone"},
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.2), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏃 Vitesse moyenne (m/s)")
    fig_speed = px.line(
        df, x="frame", y="avg_speed",
        template="plotly_dark",
        color_discrete_sequence=["#00e5ff"],
        labels={"avg_speed": "Vitesse moy (m/s)"},
    )
    fig_speed.update_layout(height=250)
    st.plotly_chart(fig_speed, use_container_width=True)

with col_right:
    st.subheader("🔥 Heatmap d'occupation")
    fig_hm = px.imshow(
        heatmap,
        color_continuous_scale="Hot",
        labels={"color": "Densité norm."},
        template="plotly_dark",
        origin="lower",
        aspect="auto",
    )
    fig_hm.update_layout(height=300, coloraxis_showscale=True)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("🚪 Flux entrant/sortant (cumulatif)")
    if flux_in_cols or flux_out_cols:
        fig_flux = go.Figure()
        for col in flux_in_cols:
            door_name = col.replace("flux_in_", "")
            fig_flux.add_trace(go.Scatter(
                x=df["frame"], y=df[col],
                name=f"IN {door_name}", mode="lines",
                line=dict(dash="solid"),
            ))
        for col in flux_out_cols:
            door_name = col.replace("flux_out_", "")
            fig_flux.add_trace(go.Scatter(
                x=df["frame"], y=df[col],
                name=f"OUT {door_name}", mode="lines",
                line=dict(dash="dash"),
            ))
        fig_flux.update_layout(
            template="plotly_dark", height=250,
            legend=dict(orientation="h", y=-0.3),
            xaxis_title="Frame", yaxis_title="Comptage cumulatif",
        )
        st.plotly_chart(fig_flux, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Temps de passage moyen
# ------------------------------------------------------------------

st.subheader("⏱ Temps de passage moyen par zone (glissant)")
passage_cols = [c for c in df.columns if c.startswith("passage_")]
if passage_cols:
    fig_pt = px.line(
        df, x="frame", y=passage_cols,
        template="plotly_dark",
        labels={"value": "Durée (s)", "variable": "Zone"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_pt.update_layout(height=280, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_pt, use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Données brutes
# ------------------------------------------------------------------

with st.expander("🗂 Données brutes (DataFrame)"):
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Télécharger CSV", csv, "kpis_log.csv", "text/csv")
