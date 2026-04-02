"""
generate_assets.py — Génère les assets visuels pour le README.

  - assets/demo.mp4          : vidéo de démonstration (simulation complète)
  - assets/screenshot_viz.png : capture d'une frame du visualiseur 3D/2D
  - assets/screenshot_dash.png: capture statique du dashboard analytics
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D  # noqa

from simulator.scene import Scene
from simulator.agent import spawn_agents
from tracker.registry import build_tracker
from analytics.kpis import KPIEngine
from viz.realtime import RealtimeViz, TRACK_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_n_frames(n, tracker_name="sort", seed=42, n_agents=8, dt=0.1, noise=0.05):
    rng    = np.random.default_rng(seed)
    scene  = Scene.default_airport_hall()
    agents = spawn_agents(n_agents, scene, rng)
    tracker = build_tracker(tracker_name, dt=dt)
    kpi_eng = KPIEngine(scene, dt=dt)

    frames = []
    for i in range(n):
        pts = []
        for ag in agents:
            ag.step(scene, dt, rng)
            pts.append(ag.generate_point_cloud(rng, noise_std=noise))
        points = np.vstack(pts) if pts else np.empty((0,3))
        confirmed = tracker.update(points)
        kpis = kpi_eng.update(confirmed, i)
        frames.append((points, confirmed, kpis))

    return frames, scene, kpi_eng


# ---------------------------------------------------------------------------
# 1. Screenshot visualiseur 3D/2D (frame 120)
# ---------------------------------------------------------------------------

print("Génération screenshot visualiseur...")
frames, scene, kpi_eng = run_n_frames(130, n_agents=8)
points, tracks, kpis = frames[120]

viz = RealtimeViz(scene, trail_length=25)
# Replay les 120 premières frames pour remplir les trails
for pts, trks, kp in frames[:121]:
    viz.render_frame(pts, trks, kp)

viz.fig.savefig("assets/screenshot_viz.png", dpi=140, bbox_inches="tight",
                facecolor=viz.fig.get_facecolor())
plt.close(viz.fig)
print("  → assets/screenshot_viz.png")


# ---------------------------------------------------------------------------
# 2. Screenshot dashboard analytics (matplotlib statique)
# ---------------------------------------------------------------------------

print("Génération screenshot dashboard...")
import pandas as pd

rows = []
for pts, trks, kp in frames:
    rows.append({
        "frame": kp.frame_idx,
        "n_tracks": len(trks),
        "avg_speed": kp.avg_speed,
        **{f"count_{z}": kp.count_per_zone.get(z, 0) for z in kp.count_per_zone},
        **{f"flux_in_{d}": kp.flux_in[d] for d in kp.flux_in},
    })
df = pd.DataFrame(rows)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#0e1117")
fig.suptitle("LiDAR Spatial Tracker — Analytics Dashboard", color="white",
             fontsize=14, fontweight="bold", y=0.98)

for ax in axes.flat:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#aaa")
    for s in ax.spines.values():
        s.set_edgecolor("#333")

# Occupation par zone
count_cols = [c for c in df.columns if c.startswith("count_")]
colors_zone = ["#4fc3f7", "#f06292"]
for col, col_c in zip(count_cols, colors_zone):
    axes[0,0].plot(df["frame"], df[col], label=col.replace("count_",""), color=col_c, lw=1.8)
axes[0,0].set_title("Occupation par zone (personnes)", color="white", fontsize=10)
axes[0,0].set_xlabel("Frame", color="#aaa", fontsize=8)
axes[0,0].legend(fontsize=8, facecolor="#111", labelcolor="white")
axes[0,0].grid(alpha=0.15)

# Vitesse moyenne
axes[0,1].plot(df["frame"], df["avg_speed"], color="#69f0ae", lw=1.8)
axes[0,1].fill_between(df["frame"], df["avg_speed"], alpha=0.15, color="#69f0ae")
axes[0,1].set_title("Vitesse moyenne (m/s)", color="white", fontsize=10)
axes[0,1].set_xlabel("Frame", color="#aaa", fontsize=8)
axes[0,1].grid(alpha=0.15)

# Heatmap
hm = kpi_eng.get_heatmap_normalized()
im = axes[1,0].imshow(hm, cmap="hot", origin="lower", aspect="auto")
axes[1,0].set_title("Heatmap d'occupation", color="white", fontsize=10)
fig.colorbar(im, ax=axes[1,0], fraction=0.046)

# Flux aux portes
flux_cols = [c for c in df.columns if c.startswith("flux_in_")]
colors_flux = ["#ffd54f", "#ff8a65", "#ce93d8"]
for col, col_c in zip(flux_cols, colors_flux):
    axes[1,1].plot(df["frame"], df[col], label=col.replace("flux_in_",""), color=col_c, lw=1.8)
axes[1,1].set_title("Flux entrant cumulatif par porte", color="white", fontsize=10)
axes[1,1].set_xlabel("Frame", color="#aaa", fontsize=8)
axes[1,1].legend(fontsize=8, facecolor="#111", labelcolor="white")
axes[1,1].grid(alpha=0.15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("assets/screenshot_dash.png", dpi=140, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print("  → assets/screenshot_dash.png")


# ---------------------------------------------------------------------------
# 3. Vidéo de démo (MP4)
# ---------------------------------------------------------------------------

print("Génération vidéo demo.mp4 (200 frames)…")
frames_vid, scene_vid, _ = run_n_frames(200, n_agents=8, seed=7)
viz2 = RealtimeViz(scene_vid, trail_length=25)

from matplotlib.animation import FuncAnimation, FFMpegWriter

def _update(i):
    pts, trks, kp = frames_vid[i]
    viz2.render_frame(pts, trks, kp)
    return []

anim = FuncAnimation(viz2.fig, _update, frames=200, interval=100, blit=False)
writer = FFMpegWriter(fps=10, bitrate=1800)
anim.save("assets/demo.mp4", writer=writer, dpi=110)
plt.close(viz2.fig)
print("  → assets/demo.mp4")

print("\nDone — tous les assets sont dans assets/")
