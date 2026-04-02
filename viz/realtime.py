"""
realtime.py — Visualisation temps réel et export vidéo MP4.

Deux modes :
  - show()   : animation Matplotlib interactive
  - export() : sauvegarde MP4 avec FFMpegWriter

Affichage :
  - Nuage de points 3D (scatter coloré par hauteur)
  - Bounding boxes 2D des tracks (vue de dessus)
  - Trajectoires des tracks (dernier N frames)
  - KPIs texte en overlay
"""

from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import List, Callable, Optional

from simulator.scene import Scene
from tracker.kalman import TrackKalman
from analytics.kpis import FrameKPIs


# Palette de couleurs pour les tracks (cycling)
TRACK_COLORS = plt.cm.tab20.colors


class RealtimeViz:
    """
    Visualiseur temps réel basé sur Matplotlib.

    Paramètres
    ----------
    scene         : scène pour afficher les zones et portes
    trail_length  : nombre de frames à afficher pour les trajectoires
    figsize       : taille de la figure
    """

    def __init__(
        self,
        scene: Scene,
        trail_length: int = 20,
        figsize: tuple = (14, 7),
    ) -> None:
        self.scene = scene
        self.trail_length = trail_length

        # Historique des positions par track
        self._trails: dict[int, list[np.ndarray]] = {}

        # Setup figure : vue 3D (gauche) + vue 2D + KPIs (droite)
        self.fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")
        self.ax3d = self.fig.add_subplot(121, projection="3d", facecolor="#16213e")
        self.ax2d = self.fig.add_subplot(122, facecolor="#16213e")

        self._setup_axes()

    def _setup_axes(self) -> None:
        """Configure les axes avec les limites de la scène."""
        sc = self.scene

        # Vue 3D
        self.ax3d.set_xlim(0, sc.width)
        self.ax3d.set_ylim(0, sc.height)
        self.ax3d.set_zlim(0, 2.2)
        self.ax3d.set_xlabel("X (m)", color="white", labelpad=4)
        self.ax3d.set_ylabel("Y (m)", color="white", labelpad=4)
        self.ax3d.set_zlabel("Z (m)", color="white", labelpad=4)
        self.ax3d.set_title("Vue 3D — nuage de points", color="white", fontsize=10)
        self.ax3d.tick_params(colors="gray")

        # Vue 2D (top-down)
        self.ax2d.set_xlim(0, sc.width)
        self.ax2d.set_ylim(0, sc.height)
        self.ax2d.set_xlabel("X (m)", color="white")
        self.ax2d.set_ylabel("Y (m)", color="white")
        self.ax2d.set_title("Vue 2D — tracks & KPIs", color="white", fontsize=10)
        self.ax2d.tick_params(colors="gray")
        for spine in self.ax2d.spines.values():
            spine.set_edgecolor("#444")

        # Dessiner les zones
        for zone in sc.zones:
            rect = patches.Rectangle(
                (zone.x_min, zone.y_min),
                zone.x_max - zone.x_min,
                zone.y_max - zone.y_min,
                linewidth=1, edgecolor="#4fc3f7", facecolor="none",
                linestyle="--", alpha=0.5,
            )
            self.ax2d.add_patch(rect)
            self.ax2d.text(
                (zone.x_min + zone.x_max) / 2, zone.y_min + 0.3,
                zone.name, color="#4fc3f7", fontsize=7, ha="center",
            )

        # Dessiner les portes
        for door in sc.doors:
            self.ax2d.plot(door.x, door.y, "D", color="#ffd54f",
                           markersize=8, zorder=5)
            self.ax2d.text(door.x, door.y + 0.5, door.name,
                           color="#ffd54f", fontsize=7, ha="center")

    # ------------------------------------------------------------------
    # Rendu d'une frame
    # ------------------------------------------------------------------

    def render_frame(
        self,
        points: np.ndarray,
        tracks: List[TrackKalman],
        kpis: Optional[FrameKPIs] = None,
    ) -> None:
        """
        Met à jour l'affichage pour une frame donnée.

        Paramètres
        ----------
        points : nuage de points (N, 3)
        tracks : tracks confirmés
        kpis   : KPIs de la frame (optionnel)
        """
        self.ax3d.cla()
        self.ax2d.cla()
        self._setup_axes()

        # --- Vue 3D : nuage de points ---
        if len(points) > 0:
            z_norm = (points[:, 2] - points[:, 2].min()) / (
                points[:, 2].max() - points[:, 2].min() + 1e-9
            )
            self.ax3d.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=z_norm, cmap="plasma", s=1.5, alpha=0.6,
            )

        # --- Vue 3D & 2D : tracks ---
        for track in tracks:
            color = TRACK_COLORS[track.track_id % len(TRACK_COLORS)]
            pos = track.position
            vel = track.velocity

            # Sphère approximative sur la vue 3D
            self.ax3d.scatter(
                pos[0], pos[1], 0.85,
                c=[color], s=80, marker="o", zorder=10,
            )

            # Trajectoire (trail)
            if track.track_id not in self._trails:
                self._trails[track.track_id] = []
            trail = self._trails[track.track_id]
            trail.append(pos.copy())
            if len(trail) > self.trail_length:
                trail.pop(0)

            if len(trail) > 1:
                trail_arr = np.array(trail)
                alphas = np.linspace(0.1, 0.8, len(trail_arr) - 1)
                for i in range(len(trail_arr) - 1):
                    self.ax2d.plot(
                        trail_arr[i:i+2, 0], trail_arr[i:i+2, 1],
                        color=color, alpha=alphas[i], lw=1.2,
                    )

            # Cercle de position + flèche vitesse
            circle = plt.Circle(pos, 0.35, color=color, fill=False, lw=1.5)
            self.ax2d.add_patch(circle)
            self.ax2d.annotate(
                "", xy=pos + vel * 0.5, xytext=pos,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )
            self.ax2d.text(
                pos[0], pos[1] + 0.5, f"T{track.track_id}",
                color=color, fontsize=7, ha="center",
            )

        # --- Overlay KPIs ---
        if kpis is not None:
            kpi_text = self._format_kpis(kpis)
            self.ax2d.text(
                0.02, 0.98, kpi_text,
                transform=self.ax2d.transAxes,
                color="white", fontsize=7.5,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1b2a", alpha=0.8),
            )

        # Frame index
        if kpis:
            self.fig.suptitle(
                f"LiDAR Spatial Tracker — Frame {kpis.frame_idx:04d}",
                color="white", fontsize=12, y=0.99,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.97])

    @staticmethod
    def _format_kpis(kpis: FrameKPIs) -> str:
        lines = [
            f"Frame {kpis.frame_idx:04d}",
            f"Vitesse moy : {kpis.avg_speed:.2f} m/s",
            "── Densité par zone ──",
        ]
        for zone, cnt in kpis.count_per_zone.items():
            dens = kpis.density_per_zone[zone]
            lines.append(f"  {zone}: {cnt} pers ({dens:.3f}/m²)")
        lines.append("── Flux portes ──")
        for door in kpis.flux_in:
            lines.append(
                f"  {door}: +{kpis.flux_in[door]} / -{kpis.flux_out[door]}"
            )
        lines.append("── Tps passage moy ──")
        for zone, t in kpis.avg_passage_time.items():
            lines.append(f"  {zone}: {t:.1f}s")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Show interactif
    # ------------------------------------------------------------------

    def animate(
        self,
        frame_generator: Callable,
        n_frames: int,
        interval: int = 100,
        output_path: Optional[str] = None,
        fps: int = 10,
    ) -> None:
        """
        Lance une animation à partir d'un générateur de frames.

        Paramètres
        ----------
        frame_generator : callable(frame_idx) → (points, tracks, kpis)
        n_frames        : nombre total de frames
        interval        : délai entre frames en ms (pour affichage interactif)
        output_path     : si fourni, export MP4 au lieu d'afficher
        fps             : images par seconde pour l'export MP4
        """

        def _update(frame_idx: int):
            points, tracks, kpis = frame_generator(frame_idx)
            self.render_frame(points, tracks, kpis)
            return []

        anim = FuncAnimation(
            self.fig, _update,
            frames=n_frames,
            interval=interval,
            blit=False,
        )

        if output_path:
            print(f"Export vidéo → {output_path}")
            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(output_path, writer=writer, dpi=120)
            print("Export terminé.")
        else:
            plt.show()
