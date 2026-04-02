"""
main.py — Point d'entrée principal du LiDAR Spatial Tracker.

Usage :
  python main.py                      # simulation interactive Matplotlib
  python main.py --export video.mp4   # export vidéo MP4
  python main.py --save-data          # sauvegarde données pour le dashboard
  python main.py --no-viz             # simulation silencieuse (benchmark)

Options :
  --agents N        Nombre d'agents (défaut: 6)
  --frames N        Nombre de frames (défaut: 200)
  --dt FLOAT        Pas de temps en secondes (défaut: 0.1)
  --seed INT        Seed aléatoire (défaut: 42)
  --noise FLOAT     Bruit LiDAR en mètres (défaut: 0.05)
  --tracker STRING  Tracker à utiliser : simple, sort  (défaut: simple)
"""

from __future__ import annotations
import argparse
import time
import numpy as np

from simulator.scene import Scene
from simulator.agent import spawn_agents
from tracker.registry import build_tracker, list_trackers
from analytics.kpis import KPIEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiDAR Spatial Tracker")
    p.add_argument("--agents",  type=int,   default=6,       help="Nombre d'agents")
    p.add_argument("--frames",  type=int,   default=200,     help="Frames à simuler")
    p.add_argument("--dt",      type=float, default=0.1,     help="Pas de temps (s)")
    p.add_argument("--seed",    type=int,   default=42,      help="Seed aléatoire")
    p.add_argument("--noise",   type=float, default=0.05,    help="Bruit capteur (m)")
    p.add_argument("--export",  type=str,   default=None,    help="Chemin export MP4")
    p.add_argument("--tracker", type=str,   default="simple",
                   choices=list_trackers(), help="Tracker à utiliser")
    p.add_argument("--save-data", action="store_true", help="Sauvegarde .npy")
    p.add_argument("--no-viz",    action="store_true", help="Pas de visualisation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Initialisation ---
    rng = np.random.default_rng(args.seed)
    scene = Scene.default_airport_hall()
    agents = spawn_agents(args.agents, scene, rng)
    tracker = build_tracker(args.tracker, dt=args.dt)
    print(f"Tracker      : {tracker.tracker_name}")
    kpi_eng = KPIEngine(scene, dt=args.dt)

    use_viz = not args.no_viz

    # Import viz uniquement si nécessaire (évite matplotlib sans display)
    viz = None
    if use_viz:
        import matplotlib

        if args.export is None:
            matplotlib.use("MacOSX")  # backend natif macOS (pas de Tk requis)
        else:
            matplotlib.use("Agg")  # headless pour export
        from viz.realtime import RealtimeViz

        viz = RealtimeViz(scene)

    # --- Boucle de simulation ---
    frames_data: list[dict] = []
    kpis_log: list = []

    print(f"Simulation : {args.agents} agents × {args.frames} frames (dt={args.dt}s)")
    t0 = time.perf_counter()

    def process_frame(frame_idx: int):
        """Traite une frame et retourne (points, tracks, kpis)."""
        all_pts = []
        for agent in agents:
            agent.step(scene, args.dt, rng)
            pts = agent.generate_point_cloud(rng, noise_std=args.noise)
            all_pts.append(pts)

        points = np.vstack(all_pts) if all_pts else np.empty((0, 3))
        confirmed = tracker.update(points)
        kpis = kpi_eng.update(confirmed, frame_idx)

        return points, confirmed, kpis

    if use_viz and viz is not None:
        # Mode animé : le viz pilote la boucle
        viz.animate(
            frame_generator=process_frame,
            n_frames=args.frames,
            interval=max(1, int(args.dt * 1000)),
            output_path=args.export,
            fps=int(1 / args.dt),
        )
    else:
        # Mode silencieux : boucle directe
        for frame_idx in range(args.frames):
            points, confirmed, kpis = process_frame(frame_idx)
            kpis_log.append(kpis)

            if frame_idx % 50 == 0:
                print(
                    f"  Frame {frame_idx:4d} | tracks={len(confirmed):2d} "
                    f"| speed={kpis.avg_speed:.2f} m/s"
                )

    elapsed = time.perf_counter() - t0
    fps_achieved = args.frames / elapsed
    print(f"\nTerminé en {elapsed:.2f}s ({fps_achieved:.1f} fps simulés)")

    # --- Sauvegarde des données ---
    if args.save_data:
        import os

        os.makedirs("data", exist_ok=True)
        np.save("data/heatmap.npy", kpi_eng.get_heatmap_normalized())
        print("Données sauvegardées dans data/")

    # --- Résumé KPIs final ---
    if kpis_log:
        last = kpis_log[-1]
        print("\n=== KPIs finaux ===")
        print(f"Flux entrant  : {last.flux_in}")
        print(f"Flux sortant  : {last.flux_out}")
        print(f"Vitesse moy   : {last.avg_speed:.2f} m/s")
        print(f"Occupation    : {last.count_per_zone}")


if __name__ == "__main__":
    main()
