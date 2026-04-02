"""
kpis.py — Calcul des 5 KPIs métier.

KPIs implémentés :
  1. Densité instantanée     : personnes par zone (par frame)
  2. Flux entrant/sortant    : comptage directionnel aux portes (cumulatif)
  3. Temps de passage moyen  : durée de traversée d'une zone (glissant 60s)
  4. Heatmap d'occupation    : densité spatiale agrégée sur N frames
  5. Vitesse moyenne         : norme vecteur vitesse Kalman (par frame)
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

from simulator.scene import Scene
from tracker.base import TrackState


@dataclass
class FrameKPIs:
    """KPIs calculés pour une frame donnée."""
    frame_idx: int
    density_per_zone: Dict[str, float]        # personnes / m²
    count_per_zone: Dict[str, int]            # nombre de personnes par zone
    avg_speed: float                           # m/s moyen sur tous les tracks
    flux_in: Dict[str, int]                   # cumul entrées par porte
    flux_out: Dict[str, int]                  # cumul sorties par porte
    avg_passage_time: Dict[str, float]        # secondes, par zone (glissant)


class KPIEngine:
    """
    Moteur de calcul des KPIs à partir des tracks confirmés.

    Paramètres
    ----------
    scene         : scène avec zones et portes
    dt            : pas de temps entre frames (s)
    heatmap_res   : résolution de la heatmap (cellules par mètre)
    passage_window: fenêtre glissante pour le temps de passage (frames)
    """

    def __init__(
        self,
        scene: Scene,
        dt: float = 0.1,
        heatmap_res: float = 2.0,
        passage_window: int = 600,   # ~60s à 10fps
    ) -> None:
        self.scene = scene
        self.dt = dt
        self.passage_window = passage_window

        # Heatmap : grille discrète de la scène
        nx = int(np.ceil(scene.width  * heatmap_res))
        ny = int(np.ceil(scene.height * heatmap_res))
        self.heatmap = np.zeros((ny, nx), dtype=np.float32)
        self._heatmap_res = heatmap_res

        # Compteurs flux cumulatifs
        self.flux_in:  Dict[str, int] = {d.name: 0 for d in scene.doors}
        self.flux_out: Dict[str, int] = {d.name: 0 for d in scene.doors}

        # Suivi par track : zone courante et frame d'entrée dans la zone
        self._track_zone: Dict[int, Optional[str]] = {}
        self._track_entry_frame: Dict[int, int] = {}
        self._track_prev_pos: Dict[int, np.ndarray] = {}

        # Historique des temps de passage (par zone, fenêtre glissante)
        self._passage_times: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=passage_window)
        )

    # ------------------------------------------------------------------
    # Mise à jour par frame
    # ------------------------------------------------------------------

    def update(self, tracks: List[TrackState], frame_idx: int) -> FrameKPIs:
        """
        Calcule les KPIs pour la frame courante.

        Paramètres
        ----------
        tracks    : tracks confirmés après mise à jour du tracker
        frame_idx : indice de frame courant

        Retourne
        --------
        FrameKPIs — snapshot des KPIs pour cette frame
        """
        count_per_zone: Dict[str, int] = {z.name: 0 for z in self.scene.zones}
        speeds: List[float] = []

        for track in tracks:
            pos = track.position
            speeds.append(track.speed)

            # Mise à jour heatmap
            self._update_heatmap(pos)

            # Vérification zones
            current_zone = self.scene.zone_at(pos[0], pos[1])
            current_zone_name = current_zone.name if current_zone else None

            if current_zone_name:
                count_per_zone[current_zone_name] += 1

            # Détection de changement de zone → temps de passage
            prev_zone_name = self._track_zone.get(track.track_id)
            if prev_zone_name != current_zone_name:
                # Sortie de zone précédente
                if prev_zone_name is not None and track.track_id in self._track_entry_frame:
                    elapsed_frames = frame_idx - self._track_entry_frame[track.track_id]
                    elapsed_sec = elapsed_frames * self.dt
                    self._passage_times[prev_zone_name].append(elapsed_sec)

                # Entrée dans nouvelle zone
                if current_zone_name is not None:
                    self._track_entry_frame[track.track_id] = frame_idx

                self._track_zone[track.track_id] = current_zone_name

            # Vérification flux aux portes
            if track.track_id in self._track_prev_pos:
                prev_pos = self._track_prev_pos[track.track_id]
                for door in self.scene.doors:
                    if door.is_entry_event(prev_pos, pos):
                        self.flux_in[door.name] += 1
                    elif door.is_exit_event(prev_pos, pos):
                        self.flux_out[door.name] += 1

            self._track_prev_pos[track.track_id] = pos.copy()

        # Densité par zone (personnes / m²)
        density_per_zone = {
            z.name: count_per_zone[z.name] / z.area
            for z in self.scene.zones
        }

        # Temps de passage moyen glissant
        avg_passage_time = {
            z.name: (
                float(np.mean(list(self._passage_times[z.name])))
                if self._passage_times[z.name] else 0.0
            )
            for z in self.scene.zones
        }

        return FrameKPIs(
            frame_idx=frame_idx,
            density_per_zone=density_per_zone,
            count_per_zone=count_per_zone,
            avg_speed=float(np.mean(speeds)) if speeds else 0.0,
            flux_in=dict(self.flux_in),
            flux_out=dict(self.flux_out),
            avg_passage_time=avg_passage_time,
        )

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------

    def _update_heatmap(self, pos: np.ndarray) -> None:
        """Incrémente la cellule heatmap correspondant à la position."""
        xi = int(np.clip(pos[0] * self._heatmap_res, 0, self.heatmap.shape[1] - 1))
        yi = int(np.clip(pos[1] * self._heatmap_res, 0, self.heatmap.shape[0] - 1))
        self.heatmap[yi, xi] += 1.0

    def get_heatmap_normalized(self) -> np.ndarray:
        """Heatmap normalisée [0, 1] pour l'affichage."""
        max_val = self.heatmap.max()
        if max_val == 0:
            return self.heatmap.copy()
        return self.heatmap / max_val

    def reset_heatmap(self) -> None:
        self.heatmap[:] = 0.0
