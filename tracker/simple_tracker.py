"""
simple_tracker.py — Tracker basique : DBSCAN + Kalman [x,y,vx,vy] + distance euclidienne.

C'est le tracker d'origine du projet. Association par distance centroïde,
sans IoU. Simple, rapide, bonne baseline.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.base import BaseTracker, TrackState
from tracker.clustering import DBSCANDetector, Detection
from tracker.kalman import TrackKalman


class SimpleTracker(BaseTracker):
    """
    Tracker multi-objets : DBSCAN + Kalman + Hungarian (distance euclidienne).

    Paramètres
    ----------
    dt              : pas de temps entre frames (s)
    max_age         : frames sans association avant suppression
    min_hits        : hits pour confirmer un track
    dist_threshold  : distance max (m) pour association
    dbscan_eps      : rayon DBSCAN (m)
    dbscan_min_pts  : points min DBSCAN
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_age: int = 5,
        min_hits: int = 3,
        dist_threshold: float = 2.0,
        dbscan_eps: float = 0.8,
        dbscan_min_pts: int = 5,
    ) -> None:
        self.dt = dt
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.detector = DBSCANDetector(eps=dbscan_eps, min_samples=dbscan_min_pts)
        self._tracks: list[TrackKalman] = []
        TrackKalman.reset_id_counter()

    @property
    def tracker_name(self) -> str:
        return "simple"

    def reset(self) -> None:
        self._tracks = []
        TrackKalman.reset_id_counter()

    def update(self, points: np.ndarray) -> list[TrackState]:
        detections = self.detector.detect(points)

        for t in self._tracks:
            t.predict()

        matched, unmatched_dets, _ = self._associate(detections)

        for det_idx, trk_idx in matched:
            self._tracks[trk_idx].update(detections[det_idx].centroid)

        for det_idx in unmatched_dets:
            self._tracks.append(TrackKalman(
                initial_pos=detections[det_idx].centroid,
                dt=self.dt,
                min_hits=self.min_hits,
            ))

        self._tracks = [t for t in self._tracks if t.frames_since_update <= self.max_age]

        return [
            TrackState(
                track_id=t.track_id,
                position=t.position,
                velocity=t.velocity,
                speed=t.speed,
                confirmed=t.confirmed,
            )
            for t in self._tracks if t.confirmed
        ]

    def _associate(
        self,
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self._tracks)))

        cost = np.zeros((len(detections), len(self._tracks)))
        for d_i, det in enumerate(detections):
            for t_i, trk in enumerate(self._tracks):
                cost[d_i, t_i] = np.linalg.norm(det.centroid - trk.position)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_dets, unmatched_trks = [], list(range(len(detections))), list(range(len(self._tracks)))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self.dist_threshold:
                continue
            matched.append((r, c))
            unmatched_dets.remove(r)
            unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks
