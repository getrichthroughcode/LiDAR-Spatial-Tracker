"""
sort_tracker.py — SORT (Simple Online and Realtime Tracking).

Référence : Bewley et al., 2016 — https://arxiv.org/abs/1602.00763

Différences clés vs SimpleTracker :
  - Kalman state : [cx, cy, w, h, vcx, vcy]  (centre + taille bbox + vitesses)
  - Association  : IoU 2D sur bounding boxes (pas distance centroïde)
  - Coût Hungarian : 1 - IoU
  - Seuil        : iou_threshold (défaut 0.3)

Adapté au domaine spatial (LiDAR top-down) : les bboxes viennent du clustering
DBSCAN sur les projections XY des nuages de points.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

from tracker.base import BaseTracker, TrackState
from tracker.clustering import DBSCANDetector, Detection


# ---------------------------------------------------------------------------
# Kalman spécialisé SORT : état [cx, cy, w, h, vcx, vcy]
# ---------------------------------------------------------------------------

class SORTKalman:
    """
    Filtre de Kalman pour SORT — modèle boîte englobante à vitesse constante.

    État  : [cx, cy, w, h, vcx, vcy]
    Mesure: [cx, cy, w, h]
    """

    _id_counter: int = 0

    def __init__(self, bbox: np.ndarray, dt: float = 0.1, min_hits: int = 3) -> None:
        """
        Paramètres
        ----------
        bbox : [xmin, ymin, xmax, ymax]
        """
        SORTKalman._id_counter += 1
        self.track_id = SORTKalman._id_counter
        self.hits = 1
        self.age = 1
        self.frames_since_update = 0
        self.confirmed = False
        self.min_hits = min_hits

        self.kf = KalmanFilter(dim_x=6, dim_z=4)

        # Transition : cx, cy, w, h évoluent avec vitesse constante
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0],
            [0, 1, 0, 0,  0, dt],
            [0, 0, 1, 0,  0,  0],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ], dtype=float)

        # Observation : on mesure [cx, cy, w, h]
        self.kf.H = np.eye(4, 6)

        # Bruits
        self.kf.R = np.diag([0.25, 0.25, 0.5, 0.5])           # mesure
        self.kf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.5, 0.5])   # processus
        self.kf.P = np.diag([1.0, 1.0, 1.0, 1.0, 100.0, 100.0])

        cx, cy, w, h = _bbox_to_cxcywh(bbox)
        self.kf.x = np.array([cx, cy, w, h, 0.0, 0.0]).reshape(6, 1)

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0

    def predict(self) -> np.ndarray:
        """Prédit et retourne la bbox [xmin, ymin, xmax, ymax]."""
        self.kf.predict()
        self.age += 1
        self.frames_since_update += 1
        return _cxcywh_to_bbox(self.kf.x[:4, 0])

    def update(self, bbox: np.ndarray) -> None:
        """Met à jour avec une mesure bbox [xmin, ymin, xmax, ymax]."""
        z = _bbox_to_cxcywh(bbox).reshape(4, 1)
        self.kf.update(z)
        self.hits += 1
        self.frames_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True

    @property
    def position(self) -> np.ndarray:
        """Centroïde [cx, cy]."""
        return self.kf.x[:2, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Vitesse [vcx, vcy]."""
        return self.kf.x[4:6, 0].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def predicted_bbox(self) -> np.ndarray:
        return _cxcywh_to_bbox(self.kf.x[:4, 0])


# ---------------------------------------------------------------------------
# Helpers bbox ↔ cxcywh
# ---------------------------------------------------------------------------

def _bbox_to_cxcywh(bbox: np.ndarray) -> np.ndarray:
    """[xmin, ymin, xmax, ymax] → [cx, cy, w, h]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    return np.array([cx, cy, max(w, 1e-3), max(h, 1e-3)])


def _cxcywh_to_bbox(cxcywh: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] → [xmin, ymin, xmax, ymax]"""
    cx, cy, w, h = cxcywh
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


# ---------------------------------------------------------------------------
# IoU 2D
# ---------------------------------------------------------------------------

def iou_2d(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    IoU entre deux bboxes 2D [xmin, ymin, xmax, ymax].
    Retourne 0 si pas de recouvrement.
    """
    ix1 = max(bbox1[0], bbox2[0])
    iy1 = max(bbox1[1], bbox2[1])
    ix2 = min(bbox1[2], bbox2[2])
    iy2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(dets: list[np.ndarray], trks: list[np.ndarray]) -> np.ndarray:
    """Matrice IoU (n_dets × n_trks)."""
    mat = np.zeros((len(dets), len(trks)))
    for i, d in enumerate(dets):
        for j, t in enumerate(trks):
            mat[i, j] = iou_2d(d, t)
    return mat


# ---------------------------------------------------------------------------
# SORTTracker
# ---------------------------------------------------------------------------

class SORTTracker(BaseTracker):
    """
    SORT — Simple Online and Realtime Tracking.

    Association via IoU sur bounding boxes 2D, Kalman [cx,cy,w,h,vcx,vcy].

    Paramètres
    ----------
    dt              : pas de temps (s)
    max_age         : frames sans association avant suppression
    min_hits        : hits avant confirmation
    iou_threshold   : IoU minimum pour associer (défaut 0.3)
    dbscan_eps      : rayon DBSCAN (m)
    dbscan_min_pts  : points min DBSCAN
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_age: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        dbscan_eps: float = 0.8,
        dbscan_min_pts: int = 5,
    ) -> None:
        self.dt = dt
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.detector = DBSCANDetector(eps=dbscan_eps, min_samples=dbscan_min_pts)
        self._tracks: list[SORTKalman] = []
        SORTKalman.reset_id_counter()

    @property
    def tracker_name(self) -> str:
        return "sort"

    def reset(self) -> None:
        self._tracks = []
        SORTKalman.reset_id_counter()

    def update(self, points: np.ndarray) -> list[TrackState]:
        detections = self.detector.detect(points)

        # Prédiction
        for t in self._tracks:
            t.predict()

        # Association par IoU
        matched, unmatched_dets, _ = self._associate(detections)

        for det_idx, trk_idx in matched:
            self._tracks[trk_idx].update(detections[det_idx].bbox)

        for det_idx in unmatched_dets:
            self._tracks.append(SORTKalman(
                bbox=detections[det_idx].bbox,
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

        det_bboxes = [d.bbox for d in detections]
        trk_bboxes = [t.predicted_bbox for t in self._tracks]

        iou_mat = iou_matrix(det_bboxes, trk_bboxes)
        # Coût = 1 - IoU  →  on minimise
        cost = 1.0 - iou_mat

        row_ind, col_ind = linear_sum_assignment(cost)

        matched, unmatched_dets, unmatched_trks = [], list(range(len(detections))), list(range(len(self._tracks)))
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] < self.iou_threshold:
                continue  # IoU trop faible → non associé
            matched.append((r, c))
            unmatched_dets.remove(r)
            unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks
