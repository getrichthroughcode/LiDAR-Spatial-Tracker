"""
clustering.py — Détection d'objets par DBSCAN sur un nuage de points 3D.

Pour chaque frame, on regroupe les points en clusters (= personnes détectées).
On utilise uniquement les coordonnées XY (projection au sol) pour le clustering,
ce qui est cohérent avec un LiDAR monté en hauteur regardant vers le bas.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import DBSCAN


@dataclass
class Detection:
    """
    Objet détecté dans une frame.

    Attributs
    ----------
    centroid   : [x, y] centre du cluster
    bbox       : [x_min, y_min, x_max, y_max]
    n_points   : nombre de points dans le cluster
    height_est : hauteur estimée (z_max du cluster)
    """
    centroid: np.ndarray      # shape (2,)
    bbox: np.ndarray          # shape (4,)
    n_points: int
    height_est: float


class DBSCANDetector:
    """
    Wrapper DBSCAN pour détecter des piétons dans un nuage de points 3D.

    Paramètres
    ----------
    eps         : rayon de voisinage (m) — ~0.8 pour des groupes piétons
    min_samples : points minimum pour former un cluster
    min_points  : clusters avec moins de points sont ignorés (bruit)
    """

    def __init__(
        self,
        eps: float = 0.8,
        min_samples: int = 5,
        min_points: int = 4,
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.min_points = min_points
        self._dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)

    def detect(self, points: np.ndarray) -> list[Detection]:
        """
        Applique DBSCAN sur un nuage de points et retourne les détections.

        Paramètres
        ----------
        points : np.ndarray de forme (N, 3) — coordonnées [x, y, z]

        Retourne
        --------
        list[Detection] — une détection par cluster valide
        """
        if len(points) < self.min_samples:
            return []

        # Clustering sur XY uniquement (projection au sol)
        xy = points[:, :2]
        labels = self._dbscan.fit_predict(xy)

        detections = []
        for label in set(labels):
            if label == -1:  # bruit DBSCAN
                continue
            mask = labels == label
            cluster_pts = points[mask]

            if len(cluster_pts) < self.min_points:
                continue

            cluster_xy = cluster_pts[:, :2]
            centroid = cluster_xy.mean(axis=0)
            bbox = np.array([
                cluster_xy[:, 0].min(),
                cluster_xy[:, 1].min(),
                cluster_xy[:, 0].max(),
                cluster_xy[:, 1].max(),
            ])
            height_est = float(cluster_pts[:, 2].max())

            detections.append(Detection(
                centroid=centroid,
                bbox=bbox,
                n_points=int(mask.sum()),
                height_est=height_est,
            ))

        return detections
