"""
kalman.py — Filtre de Kalman par track.

Modèle d'état : [x, y, vx, vy]  (position + vitesse en 2D)
Mesure       : [x, y]            (centroïde de détection)

Implémenté avec filterpy pour rester léger et lisible.
"""

from __future__ import annotations
import numpy as np
from filterpy.kalman import KalmanFilter


def make_kalman_filter(
    initial_pos: np.ndarray,
    dt: float = 0.1,
    process_noise: float = 0.5,
    measurement_noise: float = 0.5,
) -> KalmanFilter:
    """
    Crée et initialise un filtre de Kalman pour un track piéton.

    Paramètres
    ----------
    initial_pos       : [x, y] position initiale (m)
    dt                : pas de temps (s)
    process_noise     : variance du bruit de processus (incertitude mouvement)
    measurement_noise : variance du bruit de mesure (incertitude capteur)

    État : x = [x, y, vx, vy]^T
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Matrice de transition (modèle à vitesse constante)
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1],
    ], dtype=float)

    # Matrice d'observation : on observe uniquement x, y
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    # Bruit de mesure R
    kf.R = np.eye(2) * measurement_noise

    # Bruit de processus Q (via modèle d'accélération discrète)
    q = process_noise
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4
    kf.Q = q * np.array([
        [dt4/4, 0,     dt3/2, 0    ],
        [0,     dt4/4, 0,     dt3/2],
        [dt3/2, 0,     dt2,   0    ],
        [0,     dt3/2, 0,     dt2  ],
    ], dtype=float)

    # Covariance initiale (incertitude initiale élevée sur vitesse)
    kf.P = np.diag([1.0, 1.0, 10.0, 10.0])

    # État initial
    kf.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0]).reshape(4, 1)

    return kf


class TrackKalman:
    """
    Encapsule un KalmanFilter avec l'état du track associé.

    Attributs
    ----------
    track_id       : identifiant unique du track
    kf             : filtre de Kalman
    hits           : nombre de fois où le track a été associé à une détection
    age            : nombre total de frames depuis la création
    frames_since_update : frames sans association (pour suppression)
    confirmed      : True si hits >= min_hits
    """

    _id_counter: int = 0

    def __init__(
        self,
        initial_pos: np.ndarray,
        dt: float = 0.1,
        min_hits: int = 3,
    ) -> None:
        TrackKalman._id_counter += 1
        self.track_id = TrackKalman._id_counter
        self.kf = make_kalman_filter(initial_pos, dt=dt)
        self.hits: int = 1
        self.age: int = 1
        self.frames_since_update: int = 0
        self.min_hits = min_hits
        self.confirmed: bool = False

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0

    def predict(self) -> np.ndarray:
        """Avance le filtre d'un pas de temps. Retourne la position prédite [x, y]."""
        self.kf.predict()
        self.age += 1
        self.frames_since_update += 1
        return self.kf.x[:2, 0]

    def update(self, measurement: np.ndarray) -> None:
        """Met à jour le filtre avec une mesure [x, y]."""
        self.kf.update(measurement.reshape(2, 1))
        self.hits += 1
        self.frames_since_update = 0
        if self.hits >= self.min_hits:
            self.confirmed = True

    @property
    def position(self) -> np.ndarray:
        """Position estimée courante [x, y]."""
        return self.kf.x[:2, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Vitesse estimée [vx, vy]."""
        return self.kf.x[2:, 0].copy()

    @property
    def speed(self) -> float:
        """Norme du vecteur vitesse (m/s)."""
        return float(np.linalg.norm(self.velocity))
