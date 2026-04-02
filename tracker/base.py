"""
base.py — Interface commune pour tous les trackers multi-objets.

Chaque tracker doit :
  - accepter un nuage de points (N, 3) via update()
  - retourner une liste de TrackState (tracks confirmés)
  - exposer reset() pour réinitialiser entre simulations
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class TrackState:
    """
    État d'un track confirmé — interface commune à tous les trackers.

    Attributs
    ----------
    track_id  : identifiant unique et stable du track
    position  : [x, y] en mètres (estimé)
    velocity  : [vx, vy] en m/s (estimé)
    speed     : norme du vecteur vitesse (m/s)
    confirmed : True une fois que le track a été vu N fois consécutives
    """
    track_id: int
    position: np.ndarray   # shape (2,)
    velocity: np.ndarray   # shape (2,)
    speed: float
    confirmed: bool


class BaseTracker(ABC):
    """
    Classe de base abstraite pour les trackers multi-objets.

    Le contrat minimal :
      - update(points) → list[TrackState]  appelé à chaque frame
      - reset()                             réinitialise l'état interne

    Les paramètres communs (dt, max_age, min_hits) sont définis ici
    à titre indicatif ; chaque sous-classe les accepte dans __init__.
    """

    @abstractmethod
    def update(self, points: np.ndarray) -> list[TrackState]:
        """
        Traite un nuage de points et retourne les tracks confirmés.

        Paramètres
        ----------
        points : np.ndarray (N, 3) — coordonnées [x, y, z]

        Retourne
        --------
        list[TrackState] — tracks confirmés après mise à jour
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Réinitialise l'état interne du tracker (tracks, ID counter, etc.)."""
        ...

    @property
    @abstractmethod
    def tracker_name(self) -> str:
        """Nom court du tracker (ex: 'simple', 'sort')."""
        ...
