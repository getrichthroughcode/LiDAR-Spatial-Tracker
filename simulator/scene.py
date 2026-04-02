"""
scene.py — Définition de la scène, zones et portes.

La scène est un espace rectangulaire 2D (projeté en 3D) contenant :
- Des zones nommées (pour le calcul de densité et temps de passage)
- Des portes (pour le comptage flux entrant/sortant)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class Zone:
    """Zone rectangulaire nommée dans la scène."""
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass
class Door:
    """
    Porte = segment orienté sur un bord de la scène.
    direction : 'north', 'south', 'east', 'west'
    """
    name: str
    x: float          # centre de la porte
    y: float
    width: float = 1.5
    direction: str = "north"  # côté de la scène

    def is_entry_event(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> bool:
        """Retourne True si le mouvement prev→curr franchit la porte vers l'intérieur."""
        return self._crossed(prev_pos, curr_pos, inward=True)

    def is_exit_event(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> bool:
        return self._crossed(prev_pos, curr_pos, inward=False)

    def _crossed(self, prev: np.ndarray, curr: np.ndarray, inward: bool) -> bool:
        half = self.width / 2
        if self.direction in ("north", "south"):
            if not (self.x - half <= curr[0] <= self.x + half):
                return False
            if self.direction == "north":
                crossed = prev[1] < self.y <= curr[1]
            else:
                crossed = prev[1] > self.y >= curr[1]
        else:  # east / west
            if not (self.y - half <= curr[1] <= self.y + half):
                return False
            if self.direction == "east":
                crossed = prev[0] < self.x <= curr[0]
            else:
                crossed = prev[0] > self.x >= curr[0]
        return crossed if inward else not crossed


@dataclass
class Scene:
    """
    Scène principale : espace rectangulaire avec zones et portes.

    Paramètres
    ----------
    width, height : dimensions en mètres
    floor_z       : altitude du sol (z fixe pour la 3D)
    zones         : liste de zones nommées
    doors         : liste de portes (entrées/sorties)
    """
    width: float = 20.0
    height: float = 15.0
    floor_z: float = 0.0
    zones: List[Zone] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)

    @classmethod
    def default_airport_hall(cls) -> "Scene":
        """Scène de démonstration : hall d'aéroport simplifié."""
        zones = [
            Zone("waiting_area",   0, 10, 0, 15),
            Zone("transit_zone",  10, 20, 0, 15),
        ]
        doors = [
            Door("gate_A", x=0,    y=7.5,  width=2.0, direction="west"),
            Door("gate_B", x=20,   y=7.5,  width=2.0, direction="east"),
            Door("gate_C", x=10,   y=0,    width=2.0, direction="south"),
        ]
        return cls(width=20.0, height=15.0, zones=zones, doors=doors)

    def random_position(self, rng: np.random.Generator) -> np.ndarray:
        """Position 2D aléatoire dans la scène."""
        x = rng.uniform(0.5, self.width - 0.5)
        y = rng.uniform(0.5, self.height - 0.5)
        return np.array([x, y])

    def clamp_position(self, pos: np.ndarray) -> np.ndarray:
        """Force la position dans les limites de la scène."""
        return np.clip(pos, [0.0, 0.0], [self.width, self.height])

    def zone_at(self, x: float, y: float) -> Zone | None:
        for z in self.zones:
            if z.contains(x, y):
                return z
        return None
