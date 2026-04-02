"""
agent.py — Modèle d'agent piéton + génération de nuage de points LiDAR synthétique.

Chaque agent suit une trajectoire brownienne bornée dans la scène.
Sa signature LiDAR est une ellipse de points 3D bruités simulant une
détection réaliste (bruit gaussien, hauteur variable).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from simulator.scene import Scene


# --- Paramètres physiques par défaut ---
PERSON_HEIGHT_MEAN = 1.70   # m
PERSON_HEIGHT_STD  = 0.10
PERSON_RADIUS_XY   = 0.30   # rayon ellipse horizontale (m)
POINTS_PER_AGENT   = 40     # points LiDAR par agent par frame
LIDAR_NOISE_STD    = 0.05   # bruit gaussien (m)


@dataclass
class Agent:
    """
    Piéton synthétique avec état cinématique [x, y, vx, vy].

    Attributs
    ----------
    agent_id    : identifiant unique
    position    : np.ndarray [x, y] en mètres
    velocity    : np.ndarray [vx, vy] en m/s
    height      : taille du piéton en m
    active      : False = hors scène (sorti par une porte)
    _entry_time : frame d'apparition dans la scène
    """
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray
    height: float = PERSON_HEIGHT_MEAN
    active: bool = True
    _entry_time: int = 0
    _prev_position: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Mise à jour de la dynamique
    # ------------------------------------------------------------------

    def step(
        self,
        scene: Scene,
        dt: float,
        rng: np.random.Generator,
        max_speed: float = 1.5,
        accel_noise: float = 0.4,
    ) -> None:
        """
        Avance l'agent d'un pas de temps dt.

        Modèle : marche aléatoire bornée (random walk with reflecting walls).
        L'accélération est un bruit blanc gaussian scalé par accel_noise.
        """
        if not self.active:
            return

        self._prev_position = self.position.copy()

        # Bruit d'accélération (modèle de mouvement piéton réaliste)
        accel = rng.normal(0, accel_noise, size=2)
        self.velocity = np.clip(self.velocity + accel * dt, -max_speed, max_speed)

        # Intégration Euler
        new_pos = self.position + self.velocity * dt

        # Réflexion sur les murs (évite la sortie hors scène)
        for dim, limit in enumerate([scene.width, scene.height]):
            if new_pos[dim] < 0.0:
                new_pos[dim] = -new_pos[dim]
                self.velocity[dim] *= -1
            elif new_pos[dim] > limit:
                new_pos[dim] = 2 * limit - new_pos[dim]
                self.velocity[dim] *= -1

        self.position = scene.clamp_position(new_pos)

    # ------------------------------------------------------------------
    # Génération du nuage de points LiDAR
    # ------------------------------------------------------------------

    def generate_point_cloud(
        self,
        rng: np.random.Generator,
        n_points: int = POINTS_PER_AGENT,
        noise_std: float = LIDAR_NOISE_STD,
        occlusion_fraction: float = 0.0,
    ) -> np.ndarray:
        """
        Génère un nuage de points 3D simulant la détection LiDAR de cet agent.

        Modèle : cylindre vertical (ellipse XY × hauteur Z) avec bruit gaussien.
        L'occlusion partielle est simulée en supprimant un secteur angulaire.

        Retourne
        --------
        points : np.ndarray de forme (n_visible, 3)
        """
        if not self.active:
            return np.empty((0, 3))

        # Angles répartis sur le cylindre (vue de dessus = rayon + hauteur)
        angles = rng.uniform(0, 2 * np.pi, n_points)
        # Occlusion : suppression d'un secteur (ex. mur derrière l'agent)
        if occlusion_fraction > 0:
            occ_start = rng.uniform(0, 2 * np.pi)
            occ_end = occ_start + occlusion_fraction * 2 * np.pi
            mask = ~((angles >= occ_start % (2 * np.pi)) &
                     (angles <= occ_end % (2 * np.pi)))
            angles = angles[mask]

        n_vis = len(angles)
        if n_vis == 0:
            return np.empty((0, 3))

        # Coordonnées cylindriques → cartésiennes
        r = rng.uniform(0, PERSON_RADIUS_XY, n_vis)
        x = self.position[0] + r * np.cos(angles)
        y = self.position[1] + r * np.sin(angles)
        z = rng.uniform(0.1, self.height, n_vis)  # hauteur non uniforme

        # Bruit gaussien 3D
        noise = rng.normal(0, noise_std, (n_vis, 3))
        points = np.column_stack([x, y, z]) + noise

        return points.astype(np.float32)


# ------------------------------------------------------------------
# Fabrique de scènes peuplées
# ------------------------------------------------------------------

def spawn_agents(
    n_agents: int,
    scene: Scene,
    rng: np.random.Generator,
    frame_index: int = 0,
    max_speed: float = 1.2,
) -> list[Agent]:
    """Crée n_agents à des positions aléatoires dans la scène."""
    agents = []
    for i in range(n_agents):
        pos = scene.random_position(rng)
        vel = rng.uniform(-max_speed / 2, max_speed / 2, size=2)
        height = rng.normal(PERSON_HEIGHT_MEAN, PERSON_HEIGHT_STD)
        height = np.clip(height, 1.4, 2.1)
        agents.append(Agent(
            agent_id=i,
            position=pos,
            velocity=vel,
            height=float(height),
            active=True,
            _entry_time=frame_index,
        ))
    return agents
