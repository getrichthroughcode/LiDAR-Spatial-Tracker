"""
registry.py — Registre des trackers disponibles.

Ajouter un nouveau tracker :
  1. Créer une classe qui hérite de BaseTracker
  2. L'enregistrer dans REGISTRY avec une clé string

Usage :
  tracker = build_tracker("sort", dt=0.1, max_age=5)
  tracker = build_tracker("simple", dt=0.1, dist_threshold=1.5)
"""

from __future__ import annotations
from tracker.base import BaseTracker
from tracker.simple_tracker import SimpleTracker
from tracker.sort_tracker import SORTTracker

REGISTRY: dict[str, type[BaseTracker]] = {
    "simple": SimpleTracker,
    "sort":   SORTTracker,
}


def build_tracker(name: str, **kwargs) -> BaseTracker:
    """
    Instancie un tracker par son nom.

    Paramètres
    ----------
    name   : clé dans REGISTRY ('simple', 'sort', ...)
    kwargs : paramètres passés au constructeur du tracker

    Lève ValueError si le nom est inconnu.
    """
    if name not in REGISTRY:
        available = ", ".join(REGISTRY)
        raise ValueError(f"Tracker inconnu : '{name}'. Disponibles : {available}")
    return REGISTRY[name](**kwargs)


def list_trackers() -> list[str]:
    return list(REGISTRY)
