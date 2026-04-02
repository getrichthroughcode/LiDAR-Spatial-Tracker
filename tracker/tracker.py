"""
tracker.py — Facade de rétrocompatibilité.

Importe SimpleTracker sous le nom MultiObjectTracker pour ne pas casser
le code existant. Préférer l'usage direct de registry.build_tracker().
"""

from tracker.simple_tracker import SimpleTracker as MultiObjectTracker

__all__ = ["MultiObjectTracker"]
