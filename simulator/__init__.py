"""Simulator package — génération de nuages de points LiDAR synthétiques."""
from simulator.scene import Scene, Zone, Door
from simulator.agent import Agent, spawn_agents

__all__ = ["Scene", "Zone", "Door", "Agent", "spawn_agents"]
