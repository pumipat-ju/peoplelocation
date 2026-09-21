"""Travel-time gate API."""

from .config import normalize_topology_config

def normalize_travel_time_config(config, known_cameras=None):
    return normalize_topology_config(config, known_cameras=known_cameras)
