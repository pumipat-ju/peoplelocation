"""Topology subsystem extracted from backend.main."""

from .config import (
    TOPOLOGY_SCHEMA_VERSION,
    TOPOLOGY_SUPPORTED_VERSIONS,
    configure_topology_dependencies,
    default_topology_config,
    load_topology_config,
    normalize_topology_config,
    validate_topology_config,
)
from .gates import point_in_polygon

__all__ = [
    "TOPOLOGY_SCHEMA_VERSION",
    "TOPOLOGY_SUPPORTED_VERSIONS",
    "configure_topology_dependencies",
    "default_topology_config",
    "load_topology_config",
    "normalize_topology_config",
    "validate_topology_config",
    "point_in_polygon",
]
