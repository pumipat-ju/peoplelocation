"""Topology configuration and travel-time normalization."""
import json
import os
import threading
import numpy as np

TOPOLOGY_SCHEMA_VERSION = 2
TOPOLOGY_SUPPORTED_VERSIONS = {1, TOPOLOGY_SCHEMA_VERSION}

_dependency_namespace = None
def configure_topology_dependencies(namespace):
    global _dependency_namespace
    _dependency_namespace = namespace
    globals().update(namespace)

def sync_topology_dependencies():
    if _dependency_namespace is not None:
        globals().update(_dependency_namespace)

def default_topology_config():
    return {
        "version": TOPOLOGY_SCHEMA_VERSION,
        "enforce": False,
        "transitions": [],
    }


def _topology_time_value(value, field_name, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{field_name} must be a non-negative number"
        ) from error
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return parsed


def normalize_topology_config(config, known_cameras=None):
    """Validate and normalize topology without writing persistent data.

    Version 1 travel-time field names remain readable so an existing file is
    not migrated on startup. Every in-memory rule uses the version 2 contract.
    """
    if not isinstance(config, dict):
        raise ValueError("Topology config must be an object")

    version = config.get("version", 1)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in TOPOLOGY_SUPPORTED_VERSIONS
    ):
        raise ValueError(
            "Unsupported topology version; expected version 1 or 2"
        )

    enforce = config.get("enforce", False)
    if not isinstance(enforce, bool):
        raise ValueError("Topology enforce must be a boolean")

    transitions = config.get("transitions", [])
    if not isinstance(transitions, list):
        raise ValueError("Topology transitions must be a list")

    known = None
    if known_cameras is not None:
        known = {
            camera.strip()
            for camera in known_cameras
            if isinstance(camera, str) and camera.strip()
        }

    normalized_rules = []
    camera_pairs = set()
    for index, rule in enumerate(transitions):
        if not isinstance(rule, dict):
            raise ValueError(f"Topology transition {index} must be an object")

        from_camera = rule.get("from_camera")
        to_camera = rule.get("to_camera")
        if not isinstance(from_camera, str) or not from_camera.strip():
            raise ValueError(
                f"Topology transition {index} needs a source camera"
            )
        if not isinstance(to_camera, str) or not to_camera.strip():
            raise ValueError(
                f"Topology transition {index} needs a destination camera"
            )
        from_camera = from_camera.strip()
        to_camera = to_camera.strip()
        if from_camera == to_camera:
            raise ValueError(
                f"Topology transition {index} must reference two cameras"
            )

        if known is not None:
            unknown = sorted({from_camera, to_camera} - known)
            if unknown:
                raise ValueError(
                    "Topology transition references unknown camera(s): "
                    + ", ".join(unknown)
                )

        pair = (from_camera, to_camera)
        if pair in camera_pairs:
            raise ValueError(
                f"Duplicate topology transition: {from_camera} -> {to_camera}"
            )
        camera_pairs.add(pair)

        min_value = rule.get(
            "min_travel_time_sec",
            rule.get("min_travel_sec", 0.0),
        )
        max_value = rule.get(
            "max_travel_time_sec",
            rule.get("max_travel_sec"),
        )
        min_time = _topology_time_value(
            min_value,
            "min_travel_time_sec",
        )
        max_time = _topology_time_value(
            max_value,
            "max_travel_time_sec",
            allow_none=True,
        )
        if max_time is not None and max_time < min_time:
            raise ValueError(
                "max_travel_time_sec must be greater than or equal to "
                "min_travel_time_sec"
            )

        overlap_allowed = rule.get("overlap_allowed", False)
        if not isinstance(overlap_allowed, bool):
            raise ValueError("overlap_allowed must be a boolean")

        normalized_rules.append({
            "from_camera": from_camera,
            "to_camera": to_camera,
            "min_travel_time_sec": min_time,
            "max_travel_time_sec": max_time,
            "overlap_allowed": overlap_allowed,
        })

    return {
        "version": TOPOLOGY_SCHEMA_VERSION,
        "enforce": enforce,
        "transitions": normalized_rules,
    }


def validate_topology_config(config, known_cameras=None):
    normalize_topology_config(config, known_cameras=known_cameras)
    return True


def load_topology_config():
    if not os.path.isfile(TOPOLOGY_CONFIG_PATH):
        return default_topology_config()
    try:
        with open(TOPOLOGY_CONFIG_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return normalize_topology_config(config)
    except Exception as error:
        logger.error(
            "Topology config invalid; cross-camera matching is fail-closed: %s",
            error,
        )
        return {
            "version": TOPOLOGY_SCHEMA_VERSION,
            "enforce": True,
            "transitions": [],
            "_validation_error": str(error),
        }
