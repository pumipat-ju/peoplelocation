"""Assignment boundary for the GlobalIdentityManager.

Batch assignment remains implemented by the manager in this behavior-preserving
extraction; this module exposes the subsystem boundary without duplicating it.
"""

def assign_batch(manager, camera, detections, prev_assignments=None):
    return manager.assign_batch(camera, detections, prev_assignments=prev_assignments)

def assign_global_batch(manager, observations, **kwargs):
    return manager.assign_global_batch(observations, **kwargs)
