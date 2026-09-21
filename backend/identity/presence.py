"""Presence and local-track ownership boundary."""

def reconcile_local_track_presence(manager, *args, **kwargs):
    return manager._reconcile_local_track_presence(*args, **kwargs)

def enforce_local_track_presence_invariant(manager, *args, **kwargs):
    return manager._enforce_local_track_presence_invariant(*args, **kwargs)
