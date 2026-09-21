"""Pending and unresolved handoff boundary."""

def unresolved_temporal_consensus(manager, record):
    return manager._unresolved_temporal_consensus(record)

def discard_unresolved_handoffs(manager, *args, **kwargs):
    return manager.discard_unresolved_handoffs(*args, **kwargs)
