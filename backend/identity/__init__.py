"""Global identity subsystem extracted from backend.main."""

from .manager import GlobalIdentityManager, configure_identity_dependencies
from .state_machine import GlobalIDStateMachine, ObservationState

__all__ = [
    "GlobalIdentityManager",
    "GlobalIDStateMachine",
    "ObservationState",
    "configure_identity_dependencies",
]
