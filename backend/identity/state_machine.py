"""Deterministic Global-ID observation state machine.

This layer records and validates the lifecycle of one observation. It does not
score embeddings or make a policy decision; the existing manager remains the
policy owner. Centralizing lifecycle bookkeeping ensures every commit path
(fixed claim, transition recovery, batch assignment, or fallback) reports the
same terminal state.
"""

from dataclasses import dataclass, field
from enum import Enum


class ObservationState(str, Enum):
    OBSERVED = "observed"
    CANDIDATES_BUILT = "candidates_built"
    GATED = "gated"
    TOPOLOGY_CLASSIFIED = "topology_classified"
    JOINT_ASSIGNMENT = "joint_assignment"
    PENDING_AMBIGUOUS = "pending_ambiguous"
    REJECTED = "rejected"
    COMMITTED_EXISTING = "committed_existing"
    NEW_ID_ELIGIBLE = "new_id_eligible"
    COMMITTED_NEW = "committed_new"


_TERMINAL = {
    ObservationState.PENDING_AMBIGUOUS,
    ObservationState.REJECTED,
    ObservationState.COMMITTED_EXISTING,
    ObservationState.COMMITTED_NEW,
}


@dataclass
class ObservationDecision:
    key: tuple
    state: ObservationState = ObservationState.OBSERVED
    gid: int | None = None
    source: str | None = None
    reason: str | None = None
    history: list[str] = field(default_factory=lambda: [ObservationState.OBSERVED.value])
    control_flow_authorized: bool = False
    new_id_gate_passed: bool = False


class GlobalIDStateMachine:
    """Track one terminal outcome per canonical observation key."""

    def __init__(self, max_records=2048):
        self.max_records = int(max_records)
        self._records = {}

    @staticmethod
    def canonical_key(camera, local_id, generation=None):
        return (str(camera), int(local_id), generation)

    def begin(self, camera, local_id, generation=None):
        key = self.canonical_key(camera, local_id, generation)
        record = self._records.get(key)
        if record is None or record.state in _TERMINAL:
            record = ObservationDecision(key=key)
            self._records[key] = record
        return record

    def advance(self, record, state, *, gid=None, source=None, reason=None):
        target = ObservationState(state)
        if record.state in _TERMINAL and target != record.state:
            raise ValueError(f"terminal observation cannot transition: {record.state} -> {target}")
        record.state = target
        record.history.append(target.value)
        if gid is not None:
            record.gid = int(gid)
        if source is not None:
            record.source = str(source)
        if reason is not None:
            record.reason = str(reason)
        return record

    def record_commit(self, camera, local_id, generation, gid, source, reason=None):
        record = self.begin(camera, local_id, generation)
        self.authorize_commit(record, source)
        target = ObservationState.COMMITTED_NEW if str(source) == "new" else ObservationState.COMMITTED_EXISTING
        return self.advance(record, target, gid=gid, source=source, reason=reason)

    def authorize_commit(self, record, source):
        """Gate every commit after joint-assignment control flow."""
        if record.state == ObservationState.OBSERVED:
            # Compatibility callers that invoke the legacy single-observation
            # API are still routed through the complete state pipeline.
            for state in (
                ObservationState.CANDIDATES_BUILT,
                ObservationState.GATED,
                ObservationState.TOPOLOGY_CLASSIFIED,
                ObservationState.JOINT_ASSIGNMENT,
            ):
                self.advance(record, state)
        if record.state == ObservationState.NEW_ID_ELIGIBLE:
            pass
        elif record.state != ObservationState.JOINT_ASSIGNMENT:
            raise ValueError(f"commit bypassed joint assignment: {record.state}")
        if str(source) == "new":
            if record.state == ObservationState.JOINT_ASSIGNMENT:
                self.advance(record, ObservationState.NEW_ID_ELIGIBLE)
            record.new_id_gate_passed = True
        record.control_flow_authorized = True
        return record

    def record_pending(self, camera, local_id, generation, reason=None):
        record = self.begin(camera, local_id, generation)
        return self.advance(record, ObservationState.PENDING_AMBIGUOUS, reason=reason)

    def snapshot(self):
        return [
            {
                "camera": key[0],
                "local_id": key[1],
                "generation": key[2],
                "state": record.state.value,
                "gid": record.gid,
                "source": record.source,
                "reason": record.reason,
                "history": list(record.history),
                "control_flow_authorized": record.control_flow_authorized,
                "new_id_gate_passed": record.new_id_gate_passed,
            }
            for key, record in sorted(self._records.items(), key=lambda item: repr(item[0]))
        ][-self.max_records:]
