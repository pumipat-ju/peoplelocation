import unittest

from backend.identity.state_machine import GlobalIDStateMachine, ObservationState


class GlobalIDStateMachineContractTests(unittest.TestCase):
    def test_existing_commit_is_terminal_and_canonical(self):
        machine = GlobalIDStateMachine()
        record = machine.begin("cam-b", 11, 2)
        for state in (ObservationState.CANDIDATES_BUILT, ObservationState.GATED,
                      ObservationState.TOPOLOGY_CLASSIFIED, ObservationState.JOINT_ASSIGNMENT):
            machine.advance(record, state)
        machine.record_commit("cam-b", 11, 2, 7, "global-cross-camera")
        snapshot = machine.snapshot()
        self.assertEqual("committed_existing", snapshot[0]["state"])
        self.assertEqual(7, snapshot[0]["gid"])

    def test_pending_and_new_identity_are_distinct(self):
        machine = GlobalIDStateMachine()
        machine.record_pending("cam-a", 3, 0, "unknown_topology")
        machine.record_commit("cam-c", 4, 0, 8, "new")
        self.assertEqual(
            {"pending_ambiguous", "committed_new"},
            {item["state"] for item in machine.snapshot()},
        )

    def test_terminal_transition_is_rejected(self):
        machine = GlobalIDStateMachine()
        record = machine.record_pending("cam-a", 3, 0, "ambiguous")
        with self.assertRaises(ValueError):
            machine.advance(record, ObservationState.COMMITTED_NEW)


if __name__ == "__main__":
    unittest.main()
