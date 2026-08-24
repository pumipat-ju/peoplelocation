import os
import unittest
from unittest.mock import patch

os.environ.setdefault("REID_ENABLED", "false")

from backend import main


TOPOLOGY = {
    "version": 1,
    "enforce": True,
    "transitions": [{
        "from_camera": "cam1", "to_camera": "cam2",
        "min_travel_sec": 2.0, "max_travel_sec": 12.0,
    }],
}


class TopologyTravelTimeTests(unittest.TestCase):

    def setUp(self):
        self.manager = main.GlobalIdentityManager()
        self.identity = {
            "last_cam": "cam1", "last_seen": 100.0,
        }

    def _gate(self, event_time, position=(25, 5)):
        return self.manager._topology_gate(
            self.identity, "cam2", {"map_pos": position}, event_time
        )

    def test_allowed_route_in_travel_window_passes(self):
        with patch.object(main, "topology_config", TOPOLOGY):
            self.assertEqual((True, "topology_allowed"), self._gate(105.0))

    def test_route_that_is_too_fast_is_hard_rejected(self):
        with patch.object(main, "topology_config", TOPOLOGY):
            self.assertEqual(
                (False, "travel_time_outside_window"), self._gate(100.5)
            )

    def test_camera_pair_without_transition_is_hard_rejected(self):
        with patch.object(main, "topology_config", TOPOLOGY):
            self.assertEqual(
                (False, "topology_transition_not_allowed"),
                self.manager._topology_gate(
                    self.identity, "cam3", {"map_pos": (25, 5)}, 105.0
                ),
            )


if __name__ == "__main__":
    unittest.main()
