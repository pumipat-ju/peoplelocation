import os
import unittest
from unittest.mock import mock_open, patch

import numpy as np

os.environ.setdefault("REID_ENABLED", "false")

from backend import main


TOPOLOGY = {
    "version": 2,
    "enforce": True,
    "transitions": [{
        "from_camera": "cam1", "to_camera": "cam2",
        "min_travel_time_sec": 2.0,
        "max_travel_time_sec": 12.0,
        "overlap_allowed": False,
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
                (False, "topology_travel_too_fast"), self._gate(100.5)
            )

    def test_route_that_is_too_slow_is_hard_rejected(self):
        with patch.object(main, "topology_config", TOPOLOGY):
            self.assertEqual(
                (False, "topology_travel_too_slow"), self._gate(113.0)
            )

    def test_camera_pair_without_transition_is_hard_rejected(self):
        with patch.object(main, "topology_config", TOPOLOGY):
            self.assertEqual(
                (False, "topology_transition_not_allowed"),
                self.manager._topology_gate(
                    self.identity, "cam3", {"map_pos": (25, 5)}, 105.0
                ),
            )

    def test_simultaneous_presence_requires_explicit_overlap_permission(self):
        with patch.object(main, "topology_config", TOPOLOGY):
            self.assertEqual(
                (False, "topology_overlap_not_allowed"),
                self._gate(100.0),
            )

        overlap_topology = {
            **TOPOLOGY,
            "transitions": [{
                **TOPOLOGY["transitions"][0],
                "overlap_allowed": True,
            }],
        }
        with patch.object(main, "topology_config", overlap_topology):
            self.assertEqual(
                (True, "topology_overlap_allowed"),
                self._gate(100.0),
            )

    def test_missing_maximum_disables_only_the_upper_time_gate(self):
        no_maximum = {
            **TOPOLOGY,
            "transitions": [{
                **TOPOLOGY["transitions"][0],
                "max_travel_time_sec": None,
            }],
        }
        with patch.object(main, "topology_config", no_maximum):
            self.assertEqual(
                (True, "topology_allowed"),
                self._gate(1000.0),
            )

    def test_topology_diagnostics_expose_the_rule_and_event_delta(self):
        with patch.object(main, "topology_config", TOPOLOGY):
            details = self.manager._topology_gate_details(
                self.identity,
                "cam2",
                {"map_pos": (25, 5)},
                105.0,
            )

        self.assertTrue(details["passed"])
        self.assertEqual("cam1", details["source_camera"])
        self.assertEqual("cam2", details["destination_camera"])
        self.assertEqual(5.0, details["event_time_delta_sec"])
        self.assertFalse(details["overlap_allowed"])
        self.assertEqual(TOPOLOGY["transitions"][0], details["topology_rule"])

    def test_disallowed_transition_is_rejected_before_appearance_scoring(self):
        embedding = main.l2_normalize(np.asarray([1.0, 0.0], dtype=np.float32))
        self.manager.identities[1] = {
            "embedding": embedding,
            "gallery": [embedding],
            "gallery_mature": True,
            "last_cam": "cam1",
            "last_seen": 100.0,
            "last_event_time": 100.0,
            "last_map_pos": None,
            "box_wh": (40, 60),
            "state": main.IDENTITY_ACTIVE,
            "state_updated_at": 100.0,
        }
        self.manager.next_global_id = 2
        detection = {
            "tid": 7,
            "emb": embedding,
            "box": (10, 10, 50, 70),
            "box_wh": (40, 60),
            "map_pos": None,
            "event_time": 105.0,
            "local_track_confirmed": True,
            "conf": 0.95,
            "crop_size": (40, 60),
            "blur_variance": 100.0,
            "border_clip_ratio": 0.0,
        }

        with (
            patch.object(main, "topology_config", TOPOLOGY),
            patch.object(
                self.manager,
                "_gallery_similarity",
                wraps=self.manager._gallery_similarity,
            ) as appearance_score,
        ):
            result = self.manager.assign_global_batch(
                {"cam3": [detection]},
                event_time=105.0,
            )["cam3"][0]

        self.assertNotEqual(1, result["gid"])
        appearance_score.assert_not_called()
        self.assertEqual(
            105.0,
            self.manager.identities[result["gid"]]["last_event_time"],
        )
        tracklet = self.manager.tracklets[("cam3", 7)]
        self.assertEqual(105.0, tracklet["last_event_time"])
        self.assertEqual(105.0, tracklet["samples"][0]["event_time"])
        failure = self.manager.last_global_batch_diagnostics["gate_failures"][0]
        self.assertEqual("topology_transition_not_allowed", failure["reason"])
        topology = self.manager.last_global_batch_diagnostics[
            "topology_gate_decisions"
        ][0]
        self.assertEqual("cam1", topology["source_camera"])
        self.assertEqual("cam3", topology["destination_camera"])


class TopologyValidationTests(unittest.TestCase):

    def test_legacy_travel_fields_are_preserved_in_version_two_contract(self):
        normalized = main.normalize_topology_config({
            "version": 1,
            "enforce": True,
            "transitions": [{
                "from_camera": "cam1",
                "to_camera": "cam2",
                "min_travel_sec": 1.5,
                "max_travel_sec": 9.0,
            }],
        })

        self.assertEqual(2, normalized["version"])
        self.assertEqual({
            "from_camera": "cam1",
            "to_camera": "cam2",
            "min_travel_time_sec": 1.5,
            "max_travel_time_sec": 9.0,
            "overlap_allowed": False,
        }, normalized["transitions"][0])

    def test_referenced_cameras_and_explicit_types_are_validated(self):
        with self.assertRaisesRegex(ValueError, "unknown camera"):
            main.normalize_topology_config(
                TOPOLOGY,
                known_cameras={"cam1"},
            )

        malformed = {
            **TOPOLOGY,
            "transitions": [{
                **TOPOLOGY["transitions"][0],
                "overlap_allowed": "false",
            }],
        }
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            main.normalize_topology_config(malformed)

    def test_malformed_persistent_config_fails_closed_and_visibly(self):
        malformed_json = """{
            "version": 2,
            "enforce": true,
            "transitions": [{
                "from_camera": "cam1",
                "to_camera": "cam2",
                "min_travel_time_sec": 5,
                "max_travel_time_sec": 2,
                "overlap_allowed": false
            }]
        }"""
        with (
            patch.object(main.os.path, "isfile", return_value=True),
            patch("builtins.open", mock_open(read_data=malformed_json)),
        ):
            config = main.load_topology_config()

        self.assertTrue(config["enforce"])
        self.assertEqual([], config["transitions"])
        self.assertIn("_validation_error", config)

        manager = main.GlobalIdentityManager()
        identity = {"last_cam": "cam1", "last_seen": 100.0}
        with patch.object(main, "topology_config", config):
            self.assertEqual(
                (False, "topology_config_invalid"),
                manager._topology_gate(identity, "cam2", {}, 105.0),
            )


if __name__ == "__main__":
    unittest.main()
