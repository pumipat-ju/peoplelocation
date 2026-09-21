import os
import unittest
from unittest.mock import Mock

import numpy as np

os.environ["IDENTITY_DB_PATH"] = ":memory:"
os.environ["REID_ENABLED"] = "false"

from backend import main


class GlobalMultiCameraBatchMatchingTests(unittest.TestCase):
    def setUp(self):
        self.manager = main.GlobalIdentityManager()
        self.manager.identities = {
            gid: {
                "state": main.IDENTITY_ACTIVE,
                "last_cam": "cam1",
                "last_seen": 99.0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "gallery": [np.array([1.0, 0.0], dtype=np.float32)],
                "box_wh": (40, 80),
                "last_map_pos": None,
            }
            for gid in (1, 2)
        }
        self.manager.cleanup = lambda *args, **kwargs: None
        self.manager._can_match = lambda *args: True
        self.manager._hard_gate_reason = lambda *args: None
        self.manager._accept_match = lambda *args: True
        self.manager._record_tracklet_sample = lambda *args: (False, "test")
        self.manager._commit_assignment = lambda gid, *args, **_kwargs: {
            "gid": gid, "score": args[-2], "source": args[-1]
        }
        self.manager._new_identity = (
            lambda *args, **_kwargs: self.fail("unexpected new identity")
        )

    @staticmethod
    def _det(track_id):
        return {
            "tid": track_id,
            "emb": np.array([1.0, 0.0], dtype=np.float32),
            "box": (0, 0, 40, 80),
            "box_wh": (40, 80),
            "map_pos": None,
            "overlap": False,
            "local_track_confirmed": True,
            "detector_confidence": 0.95,
            "crop_size": (40, 80),
            "blur_variance": 100.0,
            "border_clip_ratio": 0.0,
        }

    def _reservation_test_manager(self, gids):
        manager = main.GlobalIdentityManager()
        manager.identities = {
            gid: {
                "state": main.IDENTITY_ACTIVE,
                "last_cam": "cam1",
                "last_seen": 99.0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "gallery": [np.array([1.0, 0.0], dtype=np.float32)],
                "gallery_mature": True,
                "box_wh": (40, 80),
                "last_map_pos": None,
            }
            for gid in gids
        }
        manager.cleanup = lambda *args, **kwargs: None
        manager._hard_gate_reason = lambda *args: None
        manager._accept_match = lambda *args: True
        manager._trusted_assignment_claim = lambda *args, **kwargs: None
        manager._soft_local_continuity_context = lambda *args, **kwargs: None
        manager._record_tracklet_sample = lambda *args: (False, "test")
        manager._commit_assignment = lambda gid, *args, **kwargs: {
            "gid": gid,
            "score": args[-2],
            "source": args[-1],
        }
        return manager

    def test_hungarian_prevents_greedy_identity_conflict_within_camera(self):
        # The first row strongly prefers GID 1 but can use GID 2; the second
        # row can only use GID 1. Per-row greedy assignment would consume G1.
        scores = {
            ("cam1", 11, 1): 0.90,
            ("cam1", 11, 2): 0.80,
            ("cam1", 22, 1): 0.85,
            ("cam1", 22, 2): 0.10,
        }

        def pair_score(gid, _identity, cam_name, detection, *_args):
            score = scores[(cam_name, detection["tid"], gid)]
            return {
                "score": score,
                "appearance": score,
                "cross_camera": True,
            }

        self.manager._pair_score = pair_score
        result = self.manager.assign_global_batch({
            "cam1": [self._det(11), self._det(22)],
        }, event_time=100.0)

        self.assertEqual(2, result["cam1"][0]["gid"])
        self.assertEqual(1, result["cam1"][1]["gid"])
        self.assertEqual({1, 2}, set(self.manager.identities))

    def test_hysteresis_reservations_are_scoped_per_camera(self):
        expected = {"cam1": 1, "cam2": 1, "cam3": 3}
        camera_orders = (
            ("cam1", "cam2", "cam3"),
            ("cam3", "cam2", "cam1"),
        )

        for camera_order in camera_orders:
            with self.subTest(camera_order=camera_order):
                manager = self._reservation_test_manager((1, 2, 3))
                manager.local_to_global[("cam1", 11)] = {
                    "gid": 1,
                    "last_seen": 99.9,
                    "generation": None,
                }

                scores = {
                    ("cam1", 11, 1): 0.70,
                    ("cam1", 11, 2): 0.80,
                    ("cam1", 11, 3): 0.05,
                    ("cam2", 22, 1): 0.90,
                    ("cam2", 22, 2): 0.10,
                    ("cam2", 22, 3): 0.05,
                    ("cam3", 33, 1): 0.05,
                    ("cam3", 33, 2): 0.10,
                    ("cam3", 33, 3): 0.95,
                }

                def pair_score(gid, _identity, cam_name, detection, *_args):
                    score = scores[(cam_name, detection["tid"], gid)]
                    return {
                        "score": score,
                        "appearance": score,
                        "cross_camera": cam_name != "cam1",
                    }

                manager._pair_score = pair_score
                detections = {
                    "cam1": [self._det(11)],
                    "cam2": [self._det(22)],
                    "cam3": [self._det(33)],
                }
                result = manager.assign_global_batch(
                    {
                        camera: detections[camera]
                        for camera in camera_order
                    },
                    event_time=100.0,
                    batch_id="three-camera-reservation",
                )

                actual = {
                    camera: result[camera][0]["gid"]
                    for camera in detections
                }
                self.assertEqual(expected, actual)
                self.assertEqual(2, len(set(actual.values())))

    def test_fixed_claim_in_other_camera_blocks_non_overlap_handoff(self):
        camera_orders = (
            ("cam_2", "cam_1"),
            ("cam_1", "cam_2"),
        )

        for camera_order in camera_orders:
            with self.subTest(camera_order=camera_order):
                manager = self._reservation_test_manager((2, 3))
                for identity in manager.identities.values():
                    identity["last_cam"] = "cam_2"

                def trusted_claim(cam_name, detection, *_args, **_kwargs):
                    if cam_name == "cam_2" and detection["tid"] == 1:
                        return {
                            "gid": 2,
                            "score": 0.99,
                            "source": "local-track-verified",
                            "priority": 100,
                        }
                    return None

                scores = {2: 0.95, 3: 0.70}
                manager._trusted_assignment_claim = trusted_claim
                manager._pair_score = (
                    lambda gid, _identity, cam_name, detection, *_args: {
                        "score": scores[gid],
                        "appearance": scores[gid],
                        "cross_camera": cam_name == "cam_1",
                    }
                )

                detections = {
                    "cam_2": [self._det(1)],
                    "cam_1": [self._det(3)],
                }
                result = manager.assign_global_batch(
                    {
                        camera: detections[camera]
                        for camera in camera_order
                    },
                    event_time=100.0,
                    batch_id="cross-camera-fixed-claim-candidate",
                )

                self.assertEqual(2, result["cam_2"][0]["gid"])
                self.assertEqual(3, result["cam_1"][0]["gid"])
                cam1_row = next(
                    row
                    for row in manager.last_global_batch_diagnostics["rows"]
                    if row["camera"] == "cam_1" and row["track_id"] == 3
                )
                self.assertEqual([3], cam1_row["candidate_gids"])
                gid2_candidate = next(
                    candidate
                    for candidate in cam1_row["candidates"]
                    if candidate["gid"] == 2
                )
                self.assertFalse(gid2_candidate["hard_gate_passed"])
                self.assertEqual(
                    "trusted_active_source_ownership",
                    gid2_candidate["hard_gate_reason"],
                )
                self.assertIsNone(gid2_candidate["score"])

    def test_transition_recovery_cannot_override_valid_short_gap_mapping(self):
        manager = self._reservation_test_manager((2, 3))
        detection = self._det(3)
        detection["camera_generation"] = 0
        manager.local_to_global[("cam_1", 3)] = {
            "gid": 2,
            "last_seen": 99.867,
            "generation": 0,
        }
        manager._presence_allows_local_claim = lambda *args, **kwargs: True
        manager._transition_lost_candidates = Mock(return_value=[{
            "gid": 3,
            "tid": 1,
            "ts": 99.0,
            "lost_ts": 99.8,
            "box": (650, 150, 800, 420),
            "box_wh": (150, 270),
        }])
        manager._trusted_assignment_claim = lambda *args, **kwargs: {
            "gid": 2,
            "score": 0.95,
            "source": "local-track-verified",
            "priority": 2,
        }

        result = manager.assign_global_batch(
            {"cam_1": [detection]},
            event_time=100.0,
            batch_id="short-gap-valid-mapping",
        )

        self.assertEqual(2, result["cam_1"][0]["gid"])
        self.assertEqual(
            "local-track-verified",
            result["cam_1"][0]["source"],
        )
        manager._transition_lost_candidates.assert_not_called()

    def test_local_id_swap_result_is_independent_of_row_order(self):
        row_orders = ((20, 10), (10, 20))

        for row_order in row_orders:
            with self.subTest(row_order=row_order):
                manager = self._reservation_test_manager((1, 2))
                manager.local_to_global.update({
                    ("cam1", 10): {
                        "gid": 1,
                        "last_seen": 99.9,
                        "generation": None,
                    },
                    ("cam1", 20): {
                        "gid": 2,
                        "last_seen": 99.9,
                        "generation": None,
                    },
                })
                scores = {
                    (10, 1): 0.70,
                    (10, 2): 0.85,
                    (20, 1): 0.85,
                    (20, 2): 0.70,
                }
                manager._pair_score = (
                    lambda gid, _identity, _camera, detection, *_args: {
                        "score": scores[(detection["tid"], gid)],
                        "appearance": scores[(detection["tid"], gid)],
                        "cross_camera": False,
                    }
                )

                detections = [self._det(track_id) for track_id in row_order]
                result = manager.assign_global_batch(
                    {"cam1": detections},
                    event_time=100.0,
                    batch_id="local-id-swap-row-order",
                )["cam1"]
                by_track_id = {
                    detection["tid"]: assignment["gid"]
                    for detection, assignment in zip(detections, result)
                }

                self.assertEqual({10: 2, 20: 1}, by_track_id)
                self.assertEqual(2, len(set(by_track_id.values())))

    def test_gallery_admission_diagnostics_are_exposed_per_assignment(self):
        self.manager.identities = {1: self.manager.identities[1]}
        self.manager.identities[1].update({
            "gallery_mature": True,
            "gallery_diagnostics": {
                "tracklet_sample_count": 3,
            },
        })
        self.manager._pair_score = lambda *_args: {
            "score": 0.95,
            "cross_camera": False,
        }

        self.manager.assign_global_batch(
            {"cam1": [self._det(11)]},
            event_time=100.0,
            batch_id="gallery-diagnostics",
        )

        diagnostics = self.manager.last_global_batch_diagnostics
        expected = {
            "gallery_update_accepted": False,
            "gallery_rejection_reason": "test",
            "gallery_mature": True,
            "tracklet_sample_count": 3,
            "gallery_size": 1,
        }
        for collection in ("rows", "assignments"):
            with self.subTest(collection=collection):
                entry = diagnostics[collection][0]
                self.assertEqual(
                    expected,
                    {key: entry[key] for key in expected},
                )

    def test_provisional_bootstrap_cannot_cross_camera(self):
        identity = self.manager.identities[1]
        identity.update({
            "state": main.IDENTITY_PROVISIONAL,
            "last_cam": "cam1",
            "gallery": [],
            "gallery_mature": False,
        })
        self.manager.identities = {1: identity}
        self.manager._pair_score = lambda *_args: {
            "score": 1.0,
            "cross_camera": True,
        }
        self.manager._new_identity = lambda *_args, **_kwargs: {
            "gid": 99,
            "score": 1.0,
            "source": "new",
        }

        result = self.manager.assign_global_batch(
            {"cam2": [self._det(22)]},
            event_time=100.0,
            batch_id="provisional-cross-camera",
        )

        self.assertEqual(99, result["cam2"][0]["gid"])
        self.assertEqual("new", result["cam2"][0]["source"])
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual([], diagnostics["candidate_gids"])
        self.assertEqual([], diagnostics["rows"][0]["candidate_gids"])
        self.assertEqual(
            "no_eligible_candidate",
            diagnostics["rows"][0]["new_identity_reason"],
        )

    def test_near_equal_candidates_are_deferred_instead_of_forced(self):
        self.manager._pair_score = lambda gid, *_args: {
            "score": 0.80 if gid == 1 else 0.76,
            "cross_camera": True,
        }
        self.manager._new_identity = lambda *_args, **_kwargs: {
            "gid": 99, "score": 1.0, "source": "new",
        }

        next_gid_before = self.manager.next_global_id
        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]}, event_time=100.0
        )

        self.assertIsNone(result["cam1"][0])
        self.assertEqual(next_gid_before, self.manager.next_global_id)
        self.assertEqual({1, 2}, set(self.manager.identities))
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual(
            "ambiguous_top1_top2_merge_guard",
            diagnostics["rejections"][0]["reason"],
        )
        self.assertAlmostEqual(
            0.04,
            diagnostics["rejections"][0]["top1_top2_margin"],
        )
        self.assertEqual(
            "ambiguous_cross_camera_handoff_pending",
            diagnostics["rows"][0]["pending_reason"],
        )
        self.assertIsNone(diagnostics["rows"][0]["new_identity_reason"])
        self.assertEqual("pending", diagnostics["rows"][0]["assignment_state"])

    def test_hard_gated_runner_up_is_not_an_ambiguity_candidate(self):
        first_identity = self.manager.identities[1]
        self.manager._hard_gate_reason = (
            lambda identity, *_args: (
                None
                if identity is first_identity
                else "incompatible_location"
            )
        )
        self.manager._pair_score = lambda gid, *_args: {
            "score": 0.80,
            "cross_camera": False,
        }

        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]},
            event_time=100.0,
            batch_id="hard-gated-runner-up",
        )

        self.assertEqual(1, result["cam1"][0]["gid"])
        diagnostics = self.manager.last_global_batch_diagnostics
        trace = diagnostics["rows"][0]
        self.assertEqual([1], trace["candidate_gids"])
        self.assertEqual(
            [1],
            [
                item["gid"]
                for item in trace["candidates"]
                if item["hard_gate_passed"]
            ],
        )
        self.assertIsNone(trace["top1_top2_margin"])
        self.assertEqual(
            [{
                "camera": "cam1",
                "track_id": 11,
                "gid": 2,
                "reason": "incompatible_location",
                "row": 0,
            }],
            diagnostics["gate_failures"],
        )
        self.assertFalse(
            any(
                rejection["reason"] == "ambiguous_top1_top2"
                for rejection in diagnostics["rejections"]
            )
        )

    def test_ambiguous_merge_guard_cannot_fall_through_to_same_camera_cache(self):
        self.manager._pair_score = lambda gid, *_args: {
            "score": 0.80 if gid == 1 else 0.76,
            "cross_camera": False,
        }
        self.manager.recent_same_cam = [{
            "gid": 1,
            "cam_name": "cam1",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "map_pos": None,
            "box_wh": (40, 80),
            "ts": 99.9,
        }]
        self.manager._new_identity = lambda *_args, **_kwargs: {
            "gid": 99,
            "score": 1.0,
            "source": "new",
        }

        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]},
            event_time=100.0,
            batch_id="ambiguous-cache-bypass",
        )

        self.assertEqual(99, result["cam1"][0]["gid"])
        self.assertEqual("new", result["cam1"][0]["source"])
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual(
            "ambiguous_top1_top2_merge_guard",
            diagnostics["rejections"][0]["reason"],
        )
        self.assertEqual(
            "ambiguous_top1_top2_merge_guard",
            diagnostics["rows"][0]["new_identity_reason"],
        )

    def test_same_camera_acceptance_failure_can_use_recent_cache(self):
        self.manager.identities = {1: self.manager.identities[1]}
        candidate = np.array([
            0.44,
            np.sqrt(1.0 - (0.44 ** 2)),
        ], dtype=np.float32)
        self.manager.identities[1].update({
            "embedding": candidate.copy(),
            "gallery": [candidate.copy()],
            "gallery_mature": True,
        })
        self.manager._pair_score = lambda *_args: {
            "score": 0.20,
            "appearance": 0.44,
            "cross_camera": False,
        }
        self.manager._accept_match = lambda *_args: False
        self.manager.recent_same_cam = [{
            "gid": 1,
            "cam_name": "cam1",
            "embedding": candidate.copy(),
            "map_pos": None,
            "box_wh": (40, 80),
            "ts": 99.9,
        }]

        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]},
            event_time=100.0,
            batch_id="same-camera-cache-recovery",
        )

        self.assertEqual(1, result["cam1"][0]["gid"])
        self.assertEqual("same-cam-cache", result["cam1"][0]["source"])
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual(
            "acceptance_threshold",
            diagnostics["rejections"][0]["reason"],
        )
        self.assertIsNone(diagnostics["rows"][0]["new_identity_reason"])

    def test_ambiguous_acceptance_failures_cannot_use_recent_cache(self):
        self.manager._pair_score = lambda gid, *_args: {
            "score": 0.44 if gid == 1 else 0.43,
            "appearance": 0.44 if gid == 1 else 0.43,
            "cross_camera": False,
        }
        self.manager._accept_match = lambda *_args: False
        self.manager.recent_same_cam = [
            {
                "gid": gid,
                "cam_name": "cam1",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "map_pos": None,
                "box_wh": (40, 80),
                "ts": 99.9,
            }
            for gid in (1, 2)
        ]
        self.manager._new_identity = lambda *_args, **_kwargs: {
            "gid": 99,
            "score": 1.0,
            "source": "new",
        }

        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]},
            event_time=100.0,
            batch_id="ambiguous-acceptance-cache-blocked",
        )

        self.assertEqual(99, result["cam1"][0]["gid"])
        self.assertEqual("new", result["cam1"][0]["source"])
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual(
            "ambiguous_top1_top2_merge_guard",
            diagnostics["rejections"][0]["reason"],
        )

    def test_hungarian_omitted_ambiguous_row_cannot_use_recent_cache(self):
        scores = {
            (11, 1): (0.90, True),
            (11, 2): (0.10, True),
            (12, 1): (0.10, True),
            (12, 2): (0.90, True),
            (13, 1): (0.44, False),
            (13, 2): (0.43, False),
        }

        def pair_score(gid, _identity, _camera, detection, *_args):
            score, cross_camera = scores[(detection["tid"], gid)]
            return {
                "score": score,
                "appearance": score,
                "cross_camera": cross_camera,
            }

        self.manager._pair_score = pair_score
        self.manager._accept_match = lambda *_args: False
        self.manager.recent_same_cam = [
            {
                "gid": gid,
                "cam_name": "cam1",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "map_pos": None,
                "box_wh": (40, 80),
                "ts": 99.9,
            }
            for gid in (1, 2)
        ]
        new_gids = iter((90, 91, 92))
        self.manager._new_identity = lambda *_args, **_kwargs: {
            "gid": next(new_gids),
            "score": 1.0,
            "source": "new",
        }

        result = self.manager.assign_global_batch(
            {
                "cam1": [
                    self._det(11),
                    self._det(12),
                    self._det(13),
                ]
            },
            event_time=100.0,
            batch_id="hungarian-omitted-ambiguity",
        )

        self.assertNotIn(result["cam1"][2]["gid"], {1, 2})
        self.assertEqual("new", result["cam1"][2]["source"])
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual(
            "ambiguous_top1_top2_merge_guard",
            diagnostics["rows"][2]["new_identity_reason"],
        )

    def test_same_camera_cache_does_not_bypass_cross_camera_acceptance(self):
        self.manager.identities = {1: self.manager.identities[1]}
        self.manager.identities[1]["last_cam"] = "cam2"
        self.manager._pair_score = lambda *_args: {
            "score": 0.20,
            "appearance": 1.0,
            "cross_camera": True,
        }
        self.manager._accept_match = lambda *_args: False
        self.manager.recent_same_cam = [{
            "gid": 1,
            "cam_name": "cam1",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "map_pos": None,
            "box_wh": (40, 80),
            "ts": 99.9,
        }]
        self.manager._new_identity = lambda *_args, **_kwargs: {
            "gid": 99,
            "score": 1.0,
            "source": "new",
        }

        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]},
            event_time=100.0,
            batch_id="cross-camera-cache-blocked",
        )

        self.assertEqual(99, result["cam1"][0]["gid"])
        self.assertEqual("new", result["cam1"][0]["source"])

    def test_same_camera_cache_cannot_bypass_hard_location_gate(self):
        self.manager.identities = {1: self.manager.identities[1]}
        self.manager._hard_gate_reason = (
            lambda *_args: "incompatible_location"
        )
        self.manager.recent_same_cam = [{
            "gid": 1,
            "cam_name": "cam1",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "map_pos": (0, 0),
            "box_wh": (40, 80),
            "ts": 99.9,
        }]
        self.manager._new_identity = lambda *_args, **_kwargs: {
            "gid": 99,
            "score": 1.0,
            "source": "new",
        }
        detection = self._det(11)
        detection["map_pos"] = (10000, 10000)

        result = self.manager.assign_global_batch(
            {"cam1": [detection]},
            event_time=100.0,
            batch_id="hard-gated-cache-bypass",
        )

        self.assertEqual(99, result["cam1"][0]["gid"])
        self.assertEqual("new", result["cam1"][0]["source"])
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual([], diagnostics["rows"][0]["candidate_gids"])
        self.assertEqual(
            "incompatible_location",
            diagnostics["gate_failures"][0]["reason"],
        )
        self.assertEqual(
            "all_candidates_hard_gated",
            diagnostics["rows"][0]["new_identity_reason"],
        )


if __name__ == "__main__":
    unittest.main()
