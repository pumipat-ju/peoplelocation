import os
import unittest

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
        self.manager._commit_assignment = lambda gid, *args: {
            "gid": gid, "score": args[-2], "source": args[-1]
        }
        self.manager._new_identity = lambda *args: self.fail("unexpected new identity")

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
        }

    def test_global_hungarian_prevents_per_camera_greedy_identity_conflict(self):
        # cam1 strongly prefers GID 1 but can use GID 2; cam2 can only use
        # GID 1.  Per-camera assignment of cam1 first would consume GID 1.
        scores = {
            ("cam1", 11, 1): 0.90,
            ("cam1", 11, 2): 0.80,
            ("cam2", 22, 1): 0.85,
            ("cam2", 22, 2): 0.10,
        }

        def pair_score(gid, _identity, cam_name, detection, *_args):
            return {"score": scores[(cam_name, detection["tid"], gid)],
                    "cross_camera": True}

        self.manager._pair_score = pair_score
        result = self.manager.assign_global_batch({
            "cam1": [self._det(11)], "cam2": [self._det(22)],
        }, event_time=100.0)

        self.assertEqual(2, result["cam1"][0]["gid"])
        self.assertEqual(1, result["cam2"][0]["gid"])
        self.assertEqual({1, 2}, set(self.manager.identities))

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
        self.manager._new_identity = lambda *_args: {
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
        self.manager._new_identity = lambda *_args: {
            "gid": 99, "score": 1.0, "source": "new",
        }

        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]}, event_time=100.0
        )

        self.assertEqual(99, result["cam1"][0]["gid"])
        self.assertEqual("new", result["cam1"][0]["source"])
        self.assertEqual({1, 2}, set(self.manager.identities))
        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual(
            "ambiguous_top1_top2",
            diagnostics["rejections"][0]["reason"],
        )
        self.assertAlmostEqual(
            0.04,
            diagnostics["rejections"][0]["top1_top2_margin"],
        )
        self.assertEqual(
            "ambiguous_top1_top2",
            diagnostics["rows"][0]["new_identity_reason"],
        )

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

    def test_ambiguous_row_cannot_fall_through_to_same_camera_cache(self):
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
        self.manager._new_identity = lambda *_args: {
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
            "ambiguous_top1_top2",
            diagnostics["rejections"][0]["reason"],
        )
        self.assertEqual(
            "ambiguous_top1_top2",
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
        self.manager._new_identity = lambda *_args: {
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
            "ambiguous_top1_top2",
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
        self.manager._new_identity = lambda *_args: {
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
            "ambiguous_top1_top2",
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
        self.manager._new_identity = lambda *_args: {
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
        self.manager._new_identity = lambda *_args: {
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
