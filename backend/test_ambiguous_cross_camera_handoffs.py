import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ["IDENTITY_DB_PATH"] = ":memory:"
os.environ["REID_ENABLED"] = "false"

from backend import main


def embedding(*values):
    return main.l2_normalize(np.asarray(values, dtype=np.float32))


def detection(track_id, event_time, generation=7, vector=None):
    return {
        "tid": track_id,
        "emb": (
            embedding(1.0, 1.0, 0.0)
            if vector is None
            else vector.copy()
        ),
        "box": (10, 10, 50, 90),
        "box_wh": (40, 80),
        "map_pos": None,
        "overlap": False,
        "local_track_confirmed": True,
        "detector_confidence": 0.95,
        "crop_size": (40, 80),
        "blur_variance": 100.0,
        "border_clip_ratio": 0.0,
        "event_time": float(event_time),
        "coordinator_generation": generation,
        "camera_generation": 3,
    }


class AmbiguousCrossCameraHandoffTests(unittest.TestCase):
    def setUp(self):
        self.manager = main.GlobalIdentityManager()
        self.manager.identities = {
            2: self._identity(embedding(1.0, 0.0, 0.0), 100.0, 12),
            3: self._identity(embedding(0.0, 1.0, 0.0), 100.5, 13),
        }
        self.manager.next_global_id = 4
        self.topology_patch = patch.object(
            main,
            "topology_config",
            {"version": 2, "enforce": False, "transitions": []},
        )
        self.topology_patch.start()
        self.addCleanup(self.topology_patch.stop)
        self.score_patch = patch.object(
            self.manager,
            "_pair_score",
            side_effect=self._crossed_pair_score,
        )
        self.score_patch.start()
        self.addCleanup(self.score_patch.stop)

    @staticmethod
    def _identity(vector, event_time, local_id):
        return {
            "state": main.IDENTITY_ACTIVE,
            "state_updated_at": float(event_time),
            "state_reason": "test_mature_identity",
            "last_cam": "cam_2",
            "last_seen": float(event_time),
            "last_event_time": float(event_time),
            "embedding": vector.copy(),
            "gallery": [vector.copy()],
            "gallery_mature": True,
            "box_wh": (40, 80),
            "last_map_pos": None,
            "camera_presence": {
                "cam_2": {
                    "gid": 2 if local_id == 12 else 3,
                    "camera": "cam_2",
                    "local_track_id": local_id,
                    "first_seen_event_time": float(event_time - 1.0),
                    "last_seen_event_time": float(event_time),
                    "active": True,
                    "generation": 7,
                    "assignment_source": "local-track-verified",
                }
            },
            "handoff_history": [],
        }

    @staticmethod
    def _crossed_pair_score(gid, _identity, _camera, row, *_args):
        scores = {
            (20, 2): 0.77,
            (20, 3): 0.80,
            (30, 2): 0.81,
            (30, 3): 0.78,
        }
        return {
            "gid": gid,
            "score": scores[(row["tid"], gid)],
            "appearance": 0.90,
            "quality_adjusted_appearance": 0.90,
            "tracklet_quality": 1.0,
            "motion": 0.0,
            "map": 0.0,
            "time": 1.0,
            "cross_camera": True,
            "source_type": "cross-camera",
        }

    def _batch(self, track_ids, event_time, generation=7, batch_id=None):
        return self.manager.assign_global_batch(
            {
                "cam_1": [
                    detection(track_id, event_time, generation=generation)
                    for track_id in track_ids
                ]
            },
            event_time=event_time,
            batch_id=batch_id or f"batch-{event_time:.2f}",
        )["cam_1"]

    def test_crossed_arrivals_defer_then_resolve_jointly_in_event_order(self):
        galleries_before = {
            gid: [item.copy() for item in identity["gallery"]]
            for gid, identity in self.manager.identities.items()
        }

        first = self._batch([20], 101.00, batch_id="first-ambiguous")
        self.assertEqual([None], first)
        self.assertEqual(4, self.manager.next_global_id)
        self.assertEqual({}, self.manager.local_to_global)
        for gid, expected_gallery in galleries_before.items():
            self.assertEqual(1, len(self.manager.identities[gid]["gallery"]))
            np.testing.assert_array_equal(
                expected_gallery[0],
                self.manager.identities[gid]["gallery"][0],
            )
            self.assertTrue(
                self.manager.identities[gid]["camera_presence"]["cam_2"][
                    "active"
                ]
            )
            self.assertEqual([], self.manager.identities[gid]["handoff_history"])

        self._batch([20], 101.04)
        self._batch([20], 101.08)
        self._batch([20, 30], 102.20)
        self._batch([20, 30], 102.24)
        resolved = self._batch(
            [20, 30],
            102.28,
            batch_id="joint-event-order-resolution",
        )

        self.assertEqual([2, 3], [item["gid"] for item in resolved])
        self.assertEqual(
            ["global-cross-camera", "global-cross-camera"],
            [item["source"] for item in resolved],
        )
        self.assertEqual(4, self.manager.next_global_id)
        self.assertEqual({}, self.manager.unresolved_cross_camera_handoffs)
        self.assertEqual(
            {2, 3},
            {
                mapping["gid"]
                for mapping in self.manager.local_to_global.values()
            },
        )

        diagnostics = self.manager.last_global_batch_diagnostics
        self.assertEqual(
            [
                "event_order_joint_one_to_one",
                "event_order_joint_one_to_one",
            ],
            [row["resolution_reason"] for row in diagnostics["rows"]],
        )
        self.assertEqual(
            [
                "order_preserving_joint_assignment",
                "order_preserving_joint_assignment",
            ],
            [
                row["unresolved_handoff"]["event_order_result"]
                for row in diagnostics["rows"]
            ],
        )
        self.assertEqual(2, len(diagnostics["selected"]))

        for gid in (2, 3):
            identity = self.manager.identities[gid]
            self.assertFalse(identity["camera_presence"]["cam_2"]["active"])
            self.assertTrue(identity["camera_presence"]["cam_1"]["active"])
            self.assertEqual(1, len(identity["handoff_history"]))

    def test_unresolved_samples_and_records_are_bounded_without_gid_allocation(self):
        for frame_index in range(30):
            result = self._batch(
                [20],
                101.0 + (frame_index * 0.04),
                batch_id=f"bounded-{frame_index}",
            )
            self.assertEqual([None], result)

        self.assertEqual(4, self.manager.next_global_id)
        self.assertEqual(1, len(self.manager.unresolved_cross_camera_handoffs))
        record = next(iter(self.manager.unresolved_cross_camera_handoffs.values()))
        self.assertEqual(main.AMBIGUOUS_HANDOFF_MAX_SAMPLES, record["sample_count"])
        self.assertLessEqual(
            len(record["samples"]),
            main.AMBIGUOUS_HANDOFF_MAX_SAMPLES,
        )
        status_record = self.manager.identity_state_diagnostics()[
            "unresolved_cross_camera_handoffs"
        ][0]
        self.assertNotIn("samples", status_record)
        self.assertNotIn("emb", status_record)

    def test_later_scoring_uses_quality_approved_aggregate_prototype(self):
        first_vector = embedding(1.0, 0.0, 1.0)
        second_vector = embedding(0.0, 1.0, 1.0)
        first_row = detection(20, 101.0, vector=first_vector)
        second_row = detection(20, 101.04, vector=second_vector)

        first = self.manager.assign_global_batch(
            {"cam_1": [first_row]},
            event_time=101.0,
            batch_id="aggregate-first",
        )["cam_1"]
        self.assertEqual([None], first)

        observed_embeddings = []

        def recording_score(gid, identity, camera, row, *args):
            observed_embeddings.append(row["emb"].copy())
            return self._crossed_pair_score(
                gid,
                identity,
                camera,
                row,
                *args,
            )

        with patch.object(
            self.manager,
            "_pair_score",
            side_effect=recording_score,
        ):
            second = self.manager.assign_global_batch(
                {"cam_1": [second_row]},
                event_time=101.04,
                batch_id="aggregate-second",
            )["cam_1"]

        self.assertEqual([None], second)
        expected = main.l2_normalize(first_vector + second_vector)
        self.assertEqual(2, len(observed_embeddings))
        for observed in observed_embeddings:
            np.testing.assert_allclose(expected, observed, atol=1e-6)

    def test_bounded_expiry_allocates_one_gid_then_local_continuity_reuses_it(self):
        self.assertEqual([None], self._batch([20], 101.0))
        self.assertEqual([None], self._batch([20], 101.5))
        self.assertEqual([None], self._batch([20], 102.0))

        expired = self._batch(
            [20],
            103.01,
            batch_id="bounded-expiry",
        )[0]
        self.assertEqual(4, expired["gid"])
        self.assertEqual("new", expired["source"])
        self.assertEqual(5, self.manager.next_global_id)
        self.assertEqual(
            "ambiguous_handoff_window_exhausted",
            self.manager.last_global_batch_diagnostics["rows"][0][
                "new_identity_reason"
            ],
        )
        self.assertEqual({}, self.manager.unresolved_cross_camera_handoffs)

        continued = self._batch([20], 103.05)[0]
        self.assertEqual(4, continued["gid"])
        self.assertEqual(5, self.manager.next_global_id)

    def test_camera_reset_discards_old_generation_pending_evidence(self):
        self.assertEqual([None], self._batch([20], 101.0, generation=7))
        old_key = next(iter(self.manager.unresolved_cross_camera_handoffs))
        self.assertEqual(7, old_key[2])

        cleanup = self.manager.reset_camera_local_state("cam_1")
        self.assertEqual(1, cleanup["unresolved_handoffs_removed"])
        self.assertEqual({}, self.manager.unresolved_cross_camera_handoffs)

        self.assertEqual([None], self._batch([20], 101.04, generation=8))
        new_key = next(iter(self.manager.unresolved_cross_camera_handoffs))
        self.assertEqual(8, new_key[2])
        self.assertNotEqual(old_key, new_key)
        self.assertEqual({}, self.manager.local_to_global)

    def test_coordinator_epoch_invalidation_discards_pending_evidence(self):
        coordinator = main.GlobalAssignmentCoordinator(
            lambda: self.manager,
            window_sec=60.0,
        )
        self.addCleanup(coordinator.stop)

        preview = coordinator.submit(
            "cam_1",
            [detection(20, 101.0, generation=0)],
            event_time=101.0,
        )
        self.assertEqual([None], preview)
        self.assertTrue(coordinator.flush())
        self.assertEqual(1, len(self.manager.unresolved_cross_camera_handoffs))

        coordinator.discard_camera("cam_1")
        self.assertEqual({}, self.manager.unresolved_cross_camera_handoffs)
        self.assertEqual(1, coordinator.camera_epochs["cam_1"])

    def test_processing_wall_time_does_not_expire_event_time_window(self):
        with patch.object(main.time, "time", return_value=10000.0):
            self.assertEqual([None], self._batch([20], 101.0))
        with patch.object(main.time, "time", return_value=20000.0):
            self.assertEqual([None], self._batch([20], 101.04))

        record = next(iter(self.manager.unresolved_cross_camera_handoffs.values()))
        self.assertAlmostEqual(0.04, record["last_event_time"] - 101.0)
        self.assertEqual(4, self.manager.next_global_id)

        self.assertEqual([None], self._batch([20], 100.5))
        record = next(iter(self.manager.unresolved_cross_camera_handoffs.values()))
        self.assertEqual(2, record["sample_count"])
        self.assertEqual("event_time_regression", record["last_quality_rejection_reason"])


if __name__ == "__main__":
    unittest.main()
