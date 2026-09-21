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
        "frame_index": int(round(float(event_time) * 100)),
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
        commit_events = [
            item
            for item in diagnostics["unresolved_handoff_forensic_events"]
            if item["event"] == "pending_committed_existing_gid"
        ]
        self.assertEqual([2, 3], [item["committed_gid"] for item in commit_events])
        self.assertEqual(
            ["global-cross-camera", "global-cross-camera"],
            [item["assignment_source"] for item in commit_events],
        )

        for gid in (2, 3):
            identity = self.manager.identities[gid]
            self.assertFalse(identity["camera_presence"]["cam_2"]["active"])
            self.assertTrue(identity["camera_presence"]["cam_1"]["active"])
            self.assertEqual(1, len(identity["handoff_history"]))

    def test_pending_sample_forensics_include_candidates_gates_and_policy(self):
        blocked_identity = self._identity(
            embedding(0.0, 0.0, 1.0),
            100.75,
            14,
        )
        blocked_identity["box_wh"] = (4000, 8000)
        self.manager.identities[4] = blocked_identity
        self.manager.next_global_id = 5

        self.assertEqual(
            [None],
            self._batch([20], 101.0, batch_id="forensic-pending"),
        )

        events = self.manager.last_global_batch_diagnostics[
            "unresolved_handoff_forensic_events"
        ]
        self.assertEqual(1, len(events))
        sample = events[0]
        self.assertEqual("pending_sample", sample["event"])
        self.assertEqual(10100, sample["frame"])
        self.assertEqual("cam_1", sample["camera"])
        self.assertEqual(20, sample["local_id"])
        self.assertIsNone(sample["previous_gid"])
        self.assertIsNone(sample["expected_gid"])
        self.assertEqual([2, 3, 4], sample["candidate_gids"])
        self.assertEqual([2, 3], sample["viable_candidate_gids"])
        self.assertEqual(0.0, sample["unresolved_age_sec"])
        self.assertEqual(main.AMBIGUOUS_HANDOFF_MIN_SAMPLES, sample["min_samples"])
        self.assertEqual(
            main.AMBIGUOUS_HANDOFF_MIN_SOLO_EVENT_SEC,
            sample["min_solo_event_sec"],
        )
        self.assertEqual(main.AMBIGUOUS_HANDOFF_MAX_EVENT_SEC, sample["max_event_sec"])
        self.assertEqual(
            "ambiguous_top1_top2_merge_guard",
            sample["rejection_reason"],
        )
        self.assertEqual("unresolved-cross-camera", sample["assignment_source"])
        self.assertEqual(3, len(sample["candidates"]))
        candidates = {item["gid"]: item for item in sample["candidates"]}
        for candidate in (candidates[2], candidates[3]):
            self.assertIn("score", candidate)
            self.assertEqual(0.90, candidate["appearance"])
            self.assertEqual(0.0, candidate["motion"])
            self.assertTrue(candidate["hard_gate_passed"])
            self.assertIsNone(candidate["hard_gate_reason"])
        self.assertFalse(candidates[4]["hard_gate_passed"])
        self.assertEqual(
            "incompatible_box_size",
            candidates[4]["hard_gate_reason"],
        )

        state_trace = self.manager.identity_state_diagnostics()[
            "unresolved_handoff_forensic_trace"
        ]
        self.assertEqual(events, state_trace)

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

    def test_stable_temporal_top1_resolves_existing_gid_at_bounded_expiry(self):
        resolved = None
        for observation in range(56):
            event_time = 101.0 + (2.0 * observation / 55.0)
            row = detection(20, event_time)
            if observation < 45:
                row["border_clip_ratio"] = 0.5
            result = self.manager.assign_global_batch(
                {"cam_1": [row]},
                event_time=event_time,
                batch_id=f"stable-top1-{observation}",
            )["cam_1"][0]
            if observation < 55:
                self.assertIsNone(result)
            else:
                resolved = result

        self.assertEqual(3, resolved["gid"])
        self.assertEqual("global-cross-camera", resolved["source"])
        self.assertEqual(4, self.manager.next_global_id)
        self.assertEqual(
            "temporal_consistent_top1_at_window_end",
            self.manager.last_global_batch_diagnostics["rows"][0][
                "resolution_reason"
            ],
        )
        self.assertEqual({}, self.manager.unresolved_cross_camera_handoffs)
        forensic_events = self.manager.last_global_batch_diagnostics[
            "unresolved_handoff_forensic_events"
        ]
        self.assertEqual(
            ["pending_sample", "pending_committed_existing_gid"],
            [item["event"] for item in forensic_events],
        )
        committed = forensic_events[-1]
        self.assertEqual(3, committed["committed_gid"])
        self.assertGreaterEqual(committed["unresolved_age_sec"], 2.0)
        self.assertEqual(
            1.0,
            self.manager.last_global_batch_diagnostics[
                "temporal_consistency_decisions"
            ][0]["dominance_ratio"],
        )
        decision = self.manager.last_global_batch_diagnostics[
            "temporal_consistency_decisions"
        ][0]
        self.assertEqual(56, decision["observation_count"])
        self.assertEqual(11, decision["qualified_sample_count"])
        self.assertEqual({3: 56}, decision["top1_counts"])

        continued = self._batch([20], 103.05)[0]
        self.assertEqual(3, continued["gid"])
        self.assertEqual(4, self.manager.next_global_id)

    def test_oscillating_temporal_top1_does_not_force_merge(self):
        def alternating_score(gid, identity, camera, row, *args):
            top_gid = (
                3
                if float(row["event_time"]) in {101.0, 102.0}
                else 2
            )
            pair = self._crossed_pair_score(
                gid,
                identity,
                camera,
                row,
                *args,
            )
            pair["score"] = 0.80 if gid == top_gid else 0.77
            return pair

        with patch.object(
            self.manager,
            "_pair_score",
            side_effect=alternating_score,
        ):
            for event_time in (101.0, 101.5, 102.0):
                self.assertEqual([None], self._batch([20], event_time))
            expired = self._batch([20], 103.01)[0]

        self.assertEqual(4, expired["gid"])
        self.assertEqual("new", expired["source"])
        decision = self.manager.last_global_batch_diagnostics[
            "temporal_consistency_decisions"
        ][0]
        self.assertFalse(decision["eligible"])
        self.assertEqual(
            "top1_temporal_consistency_below_threshold",
            decision["reason"],
        )
        self.assertEqual({2: 2, 3: 2}, decision["top1_counts"])

    def test_temporal_resolution_rechecks_current_hard_gate(self):
        for event_time in (101.0, 101.5, 102.0):
            self.assertEqual([None], self._batch([20], event_time))

        blocked = detection(20, 103.01)
        blocked["box_wh"] = (4000, 8000)
        expired = self.manager.assign_global_batch(
            {"cam_1": [blocked]},
            event_time=103.01,
            batch_id="hard-gate-at-expiry",
        )["cam_1"][0]

        self.assertEqual(4, expired["gid"])
        self.assertEqual("new", expired["source"])
        decision = self.manager.last_global_batch_diagnostics[
            "temporal_consistency_decisions"
        ][0]
        self.assertFalse(decision["eligible"])
        self.assertEqual("incompatible_box_size", decision["reason"])

    def test_temporal_resolution_rejects_same_camera_ownership_conflict(self):
        for event_time in (101.0, 101.5, 102.0):
            self.assertEqual([None], self._batch([20], event_time))

        self.manager.identities[3]["camera_presence"]["cam_1"] = {
            "gid": 3,
            "camera": "cam_1",
            "local_track_id": 99,
            "first_seen_event_time": 102.5,
            "last_seen_event_time": 103.0,
            "active": True,
            "generation": 7,
            "assignment_source": "local-track-verified",
            "inactive_reason": None,
            "deactivated_event_time": None,
        }
        expired = self._batch([20], 103.01)[0]

        self.assertEqual(4, expired["gid"])
        self.assertEqual("new", expired["source"])
        decision = self.manager.last_global_batch_diagnostics[
            "temporal_consistency_decisions"
        ][0]
        self.assertFalse(decision["eligible"])
        self.assertEqual(
            "same_camera_presence_owned_by_other_local",
            decision["reason"],
        )

    def test_temporal_resolution_rejects_topology_conflict(self):
        for event_time in (101.0, 101.5, 102.0):
            self.assertEqual([None], self._batch([20], event_time))

        with patch.object(
            main,
            "topology_config",
            {"version": 2, "enforce": True, "transitions": []},
        ):
            expired = self._batch([20], 103.01)[0]

        self.assertEqual(4, expired["gid"])
        self.assertEqual("new", expired["source"])
        decision = self.manager.last_global_batch_diagnostics[
            "temporal_consistency_decisions"
        ][0]
        self.assertFalse(decision["eligible"])
        self.assertEqual(
            "topology_transition_not_allowed",
            decision["reason"],
        )

    def test_temporal_resolution_is_order_independent_with_three_cameras(self):
        def run(camera_order):
            manager = main.GlobalIdentityManager()
            third_identity = self._identity(
                embedding(0.0, 0.0, 1.0),
                100.25,
                40,
            )
            third_presence = third_identity["camera_presence"].pop("cam_2")
            third_presence.update({
                "gid": 4,
                "camera": "cam_3",
                "local_track_id": 40,
            })
            third_identity["last_cam"] = "cam_3"
            third_identity["camera_presence"]["cam_3"] = third_presence
            manager.identities = {
                2: self._identity(embedding(1.0, 0.0, 0.0), 100.0, 12),
                3: self._identity(embedding(0.0, 1.0, 0.0), 100.5, 13),
                4: third_identity,
            }
            manager.next_global_id = 5
            manager.local_to_global = {
                ("cam_2", 12): {
                    "gid": 2,
                    "last_seen": 100.0,
                    "generation": 7,
                },
                ("cam_3", 40): {
                    "gid": 4,
                    "last_seen": 100.25,
                    "generation": 7,
                },
            }

            def three_camera_score(gid, identity, camera, row, *args):
                if gid == 4:
                    return {
                        "gid": 4,
                        "score": 0.20,
                        "appearance": 0.20,
                        "quality_adjusted_appearance": 0.20,
                        "tracklet_quality": 1.0,
                        "motion": 0.0,
                        "map": 0.0,
                        "time": 1.0,
                        "cross_camera": True,
                        "source_type": "cross-camera",
                    }
                return self._crossed_pair_score(
                    gid,
                    identity,
                    camera,
                    row,
                    *args,
                )

            with patch.object(
                manager,
                "_pair_score",
                side_effect=three_camera_score,
            ):
                result = None
                for event_time in (101.0, 101.5, 102.0, 103.01):
                    detections_by_camera = {
                        "cam_1": [detection(20, event_time)],
                        "cam_2": [detection(
                            12,
                            event_time,
                            vector=embedding(1.0, 0.0, 0.0),
                        )],
                        "cam_3": [detection(
                            40,
                            event_time,
                            vector=embedding(0.0, 0.0, 1.0),
                        )],
                    }
                    camera_detections = {
                        camera: detections_by_camera[camera]
                        for camera in camera_order
                    }
                    result = manager.assign_global_batch(
                        camera_detections,
                        event_time=event_time,
                        batch_id=f"permutation-{camera_order}-{event_time}",
                    )["cam_1"][0]
            decision = manager.last_global_batch_diagnostics[
                "temporal_consistency_decisions"
            ][0]
            return result, manager.next_global_id, decision

        forward = run(("cam_1", "cam_2", "cam_3"))
        reverse = run(("cam_3", "cam_2", "cam_1"))

        for result, next_gid, decision in (forward, reverse):
            self.assertEqual(3, result["gid"])
            self.assertEqual("global-cross-camera", result["source"])
            self.assertEqual(5, next_gid)
            self.assertEqual(1.0, decision["dominance_ratio"])
        self.assertEqual(
            forward[2]["top1_counts"],
            reverse[2]["top1_counts"],
        )

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
