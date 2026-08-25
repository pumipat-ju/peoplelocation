import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

# Importing the production module creates its application-level identity store.
# Keep this test process isolated from any developer/runtime SQLite database.
os.environ["IDENTITY_DB_PATH"] = ":memory:"
os.environ["REID_ENABLED"] = "false"

from backend import main


def embedding(*values):
    return main.l2_normalize(
        np.asarray(values, dtype=np.float32)
    )


def detection(track_id, vector, event_time=None, overlap=False, forced_gid=None):
    row = {
        "tid": track_id,
        "emb": vector,
        "box": (10, 10, 50, 70),
        "box_wh": (40, 60),
        "map_pos": None,
        "overlap": overlap,
        "forced_gid": forced_gid,
        "local_track_confirmed": True,
        "conf": 0.95,
    }
    if event_time is not None:
        row["event_time"] = event_time
    return row


class RecordingIdentityManager:
    def __init__(self, assignment_delay=0.0):
        self.lock = threading.RLock()
        self.assignment_delay = float(assignment_delay)
        self.calls = []
        self.completed = threading.Event()
        self.last_global_batch_diagnostics = None

    @staticmethod
    def preview_trusted_assignments(
        _cam_name,
        detections,
        event_time=None,
        blocking=True,
    ):
        return [None for _ in detections]

    def assign_global_batch(self, camera_detections, **kwargs):
        if self.assignment_delay:
            time.sleep(self.assignment_delay)
        self.calls.append((camera_detections, kwargs))
        self.last_global_batch_diagnostics = {
            "batch_id": kwargs["batch_id"],
            "cameras": sorted(camera_detections),
        }
        self.completed.set()
        return {
            camera: [
                {"gid": index + 1, "score": 1.0, "source": "new"}
                for index, _ in enumerate(rows)
            ]
            for camera, rows in camera_detections.items()
        }


class TensorLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def int(self):
        self.value = self.value.astype(int)
        return self

    def tolist(self):
        return self.value.tolist()


class OnePersonTrackingModel:
    def __init__(self):
        self.calls = []
        self.predictor = SimpleNamespace(trackers=[object()])

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=TensorLike([[10, 10, 50, 70]]),
            id=TensorLike([7]),
            conf=TensorLike([0.95]),
        )
        return [SimpleNamespace(boxes=boxes)]


def camera_data(source_type, tracking_model):
    return {
        "url": "synthetic",
        "source_type": source_type,
        "loop_video": False,
        "processor": None,
        "src_pts": None,
        "dst_pts": None,
        "last_frame": None,
        "prev_assignments": [],
        **main.new_camera_tracker_context(),
        "tracking_model": tracking_model,
    }


class GlobalAssignmentCoordinatorTests(unittest.TestCase):
    def test_same_local_track_keeps_gid_while_next_batch_is_pending(self):
        manager = main.GlobalIdentityManager()
        coordinator = main.GlobalAssignmentCoordinator(
            lambda: manager,
            window_sec=10.0,
        )
        person = embedding(1, 0, 0)
        try:
            coordinator.submit(
                "A",
                [detection(5, person, event_time=300.00)],
                event_time=300.00,
            )
            coordinator.flush()
            established_gid = manager.local_to_global[("A", 5)]["gid"]

            preview = coordinator.submit(
                "A",
                [detection(5, person, event_time=300.04)],
                event_time=300.04,
            )

            self.assertEqual(established_gid, preview[0]["gid"])
            self.assertEqual(
                established_gid,
                manager.local_to_global[("A", 5)]["gid"],
            )
            self.assertEqual(
                "pending",
                coordinator.status()["pending_observations"][0][
                    "assignment_state"
                ],
            )
            coordinator.flush()
            self.assertEqual(
                established_gid,
                manager.local_to_global[("A", 5)]["gid"],
            )
        finally:
            coordinator.stop()

    def test_generation_reset_discards_detached_stale_batch(self):
        manager = RecordingIdentityManager()
        coordinator = main.GlobalAssignmentCoordinator(
            lambda: manager,
            window_sec=10.0,
        )
        try:
            coordinator.submit(
                "A",
                [detection(5, embedding(1, 0), event_time=400.0)],
                event_time=400.0,
            )
            with coordinator.lock:
                stale_batch = coordinator._take_pending_locked()

            coordinator.discard_camera("A")
            coordinator._execute_batch(stale_batch)

            self.assertEqual([], manager.calls)
        finally:
            coordinator.stop()

    def test_event_window_rollover_does_not_block_submitter(self):
        manager = RecordingIdentityManager(assignment_delay=0.20)
        coordinator = main.GlobalAssignmentCoordinator(
            lambda: manager,
            window_sec=1.0,
        )
        try:
            coordinator.submit(
                "A",
                [detection(1, embedding(1, 0), event_time=100.0)],
                event_time=100.0,
            )
            started = time.monotonic()
            coordinator.submit(
                "A",
                [detection(1, embedding(1, 0), event_time=102.0)],
                event_time=102.0,
            )

            self.assertLess(time.monotonic() - started, 0.05)
            self.assertTrue(manager.completed.wait(timeout=1.0))
        finally:
            coordinator.stop()

    def test_two_cameras_inside_window_reach_one_global_batch(self):
        manager = RecordingIdentityManager()
        coordinator = main.GlobalAssignmentCoordinator(
            lambda: manager,
            window_sec=0.10,
        )
        try:
            started = time.monotonic()
            result_a = coordinator.submit(
                "A",
                [detection(1, embedding(1, 0), event_time=100.00)],
                event_time=100.00,
            )
            result_b = coordinator.submit(
                "B",
                [detection(2, embedding(0, 1), event_time=100.04)],
                event_time=100.04,
            )
            submit_duration = time.monotonic() - started

            self.assertEqual([None], result_a)
            self.assertEqual([None], result_b)
            self.assertLess(submit_duration, 0.08)
            self.assertTrue(manager.completed.wait(timeout=1.0))
            self.assertEqual(1, len(manager.calls))
            rows, kwargs = manager.calls[0]
            self.assertEqual({"A", "B"}, set(rows))
            self.assertEqual(100.00, rows["A"][0]["event_time"])
            self.assertEqual(100.04, rows["B"][0]["event_time"])
            self.assertEqual(0.10, kwargs["assignment_window_sec"])
        finally:
            coordinator.stop()

    def test_missing_camera_does_not_block_present_camera(self):
        manager = RecordingIdentityManager()
        coordinator = main.GlobalAssignmentCoordinator(
            lambda: manager,
            window_sec=0.10,
        )
        try:
            started = time.monotonic()
            coordinator.submit(
                "only-camera",
                [detection(1, embedding(1, 0))],
                event_time=200.0,
            )
            self.assertLess(time.monotonic() - started, 0.08)
            self.assertTrue(manager.completed.wait(timeout=1.0))
            self.assertEqual(
                {"only-camera"},
                set(manager.calls[0][0]),
            )
        finally:
            coordinator.stop()


class TrustedEvidenceGlobalAssignmentTests(unittest.TestCase):
    def test_simultaneous_forced_overlap_claims_cannot_share_gid(self):
        # There is no explicit exception policy for overlapping cameras, so
        # even two trusted forced claims must retain strict one-to-one output.
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        now = 7000.0
        existing_gid = manager.assign_batch(
            "origin",
            [detection(1, person, event_time=now)],
            event_time=now,
        )[0]["gid"]

        results = manager.assign_global_batch(
            {
                "A": [
                    detection(
                        7,
                        person,
                        event_time=now + 0.01,
                        overlap=True,
                        forced_gid=existing_gid,
                    )
                ],
                "B": [
                    detection(
                        8,
                        person,
                        event_time=now + 0.02,
                        overlap=True,
                        forced_gid=existing_gid,
                    )
                ],
            },
            event_time=now + 0.02,
            batch_id="simultaneous-forced-overlap",
        )

        assigned_gids = [results["A"][0]["gid"], results["B"][0]["gid"]]
        self.assertEqual(1, assigned_gids.count(existing_gid))
        self.assertEqual(2, len(set(assigned_gids)))
        self.assertEqual(
            1,
            sum(
                item["gid"] == existing_gid
                for item in manager.last_global_batch_diagnostics["assignments"]
            ),
        )

    def test_two_rows_cannot_both_claim_the_only_existing_gid(self):
        # No camera-overlap policy exists yet, so the current safe contract is
        # strict one-to-one: simultaneous camera rows cannot share one GID.
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        now = time.time()
        existing_gid = manager.assign_batch(
            "origin",
            [detection(1, person)],
            event_time=now,
        )[0]["gid"]
        manager.identities[existing_gid].update({
            "state": main.IDENTITY_ACTIVE,
            "state_updated_at": now,
            "state_reason": "test_mature_identity",
            "gallery": [person.copy()],
            "embedding": person.copy(),
            "gallery_mature": True,
        })

        def deterministic_pair_score(
            gid,
            _identity,
            cam_name,
            _detection,
            *_args,
        ):
            return {
                "gid": gid,
                "score": 0.95 if cam_name == "A" else 0.90,
                "appearance": 1.0,
                "cross_camera": True,
            }

        with (
            patch.object(
                main,
                "topology_config",
                {"version": 1, "enforce": False, "transitions": []},
            ),
            patch.object(
                manager,
                "_pair_score",
                side_effect=deterministic_pair_score,
            ),
        ):
            results = manager.assign_global_batch(
                {
                    "A": [detection(7, person, event_time=now + 0.01)],
                    "B": [detection(8, person, event_time=now + 0.02)],
                },
                event_time=now + 0.02,
                batch_id="single-gid-test",
            )
        assigned_gids = [
            results["A"][0]["gid"],
            results["B"][0]["gid"],
        ]

        self.assertEqual(1, assigned_gids.count(existing_gid))
        self.assertEqual(2, len(set(assigned_gids)))
        self.assertEqual(existing_gid, results["A"][0]["gid"])
        self.assertEqual("global-cross-camera", results["A"][0]["source"])
        self.assertEqual("new", results["B"][0]["source"])
        diagnostics = manager.last_global_batch_diagnostics
        self.assertEqual(
            [{
                "camera": "A",
                "track_id": 7,
                "gid": existing_gid,
                "score": 0.95,
            }],
            diagnostics["selected"],
        )
        self.assertEqual(
            "unmatched_global_assignment",
            diagnostics["rows"][1]["new_identity_reason"],
        )
        self.assertEqual(
            ["global-cross-camera", "new"],
            [item["source"] for item in diagnostics["assignments"]],
        )
        self.assertEqual(
            ["global-cross-camera", "unmatched_global_assignment"],
            [item["reason"] for item in diagnostics["assignments"]],
        )

    def test_verified_local_gid_is_reserved_from_conflicting_hungarian_row(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        now = time.time()
        first = manager.assign_batch(
            "A",
            [detection(7, person)],
            event_time=now,
        )[0]

        results = manager.assign_global_batch(
            {
                "A": [detection(7, person, event_time=now + 0.01)],
                "B": [detection(8, person, event_time=now + 0.02)],
            },
            event_time=now + 0.02,
            batch_id="trusted-local-test",
        )

        self.assertEqual(first["gid"], results["A"][0]["gid"])
        self.assertEqual("local-track-verified", results["A"][0]["source"])
        self.assertNotEqual(
            results["A"][0]["gid"],
            results["B"][0]["gid"],
        )
        diagnostics = manager.last_global_batch_diagnostics
        self.assertEqual([], diagnostics["candidate_gids"])
        self.assertEqual([], diagnostics["selected"])
        self.assertEqual(
            ["local-track-verified", "new"],
            [item["source"] for item in diagnostics["assignments"]],
        )
        self.assertEqual(
            "no_eligible_candidate",
            diagnostics["rows"][1]["new_identity_reason"],
        )
        self.assertEqual(
            1,
            sum(
                item["gid"] == first["gid"]
                for item in diagnostics["assignments"]
            ),
        )


class IdentityLifecycleGlobalAssignmentTests(unittest.TestCase):
    def test_dormant_identity_recovers_after_active_idle_before_dormant_ttl(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        first_event_time = 5000.0
        first = manager.assign_batch(
            "A",
            [detection(7, person, event_time=first_event_time)],
            event_time=first_event_time,
        )[0]
        gid = first["gid"]
        identity = manager.identities[gid]
        identity["state"] = main.IDENTITY_ACTIVE
        identity["state_updated_at"] = first_event_time
        identity["state_reason"] = "test_active"
        identity["gallery"] = [person.copy()]
        identity["gallery_mature"] = True
        manager.local_to_global.clear()

        recovery_time = first_event_time + main.REID_MAX_IDLE_SEC + 1.0
        result = manager.assign_global_batch(
            {
                "B": [
                    detection(
                        12,
                        person,
                        event_time=recovery_time,
                    )
                ]
            },
            event_time=recovery_time,
            batch_id="dormant-recovery",
        )["B"][0]

        self.assertEqual(gid, result["gid"])
        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        self.assertEqual("cross_camera_recovery", identity["state_reason"])
        trace = manager.last_global_batch_diagnostics["rows"][0]
        candidate = next(
            item for item in trace["candidates"] if item["gid"] == gid
        )
        self.assertTrue(candidate["hard_gate_passed"])

    def test_expired_forced_gid_is_not_reused_by_global_assignment(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        first_event_time = 6000.0
        first = manager.assign_batch(
            "A",
            [detection(7, person, event_time=first_event_time)],
            event_time=first_event_time,
        )[0]
        expired_gid = first["gid"]
        identity = manager.identities[expired_gid]
        identity["state"] = main.IDENTITY_EXPIRED
        identity["state_updated_at"] = first_event_time + 1.0
        identity["state_reason"] = "test_expired"
        manager.local_to_global.clear()

        result = manager.assign_global_batch(
            {
                "B": [
                    detection(
                        12,
                        person,
                        event_time=first_event_time + 2.0,
                        overlap=True,
                        forced_gid=expired_gid,
                    )
                ]
            },
            event_time=first_event_time + 2.0,
            batch_id="expired-forced-gid",
        )["B"][0]

        self.assertNotEqual(expired_gid, result["gid"])
        self.assertEqual(main.IDENTITY_EXPIRED, identity["state"])


class ShortGapContinuityRegressionTests(unittest.TestCase):
    @staticmethod
    def _row(track_id, event_time, box, vector, confidence=0.60):
        width = box[2] - box[0]
        height = box[3] - box[1]
        return {
            "tid": track_id,
            "emb": vector,
            "box": box,
            "box_wh": (width, height),
            "map_pos": None,
            "overlap": False,
            "forced_gid": None,
            "local_track_confirmed": True,
            "conf": confidence,
            "detector_confidence": confidence,
            "crop_size": (width, height),
            "blur_variance": 100.0,
            "border_clip_ratio": 0.0,
            "event_time": event_time,
        }

    def test_new_local_id_after_three_source_frames_recovers_gid(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        first_event_time = 1000.04
        first = manager.assign_batch(
            "A",
            [
                self._row(
                    5,
                    first_event_time,
                    (20, 0, 60, 60),
                    person,
                    confidence=0.95,
                )
            ],
            event_time=first_event_time,
        )[0]
        history = [
            {
                "gid": first["gid"],
                "box": (0, 0, 40, 60),
                "center": (20, 30),
                "tid": 5,
                "cam_name": "A",
                "overlap": False,
                "ts": 1000.00,
            },
            {
                "gid": first["gid"],
                "box": (20, 0, 60, 60),
                "center": (40, 30),
                "tid": 5,
                "cam_name": "A",
                "overlap": False,
                "ts": first_event_time,
            },
        ]
        reappeared = self._row(
            12,
            1000.12,
            (60, 0, 100, 60),
            person,
        )
        reappeared["sequence_index"] = 3
        reappeared["coordinator_generation"] = 0

        # Source time advanced only two frame intervals, while deliberately
        # slow downstream processing advanced wall time by nearly 13 seconds.
        with patch.object(main.time, "time", return_value=1013.0):
            result = manager.assign_global_batch(
                {"A": [reappeared]},
                prev_assignments_by_camera={"A": history},
                event_time=1000.12,
                batch_id="short-gap-regression",
            )["A"][0]

        self.assertEqual(5, history[-1]["tid"])
        self.assertEqual(12, reappeared["tid"])
        self.assertEqual(first["gid"], result["gid"])
        trace = manager.last_global_batch_diagnostics["rows"][0]
        self.assertEqual(3, trace["sequence_index"])
        self.assertEqual(1000.12, trace["event_time"])
        self.assertIsNone(trace["previous_local_mapping"])
        self.assertEqual([first["gid"]], trace["candidate_gids"])
        self.assertTrue(trace["candidates"][0]["hard_gate_passed"])
        self.assertEqual(1.0, trace["candidates"][0]["appearance"])
        self.assertIsNone(trace["top1_top2_margin"])
        self.assertEqual("committed", trace["assignment_state"])
        self.assertEqual("short-gap-regression", trace["batch_id"])
        self.assertEqual(0, trace["generation"])
        self.assertEqual(first["gid"], trace["final_gid"])
        self.assertIsNone(trace["new_identity_reason"])

    def test_low_quality_bootstrap_matures_and_recovers_after_dormant_gap(self):
        manager = main.GlobalIdentityManager()
        bootstrap = embedding(1, 0, 0)
        usable = embedding(0, 1, 0)
        start = 3000.0

        first = manager.assign_global_batch(
            {
                "A": [
                    self._row(
                        5,
                        start,
                        (20, 0, 60, 60),
                        bootstrap,
                        confidence=0.10,
                    )
                ]
            },
            event_time=start,
            batch_id="low-quality-bootstrap-0",
        )["A"][0]
        gid = first["gid"]
        self.assertEqual(
            "low_detector_confidence",
            first["gallery_rejection_reason"],
        )

        source_by_frame = []
        for frame_index, event_time in enumerate(
            (start + 0.04, start + 0.08, start + 0.12),
            start=1,
        ):
            result = manager.assign_global_batch(
                {
                    "A": [
                        self._row(
                            5,
                            event_time,
                            (20, 0, 60, 60),
                            usable,
                            confidence=0.95,
                        )
                    ]
                },
                event_time=event_time,
                batch_id=f"low-quality-bootstrap-{frame_index}",
            )["A"][0]
            self.assertEqual(gid, result["gid"])
            source_by_frame.append(result["source"])

        identity = manager.identities[gid]
        self.assertEqual(
            "provisional-local-continuity",
            source_by_frame[0],
        )
        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        self.assertTrue(identity["gallery_mature"])
        self.assertEqual(1, len(identity["gallery"]))

        recovery_time = start + 0.12 + main.REID_MAX_IDLE_SEC + 1.0
        recovered = manager.assign_global_batch(
            {
                "A": [
                    self._row(
                        12,
                        recovery_time,
                        (20, 0, 60, 60),
                        usable,
                        confidence=0.95,
                    )
                ]
            },
            event_time=recovery_time,
            batch_id="mature-dormant-recovery",
        )["A"][0]

        self.assertEqual(gid, recovered["gid"])
        self.assertEqual("global-batch", recovered["source"])
        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        self.assertEqual("same_camera_recovery", identity["state_reason"])

    def test_low_quality_provisional_continuity_has_absolute_time_limit(self):
        manager = main.GlobalIdentityManager()
        bootstrap = embedding(1, 0, 0)
        conflicting = embedding(0, 1, 0)
        start = 3500.0

        first = manager.assign_global_batch(
            {
                "A": [
                    self._row(
                        5,
                        start,
                        (20, 0, 60, 60),
                        bootstrap,
                        confidence=0.10,
                    )
                ]
            },
            event_time=start,
            batch_id="bounded-provisional-0",
        )["A"][0]
        original_gid = first["gid"]

        for frame_index, offset in enumerate((0.9, 1.8), start=1):
            continued = manager.assign_global_batch(
                {
                    "A": [
                        self._row(
                            5,
                            start + offset,
                            (20, 0, 60, 60),
                            conflicting,
                            confidence=0.10,
                        )
                    ]
                },
                event_time=start + offset,
                batch_id=f"bounded-provisional-{frame_index}",
            )["A"][0]
            self.assertEqual(original_gid, continued["gid"])

        expired_continuity = manager.assign_global_batch(
            {
                "A": [
                    self._row(
                        5,
                        start + 2.1,
                        (20, 0, 60, 60),
                        conflicting,
                        confidence=0.10,
                    )
                ]
            },
            event_time=start + 2.1,
            batch_id="bounded-provisional-expired",
        )["A"][0]

        self.assertNotEqual(original_gid, expired_continuity["gid"])
        self.assertEqual("new", expired_continuity["source"])

    def test_provisional_continuity_yields_to_strong_competing_identity(self):
        manager = main.GlobalIdentityManager()
        person_a = embedding(1, 0, 0)
        # Each swapped row still scores 0.50 against its local-ID owner, which
        # is above LOCAL_TRACK_VERIFY_THRESHOLD, while the correct competing
        # identity scores 1.00.
        person_b = embedding(0.5, np.sqrt(0.75), 0)
        start = 3600.0

        first = manager.assign_global_batch(
            {
                "A": [
                    self._row(
                        10,
                        start,
                        (0, 0, 40, 60),
                        person_a,
                        confidence=0.10,
                    ),
                    self._row(
                        20,
                        start,
                        (100, 0, 140, 60),
                        person_b,
                        confidence=0.10,
                    ),
                ]
            },
            event_time=start,
            batch_id="provisional-competitors-seed",
        )["A"]
        expected_gids = [first[0]["gid"], first[1]["gid"]]
        self.assertTrue(
            all(
                manager.identities[gid]["state"] == main.IDENTITY_PROVISIONAL
                for gid in expected_gids
            )
        )

        switched = manager.assign_global_batch(
            {
                "A": [
                    self._row(
                        20,
                        start + 0.1,
                        (5, 0, 45, 60),
                        person_a,
                        confidence=0.95,
                    ),
                    self._row(
                        10,
                        start + 0.1,
                        (95, 0, 135, 60),
                        person_b,
                        confidence=0.95,
                    ),
                ]
            },
            event_time=start + 0.1,
            batch_id="provisional-competitors-switched-local-ids",
        )["A"]

        self.assertEqual(
            expected_gids,
            [switched[0]["gid"], switched[1]["gid"]],
        )
        self.assertNotEqual(
            "provisional-local-continuity",
            switched[0]["source"],
        )
        self.assertNotEqual(
            "provisional-local-continuity",
            switched[1]["source"],
        )

    def test_different_person_after_short_gap_does_not_reuse_gid(self):
        manager = main.GlobalIdentityManager()
        first_person = embedding(1, 0, 0)
        different_person = embedding(0, 1, 0)
        first = manager.assign_batch(
            "A",
            [self._row(5, 2000.00, (20, 0, 60, 60), first_person, 0.95)],
            event_time=2000.00,
        )[0]
        result = manager.assign_global_batch(
            {
                "A": [
                    self._row(
                        12,
                        2000.12,
                        (25, 0, 65, 60),
                        different_person,
                        0.95,
                    )
                ]
            },
            event_time=2000.12,
            batch_id="short-gap-different-person",
        )["A"][0]

        self.assertNotEqual(first["gid"], result["gid"])

    def test_occlusion_hold_gid_is_reserved_from_conflicting_row(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        other = embedding(0, 1, 0)
        now = time.time()
        first = manager.assign_batch(
            "A",
            [detection(7, person)],
            event_time=now,
        )[0]
        manager.occlusion_hold[("A", 7)] = {
            "gid": first["gid"],
            "until_ts": now + 1.0,
            "score": 0.99,
        }

        results = manager.assign_global_batch(
            {
                "A": [detection(7, other, event_time=now + 0.01)],
                "B": [detection(8, person, event_time=now + 0.02)],
            },
            event_time=now + 0.02,
            batch_id="occlusion-hold-test",
        )

        self.assertEqual(first["gid"], results["A"][0]["gid"])
        self.assertEqual("occlusion-hold", results["A"][0]["source"])
        self.assertNotEqual(
            results["A"][0]["gid"],
            results["B"][0]["gid"],
        )
        diagnostics = manager.last_global_batch_diagnostics
        self.assertEqual([], diagnostics["candidate_gids"])
        self.assertEqual([], diagnostics["selected"])
        self.assertEqual(
            ["occlusion-hold", "new"],
            [item["source"] for item in diagnostics["assignments"]],
        )
        self.assertEqual(
            "no_eligible_candidate",
            diagnostics["rows"][1]["new_identity_reason"],
        )
        self.assertEqual(
            1,
            sum(
                item["gid"] == first["gid"]
                for item in diagnostics["assignments"]
            ),
        )


class ForcedOcclusionHistoryTests(unittest.TestCase):
    camera_name = "forced-occlusion-history-test"
    detection_box = (10, 10, 50, 90)

    def _forced_map(self, previous_assignment, now):
        missing = object()
        with main.cameras_lock:
            original = main.cameras.get(self.camera_name, missing)
            main.cameras[self.camera_name] = {
                "prev_assignments": [previous_assignment],
            }
        try:
            return main.build_forced_gid_map(
                self.camera_name,
                [self.detection_box],
                event_time=now,
            )
        finally:
            with main.cameras_lock:
                if original is missing:
                    main.cameras.pop(self.camera_name, None)
                else:
                    main.cameras[self.camera_name] = original

    def test_only_fresh_history_can_force_gid_during_occlusion(self):
        now = 5000.0
        common = {
            "gid": 41,
            "box": self.detection_box,
            "center": (30, 50),
            "tid": 7,
            "cam_name": self.camera_name,
            "overlap": False,
        }
        stale = {
            **common,
            "ts": now - main.OCCLUSION_HOLD_SEC - 0.01,
        }
        fresh = {
            **common,
            "ts": now - (main.OCCLUSION_HOLD_SEC / 2.0),
        }

        self.assertEqual({}, self._forced_map(stale, now))
        self.assertEqual({0: 41}, self._forced_map(fresh, now))


class ProductionIdentityPathTests(unittest.TestCase):
    def setUp(self):
        with main.cameras_lock:
            main.cameras.clear()

    def tearDown(self):
        with main.cameras_lock:
            main.cameras.clear()

    def test_uploaded_and_live_frames_submit_to_global_coordinator(self):
        for source_type in ("video", "live"):
            with self.subTest(source_type=source_type):
                tracker = OnePersonTrackingModel()
                coordinator = MagicMock()
                coordinator.submit.return_value = [{
                    "gid": 11,
                    "score": 0.99,
                    "source": "local-track-verified",
                }]
                cam_name = f"{source_type}-camera"
                with main.cameras_lock:
                    main.cameras.clear()
                    main.cameras[cam_name] = camera_data(
                        source_type,
                        tracker,
                    )

                frame = np.full((90, 90, 3), 127, dtype=np.uint8)
                event_time = 1234.5
                with (
                    patch.object(
                        main,
                        "extract_person_embedding",
                        return_value=embedding(1, 0, 0),
                    ),
                    patch.object(
                        main,
                        "global_assignment_coordinator",
                        coordinator,
                    ),
                ):
                    output = main.process_camera_frame(
                        cam_name,
                        frame,
                        1,
                        event_time=event_time,
                    )

                self.assertEqual(frame.shape, output.shape)
                self.assertEqual(1, coordinator.submit.call_count)
                args, kwargs = coordinator.submit.call_args
                self.assertEqual(cam_name, args[0])
                self.assertEqual(1, len(args[1]))
                self.assertEqual(event_time, args[1][0]["event_time"])
                self.assertEqual(event_time, kwargs["event_time"])
                self.assertIs(
                    tracker,
                    main.cameras[cam_name]["tracking_model"],
                )
                self.assertTrue(tracker.calls[0]["persist"])
                self.assertEqual("botsort.yaml", tracker.calls[0]["tracker"])
                timing = main.cameras[cam_name]["downstream_timing"]
                self.assertEqual(1, timing["frame_index"])
                self.assertEqual(event_time, timing["event_time"])
                self.assertGreaterEqual(timing["detection_tracking_ms"], 0.0)
                self.assertGreaterEqual(timing["reid_feature_ms"], 0.0)
                self.assertGreaterEqual(timing["coordinator_submit_ms"], 0.0)
                self.assertGreaterEqual(timing["total_downstream_ms"], 0.0)

    def test_uploaded_worker_keeps_playback_flow_while_using_global_path(self):
        tracker = OnePersonTrackingModel()
        coordinator = MagicMock()
        coordinator.submit.return_value = [None]
        cam_name = "uploaded-camera"
        frame = np.full((90, 90, 3), 127, dtype=np.uint8)
        event_time = 2345.6

        with main.cameras_lock:
            main.cameras[cam_name] = camera_data("video", tracker)

        def read_one_synchronized_set():
            main.video_worker_running = False
            return {
                cam_name: {
                    "frame": frame,
                    "frame_index": 3,
                    "fps": 25.0,
                    "source_time_sec": 0.12,
                    "event_time": event_time,
                    "time_offset_sec": 0.0,
                    "source_reset": False,
                }
            }

        original_app_running = main.app.is_running
        original_video_worker_running = main.video_worker_running
        try:
            main.app.is_running = True
            main.video_worker_running = True
            with (
                patch.object(
                    main.multi_video_manager,
                    "read_synchronized_frames",
                    side_effect=read_one_synchronized_set,
                ) as read_frames,
                patch.object(
                    main,
                    "extract_person_embedding",
                    return_value=embedding(1, 0, 0),
                ),
                patch.object(
                    main,
                    "global_assignment_coordinator",
                    coordinator,
                ),
                patch.object(
                    main,
                    "publish_processed_frame",
                    return_value=True,
                ) as publish_frame,
                patch.object(main.time, "sleep"),
            ):
                main.multi_camera_worker()
        finally:
            main.app.is_running = original_app_running
            main.video_worker_running = original_video_worker_running

        read_frames.assert_called_once_with()
        coordinator.submit.assert_called_once()
        self.assertEqual(
            event_time,
            coordinator.submit.call_args.kwargs["event_time"],
        )
        publish_frame.assert_called_once()
        self.assertEqual(
            0.12,
            main.cameras[cam_name]["tracker_source_time_sec"],
        )


if __name__ == "__main__":
    unittest.main()
