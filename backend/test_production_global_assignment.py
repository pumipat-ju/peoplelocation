import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

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
    def test_two_rows_cannot_both_claim_the_only_existing_gid(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        now = time.time()
        existing_gid = manager.assign_batch(
            "origin",
            [detection(1, person)],
            event_time=now,
        )[0]["gid"]

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
