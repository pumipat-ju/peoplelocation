import asyncio
import json
import os
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("REID_ENABLED", "false")

from backend import main


def live_camera_data(source=0):
    return {
        "url": source,
        "source_type": "live",
        "loop_video": False,
        "processor": None,
        "src_pts": None,
        "dst_pts": None,
        "last_frame": None,
        "last_frame_event_time": None,
        "prev_assignments": [],
        **main.new_camera_tracker_context()
    }


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if predicate():
            return True

        time.sleep(0.005)

    return bool(predicate())


class RepeatingFakeCapture:
    def __init__(self, marker=1, delay=0.002):
        self.marker = marker
        self.delay = delay
        self.read_count = 0
        self.release_event = threading.Event()
        self.set_calls = []

    def isOpened(self):
        return not self.release_event.is_set()

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def read(self):
        if self.release_event.wait(self.delay):
            return False, None

        self.read_count += 1
        value = (self.marker + self.read_count) % 255
        return (
            True,
            np.full((8, 8, 3), value, dtype=np.uint8)
        )

    def release(self):
        self.release_event.set()


class OneFrameThenFailureCapture(RepeatingFakeCapture):
    def read(self):
        if self.release_event.wait(self.delay):
            return False, None

        self.read_count += 1

        if self.read_count > 1:
            return False, None

        return (
            True,
            np.full((8, 8, 3), self.marker, dtype=np.uint8)
        )


class LiveSourceParsingTests(unittest.TestCase):
    def test_numeric_strings_become_camera_indices(self):
        self.assertEqual(main.parse_video_source("0"), 0)
        self.assertEqual(main.parse_video_source("1"), 1)
        self.assertEqual(main.parse_video_source(" 2 "), 2)

    def test_stream_url_remains_a_string(self):
        source = "rtsp://camera.example/live"
        self.assertEqual(main.parse_video_source(source), source)

    def test_status_mask_hides_rtsp_credentials(self):
        source = "rtsp://alice:secret@camera.example:8554/live"
        masked = main.mask_video_source(source)
        self.assertNotIn("alice", masked)
        self.assertNotIn("secret", masked)
        self.assertEqual(
            masked,
            "rtsp://***:***@camera.example:8554/live"
        )
        safe_error = main.sanitize_source_error(
            f"failed to open {source} for alice",
            source
        )
        self.assertNotIn("alice", safe_error)
        self.assertNotIn("secret", safe_error)


class LiveCameraLifecycleTests(unittest.TestCase):
    def setUp(self):
        main.app.is_running = True
        main.live_camera_manager.stop_all()

        with main.cameras_lock:
            main.cameras.clear()

        with main.video_worker_lock:
            main.processed_frames.clear()
            main.processed_frame_locks.clear()

    def tearDown(self):
        main.live_camera_manager.stop_all()

        with main.cameras_lock:
            main.cameras.clear()

        with main.video_worker_lock:
            main.processed_frames.clear()
            main.processed_frame_locks.clear()

    def test_add_camera_starts_once_and_rejects_duplicate_name(self):
        worker = object()

        with patch.object(
            main.live_camera_manager,
            "start_worker",
            return_value=(worker, True)
        ) as start_worker:
            added = asyncio.run(
                main.add_camera(name="desk", url="0")
            )
            duplicate = asyncio.run(
                main.add_camera(name="desk", url="1")
            )

        self.assertEqual(added.status_code, 200)
        self.assertEqual(duplicate.status_code, 409)
        start_worker.assert_called_once_with("desk", 0)
        self.assertEqual(main.cameras["desk"]["url"], 0)
        self.assertEqual(
            main.cameras["desk"]["source_type"],
            "live"
        )

    def test_manager_opens_integer_source_once_and_stop_releases_it(self):
        manager = main.LiveCameraManager()
        capture = RepeatingFakeCapture()

        with (
            patch.object(
                main.cv2,
                "VideoCapture",
                return_value=capture
            ) as video_capture,
            patch.object(
                main,
                "process_camera_frame",
                side_effect=lambda _name, frame, _index, **_kwargs: frame
            ),
            patch.object(
                main,
                "publish_processed_frame",
                return_value=True
            )
        ):
            with main.cameras_lock:
                main.cameras["desk"] = live_camera_data(0)

            first, first_created = manager.start_worker("desk", "0")
            second, second_created = manager.start_worker("desk", 0)

            self.assertTrue(
                wait_until(
                    lambda: first.status()["captured_frames"] > 0
                )
            )
            manager.stop_worker("desk")

        self.assertIs(first, second)
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        video_capture.assert_called_once_with(0)
        self.assertTrue(capture.release_event.is_set())
        self.assertFalse(first.status()["running"])

    def test_delete_camera_stops_live_worker(self):
        with main.cameras_lock:
            main.cameras["desk"] = live_camera_data(0)

        with (
            patch.object(
                main.live_camera_manager,
                "stop_worker",
                return_value=True
            ) as stop_worker,
            patch.object(
                main,
                "reset_camera_tracker",
                return_value={}
            ) as reset_tracker
        ):
            response = asyncio.run(
                main.delete_camera("desk")
            )

        self.assertEqual(response.status_code, 200)
        stop_worker.assert_called_once_with("desk")
        reset_tracker.assert_called_once_with(
            "desk",
            reason="camera_removed"
        )
        self.assertNotIn("desk", main.cameras)

    def test_read_failure_reconnects_and_resets_only_that_tracker(self):
        manager = main.LiveCameraManager()
        captures = [
            OneFrameThenFailureCapture(marker=10),
            RepeatingFakeCapture(marker=20)
        ]
        capture_lock = threading.Lock()

        def capture_factory(_source):
            with capture_lock:
                return captures.pop(0)

        with main.cameras_lock:
            main.cameras.update({
                "A": live_camera_data(0),
                "B": live_camera_data(1)
            })

        with (
            patch.object(
                main.cv2,
                "VideoCapture",
                side_effect=capture_factory
            ),
            patch.object(
                main,
                "process_camera_frame",
                side_effect=lambda _name, frame, _index, **_kwargs: frame
            ),
            patch.object(
                main,
                "publish_processed_frame",
                return_value=True
            ),
            patch.object(
                main,
                "reset_camera_tracker",
                return_value={}
            ) as reset_tracker,
            patch.object(
                main,
                "LIVE_CAMERA_RECONNECT_INTERVAL_SEC",
                0.01
            )
        ):
            worker, _ = manager.start_worker("A", 0)
            self.assertTrue(
                wait_until(
                    lambda: reset_tracker.call_count >= 1
                    and worker.status()["processed_frames"] >= 2
                )
            )
            status = worker.status()
            manager.stop_all()

        self.assertTrue(status["running"])
        self.assertGreaterEqual(status["reconnect_count"], 1)
        self.assertEqual(
            {
                call.args[0]
                for call in reset_tracker.call_args_list
            },
            {"A"}
        )
        reset_tracker.assert_any_call(
            "A",
            reason="live_source_reconnected"
        )

    def test_camera_workers_have_private_state_and_run_concurrently(self):
        manager = main.LiveCameraManager()
        captures = {}
        processed_cameras = set()
        processed_lock = threading.Lock()

        def capture_factory(source):
            capture = RepeatingFakeCapture(marker=source * 10 + 1)
            captures[source] = capture
            return capture

        def process(camera, frame, _index, **kwargs):
            self.assertIsInstance(kwargs["event_time"], float)

            with processed_lock:
                processed_cameras.add(camera)

            return frame

        with main.cameras_lock:
            main.cameras.update({
                "A": live_camera_data(0),
                "B": live_camera_data(1)
            })

        with (
            patch.object(
                main.cv2,
                "VideoCapture",
                side_effect=capture_factory
            ),
            patch.object(
                main,
                "process_camera_frame",
                side_effect=process
            ),
            patch.object(
                main,
                "publish_processed_frame",
                return_value=True
            )
        ):
            worker_a, _ = manager.start_worker("A", 0)
            worker_b, _ = manager.start_worker("B", 1)
            self.assertTrue(
                wait_until(
                    lambda: processed_cameras == {"A", "B"}
                )
            )
            manager.stop_all()

        self.assertIsNot(worker_a, worker_b)
        self.assertIsNot(worker_a.state_lock, worker_b.state_lock)
        self.assertNotEqual(
            worker_a.instance_id,
            worker_b.instance_id
        )
        self.assertEqual(set(captures), {0, 1})

    def test_latest_frame_slot_drops_stale_frames_under_backpressure(self):
        manager = main.LiveCameraManager()
        capture = RepeatingFakeCapture(delay=0.001)

        def slow_process(_camera, frame, _index, **_kwargs):
            time.sleep(0.03)
            return frame

        with main.cameras_lock:
            main.cameras["fast"] = live_camera_data(0)

        with (
            patch.object(
                main.cv2,
                "VideoCapture",
                return_value=capture
            ),
            patch.object(
                main,
                "process_camera_frame",
                side_effect=slow_process
            ),
            patch.object(
                main,
                "publish_processed_frame",
                return_value=True
            )
        ):
            worker, _ = manager.start_worker("fast", 0)
            self.assertTrue(
                wait_until(
                    lambda: worker.status()["captured_frames"] >= 20
                )
            )
            self.assertTrue(
                wait_until(
                    lambda: worker.status()["processed_frames"] >= 1
                )
            )
            status = worker.status()
            manager.stop_all()

        self.assertEqual(status["frame_queue_capacity"], 1)
        self.assertGreater(status["dropped_frames"], 0)
        self.assertLess(
            status["processed_frames"],
            status["captured_frames"]
        )

    def test_api_status_exposes_live_diagnostics_without_credentials(self):
        source = "rtsp://alice:secret@camera.example/live"

        with main.cameras_lock:
            main.cameras["secure"] = live_camera_data(source)

        worker_status = {
            "running": True,
            "capture_open": True,
            "frame_index": 12,
            "captured_frames": 12,
            "processed_frames": 5,
            "dropped_frames": 7,
            "processing_fps": 5.0,
            "reconnect_count": 0,
            "last_error": None
        }

        with patch.object(
            main.live_camera_manager,
            "get_status",
            return_value=worker_status
        ):
            payload = json.loads(
                asyncio.run(main.get_status()).body
            )["cameras"]["secure"]

        serialized = json.dumps(payload)
        self.assertEqual(payload["source_type"], "live")
        self.assertEqual(payload["live_worker"], worker_status)
        self.assertNotIn("alice", serialized)
        self.assertNotIn("secret", serialized)


if __name__ == "__main__":
    unittest.main()
