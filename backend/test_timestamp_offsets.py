import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("REID_ENABLED", "false")

from backend import main


class SingleFrameCapture:
    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class TimestampOffsetTests(unittest.TestCase):

    def test_downstream_event_metadata_never_moves_backward(self):
        camera_data = {}

        self.assertEqual(
            100.0,
            main.canonical_observation_event_time(camera_data, 100.0),
        )
        self.assertEqual(
            100.0,
            main.canonical_observation_event_time(camera_data, 90.0),
        )

    def test_video_offsets_produce_a_shared_canonical_timeline(self):
        manager = main.MultiCameraVideoManager()
        manager.videos = {
            "cam_a": {
                "cap": SingleFrameCapture(), "fps": 25.0,
                "total_frames": 0, "loop_video": False,
                "frame_index": 0, "time_offset_sec": 0.0,
                "playback_started_at": 1000.0,
                "last_source_time_sec": None, "last_event_time": None,
                "tracker_reset_pending": False,
            },
            "cam_b": {
                "cap": SingleFrameCapture(), "fps": 25.0,
                "total_frames": 0, "loop_video": False,
                "frame_index": 0, "time_offset_sec": 5.0,
                "playback_started_at": 1000.0,
                "last_source_time_sec": None, "last_event_time": None,
                "tracker_reset_pending": False,
            },
        }
        manager.frames = {"cam_a": None, "cam_b": None}
        manager.frame_indices = {"cam_a": 0, "cam_b": 0}
        manager.running = {"cam_a": True, "cam_b": True}

        frames = manager.read_synchronized_frames()

        self.assertEqual(0.04, frames["cam_a"]["source_time_sec"])
        self.assertEqual(0.04, frames["cam_b"]["source_time_sec"])
        self.assertAlmostEqual(
            5.0,
            frames["cam_b"]["event_time"] - frames["cam_a"]["event_time"],
        )

    def test_event_time_never_moves_back_when_a_video_rewinds(self):
        manager = main.MultiCameraVideoManager()
        manager.videos = {
            "cam": {
                "cap": SingleFrameCapture(), "fps": 10.0,
                "total_frames": 0, "loop_video": False,
                "frame_index": 3, "time_offset_sec": 0.0,
                "playback_started_at": 1000.0,
                "last_source_time_sec": 0.3, "last_event_time": 1000.3,
                "tracker_reset_pending": False,
            },
        }
        manager.frames = {"cam": None}
        manager.frame_indices = {"cam": 3}
        manager.running = {"cam": True}

        frame = manager.read_synchronized_frames()["cam"]

        self.assertGreater(frame["event_time"], 1000.3)

    def test_live_processing_uses_capture_time_not_processing_time(self):
        worker = main.LiveCameraWorker("live-event-time", 0)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        worker._publish_captured_frame(
            frame,
            event_time=1234.5,
            monotonic_time=10.0,
        )

        def process_frame(_camera, source_frame, _index, event_time=None):
            self.assertEqual(1234.5, event_time)
            worker.stop_event.set()
            return source_frame

        with main.cameras_lock:
            main.cameras["live-event-time"] = {"last_frame": None}
        try:
            with (
                patch.object(main, "process_camera_frame", side_effect=process_frame),
                patch.object(main, "publish_processed_frame", return_value=True),
                patch.object(main.time, "time", return_value=9999.0),
            ):
                worker._processing_loop()
        finally:
            with main.cameras_lock:
                main.cameras.pop("live-event-time", None)

        self.assertEqual(1234.5, worker.last_frame_event_time)
        self.assertEqual(9999.0, worker.last_processing_started)


if __name__ == "__main__":
    unittest.main()
