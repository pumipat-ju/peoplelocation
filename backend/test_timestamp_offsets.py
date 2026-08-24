import os
import unittest

import numpy as np

os.environ.setdefault("REID_ENABLED", "false")

from backend import main


class SingleFrameCapture:
    def read(self):
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class TimestampOffsetTests(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
