import asyncio
import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

os.environ.setdefault(
    "REID_ENABLED",
    "false"
)

from backend import main


class StatefulFakeYOLO:

    def __init__(self, barrier=None):
        self.frames = []
        self.calls = []
        self.active_calls = 0
        self.maximum_active_calls = 0
        self.counter_lock = threading.Lock()
        self.barrier = barrier
        self.predictor = SimpleNamespace(
            trackers=[object()]
        )

    def track(self, frame, **kwargs):
        with self.counter_lock:
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls,
                self.active_calls
            )

        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)

            time.sleep(0.005)
            self.frames.append(
                int(frame[0, 0, 0])
            )
            self.calls.append(kwargs)
            return []
        finally:
            with self.counter_lock:
                self.active_calls -= 1


class LoopingFakeCapture:

    def __init__(self):
        self.read_count = 0
        self.seek_positions = []

    def read(self):
        self.read_count += 1

        if self.read_count == 1:
            return False, None

        return (
            True,
            np.full(
                (4, 4, 3),
                9,
                dtype=np.uint8
            )
        )

    def set(self, prop, value):
        self.seek_positions.append(
            (prop, value)
        )
        return True


def camera_data():
    return {
        "url": "synthetic",
        "source_type": "video",
        "loop_video": False,
        "processor": None,
        "src_pts": None,
        "dst_pts": None,
        "last_frame": None,
        "prev_assignments": [],
        **main.new_camera_tracker_context()
    }


def embedding(*values):
    return main.l2_normalize(
        np.asarray(
            values,
            dtype=np.float32
        )
    )


def detection(local_id, vector, x=0):
    return {
        "tid": local_id,
        "emb": vector,
        "box": (x, 0, x + 40, 100),
        "box_wh": (40, 100),
        "map_pos": None,
        "overlap": False,
        "forced_gid": None,
        "conf": 0.95
    }


class PerCameraTrackerTests(unittest.TestCase):

    def setUp(self):
        with main.cameras_lock:
            main.cameras.clear()

    def tearDown(self):
        with main.cameras_lock:
            main.cameras.clear()

    def test_interleaved_frames_do_not_cross_tracker_state(self):
        tracker_a = StatefulFakeYOLO()
        tracker_b = StatefulFakeYOLO()

        with main.cameras_lock:
            main.cameras.update({
                "A": camera_data(),
                "B": camera_data()
            })

        with patch.object(
            main,
            "YOLO",
            side_effect=[tracker_a, tracker_b]
        ) as factory:
            for camera, marker, frame_index in (
                ("A", 1, 1),
                ("B", 10, 1),
                ("A", 2, 2),
                ("B", 11, 2)
            ):
                frame = np.full(
                    (8, 8, 3),
                    marker,
                    dtype=np.uint8
                )
                main.process_camera_frame(
                    camera,
                    frame,
                    frame_index
                )

        self.assertEqual(tracker_a.frames, [1, 2])
        self.assertEqual(tracker_b.frames, [10, 11])
        self.assertIsNot(
            main.cameras["A"]["tracking_model"],
            main.cameras["B"]["tracking_model"]
        )
        self.assertEqual(factory.call_count, 2)

        for call in tracker_a.calls + tracker_b.calls:
            self.assertTrue(call["persist"])
            self.assertEqual(
                call["tracker"],
                "botsort.yaml"
            )

        status_a = main.get_camera_tracker_status(
            "A"
        )
        status_b = main.get_camera_tracker_status(
            "B"
        )
        self.assertNotEqual(
            status_a["tracker_instance_id"],
            status_b["tracker_instance_id"]
        )
        self.assertNotEqual(
            status_a["botsort_state_ids"],
            status_b["botsort_state_ids"]
        )
        self.assertEqual(
            status_a["active_local_track_count"],
            0
        )
        self.assertEqual(
            status_a["local_track_scope"],
            "camera"
        )
        api_status = json.loads(
            asyncio.run(
                main.get_status()
            ).body
        )["cameras"]
        self.assertEqual(
            api_status["A"]["tracker"][
                "tracker_instance_id"
            ],
            status_a["tracker_instance_id"]
        )
        self.assertEqual(
            api_status["B"]["tracker"][
                "tracker_instance_id"
            ],
            status_b["tracker_instance_id"]
        )

    def test_same_camera_calls_are_serialized_but_cameras_can_overlap(self):
        same_camera_tracker = StatefulFakeYOLO()

        with main.cameras_lock:
            main.cameras["A"] = camera_data()

        with patch.object(
            main,
            "YOLO",
            return_value=same_camera_tracker
        ):
            threads = [
                threading.Thread(
                    target=main.process_camera_frame,
                    args=(
                        "A",
                        np.full(
                            (8, 8, 3),
                            marker,
                            dtype=np.uint8
                        ),
                        marker
                    )
                )
                for marker in range(1, 5)
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=3)

        self.assertEqual(
            same_camera_tracker.maximum_active_calls,
            1
        )
        self.assertTrue(
            all(not thread.is_alive() for thread in threads)
        )

        barrier = threading.Barrier(2)
        tracker_a = StatefulFakeYOLO(barrier)
        tracker_b = StatefulFakeYOLO(barrier)

        with main.cameras_lock:
            main.cameras.clear()
            main.cameras.update({
                "A": camera_data(),
                "B": camera_data()
            })

        with patch.object(
            main,
            "YOLO",
            side_effect=[tracker_a, tracker_b]
        ):
            main.get_camera_tracking_model("A")
            main.get_camera_tracking_model("B")
            threads = [
                threading.Thread(
                    target=main.process_camera_frame,
                    args=(
                        camera,
                        np.full(
                            (8, 8, 3),
                            marker,
                            dtype=np.uint8
                        ),
                        1
                    )
                )
                for camera, marker in (
                    ("A", 1),
                    ("B", 2)
                )
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=3)

        self.assertTrue(
            all(not thread.is_alive() for thread in threads)
        )
        self.assertEqual(tracker_a.frames, [1])
        self.assertEqual(tracker_b.frames, [2])

    def test_reset_one_camera_preserves_other_tracker_and_global_memory(self):
        first_a = StatefulFakeYOLO()
        tracker_b = StatefulFakeYOLO()
        second_a = StatefulFakeYOLO()
        identity_manager = main.GlobalIdentityManager()
        identity_manager.local_to_global.update({
            ("A", 7): {
                "gid": 1,
                "last_seen": time.time()
            },
            ("B", 7): {
                "gid": 2,
                "last_seen": time.time()
            }
        })
        identity_manager.identities.update({
            1: {"last_seen": time.time()},
            2: {"last_seen": time.time()}
        })

        with main.cameras_lock:
            main.cameras.update({
                "A": camera_data(),
                "B": camera_data()
            })

        with (
            patch.object(
                main,
                "YOLO",
                side_effect=[first_a, tracker_b, second_a]
            ),
            patch.object(
                main,
                "global_identity_manager",
                identity_manager
            )
        ):
            loaded_a = main.get_camera_tracking_model("A")
            loaded_b = main.get_camera_tracking_model("B")
            main.reset_camera_tracker(
                "A",
                reason="test_reset"
            )
            reloaded_a = main.get_camera_tracking_model("A")

        self.assertIs(loaded_a, first_a)
        self.assertIs(reloaded_a, second_a)
        self.assertIs(
            main.cameras["B"]["tracking_model"],
            loaded_b
        )
        self.assertNotIn(
            ("A", 7),
            identity_manager.local_to_global
        )
        self.assertIn(
            ("B", 7),
            identity_manager.local_to_global
        )
        self.assertEqual(
            set(identity_manager.identities),
            {1, 2}
        )

    def test_video_loop_marks_only_that_source_for_tracker_reset(self):
        manager = main.MultiCameraVideoManager()
        looping_capture = LoopingFakeCapture()
        manager.videos = {
            "A": {
                "cap": looping_capture,
                "fps": 25.0,
                "total_frames": 10,
                "loop_video": True,
                "frame_index": 10,
                "tracker_reset_pending": False
            }
        }
        manager.frames = {"A": None}
        manager.frame_indices = {"A": 10}
        manager.running = {"A": True}

        frames = manager.read_synchronized_frames()

        self.assertTrue(frames["A"]["source_reset"])
        self.assertEqual(
            frames["A"]["frame_index"],
            1
        )
        self.assertEqual(
            looping_capture.seek_positions,
            [(main.cv2.CAP_PROP_POS_FRAMES, 0)]
        )


class LocalAndGlobalIdentityTests(unittest.TestCase):

    def test_unconfirmed_detection_index_is_not_persistent_local_evidence(self):
        manager = main.GlobalIdentityManager()
        row = detection(
            -1000001,
            embedding(1, 0, 0)
        )
        row["local_track_confirmed"] = False

        manager.assign_batch(
            "A",
            [row]
        )

        self.assertNotIn(
            ("A", -1000001),
            manager.local_to_global
        )

    def test_local_id_is_camera_scoped_and_cross_camera_reid_is_global(self):
        manager = main.GlobalIdentityManager()
        person_a = embedding(1, 0, 0)
        person_b = embedding(0, 1, 0)

        gid_a = manager.assign_batch(
            "A",
            [detection(7, person_a)]
        )[0]["gid"]
        gid_b = manager.assign_batch(
            "B",
            [detection(7, person_b)]
        )[0]["gid"]
        transferred = manager.assign_batch(
            "C",
            [detection(99, person_a)]
        )[0]

        self.assertNotEqual(gid_a, gid_b)
        self.assertEqual(
            transferred["gid"],
            gid_a
        )
        self.assertEqual(
            transferred["source"],
            "cross-camera"
        )

    def test_local_id_switch_does_not_swap_global_ids(self):
        manager = main.GlobalIdentityManager()
        person_a = embedding(1, 0, 0)
        person_b = embedding(0, 1, 0)
        first = manager.assign_batch(
            "A",
            [
                detection(10, person_a, 0),
                detection(20, person_b, 100)
            ]
        )
        expected = [
            item["gid"]
            for item in first
        ]
        switched = manager.assign_batch(
            "A",
            [
                detection(20, person_a, 5),
                detection(10, person_b, 95)
            ]
        )

        self.assertEqual(
            [item["gid"] for item in switched],
            expected
        )


if __name__ == "__main__":
    unittest.main()
