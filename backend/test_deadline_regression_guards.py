import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np

os.environ["IDENTITY_DB_PATH"] = ":memory:"
os.environ["REID_ENABLED"] = "false"

from backend import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def detection(track_id, event_time):
    return {
        "tid": track_id,
        "emb": np.asarray([1.0, 0.0], dtype=np.float32),
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
    }


class FakeVideoCapture:
    def __init__(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def set(self, *_args):
        return True

    def get(self, property_id):
        if property_id == main.cv2.CAP_PROP_FPS:
            return 25.0
        if property_id == main.cv2.CAP_PROP_FRAME_COUNT:
            return 100.0
        return 0.0

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame.copy()

    def release(self):
        return None


class DeadlineProductionPathTests(unittest.TestCase):

    def test_allowed_handoff_recovers_gid_without_capture_worker(self):
        manager = main.GlobalIdentityManager()
        topology = {
            "version": 2,
            "enforce": True,
            "transitions": [{
                "from_camera": "cam-a",
                "to_camera": "cam-b",
                "min_travel_time_sec": 2.0,
                "max_travel_time_sec": 10.0,
                "overlap_allowed": False,
            }],
        }

        with (
            patch.object(main, "topology_config", topology),
            patch.object(
                main.cv2,
                "VideoCapture",
                side_effect=AssertionError("global assignment opened capture"),
            ),
        ):
            first = None
            for frame_index, event_time in enumerate((100.0, 100.1, 100.2)):
                first = manager.assign_global_batch(
                    {"cam-a": [detection(7, event_time)]},
                    event_time=event_time,
                    batch_id=f"handoff-source-{frame_index}",
                )["cam-a"][0]

            source_gid = first["gid"]
            self.assertEqual(
                main.IDENTITY_ACTIVE,
                manager.identities[source_gid]["state"],
            )
            handoff = manager.assign_global_batch(
                {"cam-b": [detection(9, 102.3)]},
                event_time=102.3,
                batch_id="allowed-handoff",
            )["cam-b"][0]

        self.assertEqual(source_gid, handoff["gid"])
        self.assertEqual("global-cross-camera", handoff["source"])
        topology_decision = next(
            item
            for item in manager.last_global_batch_diagnostics[
                "topology_gate_decisions"
            ]
            if item["gid"] == source_gid
        )
        self.assertTrue(topology_decision["passed"])
        self.assertEqual("cam-a", topology_decision["source_camera"])
        self.assertEqual("cam-b", topology_decision["destination_camera"])

    def test_uploaded_offset_reaches_backend_event_time_logic_with_fake_capture(self):
        manager = main.MultiCameraVideoManager()
        with patch.object(
            main.cv2,
            "VideoCapture",
            return_value=FakeVideoCapture(),
        ):
            manager.register_video(
                "uploaded-a",
                "synthetic.mp4",
                loop_video=False,
                time_offset_sec=7.5,
            )
            manager.videos["uploaded-a"]["playback_started_at"] = 1000.0
            manager.running["uploaded-a"] = True
            frame = manager.read_synchronized_frames()["uploaded-a"]

        self.assertEqual(7.5, frame["time_offset_sec"])
        self.assertAlmostEqual(1007.54, frame["event_time"])

    def test_clean_backend_import_cannot_open_a_real_device(self):
        script = textwrap.dedent(
            """
            import os
            os.environ['IDENTITY_DB_PATH'] = ':memory:'
            os.environ['REID_ENABLED'] = 'false'
            import cv2
            def forbidden_capture(*args, **kwargs):
                raise AssertionError('backend import attempted VideoCapture')
            cv2.VideoCapture = forbidden_capture
            import backend.main
            """
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(REPOSITORY_ROOT),
            existing_python_path,
        )))
        with tempfile.TemporaryDirectory(
            prefix="peoplelocation-import-guard-"
        ) as runtime_directory:
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=runtime_directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )

        self.assertEqual(
            0,
            completed.returncode,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
