import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.prepare_m_sequence_dataset import extraction_plan, validate_detections


class PrepareMSequenceDatasetTests(unittest.TestCase):
    def test_extraction_plan_is_deterministic_and_applies_offset(self):
        metadata = {"fps": 30.0, "frame_count": 300}
        plan = extraction_plan(metadata, 5.0, 2.0, 2.0)
        self.assertEqual(len(plan), 5)
        self.assertEqual(plan[0]["source_frame_zero_based"], 150)
        self.assertEqual(plan[-1]["source_frame_zero_based"], 210)
        self.assertEqual([row["dataset_frame"] for row in plan], [1, 2, 3, 4, 5])

    def test_validation_detects_duplicate_and_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            images = Path(temporary_directory)
            cv2.imwrite(str(images / "000001.jpg"), np.zeros((100, 200, 3), np.uint8))
            valid = {
                "frame": 1, "local_track_id": 7, "x": 10, "y": 20,
                "width": 30, "height": 40, "confidence": 0.9,
            }
            invalid = {**valid, "x": 190, "width": 20}
            errors = validate_detections("cam1", images, [valid, invalid])
            reasons = [item["reason"] for item in errors]
            self.assertIn("duplicate_camera_frame_person", reasons)
            self.assertIn("bbox_out_of_bounds", reasons)

    def test_validation_detects_missing_frame(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            detection = {
                "frame": 99, "local_track_id": 1, "x": 0, "y": 0,
                "width": 10, "height": 10, "confidence": 0.9,
            }
            errors = validate_detections(
                "cam2", Path(temporary_directory), [detection]
            )
            self.assertEqual(errors[0]["reason"], "missing_frame")


if __name__ == "__main__":
    unittest.main()
