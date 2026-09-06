import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from backend.dump_reid_debug import (
    crop_quality_flags,
    prepare_output_directories,
    validate_and_convert_bbox,
    write_sample_artifacts,
)


class DumpReIDDebugTests(unittest.TestCase):
    def test_wide_crop_is_flagged_without_being_rejected(self):
        aspect_ratio, flags = crop_quality_flags(
            np.zeros((40, 120, 3), dtype=np.uint8)
        )

        self.assertEqual(3.0, aspect_ratio)
        self.assertIn("wide_or_square_crop", flags)
        self.assertIn("extreme_aspect_ratio", flags)

    def test_invalid_bbox_rejection_and_frame_clamping(self):
        bbox, reason = validate_and_convert_bbox((100, 200, 3), (10, 20, -5, 30))
        self.assertIsNone(bbox)
        self.assertEqual("non_positive_bbox_size", reason)

        bbox, reason = validate_and_convert_bbox((100, 200, 3), (-10, 20, 40, 50))
        self.assertIsNone(reason)
        self.assertEqual([0, 20, 30, 70], bbox["frame_clamped_xyxy"])
        self.assertTrue(bbox["was_clamped"])

        bbox, reason = validate_and_convert_bbox((100, 200, 3), (210, 20, 10, 10))
        self.assertIsNone(bbox)
        self.assertEqual("bbox_outside_frame", reason)

    def test_artifacts_have_consistent_shape_norm_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directories = prepare_output_directories(Path(temporary_directory) / "debug")
            crop = np.full((64, 32, 3), 127, dtype=np.uint8)
            embedding = np.array([3.0, 4.0], dtype=np.float32) / 5.0
            metadata = {
                "embedding_dimension": 2,
                "embedding_l2_norm": float(np.linalg.norm(embedding)),
                "crop_width": 32,
                "crop_height": 64,
            }

            crop_path, embedding_path, metadata_path = write_sample_artifacts(
                directories, "sample", crop, embedding, metadata
            )

            saved_crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            saved_embedding = np.load(embedding_path)
            with metadata_path.open("r", encoding="utf-8") as source:
                saved_metadata = json.load(source)

            self.assertEqual(crop.shape, saved_crop.shape)
            self.assertEqual((2,), saved_embedding.shape)
            self.assertAlmostEqual(1.0, float(np.linalg.norm(saved_embedding)), places=6)
            self.assertEqual(saved_embedding.size, saved_metadata["embedding_dimension"])
            self.assertEqual(saved_crop.shape[1], saved_metadata["crop_width"])
            self.assertEqual(saved_crop.shape[0], saved_metadata["crop_height"])

    def test_artifact_dump_does_not_mutate_assignment_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directories = prepare_output_directories(Path(temporary_directory) / "debug")
            assignment_state = {"identities": {7: {"state": "ACTIVE"}}}
            before = json.dumps(assignment_state, sort_keys=True)

            write_sample_artifacts(
                directories,
                "isolated",
                np.zeros((32, 32, 3), dtype=np.uint8),
                np.array([1.0, 0.0], dtype=np.float32),
                {"embedding_dimension": 2},
            )

            self.assertEqual(before, json.dumps(assignment_state, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
