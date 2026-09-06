import unittest

import numpy as np

from backend.run_reid_crop_ablation import (
    aspect_ratio_pad,
    quality_assessment,
    select_samples,
)


class ReIDCropAblationTests(unittest.TestCase):
    def test_aspect_ratio_padding_has_fixed_shape_and_preserves_content_ratio(self):
        crop = np.full((100, 50, 3), 255, dtype=np.uint8)
        padded = aspect_ratio_pad(crop)
        self.assertEqual((256, 128, 3), padded.shape)
        content = np.all(padded == 255, axis=2)
        ys, xs = np.where(content)
        self.assertAlmostEqual(0.5, (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1), places=2)

    def test_quality_filter_flags_aspect_and_border(self):
        record = {
            "bbox_xyxy": [0, 10, 100, 60],
            "image_width": 200,
            "image_height": 100,
        }
        result = quality_assessment(record)
        self.assertFalse(result["quality_eligible"])
        self.assertIn("suspicious_aspect_ratio", result["quality_reasons"])
        self.assertIn("touches_frame_border", result["quality_reasons"])

    def test_sample_selection_is_deterministic_and_group_bounded(self):
        records = [
            {
                "sample_id": f"s{index:02d}",
                "dataset_identity_key": "sequence:1",
                "camera": "cam1",
                "frame_index": index,
            }
            for index in range(10)
        ]
        first = select_samples(records, 4)
        second = select_samples(records, 4)
        self.assertEqual(first, second)
        self.assertEqual(4, len(first))
        self.assertEqual(["s00", "s03", "s06", "s09"], [row["sample_id"] for row in first])


if __name__ == "__main__":
    unittest.main()
