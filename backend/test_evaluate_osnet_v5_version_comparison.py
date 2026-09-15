import unittest

from backend.evaluate_osnet_v5_version_comparison import (
    IMPROVED_CROP,
    ORIGINAL_CROP,
    crop_box,
    effect,
)


class V5ComparisonTests(unittest.TestCase):
    def test_improved_crop_is_exact_raw_bbox(self):
        self.assertEqual((10, 20, 60, 90), crop_box(
            (10, 20, 60, 90), 100, 100, IMPROVED_CROP
        ))

    def test_original_crop_matches_production_margin_geometry(self):
        self.assertEqual((19, 27, 51, 78), crop_box(
            (10, 20, 60, 90), 100, 100, ORIGINAL_CROP
        ))

    def test_metric_direction_is_explicit(self):
        self.assertEqual("improved", effect("mAP", 0.1))
        self.assertEqual("improved", effect("eer", -0.1))
        self.assertEqual("not_directional", effect(
            "validation_selected_threshold", 0.1
        ))
        self.assertEqual("unchanged", effect("pair_count", 0.0))


if __name__ == "__main__":
    unittest.main()
