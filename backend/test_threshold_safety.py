import unittest
from unittest.mock import patch

from backend import main


class ThresholdSafetyTests(unittest.TestCase):

    def setUp(self):
        self.manager = object.__new__(
            main.GlobalIdentityManager
        )
        self.identity = {
            "last_cam": "cam1"
        }
        self.detection = {}

    def test_conservative_mode_disables_cross_camera_similarity_shortcut(self):
        pair = {
            "appearance": (
                main.REID_CROSS_CAM_STRONG_THRESHOLD
                + 0.01
            ),
            "score": (
                main.ASSIGN_CROSS_CAM_SCORE_THRESHOLD
                - 0.01
            ),
            "cross_camera": True
        }

        with patch.object(
            main,
            "REID_THRESHOLD_SAFETY_MODE",
            "conservative"
        ):
            self.assertFalse(
                self.manager._accept_match(
                    pair,
                    self.identity,
                    "cam2",
                    self.detection
                )
            )

    def test_validated_mode_allows_measured_strong_shortcut(self):
        pair = {
            "appearance": (
                main.REID_CROSS_CAM_STRONG_THRESHOLD
                + 0.01
            ),
            "score": (
                main.ASSIGN_CROSS_CAM_SCORE_THRESHOLD
                - 0.01
            ),
            "cross_camera": True
        }

        with patch.object(
            main,
            "REID_THRESHOLD_SAFETY_MODE",
            "validated"
        ):
            self.assertTrue(
                self.manager._accept_match(
                    pair,
                    self.identity,
                    "cam2",
                    self.detection
                )
            )


if __name__ == "__main__":
    unittest.main()
