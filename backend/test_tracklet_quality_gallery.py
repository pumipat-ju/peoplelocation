import unittest

import numpy as np

from backend import main


class TrackletQualityGalleryTests(unittest.TestCase):

    def setUp(self):
        self.manager = main.GlobalIdentityManager()
        self.manager.identities[1] = {
            "embedding": main.l2_normalize(np.array([1.0, 0.0])),
            "gallery": [],
            "gallery_diagnostics": {
                "accepted_updates": 0,
                "rejected_updates": 0,
                "last_rejection_reason": None,
                "tracklet_sample_count": 0,
                "prototype_quality": 0.0,
            },
        }

    def _detection(self, embedding, **overrides):
        detection = {
            "emb": np.asarray(embedding, dtype=np.float32),
            "detector_confidence": 0.95,
            "crop_size": (80, 160),
            "blur_variance": 100.0,
            "overlap": False,
            "border_clip_ratio": 0.0,
        }
        detection.update(overrides)
        return detection

    def test_blurred_or_occluded_samples_never_update_gallery(self):
        accepted, reason = self.manager._record_tracklet_sample(
            1, "cam1", 7,
            self._detection([1.0, 0.0], blur_variance=0.0), 1.0,
        )
        self.assertFalse(accepted)
        self.assertEqual("blurred_crop", reason)

        accepted, reason = self.manager._record_tracklet_sample(
            1, "cam1", 7,
            self._detection([1.0, 0.0], overlap=True), 2.0,
        )
        self.assertFalse(accepted)
        self.assertEqual("overlap_or_occlusion", reason)
        self.assertEqual([], self.manager.identities[1]["gallery"])

    def test_mature_diverse_tracklet_adds_one_prototype(self):
        samples = ([1.0, 0.0], [0.9, 0.2], [0.8, 0.4])
        outcomes = [
            self.manager._record_tracklet_sample(
                1, "cam1", 7, self._detection(sample), float(index)
            )
            for index, sample in enumerate(samples, start=1)
        ]

        self.assertEqual((False, "tracklet_not_mature"), outcomes[0])
        self.assertEqual((False, "tracklet_not_mature"), outcomes[1])
        self.assertEqual((True, None), outcomes[2])
        self.assertEqual(1, len(self.manager.identities[1]["gallery"]))


if __name__ == "__main__":
    unittest.main()
