import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.reid_experiments.common import (
    DEFAULT_MANIFEST,
    FINETUNED_CHECKPOINT,
    IMPROVED_CROP,
    ORIGINAL_CROP,
    PRETRAINED_CHECKPOINT,
    SPECS,
    FinetunedCheckpointNotAvailable,
    production_crop,
    raw_gt_crop,
    retrieval_metrics,
    validate_checkpoint,
)


class ReIDExperimentRunnerTests(unittest.TestCase):
    def test_four_runner_configurations_are_isolated(self):
        expected = (
            ("01_pretrained_osnet", "pre-trained", ORIGINAL_CROP),
            ("02_pretrained_osnet_improved_crop", "pre-trained", IMPROVED_CROP),
            ("03_finetuned_osnet", "fine-tuned", ORIGINAL_CROP),
            ("04_finetuned_osnet_improved_crop", "fine-tuned", IMPROVED_CROP),
        )
        for name, checkpoint_kind, crop_policy in expected:
            spec = SPECS[name]
            self.assertEqual(checkpoint_kind, spec.checkpoint_kind)
            self.assertEqual(crop_policy, spec.crop_policy)
        self.assertEqual(PRETRAINED_CHECKPOINT, SPECS[expected[0][0]].checkpoint_path)
        self.assertEqual(PRETRAINED_CHECKPOINT, SPECS[expected[1][0]].checkpoint_path)
        self.assertEqual(FINETUNED_CHECKPOINT, SPECS[expected[2][0]].checkpoint_path)
        self.assertEqual(FINETUNED_CHECKPOINT, SPECS[expected[3][0]].checkpoint_path)

    def test_output_directories_do_not_overlap(self):
        directories = [spec.output_directory_name for spec in SPECS.values()]
        self.assertEqual(len(directories), len(set(directories)))

    def test_every_runner_uses_same_manifest_protocol(self):
        self.assertTrue(DEFAULT_MANIFEST.name == "master_manifest_v1.jsonl")
        self.assertEqual(1, len({str(DEFAULT_MANIFEST) for _ in SPECS.values()}))

    def test_baseline_and_improved_crop_cannot_be_swapped(self):
        class Runtime:
            @staticmethod
            def extract_person_crop(frame, x1, y1, x2, y2):
                return frame[y1 + 1:y2 - 1, x1 + 1:x2 - 1]

        frame = np.arange(20 * 20 * 3).reshape(20, 20, 3)
        record = {"bbox_xyxy": [2, 3, 12, 15]}
        np.testing.assert_array_equal(frame[4:14, 3:11], production_crop(Runtime, frame, record))
        np.testing.assert_array_equal(frame[3:15, 2:12], raw_gt_crop(frame, record))

    def test_finetuned_never_falls_back_to_pretrained(self):
        with self.assertRaisesRegex(ValueError, "FINETUNED_CHECKPOINT_REQUIRED"):
            validate_checkpoint(SPECS["03_finetuned_osnet"], PRETRAINED_CHECKPOINT)
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pth"
            with self.assertRaisesRegex(
                FinetunedCheckpointNotAvailable,
                "FINETUNED_CHECKPOINT_NOT_AVAILABLE",
            ):
                validate_checkpoint(SPECS["04_finetuned_osnet_improved_crop"], missing)

    def test_pretrained_runners_are_pinned_to_market1501_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "other.pth"
            wrong.touch()
            with self.assertRaisesRegex(ValueError, "PRETRAINED_CHECKPOINT_REQUIRED"):
                validate_checkpoint(SPECS["01_pretrained_osnet"], wrong)

    def test_rank_and_map_use_cross_camera_gallery(self):
        records = [
            {"sample_id": "a", "camera": "c1", "dataset_identity_key": "p1"},
            {"sample_id": "b", "camera": "c2", "dataset_identity_key": "p1"},
            {"sample_id": "c", "camera": "c2", "dataset_identity_key": "p2"},
        ]
        similarities = np.asarray([
            [1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]
        ])
        metrics = retrieval_metrics(similarities, records)
        self.assertEqual(2, metrics["valid_query_count"])
        self.assertEqual(1.0, metrics["rank_1"])
        self.assertEqual(1.0, metrics["rank_5"])
        self.assertEqual(1.0, metrics["mAP"])


if __name__ == "__main__":
    unittest.main()
