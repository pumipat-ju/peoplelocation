import unittest

from backend.baseline_metrics import (
    classification_metrics,
    confusion_at_threshold,
    evaluate_global_id_records,
    match_frame_observations,
)


def row(sample, sequence, camera, person, frame, predicted):
    return {
        "sample_id": sample,
        "sequence": sequence,
        "camera": camera,
        "dataset_identity_key": f"{sequence}:{person}",
        "frame_index": frame,
        "predicted_global_identity": f"{sequence}:{predicted}",
    }


class BaselineMetricTests(unittest.TestCase):
    def test_confusion_and_classification_metrics(self):
        confusion = confusion_at_threshold(
            [0.9, 0.7, 0.6, 0.1], [1, 0, 1, 0], 0.65
        )
        self.assertEqual({"tp": 1, "tn": 1, "fp": 1, "fn": 1}, confusion)
        metrics = classification_metrics(confusion)
        self.assertEqual(0.5, metrics["accuracy"])
        self.assertEqual(0.5, metrics["precision"])
        self.assertEqual(0.5, metrics["recall"])
        self.assertEqual(0.5, metrics["f1"])

    def test_iou_matching_is_one_to_one_and_ignores_local_id_as_gt(self):
        gt = [
            {"sample_id": "a", "bbox_xyxy": [0, 0, 10, 10]},
            {"sample_id": "b", "bbox_xyxy": [20, 0, 30, 10]},
        ]
        predictions = [
            {"predicted_global_id": 8, "local_track_id": 999,
             "predicted_bbox_xyxy": [20, 0, 30, 10]},
            {"predicted_global_id": 7, "local_track_id": 1,
             "predicted_bbox_xyxy": [0, 0, 10, 10]},
        ]
        self.assertEqual([(0, 1, 1.0), (1, 0, 1.0)], match_frame_observations(gt, predictions))

    def test_known_switch_split_and_merge_counts(self):
        records = [
            row("a1", "legacy_sequence", "cam1", 2, 1, 10),
            row("a2", "legacy_sequence", "cam1", 2, 2, 10),
            row("a3", "legacy_sequence", "cam1", 2, 3, 20),
            row("b1", "legacy_sequence", "cam1", 3, 1, 30),
            row("b2", "legacy_sequence", "cam1", 3, 2, 30),
            row("c1", "legacy_sequence", "cam2", 4, 1, 30),
        ]
        result = evaluate_global_id_records(records)
        self.assertEqual(1, result["metrics"]["id_switches"])
        self.assertEqual(1, result["metrics"]["identity_splits"])
        self.assertEqual(1, result["metrics"]["identity_split_excess_predicted_ids"])
        self.assertEqual(1, result["metrics"]["identity_merges"])
        self.assertEqual(2, result["metrics"]["identity_merge_affected_gt_identities"])

    def test_sequences_with_same_numbers_never_merge(self):
        result = evaluate_global_id_records([
            row("l", "legacy_sequence", "cam1", 2, 1, 1),
            row("m", "m_sequence", "cam1", 2, 1, 1),
        ])
        self.assertEqual(0, result["metrics"]["identity_merges"])
        self.assertEqual(2, result["metrics"]["correct_global_ids"])

    def test_switches_are_scoped_to_camera_trajectory(self):
        result = evaluate_global_id_records([
            row("a", "s", "cam1", 1, 1, 1),
            row("b", "s", "cam2", 1, 1, 2),
        ])
        self.assertEqual(0, result["metrics"]["id_switches"])
        self.assertEqual(1, result["metrics"]["identity_splits"])


if __name__ == "__main__":
    unittest.main()
