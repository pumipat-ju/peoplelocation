import unittest

from backend.evaluate_osnet_heldout import (
    comparison_rows,
    threshold_metrics,
    validate_peoplelocation_records,
)


class EvaluateOsnetHeldoutTests(unittest.TestCase):
    def test_threshold_metrics(self):
        metrics = threshold_metrics(
            [0.9, 0.8, 0.4, 0.1], [True, False, True, False], 0.5
        )
        self.assertEqual(1, metrics["true_positives"])
        self.assertEqual(1, metrics["true_negatives"])
        self.assertEqual(1, metrics["false_positives"])
        self.assertEqual(1, metrics["false_negatives"])
        self.assertEqual(0.5, metrics["accuracy"])
        self.assertEqual(0.5, metrics["f1"])

    def test_comparison_treats_lower_eer_as_improvement(self):
        baseline = {
            key: 0.5 for key in (
                "rank_1", "rank_5", "mAP", "roc_auc", "eer", "accuracy",
                "precision", "recall", "f1", "same_id_mean_similarity",
                "different_id_mean_similarity", "similarity_gap",
            )
        }
        fine = dict(baseline)
        fine["eer"] = 0.4
        fine["mAP"] = 0.6
        rows = {row["metric"]: row for row in comparison_rows(baseline, fine)}
        self.assertEqual("improved", rows["eer"]["effect"])
        self.assertEqual("improved", rows["mAP"]["effect"])

    def test_rejects_non_peoplelocation_test_record(self):
        record = {
            "sample_id": "x", "dataset_source": "msmt17",
            "split": "test", "identity_key": "msmt17:0001",
        }
        with self.assertRaises(ValueError):
            validate_peoplelocation_records([record], "test")


if __name__ == "__main__":
    unittest.main()
