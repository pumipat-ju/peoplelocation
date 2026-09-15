import unittest

from backend.evaluate_osnet_balanced_heldout import (
    comparison_rows,
    conclusion,
    flat_metrics,
)


def metrics(value):
    result = {
        key: value for key in (
            "rank_1", "rank_5", "mAP", "roc_auc", "eer", "accuracy",
            "precision", "recall", "f1", "same_id_mean_similarity",
            "different_id_mean_similarity", "similarity_gap",
        )
    }
    result.update({
        "classification_counts": {
            "false_positives": int(value * 10),
            "false_negatives": int(value * 10),
        },
        "valid_queries": 10,
        "skipped_queries": 0,
    })
    return result


class BalancedHeldoutEvaluationTests(unittest.TestCase):
    def test_flat_metrics_includes_counts_and_queries(self):
        flat = flat_metrics(metrics(0.5))
        self.assertIn("false_positives", flat)
        self.assertIn("false_negatives", flat)
        self.assertIn("valid_queries", flat)
        self.assertIn("skipped_queries", flat)

    def test_best_respects_lower_is_better(self):
        models = {
            "pretrained": metrics(0.5),
            "finetuned_v1": metrics(0.4),
            "balanced_v2": metrics(0.3),
        }
        rows = {row["metric"]: row for row in comparison_rows(models)}
        self.assertEqual(["balanced_v2"], rows["eer"]["best"])
        self.assertEqual(["pretrained"], rows["mAP"]["best"])

    def test_conclusion_requires_retrieval_and_verification(self):
        models = {
            "pretrained": metrics(0.3),
            "finetuned_v1": metrics(0.4),
            "balanced_v2": metrics(0.5),
        }
        rows = comparison_rows(models)
        text, retrieval, verification = conclusion(rows)
        self.assertTrue(retrieval)
        self.assertFalse(verification)
        self.assertIn("did not improve both", text)


if __name__ == "__main__":
    unittest.main()
