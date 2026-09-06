import unittest

import numpy as np

from backend.evaluate_reid_verification import (
    build_pair_rows,
    compute_eer,
    compute_roc,
    evaluate_pairs,
)
from backend.reid_similarity_matrix import l2_normalize_rows


def record(sample_id, person_id, camera):
    return {
        "sample_id": sample_id,
        "ground_truth_person_id": person_id,
        "camera": camera,
        "frame_index": 1,
    }


class EvaluateReIDVerificationTests(unittest.TestCase):
    def test_perfect_scores_have_auc_one_and_eer_zero(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])

        fpr, tpr, thresholds, auc = compute_roc(scores, labels)
        eer, threshold = compute_eer(fpr, tpr, thresholds)

        self.assertAlmostEqual(1.0, auc)
        self.assertAlmostEqual(0.0, eer)
        self.assertAlmostEqual(0.8, threshold)

    def test_pair_labels_use_gt_and_cross_camera_is_explicit(self):
        records = [
            record("a1", "person-a", "cam1"),
            record("a2", "person-a", "cam2"),
            record("b1", "person-b", "cam2"),
        ]
        matrix = np.eye(3)
        pairs = build_pair_rows(matrix, records)

        self.assertEqual(3, len(pairs))
        self.assertTrue(pairs[0]["same_gt_identity"])
        self.assertEqual("cross_camera", pairs[0]["camera_relation"])
        self.assertFalse(pairs[1]["same_gt_identity"])
        self.assertEqual(
            2,
            len([pair for pair in pairs if pair["camera_relation"] == "cross_camera"]),
        )

    def test_evaluation_reports_counts_threshold_and_normalized_embeddings(self):
        embeddings, _ = l2_normalize_rows(np.array([
            [3.0, 0.0],
            [4.0, 0.0],
            [0.0, 2.0],
            [0.0, 5.0],
        ], dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)
        records = [
            record("a1", "A", "cam1"), record("a2", "A", "cam2"),
            record("b1", "B", "cam1"), record("b2", "B", "cam2"),
        ]
        pairs = build_pair_rows(embeddings @ embeddings.T, records)

        report = evaluate_pairs(pairs)

        self.assertEqual(2, report["same_id"]["count"])
        self.assertEqual(4, report["different_id"]["count"])
        self.assertAlmostEqual(1.0, report["roc_auc"])
        self.assertAlmostEqual(0.0, report["eer"])
        self.assertEqual("Youden J = TPR - FPR", report["best_threshold_method"])

    def test_tied_scores_are_processed_as_one_roc_threshold(self):
        fpr, tpr, thresholds, auc = compute_roc(
            np.array([0.5, 0.5]), np.array([1, 0])
        )

        self.assertEqual(2, len(thresholds))
        self.assertAlmostEqual(0.5, auc)
        np.testing.assert_allclose(fpr, [0.0, 1.0])
        np.testing.assert_allclose(tpr, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
