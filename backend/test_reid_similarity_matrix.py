import unittest

import numpy as np

from backend.reid_similarity_matrix import analyze_similarity, l2_normalize_rows


def record(sample_id, person_id, camera="cam1"):
    return {
        "sample_id": sample_id,
        "ground_truth_person_id": person_id,
        "camera": camera,
        "source_image": f"{sample_id}.jpg",
        "frame_index": 1,
    }


class ReIDSimilarityMatrixTests(unittest.TestCase):
    def test_normalization_produces_symmetric_unit_diagonal_matrix(self):
        embeddings, original_norms = l2_normalize_rows(
            np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        )
        matrix = embeddings @ embeddings.T

        np.testing.assert_allclose(original_norms, [3.0, 4.0])
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), [1.0, 1.0])

    def test_same_different_extraction_excludes_diagonal_and_computes_gap(self):
        embeddings, _ = l2_normalize_rows(np.array([
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ], dtype=np.float32))
        records = [record("a1", "A"), record("a2", "A", "cam2"), record("b1", "B")]

        report = analyze_similarity(embeddings @ embeddings.T, records)

        self.assertEqual(3, report["diagonal"]["count"])
        self.assertEqual(1, report["same_id_off_diagonal"]["count"])
        self.assertEqual(2, report["different_id"]["count"])
        self.assertAlmostEqual(0.8, report["same_id_off_diagonal"]["mean"], places=6)
        self.assertAlmostEqual(0.3, report["different_id"]["mean"], places=6)
        self.assertAlmostEqual(0.5, report["similarity_gap"], places=6)
        self.assertEqual(
            1,
            report["stratified_by_camera_relation"]["cross_camera"]["same_id"]["count"],
        )

    def test_hard_positive_and_negative_ranking(self):
        records = [
            record("a1", "A"), record("a2", "A"),
            record("b1", "B"), record("b2", "B"),
        ]
        matrix = np.array([
            [1.0, 0.2, 0.9, 0.4],
            [0.2, 1.0, 0.3, 0.8],
            [0.9, 0.3, 1.0, 0.6],
            [0.4, 0.8, 0.6, 1.0],
        ])

        report = analyze_similarity(matrix, records, top_k=2)

        self.assertEqual(0.2, report["hard_positives"][0]["similarity"])
        self.assertEqual("a1", report["hard_positives"][0]["sample_a"]["sample_id"])
        self.assertEqual(0.9, report["hard_negatives"][0]["similarity"])
        self.assertEqual("different_gt", report["hard_negatives"][0]["label"])

    def test_non_symmetric_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            analyze_similarity(
                np.array([[1.0, 0.5], [0.4, 1.0]]),
                [record("a", "A"), record("b", "B")],
            )


if __name__ == "__main__":
    unittest.main()
