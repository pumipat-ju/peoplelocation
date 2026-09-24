import unittest

import numpy as np

from backend.evaluate_gallery_scoring_phase_c import (
    VARIANTS, decide, score_gallery, score_matrix, support_score,
)
from backend.reid.gallery import gallery_similarity


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class GalleryScoringPhaseCTests(unittest.TestCase):
    def test_bottom_three_matches_production_and_gallery_order(self):
        query = unit(1, 0, 0, 0)
        gallery = [unit(1, 0, 0, 0), unit(0.8, 0.6, 0, 0),
                   unit(0.7, 0, 0.714, 0), unit(0, 0, 0, 1)]
        identity = {"gallery": gallery, "embedding": unit(0.9, 0.2, 0.1, 0.2)}
        baseline = gallery_similarity(
            query, identity, diversity_threshold=0.985,
            prototype_enabled=True, prototype_min_samples=2,
            prototype_min_consensus=0.70, prototype_weight=0.75,
            support_weight=0.25)
        self.assertAlmostEqual(baseline, score_gallery(query, identity, "bottom-3"), places=6)
        reversed_identity = {"gallery": list(reversed(gallery)),
                             "embedding": identity["embedding"]}
        self.assertAlmostEqual(baseline,
                               score_gallery(query, reversed_identity, "bottom-3"), places=6)

    def test_support_score_definitions_and_poor_gallery_sample(self):
        scores = [0.98, 0.91, 0.83, -0.3]
        self.assertAlmostEqual(0.83, support_score(scores, "bottom-3"))
        self.assertAlmostEqual(0.91, support_score(scores, "top-3"))
        self.assertAlmostEqual(np.mean(scores), support_score(scores, "mean-all"))
        self.assertAlmostEqual(np.median(scores), support_score(scores, "median"))
        self.assertAlmostEqual(np.mean([0.83, 0.91, 0.98]),
                               support_score(scores, "top-k-mean-margin"))
        self.assertLess(support_score(scores, "bottom-3"),
                        support_score(scores, "top-3"))

    def test_same_person_multiview_and_determinism(self):
        identities = {
            "person-a": {"gallery": [unit(1, 0, 0), unit(0.7, 0.7, 0),
                                      unit(0.65, 0, 0.76), unit(0, 1, 0)],
                         "embedding": unit(0.9, 0.3, 0.1)},
            "person-b": {"gallery": [unit(0, 0, 1), unit(0.1, 0, 0.99)],
                         "embedding": unit(0, 0, 1)},
        }
        features = {"query": unit(1, 0, 0)}
        queries = [{"sample_id": "query", "dataset_identity_key": "person-a"}]
        for variant in VARIANTS:
            keys, matrix = score_matrix(queries, identities, features, variant)
            self.assertEqual("person-a", keys[int(np.argmax(matrix[0]))])
            self.assertTrue(np.array_equal(matrix,
                                           score_matrix(queries, identities, features,
                                                        variant)[1]))

    def test_similar_looking_person_margin_guard_favors_safety(self):
        scores = np.array([0.80, 0.77, 0.20])
        self.assertEqual(0, decide(scores, "top-3"))
        self.assertIsNone(decide(scores, "top-k-mean-margin"))
        self.assertIsNone(decide(np.array([0.54, 0.40, 0.30]), "bottom-3"))


if __name__ == "__main__":
    unittest.main()
