import unittest

import numpy as np

from backend.osnet_pipeline_audit import build_default_samples, summarize_embedding


class OSNetPipelineAuditTests(unittest.TestCase):
    def test_embedding_summary_reports_shape_norm_and_non_finite_values(self):
        summary = summarize_embedding(np.array([3.0, 4.0], dtype=np.float32))

        self.assertEqual((2,), summary["shape"])
        self.assertAlmostEqual(5.0, summary["L2 norm"])
        self.assertFalse(summary["contains_nan"])
        self.assertFalse(summary["contains_inf"])

        non_finite = summarize_embedding(np.array([np.nan, np.inf], dtype=np.float32))
        self.assertTrue(non_finite["contains_nan"])
        self.assertTrue(non_finite["contains_inf"])

    def test_default_samples_do_not_share_image_storage(self):
        samples = build_default_samples()

        self.assertEqual(3, len(samples))
        for left_index, left in enumerate(samples):
            for right in samples[left_index + 1:]:
                self.assertFalse(np.shares_memory(left, right))


if __name__ == "__main__":
    unittest.main()
