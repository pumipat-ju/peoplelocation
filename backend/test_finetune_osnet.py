import unittest

import numpy as np
import torch

from backend.finetune_osnet import (
    IdentityBalancedBatchSampler,
    batch_hard_triplet_loss,
    compute_roc_eer,
    require_cuda_device,
    validation_metrics,
)


class FinetuneOsnetTests(unittest.TestCase):
    def test_identity_sampler_provides_p_by_k_batches_deterministically(self):
        records = [
            {"identity_key": f"id:{identity}"}
            for identity in range(5) for _ in range(3)
        ]
        first = IdentityBalancedBatchSampler(records, 4, 2, 7)
        second = IdentityBalancedBatchSampler(records, 4, 2, 7)
        batches = list(first)
        self.assertEqual(batches, list(second))
        self.assertEqual(2, len(batches))
        for batch in batches:
            identities = [records[index]["identity_key"] for index in batch]
            counts = list(__import__("collections").Counter(identities).values())
            self.assertEqual([2, 2, 2, 2], sorted(counts))

    def test_combined_loss_is_finite_and_has_gradients(self):
        features = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]
        ], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        logits = torch.randn(4, 2, requires_grad=True)
        loss = (
            torch.nn.functional.cross_entropy(logits, labels)
            + batch_hard_triplet_loss(features, labels)
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(features.grad)
        self.assertIsNotNone(logits.grad)

    def test_validation_metrics_include_required_values(self):
        embeddings = np.asarray([
            [1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99],
        ], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        metrics = validation_metrics(
            embeddings, ["a", "a", "b", "b"], ["c1", "c2", "c1", "c2"]
        )
        for key in (
            "rank_1", "rank_5", "mAP", "roc_auc", "eer",
            "same_id_mean_similarity", "different_id_mean_similarity",
            "similarity_gap",
        ):
            self.assertTrue(np.isfinite(metrics[key]), key)
        self.assertEqual(1.0, metrics["rank_1"])
        self.assertEqual(1.0, metrics["mAP"])

    def test_roc_eer_rejects_one_class(self):
        with self.assertRaises(ValueError):
            compute_roc_eer([0.1, 0.2], [1, 1])

    def test_cpu_device_is_rejected_without_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA_REQUIRED"):
            require_cuda_device(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
