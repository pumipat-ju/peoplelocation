import os
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torchreid

import backend.main as main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BALANCED_CHECKPOINT = (
    PROJECT_ROOT
    / "backend/reid_experiments/finetune_osnet_v2_balanced/best_checkpoint.pth"
)
PRETRAINED_CHECKPOINT = PROJECT_ROOT / "weights/osnet_x1_0_market1501.pth"


class ReIDV5CropTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(100 * 100 * 3, dtype=np.int64).reshape(
            100, 100, 3
        )

    def test_original_preserves_previous_crop_behavior(self):
        expected = main.extract_person_crop(self.frame, 10, 20, 60, 90)
        actual = main.get_reid_crop(
            self.frame, 10, 20, 60, 90, crop_mode="original"
        )
        np.testing.assert_array_equal(expected, actual)

    def test_missing_crop_mode_defaults_to_original(self):
        self.assertEqual("original", main.resolve_reid_crop_mode(None))
        self.assertEqual("original", main.REID_CROP_MODE if
                         "REID_CROP_MODE" not in os.environ else
                         main.resolve_reid_crop_mode(None))

    def test_invalid_crop_mode_warns_and_falls_back(self):
        with self.assertLogs(main.logger, level="WARNING") as captured:
            mode = main.resolve_reid_crop_mode("unsafe-mode")
        self.assertEqual("original", mode)
        self.assertIn("using original", "\n".join(captured.output))

    def test_improved_is_clipped_raw_bbox_without_margin(self):
        actual = main.get_reid_crop(
            self.frame, -10, 20, 120, 90, crop_mode="improved"
        )
        # Preserve production's existing exclusive-slice convention: clamp_bbox
        # caps x2 at width - 1.
        np.testing.assert_array_equal(self.frame[20:90, 0:99], actual)

    def test_invalid_or_empty_bbox_is_rejected(self):
        self.assertIsNone(main.get_reid_crop(
            self.frame, 10, 10, 10, 50, crop_mode="improved"
        ))
        self.assertIsNone(main.get_reid_crop(
            self.frame, 200, 200, 300, 300, crop_mode="improved"
        ))

    def test_improved_crop_does_not_mutate_bbox(self):
        bbox = [10, 20, 60, 90]
        before = list(bbox)
        actual = main.get_reid_crop(
            self.frame, *bbox, crop_mode="improved"
        )
        self.assertEqual(before, bbox)
        np.testing.assert_array_equal(self.frame[20:90, 10:60], actual)


@unittest.skipUnless(
    BALANCED_CHECKPOINT.is_file() and PRETRAINED_CHECKPOINT.is_file(),
    "checkpoint-backed tests require local ignored OSNet weights",
)
class ReIDV5CheckpointTests(unittest.TestCase):
    @staticmethod
    def _model():
        return torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000,
            loss="softmax", pretrained=False,
        )

    def test_balanced_checkpoint_loads_feature_extractor_intentionally(self):
        model = self._model()
        loaded, metadata, excluded = main.load_validated_osnet_checkpoint(
            model, BALANCED_CHECKPOINT
        )
        self.assertEqual(565, loaded)
        self.assertEqual(
            ["classifier.bias", "classifier.weight"], excluded
        )
        self.assertEqual(5, metadata["epoch"])
        self.assertFalse(model.training)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        with torch.inference_mode():
            embeddings = F.normalize(
                model(torch.randn(2, 3, 256, 128, device=device)), p=2, dim=1
            )
        self.assertEqual((2, 512), tuple(embeddings.shape))
        self.assertTrue(bool(torch.isfinite(embeddings).all()))
        np.testing.assert_allclose(
            embeddings.norm(dim=1).cpu().numpy(), np.ones(2), atol=1e-5
        )

    def test_pretrained_checkpoint_remains_loadable(self):
        model = self._model()
        loaded, _, excluded = main.load_validated_osnet_checkpoint(
            model, PRETRAINED_CHECKPOINT
        )
        self.assertEqual(565, loaded)
        self.assertEqual(
            ["classifier.bias", "classifier.weight"], excluded
        )
        self.assertFalse(model.training)

    def test_non_classifier_incompatibility_is_rejected(self):
        checkpoint = torch.load(
            BALANCED_CHECKPOINT, map_location="cpu", weights_only=False
        )
        state = dict(checkpoint["state_dict"])
        removed_key = next(key for key in state if not key.startswith("classifier."))
        state.pop(removed_key)
        checkpoint = dict(checkpoint)
        checkpoint["state_dict"] = state

        original_load = torch.load
        try:
            torch.load = lambda *args, **kwargs: checkpoint
            with self.assertRaisesRegex(RuntimeError, "missing="):
                main.load_validated_osnet_checkpoint(
                    self._model(), BALANCED_CHECKPOINT
                )
        finally:
            torch.load = original_load


if __name__ == "__main__":
    unittest.main()
