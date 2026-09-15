import os
import unittest

import numpy as np

from backend.reid_versions import (
    V1_PRETRAINED_OSNET,
    V2_PRETRAINED_OSNET_IMPROVED_CROP,
    V3_FINETUNED_OSNET,
    V4_FINETUNED_OSNET_IMPROVED_CROP,
    get_reid_version,
    get_version,
    list_versions,
    version_names,
)
from backend.reid_versions.model_finetuned_osnet import (
    FineTunedCheckpointUnavailableError,
)


class ReIDVersionRegistryTests(unittest.TestCase):
    def test_registry_exposes_all_required_versions_and_configuration(self):
        self.assertEqual((
            V1_PRETRAINED_OSNET,
            V2_PRETRAINED_OSNET_IMPROVED_CROP,
            V3_FINETUNED_OSNET,
            V4_FINETUNED_OSNET_IMPROVED_CROP,
        ), version_names())
        required = {
            "version_name", "display_name", "model_architecture", "checkpoint_path",
            "crop_policy", "input_size", "preprocessing",
            "l2_normalization", "similarity_method", "status",
        }
        for configuration in list_versions():
            self.assertTrue(required.issubset(configuration))
            self.assertEqual("osnet_x1_0", configuration["model_architecture"])
            self.assertEqual({"height": 256, "width": 128}, configuration["input_size"])
            self.assertTrue(configuration["l2_normalization"])
            self.assertFalse(configuration["production_default"])

    def test_v1_crop_matches_recovered_margin_geometry(self):
        frame = np.arange(100 * 100 * 3, dtype=np.int64).reshape(100, 100, 3)
        actual = get_reid_version(V1_PRETRAINED_OSNET).crop(
            frame, (10, 10, 90, 90)
        )
        np.testing.assert_array_equal(frame[18:76, 24:76], actual)

    def test_improved_crop_is_exact_raw_gt_bbox(self):
        frame = np.arange(100 * 100 * 3, dtype=np.int64).reshape(100, 100, 3)
        actual = get_reid_version(V2_PRETRAINED_OSNET_IMPROVED_CROP).crop(
            frame, (10, 20, 60, 90)
        )
        np.testing.assert_array_equal(frame[20:90, 10:60], actual)

    def test_finetuned_versions_are_explicitly_unavailable(self):
        for name in (V3_FINETUNED_OSNET, V4_FINETUNED_OSNET_IMPROVED_CROP):
            version = get_reid_version(name)
            self.assertEqual("NOT_AVAILABLE_YET", version.status)
            self.assertIsNone(version.configuration()["checkpoint_path"])
            with self.assertRaisesRegex(
                FineTunedCheckpointUnavailableError, "NOT_AVAILABLE_YET"
            ):
                version.load_runtime()

    @unittest.skipUnless(
        os.getenv("RUN_REID_VERSION_SMOKE") == "1",
        "set RUN_REID_VERSION_SMOKE=1 for checkpoint-backed offline smoke",
    )
    def test_v1_v2_checkpoint_backed_offline_smoke(self):
        v1 = get_reid_version(V1_PRETRAINED_OSNET)
        v2 = get_reid_version(V2_PRETRAINED_OSNET_IMPROVED_CROP)
        runtime = v1.load_runtime(device="cpu")
        try:
            frame = np.full((300, 180, 3), 127, dtype=np.uint8)
            crops = [
                v1.crop(frame, (20, 10, 160, 290)),
                v2.crop(frame, (20, 10, 160, 290)),
            ]
            embeddings = runtime.appearance_extractor.extract_batch(crops)
            self.assertEqual(2, len(embeddings))
            for embedding in embeddings:
                self.assertEqual((512,), np.asarray(embedding).shape)
                self.assertTrue(np.isfinite(embedding).all())
                self.assertAlmostEqual(1.0, float(np.linalg.norm(embedding)), places=5)
        finally:
            store = getattr(runtime.global_identity_manager, "identity_store", None)
            if store is not None:
                store.close()

    def test_old_version_ids_are_not_silent_aliases(self):
        old_ids = (
            "v1_pretrained_" + "baseline",
            "v2_pretrained_" + "improved",
            "v3_finetuned_" + "baseline",
            "v4_finetuned_" + "improved",
        )
        for old_id in old_ids:
            with self.assertRaises(KeyError):
                get_version(old_id)


if __name__ == "__main__":
    unittest.main()
