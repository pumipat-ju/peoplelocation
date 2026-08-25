import copy
import os
import unittest
from unittest.mock import Mock

import numpy as np

os.environ["IDENTITY_DB_PATH"] = ":memory:"
os.environ["REID_ENABLED"] = "false"

from backend import main


class TrackletQualityGalleryTests(unittest.TestCase):

    def setUp(self):
        self.manager = main.GlobalIdentityManager()
        self._seed_identity(1)

    @staticmethod
    def _diagnostics():
        return {
            "accepted_updates": 0,
            "rejected_updates": 0,
            "last_rejection_reason": None,
            "tracklet_sample_count": 0,
            "prototype_quality": 0.0,
        }

    def _seed_identity(
        self,
        gid,
        *,
        state=None,
        embedding=None,
        gallery=None,
    ):
        state = state or main.IDENTITY_PROVISIONAL
        embedding = main.l2_normalize(
            np.asarray(
                [1.0, 0.0] if embedding is None else embedding,
                dtype=np.float32,
            )
        )
        gallery = [] if gallery is None else [
            main.l2_normalize(np.asarray(item, dtype=np.float32))
            for item in gallery
        ]
        identity = {
            "embedding": embedding.copy(),
            "gallery": gallery,
            "gallery_diagnostics": self._diagnostics(),
            "last_cam": "cam0",
            "last_seen": 0.0,
            "last_map_pos": None,
            "box_wh": (80, 160),
            "last_score": 1.0,
            "state": state,
            "state_updated_at": 0.0,
            "state_reason": "test_seed",
            "state_transitions": [{
                "from": None,
                "to": state,
                "ts": 0.0,
                "reason": "test_seed",
            }],
        }
        self.manager.identities[gid] = identity
        return identity

    def _detection(self, embedding, **overrides):
        detection = {
            "emb": np.asarray(embedding, dtype=np.float32),
            "detector_confidence": 0.95,
            "crop_size": (80, 160),
            "blur_variance": 100.0,
            "overlap": False,
            "border_clip_ratio": 0.0,
            "local_track_confirmed": True,
            "camera_generation": 1,
        }
        detection.update(overrides)
        return detection

    def _record(
        self,
        gid,
        embedding,
        timestamp,
        *,
        cam_name="cam1",
        local_id=7,
        camera_generation=1,
        **overrides,
    ):
        detection = self._detection(
            embedding,
            camera_generation=camera_generation,
            **overrides,
        )
        return self.manager._record_tracklet_sample(
            gid,
            cam_name,
            local_id,
            detection,
            float(timestamp),
        )

    def test_all_low_quality_samples_never_enter_tracklet_or_gallery(self):
        cases = (
            (
                "low_confidence",
                {"detector_confidence": 0.1},
                "low_detector_confidence",
            ),
            (
                "small_crop",
                {"crop_size": (main.REID_MIN_CROP_SIZE - 1, 160)},
                "crop_too_small",
            ),
            (
                "blurred",
                {"blur_variance": 0.0},
                "blurred_crop",
            ),
            (
                "overlap",
                {"overlap": True},
                "overlap_or_occlusion",
            ),
            (
                "border_clipped",
                {"border_clip_ratio": main.REID_MAX_BORDER_CLIP_RATIO + 0.01},
                "border_clipped",
            ),
        )

        for name, overrides, expected_reason in cases:
            with self.subTest(case=name):
                self.manager = main.GlobalIdentityManager()
                identity = self._seed_identity(1)
                original_embedding = identity["embedding"].copy()

                accepted, reason = self._record(
                    1,
                    [1.0, 0.0],
                    1.0,
                    **overrides,
                )

                self.assertFalse(accepted)
                self.assertEqual(expected_reason, reason)
                self.assertEqual({}, self.manager.tracklets)
                self.assertEqual([], identity["gallery"])
                np.testing.assert_allclose(identity["embedding"], original_embedding)

    def test_invalid_embeddings_reject_without_mutating_tracklet_or_gallery(self):
        cases = (
            ("zero", [0.0, 0.0], "zero_embedding"),
            ("nan", [np.nan, 0.0], "invalid_embedding"),
            ("positive_inf", [np.inf, 0.0], "invalid_embedding"),
            ("negative_inf", [-np.inf, 0.0], "invalid_embedding"),
            ("empty", [], "invalid_embedding"),
            (
                "wrong_dimension",
                [1.0, 0.0, 0.0],
                "embedding_dimension_mismatch",
            ),
        )

        for name, embedding, expected_reason in cases:
            with self.subTest(case=name):
                self.manager = main.GlobalIdentityManager()
                identity = self._seed_identity(1)
                original_embedding = identity["embedding"].copy()
                original_diagnostics = copy.deepcopy(
                    identity["gallery_diagnostics"]
                )

                accepted, reason = self._record(1, embedding, 1.0)

                self.assertFalse(accepted)
                self.assertEqual(expected_reason, reason)
                self.assertEqual({}, self.manager.tracklets)
                self.assertEqual([], identity["gallery"])
                np.testing.assert_allclose(identity["embedding"], original_embedding)
                self.assertEqual(
                    original_diagnostics["tracklet_sample_count"],
                    identity["gallery_diagnostics"]["tracklet_sample_count"],
                )

    def test_non_finite_quality_metadata_rejects_without_tracklet_mutation(self):
        cases = (
            ("confidence_nan", {"detector_confidence": np.nan}),
            ("confidence_inf", {"detector_confidence": np.inf}),
            ("blur_nan", {"blur_variance": np.nan}),
            ("border_nan", {"border_clip_ratio": np.nan}),
        )

        for name, overrides in cases:
            with self.subTest(case=name):
                self.manager = main.GlobalIdentityManager()
                identity = self._seed_identity(1)

                accepted, reason = self._record(
                    1,
                    [1.0, 0.0],
                    1.0,
                    **overrides,
                )

                self.assertFalse(accepted)
                self.assertEqual("invalid_quality_metadata", reason)
                self.assertEqual({}, self.manager.tracklets)
                self.assertEqual([], identity["gallery"])

    def test_near_duplicate_good_samples_mature_to_exactly_one_prototype(self):
        samples = (
            [1.0, 0.0],
            [1.0, 0.001],
            [1.0, -0.001],
        )
        outcomes = [
            self._record(1, sample, timestamp)
            for sample, timestamp in zip(samples, (1.0, 2.0, 3.0))
        ]

        self.assertFalse(np.array_equal(samples[0], samples[1]))
        self.assertFalse(np.array_equal(samples[1], samples[2]))
        self.assertEqual((False, "tracklet_not_mature"), outcomes[0])
        self.assertEqual((False, "tracklet_not_mature"), outcomes[1])
        self.assertEqual((True, None), outcomes[2])
        identity = self.manager.identities[1]
        tracklet = self.manager.tracklets[("cam1", 7)]
        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        self.assertEqual(main.REID_TRACKLET_MIN_SAMPLES, tracklet["sample_count"])
        self.assertEqual(
            main.REID_TRACKLET_MIN_SAMPLES,
            identity["gallery_diagnostics"]["tracklet_sample_count"],
        )
        self.assertEqual(1, len(tracklet["samples"]))
        self.assertEqual(1, len(identity["gallery"]))
        self.assertEqual(1, identity["gallery_diagnostics"]["accepted_updates"])
        self.assertAlmostEqual(1.0, np.linalg.norm(identity["gallery"][0]), places=6)
        np.testing.assert_allclose(
            identity["embedding"],
            identity["gallery"][0],
            atol=1e-6,
        )

    def test_provisional_bootstrap_is_local_only_until_tracklet_matures(self):
        identity = self.manager.identities[1]

        self.assertIs(identity, self.manager._assignable_identity(1))
        self.assertIsNone(self.manager._globally_matchable_identity(1))

        for timestamp in (1.0, 2.0, 3.0):
            self._record(1, [1.0, 0.0], timestamp)

        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        self.assertIs(identity, self.manager._globally_matchable_identity(1))

    def test_insufficient_new_tracklet_cannot_alter_active_gallery(self):
        self.manager = main.GlobalIdentityManager()
        existing = main.l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
        identity = self._seed_identity(
            1,
            state=main.IDENTITY_ACTIVE,
            embedding=existing,
            gallery=[existing],
        )
        original_gallery = [item.copy() for item in identity["gallery"]]
        original_embedding = identity["embedding"].copy()

        self.assertEqual(
            (False, "tracklet_not_mature"),
            self._record(1, [0.9, 0.2], 1.0, local_id=20),
        )
        self.assertEqual(
            (False, "tracklet_not_mature"),
            self._record(1, [0.9, 0.2], 2.0, local_id=20),
        )

        self.assertEqual(1, len(identity["gallery"]))
        np.testing.assert_allclose(identity["gallery"][0], original_gallery[0])
        np.testing.assert_allclose(identity["embedding"], original_embedding)
        self.assertEqual(0, identity["gallery_diagnostics"]["accepted_updates"])

    def test_one_tracklet_commits_at_most_one_permanent_prototype(self):
        for timestamp in (1.0, 2.0, 3.0):
            self._record(1, [1.0, 0.0], timestamp)

        identity = self.manager.identities[1]
        committed_gallery = [item.copy() for item in identity["gallery"]]
        committed_embedding = identity["embedding"].copy()

        for timestamp, embedding in (
            (4.0, [0.98, 0.20]),
            (5.0, [0.97, 0.24]),
            (6.0, [0.96, 0.28]),
        ):
            accepted, reason = self._record(1, embedding, timestamp)
            self.assertFalse(accepted)
            self.assertEqual("tracklet_already_committed", reason)

        self.assertEqual(1, len(identity["gallery"]))
        np.testing.assert_allclose(identity["gallery"][0], committed_gallery[0])
        np.testing.assert_allclose(identity["embedding"], committed_embedding)
        self.assertEqual(1, identity["gallery_diagnostics"]["accepted_updates"])

    def test_distinct_near_duplicate_tracklets_do_not_fill_gallery(self):
        self.manager = main.GlobalIdentityManager()
        base = main.l2_normalize(np.asarray([1.0, 0.0], dtype=np.float32))
        identity = self._seed_identity(
            1,
            state=main.IDENTITY_ACTIVE,
            embedding=base,
            gallery=[base],
        )
        tracklet_count = main.REID_GALLERY_SIZE + 2

        for tracklet_index in range(tracklet_count):
            local_id = 200 + tracklet_index
            for sample_index in range(main.REID_TRACKLET_MIN_SAMPLES):
                epsilon = (
                    0.001 * (tracklet_index + 1)
                    + 0.00001 * sample_index
                )
                self._record(
                    1,
                    [1.0, epsilon],
                    timestamp=(tracklet_index * 10) + sample_index + 1,
                    local_id=local_id,
                )

            tracklet = self.manager.tracklets[("cam1", local_id)]
            self.assertEqual(
                main.REID_TRACKLET_MIN_SAMPLES,
                tracklet["sample_count"],
            )
            self.assertEqual(1, len(tracklet["samples"]))

        self.assertEqual(1, len(identity["gallery"]))
        self.assertAlmostEqual(1.0, np.linalg.norm(identity["gallery"][0]), places=6)

    def test_distinct_tracklets_keep_gallery_bounded_and_prototypes_normalized(self):
        self.manager = main.GlobalIdentityManager()
        tracklet_count = main.REID_GALLERY_SIZE + 3
        dimension = tracklet_count
        initial = np.zeros(dimension, dtype=np.float32)
        initial[0] = 1.0
        identity = self._seed_identity(
            1,
            state=main.IDENTITY_ACTIVE,
            embedding=initial,
        )
        for tracklet_index in range(tracklet_count):
            prototype = np.zeros(dimension, dtype=np.float32)
            prototype[tracklet_index] = 1.0
            for sample_index in range(main.REID_TRACKLET_MIN_SAMPLES):
                accepted, reason = self._record(
                    1,
                    prototype,
                    timestamp=(tracklet_index * 10) + sample_index + 1,
                    local_id=100 + tracklet_index,
                )
            self.assertTrue(accepted)
            self.assertIsNone(reason)

        self.assertEqual(main.REID_GALLERY_SIZE, len(identity["gallery"]))
        for prototype in identity["gallery"]:
            self.assertTrue(np.all(np.isfinite(prototype)))
            self.assertAlmostEqual(1.0, np.linalg.norm(prototype), places=6)
        for left_index, left in enumerate(identity["gallery"]):
            for right in identity["gallery"][left_index + 1:]:
                self.assertLess(
                    abs(float(np.dot(left, right))),
                    1e-6,
                )
        expected_global_prototype = main.l2_normalize(
            np.mean(identity["gallery"], axis=0)
        )
        np.testing.assert_allclose(
            identity["embedding"],
            expected_global_prototype,
            atol=1e-6,
        )

    def test_tracklet_owner_gid_change_cannot_inherit_samples(self):
        self._seed_identity(2)

        self._record(1, [1.0, 0.0], 1.0)
        self._record(1, [1.0, 0.0], 2.0)
        accepted, reason = self._record(2, [0.0, 1.0], 3.0)

        self.assertFalse(accepted)
        self.assertEqual("tracklet_not_mature", reason)
        self.assertEqual(main.IDENTITY_PROVISIONAL, self.manager.identities[2]["state"])
        self.assertEqual([], self.manager.identities[2]["gallery"])
        self.assertEqual(
            1,
            self.manager.identities[2]["gallery_diagnostics"][
                "tracklet_sample_count"
            ],
        )

        self._record(2, [0.0, 1.0], 4.0)
        accepted, reason = self._record(2, [0.0, 1.0], 5.0)
        self.assertTrue(accepted)
        self.assertIsNone(reason)
        self.assertEqual(main.IDENTITY_ACTIVE, self.manager.identities[2]["state"])
        self.assertEqual(1, len(self.manager.identities[2]["gallery"]))

    def test_camera_generation_change_cannot_inherit_samples(self):
        self._record(1, [1.0, 0.0], 1.0, camera_generation=1)
        self._record(1, [1.0, 0.0], 2.0, camera_generation=1)

        accepted, reason = self._record(
            1,
            [1.0, 0.0],
            3.0,
            camera_generation=2,
        )

        self.assertFalse(accepted)
        self.assertEqual("tracklet_not_mature", reason)
        self.assertEqual(main.IDENTITY_PROVISIONAL, self.manager.identities[1]["state"])
        self.assertEqual([], self.manager.identities[1]["gallery"])
        self.assertEqual(
            1,
            self.manager.identities[1]["gallery_diagnostics"][
                "tracklet_sample_count"
            ],
        )

        self._record(1, [1.0, 0.0], 4.0, camera_generation=2)
        accepted, reason = self._record(
            1,
            [1.0, 0.0],
            5.0,
            camera_generation=2,
        )
        self.assertTrue(accepted)
        self.assertIsNone(reason)
        self.assertEqual(main.IDENTITY_ACTIVE, self.manager.identities[1]["state"])

    def test_mature_admission_persistence_failure_rolls_back_ram(self):
        identity = self.manager.identities[1]
        self._record(1, [1.0, 0.0], 1.0)
        self._record(1, [1.0, 0.0], 2.0)

        before = {
            "state": identity["state"],
            "state_updated_at": identity["state_updated_at"],
            "state_reason": identity["state_reason"],
            "state_transitions": copy.deepcopy(identity["state_transitions"]),
            "gallery": [item.copy() for item in identity["gallery"]],
            "embedding": identity["embedding"].copy(),
            "gallery_diagnostics": copy.deepcopy(
                identity["gallery_diagnostics"]
            ),
        }
        failing_store = Mock()
        failing_store.save_identity.side_effect = RuntimeError(
            "synthetic persistence failure"
        )
        self.manager.identity_store = failing_store

        with self.assertRaisesRegex(RuntimeError, "synthetic persistence failure"):
            self._record(1, [1.0, 0.0], 3.0)

        self.assertEqual(before["state"], identity["state"])
        self.assertEqual(before["state_updated_at"], identity["state_updated_at"])
        self.assertEqual(before["state_reason"], identity["state_reason"])
        self.assertEqual(before["state_transitions"], identity["state_transitions"])
        self.assertEqual(len(before["gallery"]), len(identity["gallery"]))
        np.testing.assert_allclose(identity["embedding"], before["embedding"])
        self.assertEqual(
            before["gallery_diagnostics"],
            identity["gallery_diagnostics"],
        )
        tracklet = self.manager.tracklets[("cam1", 7)]
        self.assertEqual(2, tracklet["sample_count"])
        self.assertFalse(tracklet["gallery_committed"])
        failing_store.save_identity.assert_called()


if __name__ == "__main__":
    unittest.main()
