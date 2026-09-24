import asyncio
import io
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from starlette.datastructures import UploadFile

from backend import embedding_view
from backend.embedding_store import EmbeddingStore, approved_identity_prototype


PROVENANCE = {
    "model_architecture": "osnet_x1_0", "checkpoint_id": "model.pth",
    "checkpoint_hash": "sha256", "preprocessing_version": "rgb_v1",
    "crop_mode": "improved", "normalization_version": "l2_v1",
}


class ArchiveSearchPhaseBTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "embeddings.sqlite3"
        self.start = datetime(2026, 9, 24, 13, 45, tzinfo=timezone(timedelta(hours=7)))
        self.store = EmbeddingStore(self.path, embedding_dim=2,
                                    identity_session_id="session-a", session_started_at=self.start)

    def save(self, gid, vector, when=None, provenance=PROVENANCE):
        self.store.select_id(gid)
        self.assertTrue(self.store.save_if_selected(
            gid, vector, captured_at=when or self.start, provenance=provenance))

    def test_session_label_is_persisted_but_uuid_remains_key(self):
        self.save(1, [1, 0])
        self.store.set_identity_session("session-b", self.start + timedelta(minutes=1))
        self.save(1, [0, 1])
        rows = self.store.list_records(global_id=1)
        self.assertEqual({"session-a", "session-b"}, {row["identity_session_id"] for row in rows})
        self.assertEqual({"Session 2026-09-24 13:45", "Session 2026-09-24 13:46"},
                         {row["session_display_name"] for row in rows})
        embedding_view.configure_store(self.store)
        listed = embedding_view.list_embeddings(global_id=1, captured_date=None,
                                                page=1, page_size=25)
        self.assertEqual(2, listed["total"])
        self.assertEqual({"Session 2026-09-24 13:45", "Session 2026-09-24 13:46"},
                         {row["session_display_name"] for row in listed["items"]})
        reopened = EmbeddingStore(self.path, embedding_dim=2, identity_session_id="session-b")
        self.assertEqual(self.start + timedelta(minutes=1), reopened.session_started_at)
        connection = sqlite3.connect(self.path)
        self.assertEqual(3, connection.execute("PRAGMA user_version").fetchone()[0])
        connection.close()

    def test_identity_aggregation_unknown_and_margin(self):
        self.save(1, [1, 0])
        self.save(1, [1, 0], self.start + timedelta(days=1))
        self.save(2, [0, 1])
        result = self.store.search_similar([1, 0], provenance=PROVENANCE, min_similarity=0.7)
        self.assertEqual("MATCH", result["status"])
        self.assertEqual(1, len(result["matches"]))
        self.assertEqual([], result["candidates"])
        self.assertEqual(2, result["matches"][0]["record_count"])
        unknown = self.store.search_similar(
            [-1, 0], provenance=PROVENANCE, min_similarity=0.7)
        self.assertEqual("UNKNOWN", unknown["status"])
        self.assertEqual([], unknown["matches"])
        self.assertEqual([], unknown["candidates"])
        self.store.set_identity_session("session-b", self.start + timedelta(minutes=1))
        self.save(3, [1, 0])
        result = self.store.search_similar([1, 0], provenance=PROVENANCE, min_similarity=0.7)
        self.assertEqual("AMBIGUOUS", result["status"])
        self.assertEqual([], result["matches"])
        self.assertGreaterEqual(len(result["candidates"]), 2)
        self.assertAlmostEqual(0.0, result["margin"])
        self.assertAlmostEqual(0.05, result["required_margin"])

    def test_ambiguous_image_response_returns_two_identities_even_with_top_k_one(self):
        self.save(1, [1, 0])
        self.save(2, [1, 0])

        class Extractor:
            def extract(self, crop):
                return np.array([1, 0], dtype=np.float32)

        embedding_view.configure_store(self.store)
        embedding_view.configure_embedding_extractor(Extractor())
        embedding_view.configure_image_search(lambda: None, lambda image, *box: image,
                                              PROVENANCE)
        success, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(success)
        uploaded = UploadFile(filename="person.png", file=io.BytesIO(encoded.tobytes()),
                              headers={"content-type": "image/png"})
        with patch.object(embedding_view, "extract_query_person", return_value=(
                np.zeros((4, 4, 3), dtype=np.uint8), 1)):
            response = asyncio.run(embedding_view.search_embedding_image(
                uploaded, top_k=1, min_similarity=0.7))
        self.assertEqual("AMBIGUOUS", response["status"])
        self.assertEqual([], response["matches"])
        self.assertEqual(0, response["total_matches"])
        self.assertEqual(2, len(response["candidates"]))
        self.assertEqual({1, 2}, {item["global_id"] for item in response["candidates"]})
        for item in response["candidates"]:
            self.assertEqual("Session 2026-09-24 13:45", item["session_display_name"])
            self.assertAlmostEqual(1.0, item["similarity"])
        self.assertAlmostEqual(response["top1_score"] - response["top2_score"],
                               response["margin"])
        self.assertAlmostEqual(0.05, response["required_margin"])

    def test_incompatible_and_legacy_are_explicitly_skipped(self):
        self.save(1, [1, 0], provenance={**PROVENANCE, "checkpoint_hash": "other"})
        self.save(2, [1, 0], provenance={})
        result = self.store.search_similar([1, 0], provenance=PROVENANCE, min_similarity=0.7)
        self.assertEqual("UNKNOWN", result["status"])
        self.assertEqual("NO_COMPATIBLE_RECORDS", result["reason"])
        self.assertEqual(2, sum(result["skipped_records"].values()))

    def test_archive_requires_approved_gallery_prototype(self):
        identity = {"gallery_mature": False, "gallery": [np.array([1, 0])],
                    "embedding": np.array([0, 1], dtype=np.float32)}
        manager = SimpleNamespace(lock=threading.RLock(), identities={1: identity})
        self.assertIsNone(approved_identity_prototype(manager, 1))
        identity["gallery_mature"] = True
        prototype = approved_identity_prototype(manager, 1)
        np.testing.assert_array_equal([0, 1], prototype)
        identity["embedding"][0] = 1
        np.testing.assert_array_equal([0, 1], prototype)

    def test_archive_cannot_save_old_batch_into_new_session(self):
        self.store.select_id(1)
        self.store.set_identity_session("session-b", self.start + timedelta(minutes=1))
        self.store.select_id(1)
        saved = self.store.save_if_selected(
            1, [1, 0], captured_at=self.start, provenance=PROVENANCE,
            expected_session_id="session-a")
        self.assertFalse(saved)
        self.assertEqual([1], self.store.selected_ids())
        self.assertEqual([], self.store.list_records())

    def test_v2_migration_preserves_rows_and_uses_earliest_capture_as_fallback_start(self):
        self.save(1, [1, 0])
        connection = sqlite3.connect(self.path)
        connection.execute("DROP TABLE identity_sessions")
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
        connection.close()
        migrated = EmbeddingStore(self.path, embedding_dim=2, identity_session_id="session-new")
        records = migrated.list_records()
        self.assertEqual(1, len(records))
        self.assertEqual("session-a", records[0]["identity_session_id"])
        self.assertEqual("Session 2026-09-24 13:45", records[0]["session_display_name"])
        self.assertTrue(self.path.with_name(self.path.name + ".pre_v3.bak").exists())

    def test_query_detects_and_crops_largest_person_before_extraction(self):
        class Boxes:
            xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(
                [[0, 0, 4, 4], [1, 1, 8, 9]], dtype=np.float32)))

        class Detector:
            def predict(self, image, **kwargs):
                self.last_kwargs = kwargs
                return [SimpleNamespace(boxes=Boxes())]

        class Extractor:
            def extract(self, crop):
                self.crop = crop
                return np.array([1, 0], dtype=np.float32)

        detector, extractor = Detector(), Extractor()
        embedding_view.configure_store(self.store)
        embedding_view.configure_embedding_extractor(extractor)
        embedding_view.configure_image_search(lambda: detector,
                                              lambda image, *box: image[box[1]:box[3], box[0]:box[2]],
                                              PROVENANCE)
        self.save(1, [1, 0])
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        uploaded = UploadFile(filename="person.png", file=io.BytesIO(encoded.tobytes()),
                              headers={"content-type": "image/png"})
        response = asyncio.run(embedding_view.search_embedding_image(
            uploaded, top_k=3, min_similarity=0.7))
        self.assertEqual("MATCH", response["status"])
        self.assertEqual("largest_person_bbox", response["selection_policy"])
        self.assertEqual((8, 7, 3), extractor.crop.shape)
        self.assertEqual([0], detector.last_kwargs["classes"])

    def test_query_without_person_does_not_extract_or_search(self):
        class Detector:
            def predict(self, image, **kwargs):
                return [SimpleNamespace(boxes=SimpleNamespace(xyxy=SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: np.empty((0, 4))))))]

        class Extractor:
            def extract(self, crop):
                raise AssertionError("extract should not run without a person")

        embedding_view.configure_store(self.store)
        embedding_view.configure_embedding_extractor(Extractor())
        embedding_view.configure_image_search(Detector, lambda image, *box: image, PROVENANCE)
        success, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(success)
        uploaded = UploadFile(filename="empty.png", file=io.BytesIO(encoded.tobytes()),
                              headers={"content-type": "image/png"})
        response = asyncio.run(embedding_view.search_embedding_image(
            uploaded, top_k=3, min_similarity=0.7))
        self.assertEqual("NO_PERSON_DETECTED", response["status"])
        self.assertEqual([], response["matches"])


if __name__ == "__main__":
    unittest.main()
