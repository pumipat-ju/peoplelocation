import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.embedding_store import EmbeddingStore


class EmbeddingPersistencePhaseATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "embeddings.sqlite3"
        self.when = datetime(2026, 9, 24, 12, 0)
        self.vector = np.array([1.0, 0.0], dtype=np.float32)
        self.provenance = {
            "model_architecture": "osnet_x1_0",
            "checkpoint_id": "model.pth",
            "checkpoint_hash": "abc123",
            "preprocessing_version": "imagenet_rgb_v1",
            "crop_mode": "improved",
            "normalization_version": "l2_v1",
        }

    def test_same_day_reused_gid_is_separate_across_sessions_and_restart(self):
        store = EmbeddingStore(self.path, embedding_dim=2, identity_session_id="session-a")
        store.select_id(1)
        self.assertTrue(store.save_if_selected(1, self.vector, captured_at=self.when,
                                               provenance=self.provenance))
        store.set_identity_session("session-b")
        store.select_id(1)
        self.assertTrue(store.save_if_selected(1, self.vector, captured_at=self.when,
                                               provenance=self.provenance))
        restarted = EmbeddingStore(self.path, embedding_dim=2, identity_session_id="session-c")
        records = restarted.list_records(global_id=1)
        self.assertEqual({"session-a", "session-b"},
                         {row["identity_session_id"] for row in records})
        self.assertEqual(2, len(records))
        decision = restarted.search_similar(
            self.vector, provenance=self.provenance, min_similarity=0.7)
        self.assertEqual(2, decision["compatible_identities"])
        self.assertEqual("AMBIGUOUS", decision["status"])
        for row in records:
            for key, value in self.provenance.items():
                self.assertEqual(value, row[key])
            self.assertTrue(row["created_at"])

    def test_legacy_migration_keeps_rows_and_creates_backup(self):
        conn = sqlite3.connect(self.path)
        with conn:
            conn.execute("""CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, global_id INTEGER NOT NULL,
                camera_name TEXT, embedding BLOB NOT NULL, embedding_dim INTEGER NOT NULL,
                captured_date TEXT NOT NULL, captured_time TEXT NOT NULL,
                UNIQUE(global_id, captured_date))""")
            conn.execute("CREATE TABLE selected_embedding_ids (global_id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO selected_embedding_ids VALUES (2)")
            conn.execute("INSERT INTO embeddings (global_id, camera_name, embedding, embedding_dim, captured_date, captured_time) VALUES (?, ?, ?, ?, ?, ?)",
                         (1, "cam1", self.vector.tobytes(), 2, "2026-09-24", "12:00:00"))
        conn.close()
        store = EmbeddingStore(self.path, embedding_dim=2, identity_session_id="new-session")
        self.assertEqual("legacy", store.list_records()[0]["identity_session_id"])
        self.assertTrue(self.path.with_name(self.path.name + ".pre_v2.bak").exists())
        conn = sqlite3.connect(self.path)
        with conn:
            self.assertEqual(3, conn.execute("PRAGMA user_version").fetchone()[0])
            self.assertEqual(1, conn.execute("SELECT count(*) FROM embeddings_legacy_v1").fetchone()[0])
            self.assertEqual(("legacy", 2), conn.execute(
                "SELECT identity_session_id, global_id FROM selected_embedding_ids"
            ).fetchone())
        conn.close()
        store.select_id(1)
        self.assertTrue(store.save_if_selected(1, self.vector, captured_at=self.when,
                                               provenance=self.provenance))
        self.assertEqual(2, len(store.list_records(global_id=1)))


if __name__ == "__main__":
    unittest.main()
