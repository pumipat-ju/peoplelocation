"""Optional, write-only SQLite archive for selected Global ID embeddings.

This module does not participate in identity matching or tracking. Instantiate
it separately and pass a copy of an embedding after Global ID assignment.
"""

from datetime import datetime
from pathlib import Path
import logging
import sqlite3
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, db_path="database/embeddings.sqlite3", selected_ids=(), embedding_dim=512):
        self.db_path = Path(db_path)
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._lock = Lock()
        self._selected_ids = set()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id INTEGER NOT NULL,
                    camera_name TEXT,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    captured_date TEXT NOT NULL,
                    captured_time TEXT NOT NULL,
                    UNIQUE(global_id, captured_date)
                )
            """)
            conn.execute("CREATE TABLE IF NOT EXISTS selected_embedding_ids (global_id INTEGER PRIMARY KEY CHECK(global_id > 0))")
            for gid in selected_ids:
                conn.execute("INSERT OR IGNORE INTO selected_embedding_ids(global_id) VALUES (?)", (self._valid_id(gid),))
            self._selected_ids = {row[0] for row in conn.execute("SELECT global_id FROM selected_embedding_ids")}

    @staticmethod
    def _valid_id(global_id):
        if isinstance(global_id, bool) or not isinstance(global_id, (int, np.integer)):
            raise ValueError("global_id must be a positive integer")
        gid = int(global_id)
        if gid <= 0:
            raise ValueError("global_id must be a positive integer")
        return gid

    def _connect(self):
        return sqlite3.connect(str(self.db_path), timeout=5)

    def select_id(self, global_id):
        """Start archiving a Global ID in this process."""
        with self._lock:
            gid = self._valid_id(global_id)
            with self._connect() as conn:
                conn.execute("INSERT OR IGNORE INTO selected_embedding_ids(global_id) VALUES (?)", (gid,))
            self._selected_ids.add(gid)

    def unselect_id(self, global_id):
        """Stop archiving a Global ID in this process."""
        with self._lock:
            gid = self._valid_id(global_id)
            with self._connect() as conn:
                conn.execute("DELETE FROM selected_embedding_ids WHERE global_id = ?", (gid,))
            self._selected_ids.discard(gid)

    def selected_ids(self):
        with self._lock:
            with self._connect() as conn:
                return [row[0] for row in conn.execute(
                    "SELECT global_id FROM selected_embedding_ids ORDER BY global_id"
                )]

    def save_if_selected(self, global_id, embedding, camera_name=None, captured_at=None):
        """Return True only when a new row is saved; duplicates return False.

        captured_at defaults to the machine's local time. Pass a local datetime
        explicitly when archiving frames captured at another time.
        """
        gid = self._valid_id(global_id)
        # Read the selection from SQLite: API requests and camera processing may
        # run in different worker processes with different in-memory sets.
        with self._connect() as conn:
            chosen = conn.execute(
                "SELECT 1 FROM selected_embedding_ids WHERE global_id = ?", (gid,)
            ).fetchone()
        if chosen is None:
            return False

        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("embedding is empty")
        invalid_count = int(np.count_nonzero(~np.isfinite(vector)))
        if invalid_count:
            raise ValueError(
                f"embedding contains {invalid_count} NaN/Infinity values "
                f"(dimension={vector.size})"
            )
        if vector.size != self.embedding_dim:
            logger.warning(
                "[EmbeddingDB] Global ID %s: expected %s dimensions, got %s; storing actual dimension",
                gid, self.embedding_dim, vector.size,
            )
        moment = captured_at if captured_at is not None else datetime.now().astimezone()
        if not isinstance(moment, datetime):
            raise TypeError("captured_at must be a datetime")
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO embeddings
                    (global_id, camera_name, embedding, embedding_dim, captured_date, captured_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                gid, camera_name, vector.tobytes(), int(vector.size),
                moment.date().isoformat(), moment.time().isoformat(timespec="seconds"),
            ))
            saved = cursor.rowcount == 1
            # The selection is one-shot. A prior record for the same ID/day
            # also completes the request without adding a duplicate.
            conn.execute("DELETE FROM selected_embedding_ids WHERE global_id = ?", (gid,))
        if saved:
            logger.info("[EmbeddingDB] Saved Global ID %s on %s from %s", gid,
                        moment.date().isoformat(), camera_name)
        return saved

    def delete_record(self, record_id):
        """Delete exactly one embedding row by its database record ID."""
        if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id <= 0:
            raise ValueError("record_id must be a positive integer")
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM embeddings WHERE id = ?", (record_id,))
            return cursor.rowcount == 1

    def search_similar(self, embedding, top_k=10, min_similarity=0.0):
        """Rank stored embeddings by cosine similarity to a query vector."""
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query.size == 0 or not np.all(np.isfinite(query)):
            raise ValueError("query embedding is empty or contains NaN/Infinity")
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-8:
            raise ValueError("query embedding has zero norm")
        query = query / query_norm
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, global_id, camera_name, embedding, embedding_dim, "
                "captured_date, captured_time FROM embeddings"
            ).fetchall()
        matches = []
        for record_id, gid, camera, raw, dimension, date, captured_time in rows:
            if int(dimension) != query.size or len(raw) != int(dimension) * 4:
                continue
            candidate = np.frombuffer(raw, dtype=np.float32)
            norm = float(np.linalg.norm(candidate))
            if norm < 1e-8 or not np.all(np.isfinite(candidate)):
                continue
            similarity = float(np.dot(query, candidate / norm))
            if similarity >= float(min_similarity):
                matches.append({
                    "id": int(record_id), "global_id": int(gid),
                    "camera_name": camera, "embedding_dim": int(dimension),
                    "captured_date": date, "captured_time": captured_time,
                    "similarity": similarity,
                })
        matches.sort(key=lambda item: (-item["similarity"], item["global_id"]))
        return matches[:max(1, int(top_k))]

    def list_records(self, global_id=None, captured_date=None):
        """Return metadata only; embeddings stay in the database."""
        clauses, params = [], []
        if global_id is not None:
            clauses.append("global_id = ?")
            params.append(self._valid_id(global_id))
        if captured_date is not None:
            clauses.append("captured_date = ?")
            params.append(str(captured_date))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, global_id, camera_name, embedding_dim, captured_date, "
                "captured_time FROM embeddings" + where
                + " ORDER BY captured_date DESC, global_id ASC, captured_time DESC, id DESC",
                params,
            ).fetchall()
        return [dict(zip(("id", "global_id", "camera_name", "embedding_dim",
                          "captured_date", "captured_time"), row)) for row in rows]
