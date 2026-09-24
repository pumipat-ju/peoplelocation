"""Optional, write-only SQLite archive for selected Global ID embeddings.

This module does not participate in identity matching or tracking. Instantiate
it separately and pass a copy of an embedding after Global ID assignment.
"""

from datetime import datetime
from collections import defaultdict
from statistics import median
from pathlib import Path
import logging
import sqlite3
from threading import Lock
import uuid

import numpy as np

logger = logging.getLogger(__name__)


def approved_identity_prototype(manager, global_id):
    """Copy only a prototype backed by a quality-approved gallery."""
    identity = manager.identities.get(global_id)
    if not identity or not identity.get("gallery_mature") or not identity.get("gallery"):
        return None
    if identity.get("embedding") is None:
        return None
    try:
        candidate = np.asarray(identity["embedding"], dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if candidate.size == 0 or not np.all(np.isfinite(candidate)):
        return None
    norm = float(np.linalg.norm(candidate))
    return candidate.copy() if norm > 1e-8 else None


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class EmbeddingStore:
    SCHEMA_VERSION = 3
    LEGACY_SESSION = "legacy"

    def __init__(self, db_path="data/embeddings.sqlite3", selected_ids=(), embedding_dim=512,
                 identity_session_id=None, session_started_at=None):
        self.db_path = Path(db_path)
        self.embedding_dim = int(embedding_dim)
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._lock = Lock()
        self.identity_session_id = self._valid_session(identity_session_id or str(uuid.uuid4()))
        self.session_started_at = session_started_at or datetime.now().astimezone()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.set_identity_session(self.identity_session_id, self.session_started_at)
        for gid in selected_ids:
            self.select_id(gid)

    @staticmethod
    def _valid_session(session_id):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("identity_session_id must be a nonempty string")
        return session_id.strip()

    @staticmethod
    def _create_schema(conn):
        conn.execute("""
            CREATE TABLE embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_session_id TEXT NOT NULL,
                global_id INTEGER NOT NULL,
                camera_name TEXT,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                captured_date TEXT NOT NULL,
                captured_time TEXT NOT NULL,
                model_architecture TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                checkpoint_hash TEXT NOT NULL,
                preprocessing_version TEXT NOT NULL,
                crop_mode TEXT NOT NULL,
                normalization_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(identity_session_id, global_id, captured_date)
            )
        """)
        conn.execute("""
            CREATE TABLE selected_embedding_ids (
                identity_session_id TEXT NOT NULL,
                global_id INTEGER NOT NULL CHECK(global_id > 0),
                PRIMARY KEY(identity_session_id, global_id)
            )
        """)

    def _initialize(self):
        with self._lock, self._connect() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > self.SCHEMA_VERSION:
                raise RuntimeError(f"unsupported embedding schema version: {version}")
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            fresh_database = version == 0 and "embeddings" not in tables
            if version == 0 and "embeddings" in tables:
                backup_path = self.db_path.with_name(self.db_path.name + ".pre_v2.bak")
                if backup_path.exists():
                    raise RuntimeError(f"migration backup already exists: {backup_path}")
                with sqlite3.connect(str(backup_path), factory=_ClosingConnection) as backup:
                    conn.backup(backup)
                logger.info("[EmbeddingDB] Pre-migration backup: %s", backup_path)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("ALTER TABLE embeddings RENAME TO embeddings_legacy_v1")
                if "selected_embedding_ids" in tables:
                    conn.execute("ALTER TABLE selected_embedding_ids RENAME TO selected_embedding_ids_legacy_v1")
                self._create_schema(conn)
                conn.execute("""
                    INSERT INTO embeddings
                        (id, identity_session_id, global_id, camera_name, embedding,
                         embedding_dim, captured_date, captured_time, model_architecture,
                         checkpoint_id, checkpoint_hash, preprocessing_version, crop_mode,
                         normalization_version, created_at)
                    SELECT id, 'legacy', global_id, camera_name, embedding, embedding_dim,
                           captured_date, captured_time, 'unknown', 'unknown', 'unknown',
                           'unknown', 'unknown', 'unknown',
                           captured_date || 'T' || captured_time
                    FROM embeddings_legacy_v1
                """)
                if "selected_embedding_ids" in tables:
                    conn.execute("""
                        INSERT INTO selected_embedding_ids(identity_session_id, global_id)
                        SELECT 'legacy', global_id FROM selected_embedding_ids_legacy_v1
                    """)
                conn.execute("PRAGMA user_version = 2")
                conn.commit()
                version = 2
            if version == 0 and "embeddings" not in tables:
                self._create_schema(conn)
                conn.execute("PRAGMA user_version = 2")
                conn.commit()
                version = 2
            if version == 2:
                if not fresh_database and "embeddings_legacy_v1" not in tables:
                    backup_path = self.db_path.with_name(self.db_path.name + ".pre_v3.bak")
                    if backup_path.exists():
                        raise RuntimeError(f"migration backup already exists: {backup_path}")
                    with sqlite3.connect(str(backup_path), factory=_ClosingConnection) as backup:
                        conn.backup(backup)
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("CREATE TABLE identity_sessions (identity_session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL)")
                conn.execute("""INSERT INTO identity_sessions
                    SELECT identity_session_id, MIN(created_at) FROM embeddings
                    GROUP BY identity_session_id""")
                conn.execute("PRAGMA user_version = 3")
                version = 3
            if version != self.SCHEMA_VERSION:
                raise RuntimeError(f"unsupported embedding schema version: {version}")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)")}
            if "identity_session_id" not in columns:
                raise RuntimeError("embedding schema is missing session IDs")

    def set_identity_session(self, identity_session_id, started_at=None):
        moment = started_at or datetime.now().astimezone()
        if not isinstance(moment, datetime):
            raise TypeError("started_at must be a datetime")
        with self._lock:
            session_id = self._valid_session(identity_session_id)
            with self._connect() as conn:
                conn.execute("INSERT OR IGNORE INTO identity_sessions VALUES (?, ?)",
                             (session_id, moment.isoformat()))
                stored = conn.execute("SELECT started_at FROM identity_sessions WHERE identity_session_id = ?",
                                      (session_id,)).fetchone()[0]
            self.identity_session_id = session_id
            self.session_started_at = datetime.fromisoformat(stored)

    @staticmethod
    def session_display_name(session_id, started_at):
        if session_id == EmbeddingStore.LEGACY_SESSION:
            return "Legacy session"
        return "Session " + datetime.fromisoformat(started_at).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _valid_id(global_id):
        if isinstance(global_id, bool) or not isinstance(global_id, (int, np.integer)):
            raise ValueError("global_id must be a positive integer")
        gid = int(global_id)
        if gid <= 0:
            raise ValueError("global_id must be a positive integer")
        return gid

    def _connect(self):
        return sqlite3.connect(str(self.db_path), timeout=5, factory=_ClosingConnection)

    def select_id(self, global_id):
        """Start archiving a Global ID in this process."""
        with self._lock:
            gid = self._valid_id(global_id)
            with self._connect() as conn:
                conn.execute("INSERT OR IGNORE INTO selected_embedding_ids(identity_session_id, global_id) VALUES (?, ?)",
                             (self.identity_session_id, gid))
            logger.info("[EmbeddingDB] Archive selected | session=%s gid=%s",
                        self.identity_session_id, gid)

    def unselect_id(self, global_id):
        """Stop archiving a Global ID in this process."""
        with self._lock:
            gid = self._valid_id(global_id)
            with self._connect() as conn:
                conn.execute("DELETE FROM selected_embedding_ids WHERE identity_session_id = ? AND global_id = ?",
                             (self.identity_session_id, gid))

    def selected_ids(self):
        with self._lock:
            with self._connect() as conn:
                return [row[0] for row in conn.execute(
                    "SELECT global_id FROM selected_embedding_ids WHERE identity_session_id = ? ORDER BY global_id",
                    (self.identity_session_id,)
                )]

    def save_if_selected(self, global_id, embedding, camera_name=None, captured_at=None,
                         provenance=None, expected_session_id=None):
        """Return True only when a new row is saved; duplicates return False.

        captured_at defaults to the machine's local time. Pass a local datetime
        explicitly when archiving frames captured at another time.
        """
        gid = self._valid_id(global_id)
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
        metadata = dict(provenance or {})
        fields = ("model_architecture", "checkpoint_id", "checkpoint_hash",
                  "preprocessing_version", "crop_mode", "normalization_version")
        values = tuple(str(metadata.get(field) or "unknown") for field in fields)
        with self._lock, self._connect() as conn:
            session_id = self.identity_session_id
            if expected_session_id is not None and session_id != expected_session_id:
                logger.info("[EmbeddingDB] Archive skipped: session changed | gid=%s expected=%s current=%s",
                            gid, expected_session_id, session_id)
                return False
            chosen = conn.execute(
                "SELECT 1 FROM selected_embedding_ids WHERE identity_session_id = ? AND global_id = ?",
                (session_id, gid),
            ).fetchone()
            if chosen is None:
                logger.debug("[EmbeddingDB] Archive skipped: GID %s is not selected in session %s",
                             gid, session_id)
                return False
            cursor = conn.execute("""
                INSERT OR IGNORE INTO embeddings
                    (identity_session_id, global_id, camera_name, embedding, embedding_dim,
                     captured_date, captured_time, model_architecture, checkpoint_id,
                     checkpoint_hash, preprocessing_version, crop_mode,
                     normalization_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, gid, camera_name, vector.tobytes(), int(vector.size),
                moment.date().isoformat(), moment.time().isoformat(timespec="seconds"),
                *values, moment.isoformat(),
            ))
            saved = cursor.rowcount == 1
            # The selection is one-shot. A prior record for the same ID/day
            # also completes the request without adding a duplicate.
            conn.execute("DELETE FROM selected_embedding_ids WHERE identity_session_id = ? AND global_id = ?",
                         (session_id, gid))
        if saved:
            logger.info("[EmbeddingDB] Saved Global ID %s on %s from %s", gid,
                        moment.date().isoformat(), camera_name)
        else:
            logger.info("[EmbeddingDB] Archive already exists for session=%s gid=%s date=%s",
                        session_id, gid, moment.date().isoformat())
        return saved

    def delete_record(self, record_id):
        """Delete exactly one embedding row by its database record ID."""
        if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id <= 0:
            raise ValueError("record_id must be a positive integer")
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM embeddings WHERE id = ?", (record_id,))
            return cursor.rowcount == 1

    def search_similar(self, embedding, top_k=10, min_similarity=0.70, provenance=None,
                       min_margin=0.05):
        """Rank compatible identities, then apply absolute-score and margin gates."""
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query.size == 0 or not np.all(np.isfinite(query)):
            raise ValueError("query embedding is empty or contains NaN/Infinity")
        query_norm = float(np.linalg.norm(query))
        if query_norm < 1e-8:
            raise ValueError("query embedding has zero norm")
        query = query / query_norm
        with self._connect() as conn:
            rows = conn.execute("""SELECT e.id, e.identity_session_id, e.global_id, e.camera_name,
                e.embedding, e.embedding_dim, e.captured_date, e.captured_time,
                e.model_architecture, e.checkpoint_id, e.checkpoint_hash,
                e.preprocessing_version, e.crop_mode, e.normalization_version, s.started_at
                FROM embeddings e LEFT JOIN identity_sessions s
                ON s.identity_session_id = e.identity_session_id""").fetchall()
        fields = ("model_architecture", "checkpoint_id", "checkpoint_hash",
                  "preprocessing_version", "crop_mode", "normalization_version")
        expected = tuple(str((provenance or {}).get(key) or "unknown") for key in fields)
        groups = defaultdict(list)
        diagnostics = defaultdict(int)
        for (record_id, session_id, gid, camera, raw, dimension, date, captured_time,
             *metadata, started_at) in rows:
            if int(dimension) != query.size:
                diagnostics["embedding_dim"] += 1
                continue
            mismatch = next((field for field, actual, wanted in zip(fields, metadata, expected)
                             if actual == "unknown" or wanted == "unknown" or actual != wanted), None)
            if mismatch:
                diagnostics[mismatch] += 1
                continue
            if len(raw) != int(dimension) * 4:
                diagnostics["invalid_vector"] += 1
                continue
            candidate = np.frombuffer(raw, dtype=np.float32)
            norm = float(np.linalg.norm(candidate))
            if norm < 1e-8 or not np.all(np.isfinite(candidate)):
                diagnostics["invalid_vector"] += 1
                continue
            similarity = float(np.dot(query, candidate / norm))
            groups[(session_id, int(gid))].append((similarity, {
                    "id": int(record_id), "identity_session_id": session_id,
                    "global_id": int(gid),
                    "session_display_name": self.session_display_name(session_id, started_at),
                    "camera_name": camera, "embedding_dim": int(dimension),
                    "captured_date": date, "captured_time": captured_time,
                }))
        candidates = []
        for observations in groups.values():
            observations.sort(key=lambda item: -item[0])
            representative = dict(observations[0][1])
            representative["similarity"] = float(median(score for score, _ in observations[:3]))
            representative["record_count"] = len(observations)
            candidates.append(representative)
        candidates.sort(key=lambda item: (-item["similarity"], item["identity_session_id"], item["global_id"]))
        top1 = candidates[0]["similarity"] if candidates else None
        top2 = candidates[1]["similarity"] if len(candidates) > 1 else None
        margin = top1 - top2 if top2 is not None else None
        status = (
            "UNKNOWN" if top1 is None or top1 < float(min_similarity) else
            "AMBIGUOUS" if margin is not None and margin < float(min_margin) else
            "MATCH"
        )

        if status == "UNKNOWN":
            print("Not Found")

        reason = ("NO_COMPATIBLE_RECORDS" if not candidates else
                  "LOW_SIMILARITY" if status == "UNKNOWN" else
                  "CLOSE_CANDIDATES" if status == "AMBIGUOUS" else None)
        return {"status": status, "reason": reason,
                "matches": [item for item in candidates if item["similarity"] >= float(min_similarity)
                            ][:max(1, int(top_k))] if status == "MATCH" else [],
                "candidates": candidates[:max(2, int(top_k))] if status == "AMBIGUOUS" else [],
                "top1_score": top1, "top2_score": top2, "margin": margin,
                "required_margin": float(min_margin),
                "compatible_identities": len(candidates), "skipped_records": dict(diagnostics)}

    def list_records(self, global_id=None, captured_date=None):
        """Return metadata only; embeddings stay in the database."""
        clauses, params = [], []
        if global_id is not None:
            clauses.append("e.global_id = ?")
            params.append(self._valid_id(global_id))
        if captured_date is not None:
            clauses.append("e.captured_date = ?")
            params.append(str(captured_date))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT e.id, e.identity_session_id, e.global_id, e.camera_name, e.embedding_dim, e.captured_date, "
                "captured_time, model_architecture, checkpoint_id, checkpoint_hash, "
                "preprocessing_version, crop_mode, normalization_version, created_at, s.started_at FROM embeddings e "
                "LEFT JOIN identity_sessions s ON s.identity_session_id = e.identity_session_id" + where
                + " ORDER BY e.captured_date DESC, e.global_id ASC, e.captured_time DESC, e.id DESC",
                params,
            ).fetchall()
        records = [dict(zip(("id", "identity_session_id", "global_id", "camera_name",
                          "embedding_dim", "captured_date", "captured_time",
                          "model_architecture", "checkpoint_id", "checkpoint_hash",
                          "preprocessing_version", "crop_mode", "normalization_version",
                          "created_at", "session_started_at"), row)) for row in rows]
        for record in records:
            record["session_display_name"] = self.session_display_name(
                record["identity_session_id"], record["session_started_at"])
        return records
