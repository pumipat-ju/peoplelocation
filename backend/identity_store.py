"""Stable SQLite persistence for GlobalIdentityManager (no pickle payloads)."""

import json
import sqlite3

import numpy as np


class IdentityStore:
    SCHEMA_VERSION = 1

    def __init__(self, path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self):
        with self.connection:
            self.connection.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS identity_snapshots (global_id INTEGER PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL, payload TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS identity_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, global_id INTEGER NOT NULL, ts REAL NOT NULL, event_type TEXT NOT NULL, reason TEXT, payload TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS identity_quarantine (id INTEGER PRIMARY KEY AUTOINCREMENT, global_id INTEGER NOT NULL, ts REAL NOT NULL, reason TEXT NOT NULL, payload TEXT NOT NULL)")
            self.connection.execute("INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?)", (str(self.SCHEMA_VERSION),))

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist()}
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported persistence value: {type(value)!r}")

    @staticmethod
    def _json_hook(value):
        if set(value) == {"__ndarray__"}:
            return np.asarray(value["__ndarray__"], dtype=np.float32)
        return value

    def save_identity(self, global_id, identity, event_type, reason=None, timestamp=None):
        """Atomically replace a snapshot and add an immutable audit event."""
        timestamp = float(timestamp if timestamp is not None else identity.get("last_seen", 0.0))
        payload = json.dumps(identity, default=self._json_default, sort_keys=True)
        audit = json.dumps({"state": identity.get("state"), "last_score": identity.get("last_score")}, sort_keys=True)
        with self.connection:
            self.connection.execute("INSERT INTO identity_snapshots(global_id, state, updated_at, payload) VALUES (?, ?, ?, ?) ON CONFLICT(global_id) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at, payload=excluded.payload", (int(global_id), identity.get("state", "ACTIVE"), timestamp, payload))
            self.connection.execute("INSERT INTO identity_audit(global_id, ts, event_type, reason, payload) VALUES (?, ?, ?, ?, ?)", (int(global_id), timestamp, event_type, reason, audit))

    def load_identities(self):
        identities = {}
        for row in self.connection.execute("SELECT global_id, payload FROM identity_snapshots"):
            try:
                payload = json.loads(row["payload"], object_hook=self._json_hook)
                if isinstance(payload, dict) and payload.get("state") != "EXPIRED":
                    identities[int(row["global_id"])] = payload
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return identities

    def purge_expired_snapshots(self, before_ts):
        """TTL cleanup removes mutable snapshots only; immutable audit remains."""
        with self.connection:
            self.connection.execute(
                "DELETE FROM identity_snapshots WHERE state = 'EXPIRED' AND updated_at < ?",
                (float(before_ts),),
            )

    def quarantine_gallery_update(self, global_id, prototype, reason, timestamp):
        """Keep rejected prototype evidence auditable without restoring it."""
        payload = json.dumps({"prototype": prototype}, default=self._json_default)
        with self.connection:
            self.connection.execute(
                "INSERT INTO identity_quarantine(global_id, ts, reason, payload) VALUES (?, ?, ?, ?)",
                (int(global_id), float(timestamp), reason, payload),
            )

    def close(self):
        self.connection.close()
