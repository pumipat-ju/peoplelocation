"""Stable SQLite persistence for GlobalIdentityManager (no pickle payloads)."""

import json
import os
import sqlite3
import threading

import numpy as np


class IdentityStore:
    SCHEMA_VERSION = 1
    IDENTITY_STATES = ("PROVISIONAL", "ACTIVE", "DORMANT", "EXPIRED")

    def __init__(self, path):
        raw_path = os.fspath(path)
        self.path = (
            raw_path
            if raw_path == ":memory:"
            else os.path.abspath(raw_path)
        )
        self._lock = threading.RLock()
        self._closed = False
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self):
        with self._lock:
            self._ensure_open()
            with self.connection:
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_meta "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS identity_snapshots "
                    "(global_id INTEGER PRIMARY KEY, state TEXT NOT NULL, "
                    "updated_at REAL NOT NULL, payload TEXT NOT NULL)"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS identity_audit "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "global_id INTEGER NOT NULL, ts REAL NOT NULL, "
                    "event_type TEXT NOT NULL, reason TEXT, payload TEXT NOT NULL)"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS identity_quarantine "
                    "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "global_id INTEGER NOT NULL, ts REAL NOT NULL, "
                    "reason TEXT NOT NULL, payload TEXT NOT NULL)"
                )
                schema_row = self.connection.execute(
                    "SELECT value FROM schema_meta WHERE key = 'schema_version'"
                ).fetchone()
                if schema_row is None:
                    self.connection.execute(
                        "INSERT INTO schema_meta(key, value) "
                        "VALUES ('schema_version', ?)",
                        (str(self.SCHEMA_VERSION),),
                    )
                else:
                    try:
                        stored_version = int(schema_row["value"])
                    except (TypeError, ValueError) as error:
                        raise RuntimeError(
                            "invalid identity database schema version"
                        ) from error
                    if stored_version != self.SCHEMA_VERSION:
                        raise RuntimeError(
                            "unsupported identity database schema version: "
                            f"{stored_version}"
                        )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("identity store is closed")

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

    def save_identity(
        self,
        global_id,
        identity,
        event_type,
        reason=None,
        timestamp=None,
        preceding_events=None,
    ):
        """Atomically replace a snapshot and add immutable audit events."""
        state = identity.get("state")
        if state not in self.IDENTITY_STATES:
            raise ValueError(f"invalid identity state: {state!r}")
        if event_type == "state_transition" and not reason:
            raise ValueError("state transitions require a reason")
        if timestamp is None and event_type == "state_transition":
            timestamp = identity.get("state_updated_at")
        timestamp = float(
            timestamp
            if timestamp is not None
            else identity.get("last_seen", 0.0)
        )
        events = []
        for item in list(preceding_events or []) + [{
            "event_type": event_type,
            "reason": reason,
            "timestamp": timestamp,
        }]:
            if not isinstance(item, dict):
                raise TypeError("identity audit events must be dictionaries")
            item_type = item.get("event_type")
            item_reason = item.get("reason")
            if not isinstance(item_type, str) or not item_type:
                raise ValueError("identity audit events require an event_type")
            if item_type == "state_transition" and not item_reason:
                raise ValueError("state transitions require a reason")
            events.append({
                "event_type": item_type,
                "reason": item_reason,
                "timestamp": float(item.get("timestamp", timestamp)),
            })
        payload = json.dumps(identity, default=self._json_default, sort_keys=True)
        gid = int(global_id)
        with self._lock:
            self._ensure_open()
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                previous = self.connection.execute(
                    "SELECT state FROM identity_snapshots WHERE global_id = ?",
                    (gid,),
                ).fetchone()
                self.connection.execute(
                    "INSERT INTO identity_snapshots"
                    "(global_id, state, updated_at, payload) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(global_id) DO UPDATE SET "
                    "state=excluded.state, updated_at=excluded.updated_at, "
                    "payload=excluded.payload",
                    (gid, state, timestamp, payload),
                )
                previous_state = previous["state"] if previous else None
                for event in events:
                    audit_payload = {
                        "state": state,
                        "last_score": identity.get("last_score"),
                    }
                    if event["event_type"] == "state_transition":
                        audit_payload.update({
                            "from_state": previous_state,
                            "to_state": state,
                        })
                        previous_state = state
                    audit = json.dumps(audit_payload, sort_keys=True)
                    self.connection.execute(
                        "INSERT INTO identity_audit"
                        "(global_id, ts, event_type, reason, payload) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            gid,
                            event["timestamp"],
                            event["event_type"],
                            event["reason"],
                            audit,
                        ),
                    )
                self.connection.commit()
            except Exception:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
                raise

    def load_identities(self):
        identities = {}
        with self._lock:
            self._ensure_open()
            rows = self.connection.execute(
                "SELECT global_id, state, payload FROM identity_snapshots"
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(
                    row["payload"], object_hook=self._json_hook
                )
                state = row["state"]
                if not isinstance(payload, dict) or state not in self.IDENTITY_STATES:
                    continue
                payload["state"] = state
                if (
                    isinstance(payload.get("last_seen"), (int, float))
                    and payload.get("embedding") is not None
                ):
                    identities[int(row["global_id"])] = payload
            except (TypeError, ValueError):
                continue
        return identities

    def next_global_id(self):
        """Return a durable high-water mark, including retained audit history."""
        with self._lock:
            self._ensure_open()
            row = self.connection.execute(
                "SELECT MAX(global_id) AS max_global_id FROM ("
                "SELECT global_id FROM identity_snapshots "
                "UNION ALL SELECT global_id FROM identity_audit "
                "UNION ALL SELECT global_id FROM identity_quarantine"
                ")"
            ).fetchone()
        return int(row["max_global_id"] or 0) + 1

    def purge_expired_snapshots(self, before_ts):
        """TTL cleanup removes mutable snapshots only; immutable audit remains."""
        with self._lock:
            self._ensure_open()
            with self.connection:
                cursor = self.connection.execute(
                    "DELETE FROM identity_snapshots "
                    "WHERE state = 'EXPIRED' AND updated_at < ?",
                    (float(before_ts),),
                )
        return cursor.rowcount

    def quarantine_gallery_update(self, global_id, prototype, reason, timestamp):
        """Keep rejected prototype evidence auditable without restoring it."""
        payload = json.dumps({"prototype": prototype}, default=self._json_default)
        with self._lock:
            self._ensure_open()
            with self.connection:
                self.connection.execute(
                    "INSERT INTO identity_quarantine"
                    "(global_id, ts, reason, payload) VALUES (?, ?, ?, ?)",
                    (int(global_id), float(timestamp), reason, payload),
                )

    def status(self):
        with self._lock:
            base = {
                "path": self.path,
                "connected": not self._closed,
                "schema_version": self.SCHEMA_VERSION,
                "state_counts": {
                    state: 0 for state in self.IDENTITY_STATES
                },
                "recent_transitions": [],
            }
            if self._closed:
                return base
            count_rows = self.connection.execute(
                "SELECT state, COUNT(*) AS count "
                "FROM identity_snapshots GROUP BY state"
            ).fetchall()
            transition_rows = self.connection.execute(
                "SELECT global_id, ts, reason, payload "
                "FROM identity_audit "
                "WHERE event_type = 'state_transition' "
                "ORDER BY id DESC LIMIT 100"
            ).fetchall()

        for row in count_rows:
            base["state_counts"][row["state"]] = int(row["count"])
        for row in transition_rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            base["recent_transitions"].append({
                "gid": int(row["global_id"]),
                "ts": float(row["ts"]),
                "reason": row["reason"],
                "from": payload.get("from_state"),
                "to": payload.get("to_state", payload.get("state")),
            })
        return base

    def close(self):
        with self._lock:
            if self._closed:
                return False
            self.connection.close()
            self._closed = True
            return True
