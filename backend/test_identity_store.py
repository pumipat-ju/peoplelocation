import json
import os
import sqlite3
import tempfile
import unittest

import numpy as np

from backend.identity_store import IdentityStore


class CommitFailOnceConnection:
    def __init__(self, connection):
        self.connection = connection
        self.fail_next_commit = True

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self):
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("synthetic commit failure")
        return self.connection.commit()


class IdentityStoreTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(handle)
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        os.unlink(self.path)

    def test_restart_restores_dormant_identity_and_next_gid_is_safe(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        store.save_identity(7, {
            "state": "DORMANT", "last_seen": 10.0,
            "embedding": np.array([1.0, 0.0], dtype=np.float32), "gallery": [],
        }, "assignment", "test", 10.0)
        store.close()

        restored_store = IdentityStore(self.path)
        self.stores.append(restored_store)
        restored = restored_store.load_identities()
        self.assertEqual("DORMANT", restored[7]["state"])
        np.testing.assert_array_equal(np.array([1.0, 0.0], dtype=np.float32), restored[7]["embedding"])
        self.assertEqual(8, max(restored) + 1)

    def test_next_gid_remains_above_expired_identity_after_restart(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        store.save_identity(41, {
            "state": "EXPIRED", "last_seen": 10.0,
            "embedding": np.array([1.0, 0.0], dtype=np.float32), "gallery": [],
        }, "state_transition", "dormant_ttl_expired", 20.0)
        store.close()

        restored_store = IdentityStore(self.path)
        self.stores.append(restored_store)
        self.assertEqual(42, restored_store.next_global_id())

    def test_next_gid_uses_audit_high_water_after_snapshot_is_purged(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        store.save_identity(73, {
            "state": "EXPIRED", "last_seen": 10.0,
            "embedding": np.array([0.0, 1.0], dtype=np.float32), "gallery": [],
        }, "state_transition", "dormant_ttl_expired", 20.0)
        store.connection.execute(
            "DELETE FROM identity_snapshots WHERE global_id = ?", (73,),
        )
        store.connection.commit()
        store.close()

        restored_store = IdentityStore(self.path)
        self.stores.append(restored_store)
        self.assertEqual({}, restored_store.load_identities())
        self.assertEqual(74, restored_store.next_global_id())

    def test_transition_audit_records_from_to_reason_and_timestamp(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        store.save_identity(12, {
            "state": "ACTIVE", "last_seen": 100.0,
            "embedding": np.array([1.0, 0.0], dtype=np.float32), "gallery": [],
        }, "assignment", "new_track", 100.0)
        store.save_identity(12, {
            "state": "DORMANT", "last_seen": 100.0,
            "embedding": np.array([1.0, 0.0], dtype=np.float32), "gallery": [],
        }, "state_transition", "active_idle_timeout", 112.5)

        row = store.connection.execute(
            """
            SELECT event_type, reason, ts, payload
            FROM identity_audit
            WHERE global_id = ? AND event_type = 'state_transition'
            ORDER BY ts DESC
            LIMIT 1
            """,
            (12,),
        ).fetchone()
        self.assertIsNotNone(row)
        event_type, reason, timestamp, payload_json = row
        payload = json.loads(payload_json)
        self.assertEqual("state_transition", event_type)
        self.assertEqual("active_idle_timeout", reason)
        self.assertEqual(112.5, timestamp)
        self.assertEqual("ACTIVE", payload["from_state"])
        self.assertEqual("DORMANT", payload["to_state"])

    def test_preceding_transition_audit_is_written_before_primary_assignment(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        identity = {
            "state": "DORMANT",
            "last_seen": 100.0,
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "gallery": [],
        }
        store.save_identity(
            21,
            identity,
            "test_seed",
            "dormant_seed",
            100.0,
        )

        identity.update({
            "state": "ACTIVE",
            "state_updated_at": 101.0,
            "state_reason": "cross_camera_recovery",
            "last_seen": 101.0,
        })
        store.save_identity(
            21,
            identity,
            "assignment",
            "global-cross-camera",
            101.0,
            preceding_events=[{
                "event_type": "state_transition",
                "reason": "cross_camera_recovery",
                "timestamp": 101.0,
            }],
        )

        rows = store.connection.execute(
            """
            SELECT event_type, reason, ts, payload
            FROM identity_audit
            WHERE global_id = ? AND ts = ?
            ORDER BY rowid ASC
            """,
            (21, 101.0),
        ).fetchall()
        self.assertEqual(
            ["state_transition", "assignment"],
            [row[0] for row in rows],
        )
        self.assertEqual(
            ["cross_camera_recovery", "global-cross-camera"],
            [row[1] for row in rows],
        )
        self.assertEqual([101.0, 101.0], [row[2] for row in rows])
        transition_payload = json.loads(rows[0][3])
        self.assertEqual("DORMANT", transition_payload["from_state"])
        self.assertEqual("ACTIVE", transition_payload["to_state"])
        self.assertEqual("ACTIVE", store.load_identities()[21]["state"])

    def test_preceding_audit_and_snapshot_roll_back_when_primary_event_fails(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        identity = {
            "state": "DORMANT",
            "last_seen": 200.0,
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "gallery": [],
        }
        store.save_identity(
            31,
            identity,
            "test_seed",
            "dormant_seed",
            200.0,
        )
        store.connection.execute(
            """
            CREATE TRIGGER fail_requested_primary_assignment
            BEFORE INSERT ON identity_audit
            WHEN NEW.event_type = 'assignment'
                 AND NEW.reason = 'force_primary_failure'
            BEGIN
                SELECT RAISE(ABORT, 'forced primary audit failure');
            END
            """
        )
        store.connection.commit()

        identity.update({
            "state": "ACTIVE",
            "state_updated_at": 201.0,
            "state_reason": "cross_camera_recovery",
            "last_seen": 201.0,
        })
        with self.assertRaises(sqlite3.DatabaseError):
            store.save_identity(
                31,
                identity,
                "assignment",
                "force_primary_failure",
                201.0,
                preceding_events=[{
                    "event_type": "state_transition",
                    "reason": "cross_camera_recovery",
                    "timestamp": 201.0,
                }],
            )

        audit_count = store.connection.execute(
            """
            SELECT COUNT(*)
            FROM identity_audit
            WHERE global_id = ? AND ts = ?
            """,
            (31, 201.0),
        ).fetchone()[0]
        self.assertEqual(0, audit_count)
        restored = store.load_identities()[31]
        self.assertEqual("DORMANT", restored["state"])
        self.assertEqual(200.0, restored["last_seen"])

    def test_commit_failure_rolls_back_and_next_transaction_can_begin(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        identity = {
            "state": "DORMANT",
            "last_seen": 300.0,
            "embedding": np.array([1.0, 1.0], dtype=np.float32),
            "gallery": [],
        }
        store.save_identity(
            41,
            identity,
            "test_seed",
            "dormant_seed",
            300.0,
        )
        store.connection = CommitFailOnceConnection(store.connection)

        identity.update({
            "state": "ACTIVE",
            "state_updated_at": 301.0,
            "state_reason": "cross_camera_recovery",
            "last_seen": 301.0,
        })
        with self.assertRaisesRegex(
            sqlite3.OperationalError,
            "synthetic commit failure",
        ):
            store.save_identity(
                41,
                identity,
                "assignment",
                "global-cross-camera",
                301.0,
                preceding_events=[{
                    "event_type": "state_transition",
                    "reason": "cross_camera_recovery",
                    "timestamp": 301.0,
                }],
            )

        restored = store.load_identities()[41]
        self.assertEqual("DORMANT", restored["state"])
        self.assertEqual(300.0, restored["last_seen"])
        failed_audits = store.connection.execute(
            "SELECT COUNT(*) FROM identity_audit "
            "WHERE global_id = ? AND ts = ?",
            (41, 301.0),
        ).fetchone()[0]
        self.assertEqual(0, failed_audits)

        identity["state_updated_at"] = 302.0
        identity["last_seen"] = 302.0
        store.save_identity(
            41,
            identity,
            "assignment",
            "global-cross-camera-retry",
            302.0,
            preceding_events=[{
                "event_type": "state_transition",
                "reason": "cross_camera_recovery",
                "timestamp": 302.0,
            }],
        )
        retry = store.load_identities()[41]
        self.assertEqual("ACTIVE", retry["state"])
        self.assertEqual(302.0, retry["last_seen"])

    def test_status_tracks_idempotent_close_without_querying_closed_database(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        status = store.status()
        self.assertTrue(status["connected"])
        self.assertEqual(os.path.abspath(self.path), os.path.abspath(status["path"]))

        store.close()
        self.assertFalse(store.status()["connected"])
        store.close()

    def test_failed_serialization_does_not_create_half_committed_snapshot(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        with self.assertRaises(TypeError):
            store.save_identity(1, {"state": "ACTIVE", "bad": object()}, "assignment")
        self.assertEqual({}, store.load_identities())

    def test_incomplete_legacy_snapshot_is_skipped_at_restore(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        store.save_identity(1, {
            "state": "ACTIVE", "embedding": np.array([1.0, 0.0], dtype=np.float32),
        }, "legacy")
        self.assertEqual({}, store.load_identities())


if __name__ == "__main__":
    unittest.main()
