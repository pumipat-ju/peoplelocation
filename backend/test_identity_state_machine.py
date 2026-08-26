import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Importing backend.main creates the process-wide identity manager. Force that
# manager onto an in-memory database so test collection can never touch the
# developer's production identity database.
os.environ["IDENTITY_DB_PATH"] = ":memory:"

from backend import main
from backend.identity_store import IdentityStore


class IdentityStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = os.path.join(self.temp_dir.name, "identity-state.sqlite3")
        self.store = IdentityStore(self.db_path)
        self.addCleanup(self.store.close)
        self.manager = main.GlobalIdentityManager(identity_store=self.store)
        self.det = {
            "emb": np.array([1.0, 0.0], dtype=np.float32),
            "detector_confidence": 0.95,
            "crop_size": (80, 160),
            "blur_variance": 100.0,
            "overlap": False,
            "border_clip_ratio": 0.0,
        }

    def _seed_identity(self, state, timestamp=1.0):
        result = self.manager._new_identity(
            "cam1", 7, self.det["emb"], None, (80, 160), timestamp
        )
        gid = result["gid"]
        identity = self.manager.identities[gid]
        identity.update(
            {
                "state": state,
                "state_updated_at": timestamp,
                "state_reason": "test_seed",
                "last_seen": timestamp,
            }
        )
        self.store.save_identity(
            gid,
            identity,
            event_type="test_seed",
            reason="test_seed",
            timestamp=timestamp,
        )
        return gid, identity

    def _load_from_new_connection(self, gid):
        reader = IdentityStore(self.db_path)
        try:
            return reader.load_identities()[gid]
        finally:
            reader.close()

    def _latest_transition_audit(self, gid):
        connection = sqlite3.connect(self.db_path)
        try:
            row = connection.execute(
                """
                SELECT ts, reason, payload
                FROM identity_audit
                WHERE global_id = ? AND event_type = 'state_transition'
                ORDER BY ts DESC, rowid DESC
                LIMIT 1
                """,
                (gid,),
            ).fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(row, "state transition must be written to the audit log")
        return row[0], row[1], json.loads(row[2])

    def test_new_identity_becomes_active_only_after_mature_tracklet(self):
        result = self.manager._new_identity(
            "cam1", 7, self.det["emb"], None, (80, 160), 1.0
        )
        gid = result["gid"]
        identity = self.manager.identities[gid]
        self.assertEqual(main.IDENTITY_PROVISIONAL, identity["state"])

        for timestamp, embedding in zip(
            (1.0, 2.0, 3.0), ([1.0, 0.0], [0.9, 0.2], [0.8, 0.4])
        ):
            sample = dict(self.det, emb=np.array(embedding, dtype=np.float32))
            self.manager._record_tracklet_sample(gid, "cam1", 7, sample, timestamp)

        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        self.assertEqual("mature_tracklet", identity["state_reason"])

    def test_cleanup_persists_active_to_dormant_transition_and_audit(self):
        gid, identity = self._seed_identity(main.IDENTITY_ACTIVE, timestamp=10.0)
        transition_at = 10.0 + main.REID_MAX_IDLE_SEC + 1.0

        self.manager.cleanup(reference_time=transition_at)

        self.assertEqual(main.IDENTITY_DORMANT, identity["state"])
        persisted = self._load_from_new_connection(gid)
        self.assertEqual(main.IDENTITY_DORMANT, persisted["state"])
        self.assertEqual(transition_at, persisted["state_updated_at"])
        self.assertEqual("idle_timeout", persisted["state_reason"])

        audit_ts, reason, payload = self._latest_transition_audit(gid)
        self.assertEqual(transition_at, audit_ts)
        self.assertEqual("idle_timeout", reason)
        self.assertEqual(main.IDENTITY_ACTIVE, payload["from_state"])
        self.assertEqual(main.IDENTITY_DORMANT, payload["to_state"])

    def test_dormant_identity_can_reactivate_before_ttl(self):
        gid, identity = self._seed_identity(main.IDENTITY_DORMANT, timestamp=50.0)
        reactivated_at = 50.0 + main.IDENTITY_DORMANT_TTL_SEC - 1.0

        changed = self.manager._transition_identity(
            gid,
            identity,
            main.IDENTITY_ACTIVE,
            reactivated_at,
            "reidentified",
        )

        self.assertTrue(changed)
        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        persisted = self._load_from_new_connection(gid)
        self.assertEqual(main.IDENTITY_ACTIVE, persisted["state"])
        self.assertEqual(reactivated_at, persisted["state_updated_at"])
        self.assertEqual("reidentified", persisted["state_reason"])

    def test_commit_dormant_recovery_persists_transition_then_assignment_once(self):
        gid, identity = self._seed_identity(
            main.IDENTITY_DORMANT,
            timestamp=75.0,
        )
        reactivated_at = 76.0

        with patch.object(
            self.store,
            "save_identity",
            wraps=self.store.save_identity,
        ) as save_identity:
            result = self.manager._commit_assignment(
                gid,
                "cam2",
                9,
                self.det["emb"],
                None,
                (80, 160),
                reactivated_at,
                0.91,
                "global-cross-camera",
            )

        self.assertEqual(gid, result["gid"])
        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        self.assertEqual("cross_camera_recovery", identity["state_reason"])
        save_identity.assert_called_once()
        preceding_events = save_identity.call_args.kwargs["preceding_events"]
        self.assertEqual(
            {
                "event_type": "state_transition",
                "reason": "cross_camera_recovery",
                "timestamp": reactivated_at,
            },
            preceding_events[0],
        )
        self.assertEqual("handoff", preceding_events[1]["event_type"])
        self.assertEqual("cam1", preceding_events[1]["payload"]["handoff"]["from_camera"])
        self.assertEqual("cam2", preceding_events[1]["payload"]["handoff"]["to_camera"])

        connection = sqlite3.connect(self.db_path)
        try:
            rows = connection.execute(
                """
                SELECT event_type, reason, ts, payload
                FROM identity_audit
                WHERE global_id = ? AND ts = ?
                ORDER BY rowid ASC
                """,
                (gid, reactivated_at),
            ).fetchall()
        finally:
            connection.close()

        self.assertEqual(
            ["state_transition", "handoff", "assignment"],
            [row[0] for row in rows],
        )
        self.assertEqual(
            [
                "cross_camera_recovery",
                "confirmed_non_overlap_handoff",
                "global-cross-camera",
            ],
            [row[1] for row in rows],
        )
        transition_payload = json.loads(rows[0][3])
        self.assertEqual(
            main.IDENTITY_DORMANT,
            transition_payload["from_state"],
        )
        self.assertEqual(
            main.IDENTITY_ACTIVE,
            transition_payload["to_state"],
        )
        handoff_payload = json.loads(rows[1][3])["handoff"]
        self.assertEqual("cam1", handoff_payload["from_camera"])
        self.assertEqual("cam2", handoff_payload["to_camera"])
        persisted = self._load_from_new_connection(gid)
        self.assertEqual(main.IDENTITY_ACTIVE, persisted["state"])
        self.assertEqual("cam2", persisted["last_cam"])
        self.assertEqual(reactivated_at, persisted["last_seen"])

    def test_dormant_match_is_not_rejected_by_active_idle_limit(self):
        last_seen = 100.0
        gid, identity = self._seed_identity(main.IDENTITY_DORMANT, timestamp=last_seen)
        identity["state_updated_at"] = last_seen + main.REID_MAX_IDLE_SEC + 1.0
        now_ts = identity["state_updated_at"] + 1.0
        self.store.save_identity(
            gid,
            identity,
            event_type="test_seed",
            reason="dormant_within_ttl",
            timestamp=identity["state_updated_at"],
        )

        self.assertGreater(now_ts - identity["last_seen"], main.REID_MAX_IDLE_SEC)
        self.assertLess(
            now_ts - identity["state_updated_at"], main.IDENTITY_DORMANT_TTL_SEC
        )
        self.assertTrue(
            self.manager._can_match(identity, "cam2", now_ts, None, (80, 160))
        )

    def test_cleanup_persists_dormant_to_expired_and_never_reuses_it(self):
        dormant_at = 200.0
        gid, identity = self._seed_identity(
            main.IDENTITY_DORMANT, timestamp=dormant_at
        )
        expired_at = dormant_at + main.IDENTITY_DORMANT_TTL_SEC + 1.0

        self.manager.cleanup(reference_time=expired_at)

        self.assertEqual(main.IDENTITY_EXPIRED, identity["state"])
        persisted = self._load_from_new_connection(gid)
        self.assertEqual(main.IDENTITY_EXPIRED, persisted["state"])
        self.assertEqual(expired_at, persisted["state_updated_at"])
        self.assertEqual("dormant_ttl_expired", persisted["state_reason"])
        self.assertFalse(
            self.manager._can_match(identity, "cam2", expired_at, None, (80, 160))
        )

        replacement = self.manager._new_identity(
            "cam2", 9, self.det["emb"], None, (80, 160), expired_at + 1.0
        )
        self.assertNotEqual(gid, replacement["gid"])

        audit_ts, reason, payload = self._latest_transition_audit(gid)
        self.assertEqual(expired_at, audit_ts)
        self.assertEqual("dormant_ttl_expired", reason)
        self.assertEqual(main.IDENTITY_DORMANT, payload["from_state"])
        self.assertEqual(main.IDENTITY_EXPIRED, payload["to_state"])

    def test_manager_restart_restores_state_and_keeps_gid_monotonic(self):
        dormant_gid, _ = self._seed_identity(
            main.IDENTITY_DORMANT, timestamp=250.0
        )
        expired_gid, _ = self._seed_identity(
            main.IDENTITY_EXPIRED, timestamp=251.0
        )
        self.store.close()

        restarted_store = IdentityStore(self.db_path)
        self.addCleanup(restarted_store.close)
        restarted_manager = main.GlobalIdentityManager(
            identity_store=restarted_store
        )

        self.assertEqual(
            main.IDENTITY_DORMANT,
            restarted_manager.identities[dormant_gid]["state"],
        )
        replacement = restarted_manager._new_identity(
            "cam2", 9, self.det["emb"], None, (80, 160), 252.0
        )
        self.assertEqual(expired_gid + 1, replacement["gid"])

    def test_diagnostics_include_safe_persistence_status(self):
        self._seed_identity(main.IDENTITY_ACTIVE, timestamp=275.0)

        diagnostics = self.manager.identity_state_diagnostics()

        self.assertEqual(1, diagnostics["state_counts"][main.IDENTITY_ACTIVE])
        self.assertTrue(diagnostics["persistence"]["connected"])
        self.assertEqual(
            os.path.abspath(self.db_path),
            os.path.abspath(diagnostics["persistence"]["path"]),
        )
        self.assertNotIn("embedding", json.dumps(diagnostics))

    def test_explicit_transition_graph_accepts_valid_edges(self):
        valid_edges = (
            (main.IDENTITY_PROVISIONAL, main.IDENTITY_ACTIVE),
            (main.IDENTITY_PROVISIONAL, main.IDENTITY_DORMANT),
            (main.IDENTITY_ACTIVE, main.IDENTITY_DORMANT),
            (main.IDENTITY_DORMANT, main.IDENTITY_ACTIVE),
            (main.IDENTITY_DORMANT, main.IDENTITY_EXPIRED),
        )

        for from_state, to_state in valid_edges:
            with self.subTest(from_state=from_state, to_state=to_state):
                gid, identity = self._seed_identity(from_state, timestamp=300.0)
                changed = self.manager._transition_identity(
                    gid, identity, to_state, 301.0, "state_graph_test"
                )
                self.assertTrue(changed)
                self.assertEqual(to_state, identity["state"])

    def test_explicit_transition_graph_rejects_invalid_edges(self):
        invalid_edges = (
            (main.IDENTITY_ACTIVE, main.IDENTITY_PROVISIONAL),
            (main.IDENTITY_DORMANT, main.IDENTITY_PROVISIONAL),
            (main.IDENTITY_EXPIRED, main.IDENTITY_ACTIVE),
            (main.IDENTITY_EXPIRED, main.IDENTITY_DORMANT),
            (main.IDENTITY_ACTIVE, main.IDENTITY_EXPIRED),
            (main.IDENTITY_PROVISIONAL, main.IDENTITY_EXPIRED),
        )

        for from_state, to_state in invalid_edges:
            with self.subTest(from_state=from_state, to_state=to_state):
                gid, identity = self._seed_identity(from_state, timestamp=400.0)
                with self.assertRaises(ValueError):
                    self.manager._transition_identity(
                        gid, identity, to_state, 401.0, "invalid_transition"
                    )
                self.assertEqual(from_state, identity["state"])

    def test_invalid_persisted_identity_can_expire_exceptionally(self):
        for from_state in (main.IDENTITY_PROVISIONAL, main.IDENTITY_ACTIVE):
            with self.subTest(from_state=from_state):
                gid, identity = self._seed_identity(from_state, timestamp=500.0)
                changed = self.manager._transition_identity(
                    gid,
                    identity,
                    main.IDENTITY_EXPIRED,
                    501.0,
                    "invalid_persisted_identity",
                )
                self.assertTrue(changed)
                self.assertEqual(main.IDENTITY_EXPIRED, identity["state"])

    def test_same_state_transition_is_a_no_op(self):
        gid, identity = self._seed_identity(main.IDENTITY_ACTIVE, timestamp=600.0)

        changed = self.manager._transition_identity(
            gid, identity, main.IDENTITY_ACTIVE, 601.0, "duplicate_event"
        )

        self.assertFalse(changed)
        self.assertEqual(main.IDENTITY_ACTIVE, identity["state"])
        connection = sqlite3.connect(self.db_path)
        try:
            audit_count = connection.execute(
                """
                SELECT COUNT(*)
                FROM identity_audit
                WHERE global_id = ? AND event_type = 'state_transition'
                """,
                (gid,),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(0, audit_count)

    def test_cleanup_background_workers_closes_identity_store(self):
        with patch.object(main, "stop_background_workers") as stop_workers, patch.object(
            main.global_identity_manager, "close"
        ) as close_identity_store:
            main.cleanup_background_workers()

        stop_workers.assert_called_once_with()
        close_identity_store.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
