import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ["IDENTITY_DB_PATH"] = ":memory:"
os.environ["REID_ENABLED"] = "false"

from backend import main
from backend.identity_store import IdentityStore


def embedding(*values):
    return main.l2_normalize(np.asarray(values, dtype=np.float32))


def detection(track_id, vector, event_time, generation=1, forced_gid=None):
    return {
        "tid": track_id,
        "emb": vector,
        "box": (10, 10, 50, 90),
        "box_wh": (40, 80),
        "map_pos": None,
        "overlap": forced_gid is not None,
        "forced_gid": forced_gid,
        "local_track_confirmed": True,
        "detector_confidence": 0.95,
        "crop_size": (40, 80),
        "blur_variance": 100.0,
        "border_clip_ratio": 0.0,
        "event_time": float(event_time),
        "coordinator_generation": generation,
    }


def topology(*transitions):
    return {
        "version": 2,
        "enforce": True,
        "transitions": [
            {
                "from_camera": source,
                "to_camera": destination,
                "min_travel_time_sec": minimum,
                "max_travel_time_sec": maximum,
                "overlap_allowed": overlap,
            }
            for source, destination, minimum, maximum, overlap in transitions
        ],
    }


class GlobalPresenceHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = os.path.join(self.temp_dir.name, "presence.sqlite3")
        self.store = IdentityStore(self.db_path)
        self.addCleanup(self.store.close)
        self.manager = main.GlobalIdentityManager(identity_store=self.store)
        self.person = embedding(1.0, 0.0, 0.0)
        self.other_person = embedding(0.0, 1.0, 0.0)
        self.topology = topology(
            ("A", "B", 1.0, 10.0, False),
            ("B", "A", 1.0, 10.0, False),
            ("A", "C", 1.0, 10.0, False),
        )
        self.topology_patch = patch.object(
            main,
            "topology_config",
            self.topology,
        )
        self.topology_patch.start()
        self.addCleanup(self.topology_patch.stop)

    def _seed_mature_identity(
        self,
        camera="A",
        local_id=5,
        event_time=100.0,
        vector=None,
    ):
        vector = self.person if vector is None else vector
        result = self.manager.assign_global_batch(
            {camera: [detection(local_id, vector, event_time)]},
            event_time=event_time,
            batch_id=f"seed-{camera}-{local_id}",
        )[camera][0]
        identity = self.manager.identities[result["gid"]]
        identity.update({
            "state": main.IDENTITY_ACTIVE,
            "state_updated_at": float(event_time),
            "state_reason": "test_mature_identity",
            "embedding": vector.copy(),
            "gallery": [vector.copy()],
            "gallery_mature": True,
        })
        self.store.save_identity(
            result["gid"],
            identity,
            "test_mature_identity",
            "test_mature_identity",
            event_time,
        )
        return result["gid"]

    def _handoff(self, gid, source, destination, local_id, event_time):
        result = self.manager.assign_global_batch(
            {destination: [detection(local_id, self.person, event_time)]},
            event_time=event_time,
            batch_id=f"handoff-{source}-{destination}",
        )[destination][0]
        self.assertEqual(gid, result["gid"])
        self.assertEqual("global-cross-camera", result["source"])
        return result

    def test_non_overlap_handoff_deactivates_source_and_preserves_identity(self):
        gid = self._seed_mature_identity()
        identity = self.manager.identities[gid]
        gallery_before = [item.copy() for item in identity["gallery"]]

        result = self._handoff(gid, "A", "B", 20, 103.0)

        presence = identity["camera_presence"]
        self.assertFalse(presence["A"]["active"])
        self.assertEqual(
            "confirmed_non_overlap_handoff",
            presence["A"]["inactive_reason"],
        )
        self.assertTrue(presence["B"]["active"])
        self.assertEqual(20, presence["B"]["local_track_id"])
        self.assertEqual(1, presence["B"]["generation"])
        self.assertEqual("B", identity["last_cam"])
        self.assertEqual(len(gallery_before), len(identity["gallery"]))
        for expected, actual in zip(gallery_before, identity["gallery"]):
            np.testing.assert_array_equal(expected, actual)
        self.assertTrue(result["handoff_committed"])
        self.assertEqual(1, len(identity["handoff_history"]))

        decision = self.manager.last_global_batch_diagnostics[
            "handoff_decisions"
        ][0]
        self.assertTrue(decision["handoff_committed"])
        self.assertEqual("A", decision["from_camera"])
        self.assertEqual("B", decision["to_camera"])
        self.assertEqual(3.0, decision["event_time_delta_sec"])
        self.assertTrue(decision["hard_gate_result"]["passed"])
        self.assertEqual(1.0, decision["candidate_appearance"])

    def test_return_handoff_recovers_same_gid_with_new_local_id(self):
        gid = self._seed_mature_identity()
        self._handoff(gid, "A", "B", 20, 103.0)

        returned = self._handoff(gid, "B", "A", 31, 106.0)

        identity = self.manager.identities[gid]
        presence = identity["camera_presence"]
        self.assertTrue(presence["A"]["active"])
        self.assertEqual(31, presence["A"]["local_track_id"])
        self.assertFalse(presence["B"]["active"])
        self.assertEqual("A", identity["last_cam"])
        self.assertEqual(2, len(identity["handoff_history"]))
        self.assertEqual("B", returned["handoff"]["from_camera"])
        self.assertEqual("A", returned["handoff"]["to_camera"])

    def test_different_person_does_not_steal_presence(self):
        gid = self._seed_mature_identity()

        result = self.manager.assign_global_batch(
            {"B": [detection(20, self.other_person, 103.0)]},
            event_time=103.0,
            batch_id="different-person",
        )["B"][0]

        self.assertNotEqual(gid, result["gid"])
        identity = self.manager.identities[gid]
        self.assertTrue(identity["camera_presence"]["A"]["active"])
        self.assertNotIn("B", identity["camera_presence"])
        self.assertEqual([], identity["handoff_history"])

    def test_unconfirmed_cross_camera_observation_does_not_transfer_presence(self):
        gid = self._seed_mature_identity()
        unconfirmed = detection(20, self.person, 103.0)
        unconfirmed["local_track_confirmed"] = False

        result = self.manager.assign_global_batch(
            {"B": [unconfirmed]},
            event_time=103.0,
            batch_id="unconfirmed-handoff",
        )["B"][0]

        self.assertNotEqual(gid, result["gid"])
        identity = self.manager.identities[gid]
        self.assertTrue(identity["camera_presence"]["A"]["active"])
        self.assertNotIn("B", identity["camera_presence"])
        self.assertEqual([], identity["handoff_history"])
        self.assertEqual(
            "unconfirmed_cross_camera_observation",
            self.manager.last_global_batch_diagnostics["gate_failures"][0][
                "reason"
            ],
        )

    def test_ambiguous_candidate_does_not_transfer_presence(self):
        first_gid = self._seed_mature_identity(local_id=5, event_time=100.0)
        second_vector = self.other_person
        second_gid = self._seed_mature_identity(
            local_id=6,
            event_time=100.1,
            vector=second_vector,
        )

        def ambiguous_score(gid, *_args):
            return {
                "gid": gid,
                "score": 0.80 if gid == first_gid else 0.76,
                "appearance": 0.95,
                "quality_adjusted_appearance": 0.95,
                "tracklet_quality": 1.0,
                "motion": 0.0,
                "map": 0.0,
                "time": 1.0,
                "cross_camera": True,
                "source_type": "cross-camera",
            }

        with patch.object(self.manager, "_pair_score", side_effect=ambiguous_score):
            result = self.manager.assign_global_batch(
                {"B": [detection(20, self.person, 103.0)]},
                event_time=103.0,
                batch_id="ambiguous-handoff",
            )["B"][0]

        self.assertIsNone(result)
        for gid in (first_gid, second_gid):
            identity = self.manager.identities[gid]
            self.assertTrue(identity["camera_presence"]["A"]["active"])
            self.assertEqual([], identity["handoff_history"])
        decision = self.manager.last_global_batch_diagnostics[
            "handoff_decisions"
        ][0]
        self.assertFalse(decision["handoff_committed"])
        self.assertEqual(
            "ambiguous_cross_camera_handoff_pending",
            decision["handoff_rejection_reason"],
        )
        self.assertAlmostEqual(0.04, decision["margin"])
        self.assertIn(decision["candidate_gid"], {first_gid, second_gid})
        self.assertEqual("A", decision["from_camera"])
        self.assertEqual(0.95, decision["candidate_appearance"])
        self.assertTrue(decision["hard_gate_result"]["passed"])
        self.assertEqual(
            "unresolved-cross-camera",
            decision["assignment_source"],
        )
        self.assertEqual(
            "pending",
            self.manager.last_global_batch_diagnostics["rows"][0][
                "assignment_state"
            ],
        )

    def test_explicit_overlap_keeps_both_presences_until_source_ages(self):
        main.topology_config = topology(
            ("A", "B", 0.0, 10.0, True),
            ("B", "A", 0.0, 10.0, True),
        )
        gid = self._seed_mature_identity()

        self._handoff(gid, "A", "B", 20, 101.0)

        identity = self.manager.identities[gid]
        self.assertTrue(identity["camera_presence"]["A"]["active"])
        self.assertTrue(identity["camera_presence"]["B"]["active"])
        self.assertEqual(
            "confirmed_overlap_handoff",
            identity["handoff_history"][-1]["reason"],
        )

        later = 101.0 + main.REID_MAX_IDLE_SEC + 0.1
        self.manager.assign_global_batch(
            {"B": [detection(20, self.person, later)]},
            event_time=later,
            batch_id="overlap-presence-aging",
        )
        self.assertFalse(identity["camera_presence"]["A"]["active"])
        self.assertTrue(identity["camera_presence"]["B"]["active"])

    def test_one_to_one_allows_only_one_non_overlap_handoff(self):
        gid = self._seed_mature_identity()

        results = self.manager.assign_global_batch(
            {
                "B": [detection(20, self.person, 103.0)],
                "C": [detection(30, self.person, 103.1)],
            },
            event_time=103.1,
            batch_id="one-to-one-handoff",
        )

        assigned = [results["B"][0]["gid"], results["C"][0]["gid"]]
        self.assertEqual(1, assigned.count(gid))
        self.assertEqual(2, len(set(assigned)))
        active = [
            camera
            for camera, record in self.manager.identities[gid][
                "camera_presence"
            ].items()
            if record["active"]
        ]
        self.assertEqual(1, len(active))
        self.assertEqual(1, len(self.manager.identities[gid]["handoff_history"]))

    def test_presence_and_handoff_audit_restore_from_temporary_database(self):
        gid = self._seed_mature_identity()
        self._handoff(gid, "A", "B", 20, 103.0)
        self.store.close()

        restarted_store = IdentityStore(self.db_path)
        self.addCleanup(restarted_store.close)
        restarted = main.GlobalIdentityManager(identity_store=restarted_store)

        identity = restarted.identities[gid]
        self.assertFalse(identity["camera_presence"]["A"]["active"])
        self.assertTrue(identity["camera_presence"]["B"]["active"])
        self.assertEqual("B", identity["last_cam"])
        self.assertEqual(1, len(identity["handoff_history"]))
        self.assertEqual("A", identity["handoff_history"][0]["from_camera"])
        self.assertEqual("B", identity["handoff_history"][0]["to_camera"])
        persistence_handoff = restarted_store.status()["recent_handoffs"][0]
        self.assertEqual(gid, persistence_handoff["gid"])
        self.assertEqual("A", persistence_handoff["from_camera"])
        self.assertEqual("B", persistence_handoff["to_camera"])

        connection = sqlite3.connect(self.db_path)
        try:
            row = connection.execute(
                "SELECT payload FROM identity_audit "
                "WHERE global_id = ? AND event_type = 'handoff'",
                (gid,),
            ).fetchone()
        finally:
            connection.close()
        self.assertIsNotNone(row)
        handoff = json.loads(row[0])["handoff"]
        self.assertEqual("A", handoff["from_camera"])
        self.assertEqual("B", handoff["to_camera"])

    def test_handoff_uses_observation_event_time_not_processing_time(self):
        gid = self._seed_mature_identity(event_time=100.0)

        with patch.object(main.time, "time", return_value=10000.0):
            result = self.manager.assign_global_batch(
                {"B": [detection(20, self.person, 103.0)]},
                event_time=10000.0,
                batch_id="event-time-handoff",
            )["B"][0]

        self.assertEqual(gid, result["gid"])
        self.assertEqual(103.0, result["handoff"]["entry_event_time"])
        self.assertEqual(3.0, result["handoff"]["event_time_delta_sec"])
        self.assertEqual(
            3.0,
            result["handoff"]["topology_result"]["event_time_delta_sec"],
        )


if __name__ == "__main__":
    unittest.main()
