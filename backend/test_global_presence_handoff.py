import json
import os
import sqlite3
import tempfile
import unittest
from itertools import permutations
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

    def _seed_cam2_then_handoff_two_identities_to_cam1(
        self,
        overlap_allowed=False,
    ):
        main.topology_config = topology(
            ("cam2", "cam1", 1.0, 10.0, overlap_allowed),
            ("cam1", "cam2", 1.0, 10.0, overlap_allowed),
        )
        self.manager.next_global_id = 2
        vectors = {
            20: self.person,
            30: self.other_person,
        }
        source_detections = []
        for track_id, vector in vectors.items():
            item = detection(track_id, vector, 100.0)
            x_offset = 0 if track_id == 20 else 240
            item["box"] = (10 + x_offset, 10, 50 + x_offset, 90)
            source_detections.append(item)

        seeded = self.manager.assign_global_batch(
            {"cam2": source_detections},
            event_time=100.0,
            batch_id="seed-cam2-g2-g3",
        )["cam2"]
        gids_by_tid = {
            item["tid"]: result["gid"]
            for item, result in zip(source_detections, seeded)
        }
        self.assertEqual({20: 2, 30: 3}, gids_by_tid)

        for item in source_detections:
            gid = gids_by_tid[item["tid"]]
            identity = self.manager.identities[gid]
            identity.update({
                "state": main.IDENTITY_ACTIVE,
                "state_updated_at": 100.0,
                "state_reason": "test_mature_identity",
                "embedding": item["emb"].copy(),
                "gallery": [item["emb"].copy()],
                "gallery_mature": True,
            })

        cam1_detections = [
            detection(120, self.person, 103.0),
            detection(130, self.other_person, 103.0),
        ]
        cam1_detections[0]["box"] = (10, 10, 50, 90)
        cam1_detections[1]["box"] = (250, 10, 290, 90)
        moved = self.manager.assign_global_batch(
            {"cam1": cam1_detections},
            event_time=103.0,
            batch_id="handoff-cam2-cam1-g2-g3",
        )["cam1"]
        self.assertEqual([2, 3], [item["gid"] for item in moved])

        source_history = [
            {
                "gid": gids_by_tid[item["tid"]],
                "box": item["box"],
                "box_wh": item["box_wh"],
                "tid": item["tid"],
                "cam_name": "cam2",
                "overlap": False,
                "ts": 100.0,
                "emb": item["emb"],
            }
            for item in source_detections
        ]
        return gids_by_tid, source_history, cam1_detections

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

    def test_local_transition_return_cam2_cam1_cam2_preserves_g2_g3(self):
        _, source_history, _ = (
            self._seed_cam2_then_handoff_two_identities_to_cam1()
        )
        returned_detections = [
            detection(21, self.person, 106.0),
            detection(31, self.other_person, 106.0),
        ]
        returned_detections[0]["box"] = (10, 10, 50, 90)
        returned_detections[1]["box"] = (250, 10, 290, 90)

        returned = self.manager.assign_global_batch(
            {"cam2": returned_detections},
            prev_assignments_by_camera={"cam2": source_history},
            event_time=106.0,
            batch_id="local-transition-return-cam2",
        )["cam2"]

        self.assertEqual([2, 3], [item["gid"] for item in returned])
        self.assertEqual(
            ["local-transition-recovery-v2"] * 2,
            [item["source"] for item in returned],
        )
        for gid in (2, 3):
            identity = self.manager.identities[gid]
            active_cameras = {
                camera
                for camera, record in identity["camera_presence"].items()
                if record["active"]
            }
            self.assertEqual({"cam2"}, active_cameras)
            self.assertEqual("cam1", identity["handoff_history"][-1]["from_camera"])
            self.assertEqual("cam2", identity["handoff_history"][-1]["to_camera"])
            self.assertEqual(
                "local-transition-recovery-v2",
                identity["handoff_history"][-1]["assignment_source"],
            )

    def test_local_transition_does_not_reclaim_live_other_camera_presence(self):
        _, source_history, cam1_detections = (
            self._seed_cam2_then_handoff_two_identities_to_cam1()
        )
        returned_detections = [
            detection(21, self.person, 106.0),
            detection(31, self.other_person, 106.0),
        ]
        returned_detections[0]["box"] = (10, 10, 50, 90)
        returned_detections[1]["box"] = (250, 10, 290, 90)
        live_cam1 = [
            detection(120, self.person, 106.0),
            detection(130, self.other_person, 106.0),
        ]
        live_cam1[0]["box"] = cam1_detections[0]["box"]
        live_cam1[1]["box"] = cam1_detections[1]["box"]

        results = self.manager.assign_global_batch(
            {
                "cam1": live_cam1,
                "cam2": returned_detections,
            },
            prev_assignments_by_camera={"cam2": source_history},
            event_time=106.0,
            batch_id="local-transition-live-presence-guard",
        )

        self.assertEqual([2, 3], [item["gid"] for item in results["cam1"]])
        self.assertTrue(
            all(
                item is None or item["gid"] not in {2, 3}
                for item in results["cam2"]
            )
        )
        for gid in (2, 3):
            active_cameras = {
                camera
                for camera, record in self.manager.identities[gid][
                    "camera_presence"
                ].items()
                if record["active"]
            }
            self.assertEqual({"cam1"}, active_cameras)

    def test_local_transition_preserves_overlap_allowed_presence(self):
        _, source_history, _ = (
            self._seed_cam2_then_handoff_two_identities_to_cam1(
                overlap_allowed=True,
            )
        )
        returned_detections = [
            detection(21, self.person, 106.0),
            detection(31, self.other_person, 106.0),
        ]
        returned_detections[0]["box"] = (10, 10, 50, 90)
        returned_detections[1]["box"] = (250, 10, 290, 90)

        returned = self.manager.assign_global_batch(
            {"cam2": returned_detections},
            prev_assignments_by_camera={"cam2": source_history},
            event_time=106.0,
            batch_id="local-transition-overlap-return-cam2",
        )["cam2"]

        self.assertEqual([2, 3], [item["gid"] for item in returned])
        for gid in (2, 3):
            identity = self.manager.identities[gid]
            active_cameras = {
                camera
                for camera, record in identity["camera_presence"].items()
                if record["active"]
            }
            self.assertEqual({"cam1", "cam2"}, active_cameras)
            self.assertEqual(
                "confirmed_overlap_handoff",
                identity["handoff_history"][-1]["reason"],
            )

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

    def test_live_trusted_source_cannot_be_stolen_by_destination(self):
        gid = self._seed_mature_identity(camera="A", local_id=3)

        results = self.manager.assign_global_batch(
            {
                "A": [detection(3, self.person, 103.0)],
                "B": [detection(11, self.person, 103.1)],
            },
            event_time=103.1,
            batch_id="live-source-ownership-guard",
        )

        self.assertEqual(gid, results["A"][0]["gid"])
        self.assertIsNone(results["B"][0])
        identity = self.manager.identities[gid]
        self.assertTrue(identity["camera_presence"]["A"]["active"])
        self.assertNotIn("B", identity["camera_presence"])
        self.assertEqual([], identity["handoff_history"])
        self.assertIn(
            "trusted_active_source_ownership",
            {
                item["reason"]
                for item in self.manager.last_global_batch_diagnostics[
                    "rejections"
                ]
            },
        )

    def test_destination_can_handoff_after_source_really_disappears(self):
        gid = self._seed_mature_identity(camera="A", local_id=3)

        result = self.manager.assign_global_batch(
            {"B": [detection(11, self.person, 103.0)]},
            event_time=103.0,
            batch_id="source-gone-handoff",
        )["B"][0]

        self.assertEqual(gid, result["gid"])
        self.assertTrue(result["handoff_committed"])
        self.assertFalse(
            self.manager.identities[gid]["camera_presence"]["A"]["active"]
        )
        self.assertTrue(
            self.manager.identities[gid]["camera_presence"]["B"]["active"]
        )

    def test_two_people_cross_camera_reclaim_existing_gids(self):
        first_gid = self._seed_mature_identity(
            camera="A",
            local_id=3,
            vector=self.person,
        )
        second_gid = self._seed_mature_identity(
            camera="A",
            local_id=4,
            event_time=100.1,
            vector=self.other_person,
        )

        destination = self.manager.assign_global_batch(
            {
                "B": [
                    detection(11, self.person, 103.0),
                    detection(12, self.other_person, 103.0),
                ],
            },
            event_time=103.0,
            batch_id="two-person-reclaim",
        )["B"]

        self.assertEqual(
            [first_gid, second_gid],
            [item["gid"] for item in destination],
        )
        self.assertEqual(
            ["global-cross-camera", "global-cross-camera"],
            [item["source"] for item in destination],
        )

    def test_three_camera_destination_order_is_deterministic_for_reclaim(self):
        from itertools import permutations

        expected = None
        for order in permutations(("B", "C", "A")):
            manager = main.GlobalIdentityManager(identity_store=None)
            first_gid = manager.assign_global_batch(
                {"A": [detection(3, self.person, 100.0)]},
                event_time=100.0,
                batch_id="order-seed-a",
            )["A"][0]["gid"]
            second_gid = manager.assign_global_batch(
                {"A": [detection(4, self.other_person, 100.1)]},
                event_time=100.1,
                batch_id="order-seed-a2",
            )["A"][0]["gid"]
            for gid, vector in (
                (first_gid, self.person),
                (second_gid, self.other_person),
            ):
                manager.identities[gid].update({
                    "state": main.IDENTITY_ACTIVE,
                    "embedding": vector.copy(),
                    "gallery": [vector.copy()],
                    "gallery_mature": True,
                })
            detections = {
                "A": [],
                "B": [detection(11, self.person, 103.0)],
                "C": [detection(12, self.other_person, 103.0)],
            }
            result = manager.assign_global_batch(
                {camera: detections[camera] for camera in order},
                event_time=103.0,
                batch_id="three-camera-reclaim-order",
            )
            actual = {
                "B": result["B"][0]["gid"],
                "C": result["C"][0]["gid"],
            }
            if expected is None:
                expected = actual
            self.assertEqual(expected, actual)
            self.assertEqual(first_gid, actual["B"])
            self.assertEqual(second_gid, actual["C"])

    def test_explicit_overlap_allows_live_source_and_destination_same_gid(self):
        main.topology_config = topology(
            ("A", "B", 0.0, 10.0, True),
            ("B", "A", 0.0, 10.0, True),
        )
        gid = self._seed_mature_identity(camera="A", local_id=3)

        results = self.manager.assign_global_batch(
            {
                "A": [detection(3, self.person, 103.0)],
                "B": [detection(11, self.person, 103.0)],
            },
            event_time=103.0,
            batch_id="explicit-overlap-live-source",
        )

        self.assertEqual(gid, results["A"][0]["gid"])
        self.assertEqual(gid, results["B"][0]["gid"])
        active_cameras = {
            camera
            for camera, presence in self.manager.identities[gid][
                "camera_presence"
            ].items()
            if presence["active"]
        }
        self.assertEqual({"A", "B"}, active_cameras)

    def test_local_track_reassignment_deactivates_previous_gid_presence(self):
        old_gid = self._seed_mature_identity(camera="Penpxp", local_id=27)
        new_gid = self._seed_mature_identity(
            camera="Penpxp",
            local_id=28,
            event_time=100.1,
            vector=self.other_person,
        )

        result = self.manager._commit_assignment(
            new_gid,
            "Penpxp",
            27,
            self.other_person,
            None,
            (40, 80),
            102.0,
            0.95,
            "local-track-verified",
            generation=1,
        )

        self.assertEqual(new_gid, result["gid"])
        self.assertEqual(new_gid, self.manager.local_to_global[("Penpxp", 27)]["gid"])
        old_presence = self.manager.identities[old_gid]["camera_presence"]["Penpxp"]
        new_presence = self.manager.identities[new_gid]["camera_presence"]["Penpxp"]
        self.assertFalse(old_presence["active"])
        self.assertEqual("local_track_reassigned", old_presence["inactive_reason"])
        self.assertTrue(new_presence["active"])

    def test_repeated_local_track_verified_has_single_active_presence(self):
        gid = self._seed_mature_identity(camera="Penpxp", local_id=27)
        for event_time in (101.0, 102.0, 103.0):
            result = self.manager.assign_global_batch(
                {"Penpxp": [detection(27, self.person, event_time)]},
                event_time=event_time,
                batch_id=f"repeat-local-27-{event_time}",
            )["Penpxp"][0]
            self.assertEqual(gid, result["gid"])

        active = [
            identity["camera_presence"]["Penpxp"]
            for identity in self.manager.identities.values()
            if "Penpxp" in identity.get("camera_presence", {})
            and identity["camera_presence"]["Penpxp"].get("active")
            and identity["camera_presence"]["Penpxp"].get("local_track_id") == 27
        ]
        self.assertEqual([gid], [record["gid"] for record in active])

    def test_transition_recovery_owner_change_closes_previous_presence(self):
        old_gid = self._seed_mature_identity(camera="Penpxp", local_id=27)
        new_gid = self._seed_mature_identity(
            camera="Penpxp",
            local_id=28,
            event_time=100.1,
            vector=self.other_person,
        )
        self.manager._commit_assignment(
            new_gid,
            "Penpxp",
            27,
            self.other_person,
            None,
            (40, 80),
            101.0,
            0.95,
            "local-transition-recovery-v2",
            generation=1,
        )
        self.assertFalse(
            self.manager.identities[old_gid]["camera_presence"]["Penpxp"]["active"]
        )
        self.assertEqual(
            "local_track_reassigned",
            self.manager.identities[old_gid]["camera_presence"]["Penpxp"]["inactive_reason"],
        )

    def test_cross_camera_same_local_track_cannot_have_two_gids(self):
        old_gid = self._seed_mature_identity(camera="A", local_id=27)
        new_gid = self._seed_mature_identity(
            camera="B",
            local_id=27,
            event_time=100.1,
            vector=self.other_person,
        )
        self.manager._commit_assignment(
            new_gid,
            "A",
            27,
            self.other_person,
            None,
            (40, 80),
            102.0,
            0.95,
            "global-cross-camera",
            generation=1,
        )
        self.assertFalse(
            self.manager.identities[old_gid]["camera_presence"]["A"]["active"]
        )
        self.assertTrue(
            self.manager.identities[new_gid]["camera_presence"]["A"]["active"]
        )
        self.assertFalse(
            self.manager.identities[new_gid]["camera_presence"]["B"]["active"]
        )

    def test_three_camera_ownership_guard_is_permutation_deterministic(self):
        expected = None
        for camera_order in permutations(("A", "B", "C")):
            manager = main.GlobalIdentityManager(identity_store=None)
            first = manager.assign_global_batch(
                {"A": [detection(3, self.person, 100.0)]},
                event_time=100.0,
                batch_id="seed-owner-a",
            )["A"][0]["gid"]
            second = manager.assign_global_batch(
                {"C": [detection(7, self.other_person, 100.0)]},
                event_time=100.0,
                batch_id="seed-owner-c",
            )["C"][0]["gid"]
            for gid, vector in ((first, self.person), (second, self.other_person)):
                manager.identities[gid].update({
                    "state": main.IDENTITY_ACTIVE,
                    "embedding": vector.copy(),
                    "gallery": [vector.copy()],
                    "gallery_mature": True,
                })

            detections = {
                "A": [detection(3, self.person, 103.0)],
                "B": [detection(11, self.person, 103.1)],
                "C": [detection(7, self.other_person, 103.0)],
            }
            results = manager.assign_global_batch(
                {
                    camera: detections[camera]
                    for camera in camera_order
                },
                event_time=103.1,
                batch_id="three-camera-permutation",
            )
            actual = {
                camera: (
                    None
                    if results[camera][0] is None
                    else results[camera][0]["gid"]
                )
                for camera in ("A", "B", "C")
            }
            if expected is None:
                expected = actual
            self.assertEqual(expected, actual)
            self.assertEqual(first, actual["A"])
            self.assertIsNone(actual["B"])
            self.assertEqual(second, actual["C"])


if __name__ == "__main__":
    unittest.main()
