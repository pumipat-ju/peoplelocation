import asyncio
import copy
import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ["IDENTITY_DB_PATH"] = ":memory:"
os.environ["REID_ENABLED"] = "false"

from backend import main


class StatefulFakeYOLO:

    def __init__(self, barrier=None):
        self.frames = []
        self.calls = []
        self.active_calls = 0
        self.maximum_active_calls = 0
        self.counter_lock = threading.Lock()
        self.barrier = barrier
        self.predictor = SimpleNamespace(
            trackers=[object()]
        )

    def track(self, frame, **kwargs):
        with self.counter_lock:
            self.active_calls += 1
            self.maximum_active_calls = max(
                self.maximum_active_calls,
                self.active_calls
            )

        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=2)

            time.sleep(0.005)
            self.frames.append(
                int(frame[0, 0, 0])
            )
            self.calls.append(kwargs)
            return []
        finally:
            with self.counter_lock:
                self.active_calls -= 1


class LoopingFakeCapture:

    def __init__(self):
        self.read_count = 0
        self.seek_positions = []

    def read(self):
        self.read_count += 1

        if self.read_count == 1:
            return False, None

        return (
            True,
            np.full(
                (4, 4, 3),
                9,
                dtype=np.uint8
            )
        )

    def set(self, prop, value):
        self.seek_positions.append(
            (prop, value)
        )
        return True


def camera_data():
    return {
        "url": "synthetic",
        "source_type": "video",
        "loop_video": False,
        "processor": None,
        "src_pts": None,
        "dst_pts": None,
        "last_frame": None,
        "prev_assignments": [],
        **main.new_camera_tracker_context()
    }


def embedding(*values):
    return main.l2_normalize(
        np.asarray(
            values,
            dtype=np.float32
        )
    )


def detection(local_id, vector, x=0):
    return {
        "tid": local_id,
        "emb": vector,
        "box": (x, 0, x + 40, 100),
        "box_wh": (40, 100),
        "map_pos": None,
        "overlap": False,
        "forced_gid": None,
        "conf": 0.95
    }


def quality_detection(
    local_id,
    vector,
    x=0,
    camera_generation=1,
    local_track_confirmed=True,
):
    row = detection(local_id, vector, x=x)
    row.update({
        "detector_confidence": 0.95,
        "crop_size": (80, 160),
        "blur_variance": 100.0,
        "border_clip_ratio": 0.0,
        "camera_generation": camera_generation,
        "local_track_confirmed": local_track_confirmed,
    })
    return row


class PerCameraTrackerTests(unittest.TestCase):

    def setUp(self):
        with main.cameras_lock:
            main.cameras.clear()

    def tearDown(self):
        with main.cameras_lock:
            main.cameras.clear()

    def test_interleaved_frames_do_not_cross_tracker_state(self):
        tracker_a = StatefulFakeYOLO()
        tracker_b = StatefulFakeYOLO()

        with main.cameras_lock:
            main.cameras.update({
                "A": camera_data(),
                "B": camera_data()
            })

        with patch.object(
            main,
            "YOLO",
            side_effect=[tracker_a, tracker_b]
        ) as factory:
            for camera, marker, frame_index in (
                ("A", 1, 1),
                ("B", 10, 1),
                ("A", 2, 2),
                ("B", 11, 2)
            ):
                frame = np.full(
                    (8, 8, 3),
                    marker,
                    dtype=np.uint8
                )
                main.process_camera_frame(
                    camera,
                    frame,
                    frame_index
                )

        self.assertEqual(tracker_a.frames, [1, 2])
        self.assertEqual(tracker_b.frames, [10, 11])
        self.assertIsNot(
            main.cameras["A"]["tracking_model"],
            main.cameras["B"]["tracking_model"]
        )
        self.assertEqual(factory.call_count, 2)

        for call in tracker_a.calls + tracker_b.calls:
            self.assertTrue(call["persist"])
            self.assertEqual(
                call["tracker"],
                "botsort.yaml"
            )

        status_a = main.get_camera_tracker_status(
            "A"
        )
        status_b = main.get_camera_tracker_status(
            "B"
        )
        self.assertNotEqual(
            status_a["tracker_instance_id"],
            status_b["tracker_instance_id"]
        )
        self.assertNotEqual(
            status_a["botsort_state_ids"],
            status_b["botsort_state_ids"]
        )
        self.assertEqual(
            status_a["active_local_track_count"],
            0
        )
        self.assertEqual(
            status_a["local_track_scope"],
            "camera"
        )
        api_status = json.loads(
            asyncio.run(
                main.get_status()
            ).body
        )["cameras"]
        self.assertEqual(
            api_status["A"]["tracker"][
                "tracker_instance_id"
            ],
            status_a["tracker_instance_id"]
        )
        self.assertEqual(
            api_status["B"]["tracker"][
                "tracker_instance_id"
            ],
            status_b["tracker_instance_id"]
        )

    def test_same_camera_calls_are_serialized_but_cameras_can_overlap(self):
        same_camera_tracker = StatefulFakeYOLO()

        with main.cameras_lock:
            main.cameras["A"] = camera_data()

        with patch.object(
            main,
            "YOLO",
            return_value=same_camera_tracker
        ):
            threads = [
                threading.Thread(
                    target=main.process_camera_frame,
                    args=(
                        "A",
                        np.full(
                            (8, 8, 3),
                            marker,
                            dtype=np.uint8
                        ),
                        marker
                    )
                )
                for marker in range(1, 5)
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=3)

        self.assertEqual(
            same_camera_tracker.maximum_active_calls,
            1
        )
        self.assertTrue(
            all(not thread.is_alive() for thread in threads)
        )

        barrier = threading.Barrier(2)
        tracker_a = StatefulFakeYOLO(barrier)
        tracker_b = StatefulFakeYOLO(barrier)

        with main.cameras_lock:
            main.cameras.clear()
            main.cameras.update({
                "A": camera_data(),
                "B": camera_data()
            })

        with patch.object(
            main,
            "YOLO",
            side_effect=[tracker_a, tracker_b]
        ):
            main.get_camera_tracking_model("A")
            main.get_camera_tracking_model("B")
            threads = [
                threading.Thread(
                    target=main.process_camera_frame,
                    args=(
                        camera,
                        np.full(
                            (8, 8, 3),
                            marker,
                            dtype=np.uint8
                        ),
                        1
                    )
                )
                for camera, marker in (
                    ("A", 1),
                    ("B", 2)
                )
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=3)

        self.assertTrue(
            all(not thread.is_alive() for thread in threads)
        )
        self.assertEqual(tracker_a.frames, [1])
        self.assertEqual(tracker_b.frames, [2])

    def test_reset_one_camera_preserves_other_tracker_and_global_memory(self):
        first_a = StatefulFakeYOLO()
        tracker_b = StatefulFakeYOLO()
        second_a = StatefulFakeYOLO()
        identity_manager = main.GlobalIdentityManager()
        identity_manager.local_to_global.update({
            ("A", 7): {
                "gid": 1,
                "last_seen": time.time()
            },
            ("B", 7): {
                "gid": 2,
                "last_seen": time.time()
            }
        })
        identity_manager.identities.update({
            1: {"last_seen": time.time()},
            2: {"last_seen": time.time()}
        })

        with main.cameras_lock:
            main.cameras.update({
                "A": camera_data(),
                "B": camera_data()
            })

        with (
            patch.object(
                main,
                "YOLO",
                side_effect=[first_a, tracker_b, second_a]
            ),
            patch.object(
                main,
                "global_identity_manager",
                identity_manager
            )
        ):
            loaded_a = main.get_camera_tracking_model("A")
            loaded_b = main.get_camera_tracking_model("B")
            main.reset_camera_tracker(
                "A",
                reason="test_reset"
            )
            reloaded_a = main.get_camera_tracking_model("A")

        self.assertIs(loaded_a, first_a)
        self.assertIs(reloaded_a, second_a)
        self.assertIs(
            main.cameras["B"]["tracking_model"],
            loaded_b
        )
        self.assertNotIn(
            ("A", 7),
            identity_manager.local_to_global
        )
        self.assertIn(
            ("B", 7),
            identity_manager.local_to_global
        )
        self.assertEqual(
            set(identity_manager.identities),
            {1, 2}
        )

    def test_reset_removes_all_camera_local_evidence_and_reports_exact_counts(self):
        identity_manager = main.GlobalIdentityManager()
        now = time.time()
        gallery_a = embedding(1, 0, 0)
        gallery_b = embedding(0, 1, 0)
        identity_manager.identities.update({
            1: {
                "state": main.IDENTITY_ACTIVE,
                "last_seen": now,
                "embedding": gallery_a.copy(),
                "gallery": [gallery_a.copy()],
            },
            2: {
                "state": main.IDENTITY_ACTIVE,
                "last_seen": now,
                "embedding": gallery_b.copy(),
                "gallery": [gallery_b.copy()],
            },
        })
        identity_manager.local_to_global.update({
            ("A", 7): {"gid": 1, "last_seen": now},
            ("B", 7): {"gid": 2, "last_seen": now},
        })
        identity_manager.tracklets.update({
            ("A", 7): {
                "gid": 1,
                "generation": 1,
                "last_seen": now,
                "sample_count": 1,
                "samples": [{"emb": gallery_a.copy(), "ts": now}],
                "gallery_committed": False,
            },
            # This key intentionally has no local mapping.
            ("A", 99): {
                "gid": 1,
                "generation": 1,
                "last_seen": now,
                "sample_count": 1,
                "samples": [{"emb": gallery_a.copy(), "ts": now}],
                "gallery_committed": False,
            },
            ("B", 7): {
                "gid": 2,
                "generation": 1,
                "last_seen": now,
                "sample_count": 1,
                "samples": [{"emb": gallery_b.copy(), "ts": now}],
                "gallery_committed": False,
            },
        })
        identity_manager.occlusion_hold.update({
            ("A", 88): {"gid": 1, "until_ts": now + 1.0, "score": 1.0},
            ("B", 88): {"gid": 2, "until_ts": now + 1.0, "score": 1.0},
        })
        identity_manager.recent_same_cam.extend([
            {"gid": 1, "cam_name": "A", "embedding": gallery_a, "ts": now},
            {"gid": 2, "cam_name": "B", "embedding": gallery_b, "ts": now},
        ])
        identity_manager.recent_cross_cam.extend([
            {"gid": 1, "cam_name": "A", "embedding": gallery_a, "ts": now},
            {"gid": 2, "cam_name": "B", "embedding": gallery_b, "ts": now},
        ])

        with main.cameras_lock:
            main.cameras.update({
                "A": camera_data(),
                "B": camera_data(),
            })

        with (
            patch.object(
                main,
                "global_identity_manager",
                identity_manager,
            ),
            patch.object(
                main.global_assignment_coordinator,
                "discard_camera",
                return_value=True,
            ) as discard_camera,
        ):
            result = main.reset_camera_tracker(
                "A",
                reason="test_downstream_cleanup",
            )

        discard_camera.assert_called_once_with("A")
        self.assertEqual(1, result["local_mappings_removed"])
        self.assertEqual(1, result["occlusion_holds_removed"])
        self.assertEqual(2, result["tracklets_removed"])
        self.assertEqual(1, result["recent_same_cam_removed"])
        self.assertEqual(1, result["recent_cross_cam_removed"])

        self.assertNotIn(("A", 7), identity_manager.local_to_global)
        self.assertNotIn(("A", 7), identity_manager.tracklets)
        self.assertNotIn(("A", 99), identity_manager.tracklets)
        self.assertNotIn(("A", 88), identity_manager.occlusion_hold)
        self.assertTrue(all(
            item["cam_name"] != "A"
            for item in identity_manager.recent_same_cam
        ))
        self.assertTrue(all(
            item["cam_name"] != "A"
            for item in identity_manager.recent_cross_cam
        ))

        self.assertIn(("B", 7), identity_manager.local_to_global)
        self.assertIn(("B", 7), identity_manager.tracklets)
        self.assertEqual(2, identity_manager.tracklets[("B", 7)]["gid"])
        self.assertEqual(1, identity_manager.tracklets[("B", 7)]["generation"])
        self.assertIn(("B", 88), identity_manager.occlusion_hold)
        self.assertEqual(
            ["B"],
            [item["cam_name"] for item in identity_manager.recent_same_cam],
        )
        self.assertEqual(
            ["B"],
            [item["cam_name"] for item in identity_manager.recent_cross_cam],
        )
        self.assertEqual({1, 2}, set(identity_manager.identities))
        np.testing.assert_array_equal(
            gallery_a,
            identity_manager.identities[1]["gallery"][0],
        )
        np.testing.assert_array_equal(
            gallery_b,
            identity_manager.identities[2]["gallery"][0],
        )

    def test_video_loop_marks_only_that_source_for_tracker_reset(self):
        manager = main.MultiCameraVideoManager()
        looping_capture = LoopingFakeCapture()
        manager.videos = {
            "A": {
                "cap": looping_capture,
                "fps": 25.0,
                "total_frames": 10,
                "loop_video": True,
                "frame_index": 10,
                "tracker_reset_pending": False
            }
        }
        manager.frames = {"A": None}
        manager.frame_indices = {"A": 10}
        manager.running = {"A": True}

        frames = manager.read_synchronized_frames()

        self.assertTrue(frames["A"]["source_reset"])
        self.assertEqual(
            frames["A"]["frame_index"],
            1
        )
        self.assertEqual(
            looping_capture.seek_positions,
            [(main.cv2.CAP_PROP_POS_FRAMES, 0)]
        )


class LocalAndGlobalIdentityTests(unittest.TestCase):

    def test_local_mapping_expiry_removes_its_tracklet(self):
        manager = main.GlobalIdentityManager()
        local_key = ("A", 7)
        reference_time = main.REID_MAX_IDLE_SEC + 10.0
        vector = embedding(1, 0, 0)
        result = manager._new_identity(
            "A",
            7,
            vector,
            None,
            (40, 100),
            reference_time,
        )
        manager._record_tracklet_sample(
            result["gid"],
            "A",
            7,
            quality_detection(7, vector),
            0.0,
        )
        manager.local_to_global[local_key]["last_seen"] = 0.0

        self.assertIn(local_key, manager.tracklets)

        manager.cleanup(reference_time=reference_time)

        self.assertNotIn(local_key, manager.local_to_global)
        self.assertNotIn(local_key, manager.tracklets)

    def test_unconfirmed_detection_index_is_not_persistent_local_evidence(self):
        manager = main.GlobalIdentityManager()
        row = quality_detection(
            -1000001,
            embedding(1, 0, 0),
            local_track_confirmed=False,
        )

        result = manager.assign_batch(
            "A",
            [row],
            event_time=1.0,
        )[0]
        identity = manager.identities[result["gid"]]

        self.assertNotIn(
            ("A", -1000001),
            manager.local_to_global
        )
        self.assertNotIn(
            ("A", -1000001),
            manager.tracklets,
        )
        self.assertFalse(result["gallery_update_accepted"])
        self.assertEqual(
            "unconfirmed_local_track",
            result["gallery_rejection_reason"],
        )
        self.assertEqual([], identity["gallery"])
        self.assertEqual(
            0,
            identity["gallery_diagnostics"]["accepted_updates"],
        )
        self.assertEqual(
            0,
            identity["gallery_diagnostics"]["tracklet_sample_count"],
        )

    def test_tracklet_generation_change_discards_previous_samples(self):
        manager = main.GlobalIdentityManager()
        sample_count = main.REID_TRACKLET_MIN_SAMPLES - 1
        basis = np.eye(
            main.REID_TRACKLET_MIN_SAMPLES + 1,
            dtype=np.float32,
        )
        result = manager._new_identity(
            "A",
            7,
            basis[0],
            None,
            (40, 100),
            1.0,
        )
        gid = result["gid"]

        for index in range(sample_count):
            manager._record_tracklet_sample(
                gid,
                "A",
                7,
                quality_detection(
                    7,
                    basis[index],
                    camera_generation=1,
                ),
                float(index + 1),
            )

        accepted, reason = manager._record_tracklet_sample(
            gid,
            "A",
            7,
            quality_detection(
                7,
                basis[-1],
                camera_generation=2,
            ),
            10.0,
        )

        self.assertFalse(accepted)
        self.assertEqual("tracklet_not_mature", reason)
        self.assertEqual(
            main.IDENTITY_PROVISIONAL,
            manager.identities[gid]["state"],
        )
        self.assertEqual(
            1,
            manager.identities[gid]["gallery_diagnostics"][
                "tracklet_sample_count"
            ],
        )
        self.assertEqual([], manager.identities[gid]["gallery"])

    def test_local_key_reassignment_discards_previous_gid_samples(self):
        manager = main.GlobalIdentityManager()
        sample_count = main.REID_TRACKLET_MIN_SAMPLES - 1
        basis = np.eye(
            main.REID_TRACKLET_MIN_SAMPLES + 1,
            dtype=np.float32,
        )
        first = manager._new_identity(
            "A",
            7,
            basis[0],
            None,
            (40, 100),
            1.0,
        )

        for index in range(sample_count):
            manager._record_tracklet_sample(
                first["gid"],
                "A",
                7,
                quality_detection(
                    7,
                    basis[index],
                    camera_generation=1,
                ),
                float(index + 1),
            )

        second = manager._new_identity(
            "B",
            8,
            basis[-1],
            None,
            (40, 100),
            5.0,
        )
        manager._commit_assignment(
            second["gid"],
            "A",
            7,
            basis[-1],
            None,
            (40, 100),
            6.0,
            1.0,
            "test-reassignment",
        )

        accepted, reason = manager._record_tracklet_sample(
            second["gid"],
            "A",
            7,
            quality_detection(
                7,
                basis[-1],
                camera_generation=1,
            ),
            6.0,
        )

        self.assertEqual(
            second["gid"],
            manager.local_to_global[("A", 7)]["gid"],
        )
        self.assertFalse(accepted)
        self.assertEqual("tracklet_not_mature", reason)
        self.assertEqual(
            main.IDENTITY_PROVISIONAL,
            manager.identities[second["gid"]]["state"],
        )
        self.assertEqual(
            1,
            manager.identities[second["gid"]]["gallery_diagnostics"][
                "tracklet_sample_count"
            ],
        )
        self.assertEqual([], manager.identities[second["gid"]]["gallery"])

    def test_failed_different_gid_commit_restores_all_previous_evidence(self):
        manager = main.GlobalIdentityManager()
        previous_vector = embedding(1, 0, 0)
        target_vector = embedding(0, 1, 0)
        previous = manager._new_identity(
            "A",
            7,
            previous_vector,
            None,
            (40, 100),
            1.0,
        )
        target = manager._new_identity(
            "B",
            8,
            target_vector,
            None,
            (40, 100),
            2.0,
        )
        self.assertNotEqual(previous["gid"], target["gid"])

        manager._record_tracklet_sample(
            previous["gid"],
            "A",
            7,
            quality_detection(7, previous_vector),
            1.5,
        )
        local_key = ("A", 7)
        manager.occlusion_hold[local_key] = {
            "gid": previous["gid"],
            "until_ts": 10.0,
            "score": 0.99,
        }

        local_mappings_before = copy.deepcopy(manager.local_to_global)
        tracklet_before = copy.deepcopy(manager.tracklets[local_key])
        holds_before = copy.deepcopy(manager.occlusion_hold)
        target_before = copy.deepcopy(manager.identities[target["gid"]])
        recent_same_before = copy.deepcopy(manager.recent_same_cam)
        recent_cross_before = copy.deepcopy(manager.recent_cross_cam)

        failing_store = Mock()
        failing_store.save_identity.side_effect = RuntimeError(
            "synthetic assignment persistence failure"
        )
        manager.identity_store = failing_store

        with self.assertRaisesRegex(
            RuntimeError,
            "synthetic assignment persistence failure",
        ):
            manager._commit_assignment(
                target["gid"],
                "A",
                7,
                target_vector,
                None,
                (40, 100),
                3.0,
                0.91,
                "cross-camera",
            )

        self.assertEqual(local_mappings_before, manager.local_to_global)
        self.assertEqual(holds_before, manager.occlusion_hold)
        self.assertEqual(set(manager.tracklets), {local_key})
        tracklet_after = manager.tracklets[local_key]
        self.assertEqual(
            {
                key: value
                for key, value in tracklet_before.items()
                if key != "samples"
            },
            {
                key: value
                for key, value in tracklet_after.items()
                if key != "samples"
            },
        )
        self.assertEqual(
            len(tracklet_before["samples"]),
            len(tracklet_after["samples"]),
        )
        for expected, actual in zip(
            tracklet_before["samples"],
            tracklet_after["samples"],
        ):
            np.testing.assert_allclose(expected["emb"], actual["emb"])
            self.assertEqual(
                {key: value for key, value in expected.items() if key != "emb"},
                {key: value for key, value in actual.items() if key != "emb"},
            )

        target_after = manager.identities[target["gid"]]
        self.assertEqual(
            {
                key: value
                for key, value in target_before.items()
                if key not in {"embedding", "gallery"}
            },
            {
                key: value
                for key, value in target_after.items()
                if key not in {"embedding", "gallery"}
            },
        )
        np.testing.assert_allclose(
            target_before["embedding"],
            target_after["embedding"],
        )
        self.assertEqual(len(target_before["gallery"]), len(target_after["gallery"]))
        for expected, actual in zip(
            target_before["gallery"],
            target_after["gallery"],
        ):
            np.testing.assert_allclose(expected, actual)

        for expected_cache, actual_cache in (
            (recent_same_before, manager.recent_same_cam),
            (recent_cross_before, manager.recent_cross_cam),
        ):
            self.assertEqual(len(expected_cache), len(actual_cache))
            for expected, actual in zip(expected_cache, actual_cache):
                np.testing.assert_allclose(
                    expected["embedding"],
                    actual["embedding"],
                )
                self.assertEqual(
                    {
                        key: value
                        for key, value in expected.items()
                        if key != "embedding"
                    },
                    {
                        key: value
                        for key, value in actual.items()
                        if key != "embedding"
                    },
                )
        failing_store.save_identity.assert_called()

    def test_provisional_identity_cannot_cross_camera_in_legacy_assign_batch(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1, 0, 0)
        first = manager.assign_batch(
            "A",
            [detection(7, person)],
            event_time=1.0,
        )[0]

        self.assertEqual(
            main.IDENTITY_PROVISIONAL,
            manager.identities[first["gid"]]["state"],
        )
        second = manager.assign_batch(
            "B",
            [detection(8, person)],
            event_time=1.1,
        )[0]

        self.assertNotEqual(first["gid"], second["gid"])
        self.assertEqual("new", second["source"])

    def test_legacy_acceptance_failure_can_use_recent_same_camera_cache(self):
        manager = main.GlobalIdentityManager()
        candidate = embedding(
            0.44,
            np.sqrt(1.0 - (0.44 ** 2)),
        )
        query = embedding(1.0, 0.0)
        first = manager.assign_batch(
            "A",
            [quality_detection(7, candidate)],
            event_time=10.0,
        )[0]
        gid = first["gid"]
        identity = manager.identities[gid]
        identity.update({
            "state": main.IDENTITY_ACTIVE,
            "state_updated_at": 10.0,
            "state_reason": "test_mature_identity",
            "gallery": [candidate.copy()],
            "embedding": candidate.copy(),
            "gallery_mature": True,
        })
        manager.local_to_global.clear()
        manager.tracklets.clear()

        result = manager.assign_batch(
            "A",
            [quality_detection(8, query)],
            event_time=10.1,
        )[0]

        self.assertEqual(gid, result["gid"])
        self.assertEqual("same-cam-cache", result["source"])

    def test_legacy_ambiguous_acceptance_failures_cannot_use_recent_cache(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1.0, 0.0)
        manager.identities = {
            gid: {
                "state": main.IDENTITY_ACTIVE,
                "state_updated_at": 100.0,
                "state_reason": "test_mature_identity",
                "last_cam": "A",
                "last_seen": 100.0,
                "embedding": person.copy(),
                "gallery": [person.copy()],
                "gallery_mature": True,
                "box_wh": (40, 100),
                "last_map_pos": None,
            }
            for gid in (1, 2)
        }
        manager.next_global_id = 3
        manager.recent_same_cam = [
            {
                "gid": gid,
                "cam_name": "A",
                "embedding": person.copy(),
                "map_pos": None,
                "box_wh": (40, 100),
                "ts": 100.0,
            }
            for gid in (1, 2)
        ]

        with (
            patch.object(
                manager,
                "_pair_score",
                side_effect=lambda gid, *_args: {
                    "gid": gid,
                    "score": 0.44 if gid == 1 else 0.43,
                    "appearance": 0.44 if gid == 1 else 0.43,
                    "cross_camera": False,
                },
            ),
            patch.object(
                manager,
                "_accept_match",
                return_value=False,
            ) as accept_match,
        ):
            result = manager.assign_batch(
                "A",
                [quality_detection(8, person)],
                event_time=100.1,
            )[0]

        accept_match.assert_not_called()
        self.assertEqual(3, result["gid"])
        self.assertEqual("new", result["source"])

    def test_legacy_hungarian_omitted_ambiguous_row_cannot_use_cache(self):
        manager = main.GlobalIdentityManager()
        person = embedding(1.0, 0.0)
        manager.identities = {
            gid: {
                "state": main.IDENTITY_ACTIVE,
                "state_updated_at": 100.0,
                "state_reason": "test_mature_identity",
                "last_cam": "A",
                "last_seen": 100.0,
                "embedding": person.copy(),
                "gallery": [person.copy()],
                "gallery_mature": True,
                "box_wh": (40, 100),
                "last_map_pos": None,
            }
            for gid in (1, 2)
        }
        manager.next_global_id = 3
        manager.recent_same_cam = [
            {
                "gid": gid,
                "cam_name": "A",
                "embedding": person.copy(),
                "map_pos": None,
                "box_wh": (40, 100),
                "ts": 100.0,
            }
            for gid in (1, 2)
        ]
        scores = {
            (11, 1): (0.90, True),
            (11, 2): (0.10, True),
            (12, 1): (0.10, True),
            (12, 2): (0.90, True),
            (13, 1): (0.44, False),
            (13, 2): (0.43, False),
        }

        def pair_score(gid, _identity, _camera, row, *_args):
            score, cross_camera = scores[(row["tid"], gid)]
            return {
                "gid": gid,
                "score": score,
                "appearance": score,
                "cross_camera": cross_camera,
            }

        with (
            patch.object(manager, "_pair_score", side_effect=pair_score),
            patch.object(manager, "_accept_match", return_value=False),
        ):
            results = manager.assign_batch(
                "A",
                [
                    quality_detection(11, person),
                    quality_detection(12, person),
                    quality_detection(13, person),
                ],
                event_time=100.1,
            )

        self.assertNotIn(results[2]["gid"], {1, 2})
        self.assertEqual("new", results[2]["source"])

    def test_local_id_is_camera_scoped_and_mature_reid_is_global(self):
        manager = main.GlobalIdentityManager()
        person_a = embedding(1, 0, 0)
        person_b = embedding(0, 1, 0)

        gid_a = manager.assign_batch(
            "A",
            [detection(7, person_a)]
        )[0]["gid"]
        identity_a = manager.identities[gid_a]
        identity_a.update({
            "state": main.IDENTITY_ACTIVE,
            "state_updated_at": identity_a["last_seen"],
            "state_reason": "test_mature_identity",
            "gallery": [person_a.copy()],
            "embedding": person_a.copy(),
            "gallery_mature": True,
        })
        gid_b = manager.assign_batch(
            "B",
            [detection(7, person_b)]
        )[0]["gid"]
        transferred = manager.assign_batch(
            "C",
            [detection(99, person_a)]
        )[0]

        self.assertNotEqual(gid_a, gid_b)
        self.assertEqual(
            transferred["gid"],
            gid_a
        )
        self.assertEqual(
            transferred["source"],
            "cross-camera"
        )

    def test_local_id_switch_does_not_swap_global_ids(self):
        manager = main.GlobalIdentityManager()
        person_a = embedding(1, 0, 0)
        person_b = embedding(0, 1, 0)
        first = manager.assign_batch(
            "A",
            [
                detection(10, person_a, 0),
                detection(20, person_b, 100)
            ]
        )
        expected = [
            item["gid"]
            for item in first
        ]
        switched = manager.assign_batch(
            "A",
            [
                detection(20, person_a, 5),
                detection(10, person_b, 95)
            ]
        )

        self.assertEqual(
            [item["gid"] for item in switched],
            expected
        )


if __name__ == "__main__":
    unittest.main()
