import unittest

import numpy as np

from backend import main


class GlobalMultiCameraBatchMatchingTests(unittest.TestCase):
    def setUp(self):
        self.manager = main.GlobalIdentityManager()
        self.manager.identities = {1: {}, 2: {}}
        self.manager.cleanup = lambda: None
        self.manager._can_match = lambda *args: True
        self.manager._hard_gate_reason = lambda *args: None
        self.manager._accept_match = lambda *args: True
        self.manager._record_tracklet_sample = lambda *args: (False, "test")
        self.manager._commit_assignment = lambda gid, *args: {
            "gid": gid, "score": args[-2], "source": args[-1]
        }
        self.manager._new_identity = lambda *args: self.fail("unexpected new identity")

    @staticmethod
    def _det(track_id):
        return {"tid": track_id, "emb": np.array([1.0, 0.0], dtype=np.float32)}

    def test_global_hungarian_prevents_per_camera_greedy_identity_conflict(self):
        # cam1 strongly prefers GID 1 but can use GID 2; cam2 can only use
        # GID 1.  Per-camera assignment of cam1 first would consume GID 1.
        scores = {
            ("cam1", 11, 1): 0.90,
            ("cam1", 11, 2): 0.80,
            ("cam2", 22, 1): 0.85,
            ("cam2", 22, 2): 0.10,
        }

        def pair_score(gid, _identity, cam_name, detection, *_args):
            return {"score": scores[(cam_name, detection["tid"], gid)],
                    "cross_camera": True}

        self.manager._pair_score = pair_score
        result = self.manager.assign_global_batch({
            "cam1": [self._det(11)], "cam2": [self._det(22)],
        }, event_time=100.0)

        self.assertEqual(2, result["cam1"][0]["gid"])
        self.assertEqual(1, result["cam2"][0]["gid"])
        self.assertEqual({1, 2}, set(self.manager.identities))

    def test_near_equal_candidates_are_deferred_instead_of_forced(self):
        self.manager._pair_score = lambda gid, *_args: {
            "score": 0.80 if gid == 1 else 0.76,
            "cross_camera": True,
        }
        self.manager._new_identity = lambda *_args: {
            "gid": 99, "score": 1.0, "source": "new",
        }

        result = self.manager.assign_global_batch(
            {"cam1": [self._det(11)]}, event_time=100.0
        )

        self.assertEqual(99, result["cam1"][0]["gid"])
        self.assertEqual({1, 2}, set(self.manager.identities))


if __name__ == "__main__":
    unittest.main()
