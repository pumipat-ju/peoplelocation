import unittest
from unittest.mock import patch

import numpy as np

from backend import main


class IdentityStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.manager = main.GlobalIdentityManager()
        self.det = {
            "emb": np.array([1.0, 0.0], dtype=np.float32),
            "detector_confidence": 0.95, "crop_size": (80, 160),
            "blur_variance": 100.0, "overlap": False, "border_clip_ratio": 0.0,
        }

    def test_new_identity_becomes_active_only_after_mature_tracklet(self):
        result = self.manager._new_identity("cam1", 7, self.det["emb"], None, (80, 160), 1.0)
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

    def test_active_becomes_dormant_then_expired_without_reuse(self):
        result = self.manager._new_identity("cam1", 7, self.det["emb"], None, (80, 160), 1.0)
        identity = self.manager.identities[result["gid"]]
        identity["state"] = main.IDENTITY_ACTIVE
        with patch.object(main.time, "time", return_value=1.0 + main.REID_MAX_IDLE_SEC + 1.0):
            self.manager.cleanup()
        self.assertEqual(main.IDENTITY_DORMANT, identity["state"])
        with patch.object(main.time, "time", return_value=1.0 + main.REID_MAX_IDLE_SEC + main.IDENTITY_DORMANT_TTL_SEC + 1.0):
            self.manager.cleanup()
        self.assertEqual(main.IDENTITY_EXPIRED, identity["state"])
        self.assertFalse(self.manager._can_match(identity, "cam2", 999.0, None, (80, 160)))


if __name__ == "__main__":
    unittest.main()
