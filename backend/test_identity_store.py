import os
import tempfile
import unittest

import numpy as np

from backend.identity_store import IdentityStore


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

    def test_failed_serialization_does_not_create_half_committed_snapshot(self):
        store = IdentityStore(self.path)
        self.stores.append(store)
        with self.assertRaises(TypeError):
            store.save_identity(1, {"state": "ACTIVE", "bad": object()}, "assignment")
        self.assertEqual({}, store.load_identities())


if __name__ == "__main__":
    unittest.main()
