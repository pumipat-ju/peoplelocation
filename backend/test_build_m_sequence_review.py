import unittest

from backend.build_m_sequence_review import (
    final_identity_id,
    select_proposal_rows,
    validate_cross_camera_identity_policy,
)


class BuildMSequenceReviewTests(unittest.TestCase):
    def test_selects_interval_and_replaces_id(self):
        rows = [
            {"frame": 9, "local_track_id": 4, "confidence": 0.9},
            {"frame": 10, "local_track_id": 4, "confidence": 0.8},
            {"frame": 11, "local_track_id": 5, "confidence": 0.9},
        ]
        selected = select_proposal_rows(
            rows, [{"track": 4, "first": 10, "last": 20}], person_id=2
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["frame"], 10)
        self.assertEqual(selected[0]["local_track_id"], 2)

    def test_duplicate_frame_keeps_higher_confidence(self):
        rows = [
            {"frame": 10, "local_track_id": 4, "confidence": 0.7},
            {"frame": 10, "local_track_id": 5, "confidence": 0.9},
        ]
        selected = select_proposal_rows(
            rows,
            [
                {"track": 4, "first": 1, "last": 20},
                {"track": 5, "first": 1, "last": 20},
            ],
            person_id=3,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["confidence"], 0.9)

    def test_rejected_match_gets_camera_local_ids(self):
        self.assertEqual(
            final_identity_id(1, "cam1", "REJECTED_CROSS_CAMERA_MATCH"), 1001
        )
        self.assertEqual(
            final_identity_id(1, "cam2", "REJECTED_CROSS_CAMERA_MATCH"), 2001
        )
        self.assertEqual(final_identity_id(2, "cam1", "CONFIRMED"), 2)

    def test_only_confirmed_ids_may_be_shared(self):
        rows = {
            "cam1": [{"local_track_id": 2}, {"local_track_id": 1001}],
            "cam2": [{"local_track_id": 2}, {"local_track_id": 2001}],
        }
        identities, shared = validate_cross_camera_identity_policy(rows, [2])
        self.assertEqual(shared, {2})
        self.assertEqual(identities["cam1"], {2, 1001})


if __name__ == "__main__":
    unittest.main()
