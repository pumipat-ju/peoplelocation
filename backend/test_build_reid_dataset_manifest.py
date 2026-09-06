import unittest
from pathlib import Path

from backend.build_reid_dataset_manifest import (
    audit_review_policy,
    assert_unique_sample_ids,
    convert_mot_bbox,
    deterministic_sample_id,
    identity_audit_rows,
    make_manifest_record,
)


class BuildReIDDatasetManifestTests(unittest.TestCase):
    def test_mot_xywh_conversion_and_valid_bounds(self):
        bbox, reason = convert_mot_bbox((100, 200, 3), (10, 20, 30, 40))

        self.assertIsNone(reason)
        self.assertEqual([10, 20, 30, 40], bbox["bbox_xywh"])
        self.assertEqual([10, 20, 40, 60], bbox["bbox_xyxy"])
        self.assertEqual(1200, bbox["bbox_area"])
        self.assertEqual(0.75, bbox["aspect_ratio_width_over_height"])

    def test_out_of_bounds_and_invalid_bbox_are_rejected(self):
        bbox, reason = convert_mot_bbox((100, 200, 3), (-1, 20, 30, 40))
        self.assertIsNone(bbox)
        self.assertEqual("bbox_out_of_frame_bounds", reason)

        bbox, reason = convert_mot_bbox((100, 200, 3), (10, 20, 0, 40))
        self.assertIsNone(bbox)
        self.assertEqual("non_positive_bbox_size", reason)

    def test_sample_id_and_manifest_record_are_deterministic(self):
        first_id = deterministic_sample_id("sequence", "cam1", 7, "person-a")
        second_id = deterministic_sample_id("sequence", "cam1", 7, "person-a")
        self.assertEqual(first_id, second_id)

        arguments = {
            "sequence": "sequence",
            "camera": "cam1",
            "annotation_file": Path("labeled_data/cam1/gt1.txt"),
            "annotation_row": 1,
            "source_image": Path("labeled_data/cam1/images/000007.jpg"),
            "frame_index": 7,
            "person_id": "person-a",
            "frame_shape": (100, 200, 3),
            "bbox_xywh": (10, 20, 30, 40),
            "mot_extra_fields": ["1", "-1", "-1", "-1"],
        }
        first, first_reason = make_manifest_record(**arguments)
        second, second_reason = make_manifest_record(**arguments)
        self.assertIsNone(first_reason)
        self.assertIsNone(second_reason)
        self.assertEqual(first, second)

    def test_duplicate_sample_ids_are_detected(self):
        records = [{"sample_id": "same"}, {"sample_id": "same"}]
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            assert_unique_sample_ids(records)

    def test_suspected_label_identity_is_marked_without_changing_identity(self):
        rows = identity_audit_rows([
            {
                "sample_id": "sample",
                "ground_truth_person_id": "2",
                "camera": "cam1",
                "frame_index": 10,
            }
        ], suspected_label_ids={"2"})

        self.assertEqual("2", rows[0]["ground_truth_person_id"])
        self.assertTrue(rows[0]["suspected_label_issue"])
        self.assertIn("requires manual correction", rows[0]["audit_note"])

    def test_review_policy_allows_only_confirmed_shared_ids(self):
        decisions = [
            {"proposed_person_id": "1", "status": "REJECTED_CROSS_CAMERA_MATCH"},
            {"proposed_person_id": "2", "status": "CONFIRMED"},
            {"proposed_person_id": "3", "status": "CONFIRMED"},
        ]
        audit = audit_review_policy(decisions, ["2", "3"])
        self.assertEqual(["2", "3"], audit["confirmed_cross_camera_ids"])
        self.assertEqual(["1"], audit["rejected_cross_camera_ids"])
        self.assertEqual([], audit["policy_violations"])

    def test_review_policy_rejects_unconfirmed_shared_id(self):
        decisions = [
            {"proposed_person_id": "1", "status": "REJECTED_CROSS_CAMERA_MATCH"}
        ]
        audit = audit_review_policy(decisions, ["1"])
        self.assertIn("rejected identity 1", audit["policy_violations"][0])


if __name__ == "__main__":
    unittest.main()
