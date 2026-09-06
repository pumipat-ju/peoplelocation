import tempfile
import unittest
from pathlib import Path

from backend.build_master_reid_manifest import (
    collision_audit,
    discover_sequences,
    normalize_record,
)


class BuildMasterReIDManifestTests(unittest.TestCase):
    def test_discovers_legacy_and_named_sequence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "labeled_data"
            for camera in (root / "cam1", root / "m_sequence" / "cam2"):
                (camera / "images").mkdir(parents=True)
                (camera / "gt1.txt").write_text("", encoding="utf-8")
            sequences = discover_sequences(root)
            self.assertEqual(
                ["legacy_sequence", "m_sequence"],
                [item["sequence"] for item in sequences],
            )

    def test_normalized_identity_is_sequence_scoped_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            dataset = project / "labeled_data"
            image = dataset / "cam1" / "images" / "000001.jpg"
            annotation = dataset / "cam1" / "gt1.txt"
            image.parent.mkdir(parents=True)
            annotation.write_text("", encoding="utf-8")
            record = {
                "source_image": Path("labeled_data/cam1/images/000001.jpg"),
                "annotation_file": Path("labeled_data/cam1/gt1.txt"),
                "ground_truth_person_id": "2",
                "camera": "cam1",
                "frame_index": 1,
            }
            spec = {"root": dataset, "sequence": "legacy_sequence"}
            first = normalize_record(record, spec, project)
            second = normalize_record(record, spec, project)
            self.assertEqual(first, second)
            self.assertEqual("2", first["original_gt_person_id"])
            self.assertEqual("legacy_sequence:2", first["dataset_identity_key"])

    def test_same_original_id_in_two_sequences_is_not_namespace_collision(self):
        summaries = [
            {"sequence": "old", "original_identities": ["2"]},
            {"sequence": "new", "original_identities": ["2"]},
        ]
        records = [
            {"sequence": "old", "original_gt_person_id": "2",
             "dataset_identity_key": "old:2"},
            {"sequence": "new", "original_gt_person_id": "2",
             "dataset_identity_key": "new:2"},
        ]
        audit = collision_audit(summaries, records)
        self.assertEqual(["new", "old"], audit["unscoped_original_id_collisions"]["2"])
        self.assertEqual({}, audit["namespace_identity_key_collisions"])
        self.assertTrue(audit["unscoped_collisions_are_isolated_by_namespace"])


if __name__ == "__main__":
    unittest.main()
