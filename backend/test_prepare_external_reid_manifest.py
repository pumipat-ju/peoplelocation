import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.prepare_external_reid_manifest import (
    audit_split,
    build_external_manifest,
    namespaced_identity_key,
    parse_market1501_filename,
)


class PrepareExternalReidManifestTests(unittest.TestCase):
    def test_parses_market1501_metadata(self):
        parsed = parse_market1501_filename("0002_c1s1_000451_03.jpg")
        self.assertEqual(2, parsed["original_person_id"])
        self.assertEqual("0002", parsed["original_person_id_text"])
        self.assertEqual(1, parsed["camera_id"])
        self.assertEqual(1, parsed["sequence_id"])
        self.assertEqual(451, parsed["frame_index"])
        self.assertEqual(3, parsed["sample_index"])
        self.assertEqual(-1, parse_market1501_filename(
            "-1_c2s3_000401_04.jpg"
        )["original_person_id"])
        self.assertIsNone(parse_market1501_filename("0002_camera1.jpg"))

    def test_namespace_never_uses_local_sequence_namespaces(self):
        self.assertEqual("external_reid:0002", namespaced_identity_key(2))
        self.assertNotEqual("legacy_sequence:2", namespaced_identity_key(2))
        self.assertNotEqual("m_sequence:2", namespaced_identity_key(2))

    def test_build_is_train_only_and_marks_junk(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = root / "bounding_box_train"
            test = root / "bounding_box_test"
            train.mkdir()
            test.mkdir()
            image = np.zeros((16, 8, 3), dtype=np.uint8)
            cv2.imwrite(str(train / "0002_c1s1_000451_03.jpg"), image)
            cv2.imwrite(str(train / "-1_c2s1_000401_03.jpg"), image)
            cv2.imwrite(str(test / "0002_c3s2_000501_01.jpg"), image)
            output = root / "reid_dataset" / "external.jsonl"
            summary_path = root / "reid_dataset" / "summary.json"

            summary = build_external_manifest(
                train, test, output, summary_path, root
            )
            records = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(2, len(records))
            self.assertTrue(all(row["source_split"] == "bounding_box_train"
                                for row in records))
            junk = next(row for row in records if row["original_person_id"] == -1)
            self.assertTrue(junk["ignored_junk"])
            self.assertFalse(junk["eligible_for_training"])
            self.assertEqual(1, summary["train"]["usable_train_images"])
            self.assertEqual(1, summary["train"]["ignored_junk_images"])
            self.assertTrue(summary["test"]["excluded_from_training"])
            self.assertEqual(0, summary["test"]["manifest_records"])
            self.assertFalse(summary["policy"]["recropping_performed"])
            self.assertFalse(summary["policy"]["fine_tuning_performed"])

    def test_audit_detects_unreadable_and_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split = root / "bounding_box_train"
            split.mkdir()
            image = np.zeros((16, 8, 3), dtype=np.uint8)
            cv2.imwrite(str(split / "0002_c1s1_000451_03.jpg"), image)
            cv2.imwrite(str(split / "0002_c1s1_000451_03.jpeg"), image)
            (split / "0003_c1s1_000452_01.jpg").write_bytes(b"not an image")

            _, audit = audit_split(split, root)

            self.assertEqual(1, audit["duplicate_sample_id_count"])
            self.assertEqual(1, audit["unreadable_count"])
            self.assertEqual(0, audit["duplicate_path_count"])


if __name__ == "__main__":
    unittest.main()
