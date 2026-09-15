import tempfile
import unittest
from pathlib import Path

from backend.prepare_finetuning_dataset import (
    choose_peoplelocation_splits,
    leakage_audit,
    normalize_market1501_record,
    parse_msmt17_train_list,
)


def people_record(identity, camera, sample):
    return {
        "dataset_identity_key": identity,
        "camera": camera,
        "sample_id": sample,
    }


class PrepareFinetuningDatasetTests(unittest.TestCase):
    def test_split_is_deterministic_identity_disjoint_and_cross_camera(self):
        records = []
        for index in range(5):
            identity = f"sequence:{index}"
            records.extend([
                people_record(identity, "cam1", f"{index}-a"),
                people_record(identity, "cam2", f"{index}-b"),
            ])
        records.extend([
            people_record("sequence:5", "cam1", "5-a"),
            people_record("sequence:6", "cam2", "6-a"),
        ])

        first = choose_peoplelocation_splits(records)
        second = choose_peoplelocation_splits(list(reversed(records)))

        self.assertEqual(first, second)
        self.assertEqual({"train": 3, "val": 2, "test": 2}, {
            split: len(identities) for split, identities in first.items()
        })
        self.assertFalse(set(first["train"]) & set(first["val"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["val"]) & set(first["test"]))
        for split in ("val", "test"):
            for identity in first[split]:
                self.assertEqual(
                    {"cam1", "cam2"},
                    {row["camera"] for row in records
                     if row["dataset_identity_key"] == identity},
                )

    def test_external_record_must_be_eligible_train_source(self):
        base = {
            "source_split": "bounding_box_train",
            "eligible_for_training": True,
            "ignored_junk": False,
            "sample_id": "external-1",
            "identity_key": "external_reid:0002",
            "original_person_id": 2,
            "image_path": "bounding_box_train/example.jpg",
            "camera_id": 1,
            "sequence_id": 1,
            "frame_index": 1,
            "sample_index": 1,
        }
        normalized = normalize_market1501_record(base)
        self.assertEqual("train", normalized["split"])
        self.assertEqual("market1501:0002", normalized["identity_key"])
        self.assertEqual("pre_cropped_no_recrop", normalized["crop_policy"])
        with self.assertRaises(ValueError):
            normalize_market1501_record({
                **base, "source_split": "bounding_box_test"
            })
        with self.assertRaises(ValueError):
            normalize_market1501_record({**base, "ignored_junk": True})

    def test_msmt17_reads_only_list_train_and_validates_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            root = project / "datasets" / "msmt17" / "MSMT17_V1"
            train = root / "train" / "0002"
            train.mkdir(parents=True)
            image = train / "0002_001_03_0303morning_0001_0.jpg"
            image.write_bytes(b"source existence is sufficient for list parsing")
            list_path = root / "list_train.txt"
            list_path.write_text(
                "0002/0002_001_03_0303morning_0001_0.jpg 2\n",
                encoding="utf-8",
            )
            (root / "list_val.txt").write_text(
                "0002/forbidden.jpg 2\n", encoding="utf-8"
            )

            records, audit = parse_msmt17_train_list(
                list_path, root / "train", project
            )

            self.assertEqual(1, len(records))
            self.assertEqual("msmt17:0002", records[0]["identity_key"])
            self.assertTrue(records[0]["source_list"].endswith("/list_train.txt"))
            self.assertEqual(0, audit["invalid_label_count"])
            self.assertNotIn("forbidden.jpg", records[0]["image_path"])

    def test_leakage_and_external_test_contamination_are_detected(self):
        common = {
            "manifest_schema_version": "test",
            "dataset_version": "test",
            "usage": "training",
            "image_path": "image.jpg",
            "camera_id": "cam1",
        }
        splits = {
            "train": [{
                **common, "dataset_source": "peoplelocation", "split": "train",
                "identity_key": "peoplelocation:sequence:1", "sample_id": "a",
            }, {
                **common, "dataset_source": "market1501", "split": "train",
                "identity_key": "market1501:0002", "sample_id": "b",
                "usage": "supplemental_training_only",
                "source_split": "bounding_box_train",
                "image_path": "bounding_box_test/b.jpg",
            }],
            "val": [{
                **common, "dataset_source": "peoplelocation", "split": "val",
                "identity_key": "peoplelocation:sequence:1", "sample_id": "c",
            }],
            "test": [],
        }

        audit = leakage_audit(splits)

        self.assertEqual(1, audit["identity_leakage_count"])
        self.assertEqual(1, audit["market1501_test_contamination_count"])


if __name__ == "__main__":
    unittest.main()
