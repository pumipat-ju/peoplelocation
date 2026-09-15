"""Audit a cropped Market-1501-style dataset and build a train-only manifest.

This is an offline dataset preparation utility. It does not crop images, train a
model, or integrate the external dataset with the production runtime.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

import cv2


DATASET_SOURCE = "external_reid"
MANIFEST_SCHEMA_VERSION = "external_reid_train_manifest_v1"
MARKET1501_FILENAME = re.compile(
    r"^(?P<pid>-?\d+)_c(?P<camera>\d+)s(?P<sequence>\d+)_"
    r"(?P<frame>\d+)_(?P<index>\d+)\.(?P<extension>jpe?g)$",
    re.IGNORECASE,
)


def parse_market1501_filename(filename):
    """Return Market-1501 filename metadata, or None for a non-matching name."""
    match = MARKET1501_FILENAME.fullmatch(Path(filename).name)
    if match is None:
        return None
    values = match.groupdict()
    return {
        "original_person_id": int(values["pid"]),
        "original_person_id_text": values["pid"],
        "camera_id": int(values["camera"]),
        "sequence_id": int(values["sequence"]),
        "frame_index": int(values["frame"]),
        "sample_index": int(values["index"]),
    }


def namespaced_identity_key(person_id):
    if person_id == -1:
        return f"{DATASET_SOURCE}:-1"
    return f"{DATASET_SOURCE}:{person_id:04d}"


def deterministic_sample_id(metadata):
    canonical = ":".join([
        DATASET_SOURCE,
        str(metadata["original_person_id"]),
        str(metadata["camera_id"]),
        str(metadata["sequence_id"]),
        str(metadata["frame_index"]),
        str(metadata["sample_index"]),
    ])
    return f"external_reid_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def image_is_readable(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return image is not None and image.size > 0


def project_relative(path, project_root):
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def audit_split(split_path, project_root, include_records=False):
    files = sorted(
        (path for path in split_path.iterdir() if path.is_file()),
        key=lambda path: path.name,
    )
    records = []
    invalid_filenames = []
    unreadable_files = []
    normalized_paths = Counter()
    sample_ids = Counter()
    parsed_metadata = []

    for path in files:
        relative_path = project_relative(path, project_root)
        normalized_paths[relative_path.casefold()] += 1
        readable = image_is_readable(path)
        if not readable:
            unreadable_files.append(relative_path)

        metadata = parse_market1501_filename(path.name)
        if metadata is None:
            invalid_filenames.append(relative_path)
            continue

        parsed_metadata.append(metadata)
        sample_id = deterministic_sample_id(metadata)
        sample_ids[sample_id] += 1
        if include_records:
            is_junk = metadata["original_person_id"] == -1
            records.append({
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "dataset_source": DATASET_SOURCE,
                "source_split": split_path.name,
                "image_path": relative_path,
                "sample_id": sample_id,
                "original_person_id": metadata["original_person_id"],
                "original_person_id_text": metadata["original_person_id_text"],
                "identity_key": namespaced_identity_key(
                    metadata["original_person_id"]
                ),
                "camera_id": metadata["camera_id"],
                "sequence_id": metadata["sequence_id"],
                "frame_index": metadata["frame_index"],
                "sample_index": metadata["sample_index"],
                "ignored_junk": is_junk,
                "readable": readable,
                "eligible_for_training": readable and not is_junk,
                "already_cropped_person_image": True,
            })

    duplicate_paths = sorted(
        path for path, count in normalized_paths.items() if count > 1
    )
    duplicate_sample_ids = sorted(
        sample_id for sample_id, count in sample_ids.items() if count > 1
    )
    junk_count = sum(
        item["original_person_id"] == -1 for item in parsed_metadata
    )
    return records, {
        "split": split_path.name,
        "total_files": len(files),
        "market1501_filename_matches": len(parsed_metadata),
        "invalid_filename_count": len(invalid_filenames),
        "invalid_filenames": invalid_filenames,
        "unreadable_count": len(unreadable_files),
        "unreadable_files": unreadable_files,
        "duplicate_path_count": len(duplicate_paths),
        "duplicate_paths": duplicate_paths,
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids,
        "parsed_junk_count": junk_count,
        "camera_ids": sorted({item["camera_id"] for item in parsed_metadata}),
        "camera_image_counts": dict(sorted(Counter(
            item["camera_id"] for item in parsed_metadata
        ).items())),
    }


def identity_statistics(records):
    grouped = defaultdict(list)
    for record in records:
        if record["eligible_for_training"]:
            grouped[record["identity_key"]].append(record)
    counts = [len(rows) for rows in grouped.values()]
    cross_camera = sum(
        len({row["camera_id"] for row in rows}) > 1
        for rows in grouped.values()
    )
    return {
        "unique_identities": len(grouped),
        "identities_in_multiple_cameras": cross_camera,
        "images_per_identity": {
            "minimum": min(counts) if counts else 0,
            "maximum": max(counts) if counts else 0,
            "mean": (sum(counts) / len(counts)) if counts else 0.0,
            "median": statistics.median(counts) if counts else 0.0,
        },
        "counts_by_identity": {
            identity: len(grouped[identity]) for identity in sorted(grouped)
        },
    }


def build_external_manifest(
    train_path, test_path, output_path, summary_path, project_root, overwrite=False
):
    for split_path in (train_path, test_path):
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing dataset split: {split_path}")
    for target in (output_path, summary_path):
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    train_records, train_audit = audit_split(
        train_path, project_root, include_records=True
    )
    _, test_audit = audit_split(test_path, project_root, include_records=False)
    identity_audit = identity_statistics(train_records)
    usable_records = [
        record for record in train_records if record["eligible_for_training"]
    ]
    junk_count = sum(record["ignored_junk"] for record in train_records)

    manifest_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in train_records
    )
    summary = {
        "dataset_detected_as": "Market-1501-style pre-cropped person Re-ID dataset",
        "dataset_source": DATASET_SOURCE,
        "identity_namespace": "external_reid:<zero-padded-original-person-id>",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_path": project_relative(output_path, project_root),
        "manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest(),
        "train": {
            **train_audit,
            "manifest_records": len(train_records),
            "usable_train_images": len(usable_records),
            "ignored_junk_images": junk_count,
            **identity_audit,
        },
        "test": {
            **test_audit,
            "excluded_from_training": True,
            "manifest_records": 0,
        },
        "policy": {
            "input_images_are_already_cropped": True,
            "recropping_performed": False,
            "fine_tuning_performed": False,
            "production_modified": False,
            "camera_video_calibration_modified": False,
            "junk_person_id": -1,
            "junk_is_training_eligible": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("bounding_box_train"))
    parser.add_argument("--test", type=Path, default=Path("bounding_box_test"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reid_dataset/external_reid_train_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reid_dataset/external_reid_train_manifest_v1_summary.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build_external_manifest(
        arguments.train,
        arguments.test,
        arguments.output,
        arguments.summary,
        arguments.project_root,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
