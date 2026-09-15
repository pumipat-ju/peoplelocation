"""Prepare identity-disjoint OSNet fine-tuning split manifests offline."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics

import cv2


DATASET_VERSION = "peoplelocation-reid-finetune-v1"
SCHEMA_VERSION = "peoplelocation-reid-finetune-manifest-v1"
SPLIT_SEED = "peoplelocation-reid-split-v1"
PEOPLE_SOURCE = "peoplelocation"
MARKET_SOURCE = "market1501"
MSMT_SOURCE = "msmt17"
PRESERVED_PEOPLE_SPLITS = {
    "train": (
        "legacy_sequence:3", "m_sequence:1001", "m_sequence:2001",
    ),
    "val": ("legacy_sequence:1", "m_sequence:3"),
    "test": ("legacy_sequence:2", "m_sequence:2"),
}
MSMT_FILENAME = re.compile(
    r"^(?P<pid>\d{4})_(?P<index>\d+)_(?P<camera>\d{2})_.+\.jpg$",
    re.IGNORECASE,
)


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return records


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_people_identities(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["dataset_identity_key"]].append(record)
    return dict(grouped)


def choose_peoplelocation_splits(records):
    """Choose deterministic 3/2/2 identity splits for this small dataset.

    Validation and test each receive two cross-camera identities. Of the valid
    allocations, the one balancing val/test image counts most closely wins;
    SHA-256 of the seed and assignments is the deterministic tie breaker.
    """
    grouped = group_people_identities(records)
    preserved_identities = {
        identity for identities in PRESERVED_PEOPLE_SPLITS.values()
        for identity in identities
    }
    if set(grouped) == preserved_identities:
        return dict(PRESERVED_PEOPLE_SPLITS)
    if len(grouped) < 7:
        raise ValueError(
            "At least seven PeopleLocation identities are required for 3/2/2 splits"
        )
    cross_camera = sorted(
        identity for identity, rows in grouped.items()
        if len({row["camera"] for row in rows}) > 1
    )
    if len(cross_camera) < 5:
        raise ValueError(
            "At least five cross-camera identities are required to keep two in "
            "validation, two in held-out test, and one in training"
        )

    candidates = []
    for validation_ids in itertools.combinations(cross_camera, 2):
        remaining = [item for item in cross_camera if item not in validation_ids]
        for test_ids in itertools.combinations(remaining, 2):
            val_images = sum(len(grouped[item]) for item in validation_ids)
            test_images = sum(len(grouped[item]) for item in test_ids)
            tie_text = json.dumps(
                {"validation": validation_ids, "test": test_ids}, sort_keys=True
            )
            tie_breaker = hashlib.sha256(
                f"{SPLIT_SEED}:{tie_text}".encode("utf-8")
            ).hexdigest()
            candidates.append((
                abs(val_images - test_images),
                tie_breaker,
                tuple(validation_ids),
                tuple(test_ids),
            ))
    _, _, validation_ids, test_ids = min(candidates)
    held_out = set(validation_ids) | set(test_ids)
    train_ids = tuple(sorted(set(grouped) - held_out))
    return {
        "train": train_ids,
        "val": tuple(sorted(validation_ids)),
        "test": tuple(sorted(test_ids)),
    }


def normalize_people_record(record, split):
    return {
        "manifest_schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_source": PEOPLE_SOURCE,
        "split": split,
        "usage": {
            "train": "training",
            "val": "validation_model_selection",
            "test": "held_out_final_evaluation_only",
        }[split],
        "sample_id": record["sample_id"],
        "identity_key": f"peoplelocation:{record['dataset_identity_key']}",
        "original_person_id": record["original_gt_person_id"],
        "image_path": record["source_image"],
        "camera_id": record["camera"],
        "sequence_id": record["sequence"],
        "frame_index": record["frame_index"],
        "bbox_xyxy": record["bbox_xyxy"],
        "crop_policy": "raw_gt_bbox_no_margin",
        "already_cropped_person_image": False,
    }


def normalize_market1501_record(record):
    if record.get("source_split") != "bounding_box_train":
        raise ValueError("External non-training split record is forbidden")
    if not record.get("eligible_for_training") or record.get("ignored_junk"):
        raise ValueError("External junk or ineligible record is forbidden")
    return {
        "manifest_schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "dataset_source": MARKET_SOURCE,
        "split": "train",
        "usage": "supplemental_training_only",
        "sample_id": record["sample_id"],
        "identity_key": f"market1501:{record['original_person_id']:04d}",
        "original_person_id": record["original_person_id"],
        "image_path": record["image_path"],
        "camera_id": f"c{record['camera_id']}",
        "sequence_id": f"s{record['sequence_id']}",
        "frame_index": record["frame_index"],
        "sample_index": record["sample_index"],
        "source_split": "bounding_box_train",
        "bbox_xyxy": None,
        "crop_policy": "pre_cropped_no_recrop",
        "already_cropped_person_image": True,
    }


def parse_msmt17_train_list(list_path, train_root, project_root):
    """Parse only MSMT17 list_train.txt and validate every listed source row."""
    records = []
    invalid_labels = []
    missing_images = []
    escaped_paths = []
    record_keys = Counter()
    resolved_train_root = train_root.resolve()
    source_list = list_path.resolve().relative_to(
        project_root.resolve()
    ).as_posix()
    with list_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            parts = line.split()
            if len(parts) != 2:
                invalid_labels.append({"line": line_number, "value": line})
                continue
            relative_text, label_text = parts
            try:
                label = int(label_text)
            except ValueError:
                invalid_labels.append({"line": line_number, "value": line})
                continue
            relative_path = Path(relative_text)
            image_path = (train_root / relative_path).resolve()
            try:
                image_path.relative_to(resolved_train_root)
            except ValueError:
                escaped_paths.append({"line": line_number, "path": relative_text})
                continue
            filename_match = MSMT_FILENAME.fullmatch(relative_path.name)
            expected_pid = f"{label:04d}"
            if (
                label < 0
                or filename_match is None
                or relative_path.parts[0] != expected_pid
                or filename_match.group("pid") != expected_pid
            ):
                invalid_labels.append({"line": line_number, "value": line})
                continue
            project_image_path = image_path.relative_to(
                project_root.resolve()
            ).as_posix()
            if not image_path.is_file():
                missing_images.append(project_image_path)
            canonical = f"msmt17:{label}:{relative_path.as_posix()}"
            sample_id = (
                "msmt17_"
                + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
            )
            record_keys[(relative_path.as_posix().casefold(), label)] += 1
            records.append({
                "manifest_schema_version": SCHEMA_VERSION,
                "dataset_version": DATASET_VERSION,
                "dataset_source": MSMT_SOURCE,
                "split": "train",
                "usage": "supplemental_training_only",
                "sample_id": sample_id,
                "identity_key": f"msmt17:{label:04d}",
                "original_person_id": label,
                "image_path": project_image_path,
                "camera_id": f"c{int(filename_match.group('camera'))}",
                "sequence_id": None,
                "frame_index": None,
                "sample_index": int(filename_match.group("index")),
                "bbox_xyxy": None,
                "crop_policy": "pre_cropped_no_recrop",
                "already_cropped_person_image": True,
                "source_split": "train",
                "source_list": source_list,
                "source_list_line": line_number,
            })
    duplicate_records = sorted(
        f"{path}|{label}" for (path, label), count in record_keys.items()
        if count > 1
    )
    audit = {
        "source_list": source_list,
        "source_list_sha256": sha256_file(list_path),
        "listed_records": len(records),
        "invalid_label_count": len(invalid_labels),
        "invalid_labels": invalid_labels,
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "escaped_path_count": len(escaped_paths),
        "escaped_paths": escaped_paths,
        "duplicate_source_record_count": len(duplicate_records),
        "duplicate_source_records": duplicate_records,
    }
    return records, audit


def validate_images(records, project_root):
    missing = []
    unreadable = []
    empty_crops = []
    out_of_bounds = []
    cached_images = {}
    for record in records:
        path = (project_root / record["image_path"]).resolve()
        if not path.is_file():
            missing.append(record["sample_id"])
            continue
        key = str(path)
        image = cached_images.get(key)
        if image is None:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            cached_images[key] = image
        if image is None or image.size == 0:
            unreadable.append(record["sample_id"])
            continue
        bbox = record["bbox_xyxy"]
        if bbox is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in bbox)
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            out_of_bounds.append(record["sample_id"])
            continue
        if x2 <= x1 or y2 <= y1 or image[y1:y2, x1:x2].size == 0:
            empty_crops.append(record["sample_id"])
    return {
        "missing_image_path_count": len(missing),
        "missing_image_sample_ids": missing,
        "unreadable_image_count": len(unreadable),
        "unreadable_sample_ids": unreadable,
        "empty_crop_count": len(empty_crops),
        "empty_crop_sample_ids": empty_crops,
        "out_of_bounds_crop_count": len(out_of_bounds),
        "out_of_bounds_sample_ids": out_of_bounds,
    }


def split_statistics(records, dataset_source=None):
    selected = [
        record for record in records
        if dataset_source is None or record["dataset_source"] == dataset_source
    ]
    grouped = defaultdict(list)
    for record in selected:
        grouped[record["identity_key"]].append(record)
    counts = [len(rows) for rows in grouped.values()]
    return {
        "identities": len(grouped),
        "images": len(selected),
        "cross_camera_identities": sum(
            len({row["camera_id"] for row in rows}) > 1
            for rows in grouped.values()
        ),
        "cameras": dict(sorted(Counter(
            row["camera_id"] for row in selected
        ).items())),
        "images_per_identity": {
            "minimum": min(counts) if counts else 0,
            "maximum": max(counts) if counts else 0,
            "mean": sum(counts) / len(counts) if counts else 0.0,
            "median": statistics.median(counts) if counts else 0.0,
        },
    }


def leakage_audit(split_records):
    identity_sets = {
        split: {
            row["identity_key"] for row in rows
        }
        for split, rows in split_records.items()
    }
    overlaps = {
        "train_val": sorted(identity_sets["train"] & identity_sets["val"]),
        "train_test": sorted(identity_sets["train"] & identity_sets["test"]),
        "val_test": sorted(identity_sets["val"] & identity_sets["test"]),
    }
    all_records = [row for rows in split_records.values() for row in rows]
    sample_counts = Counter(row["sample_id"] for row in all_records)
    duplicate_sample_ids = sorted(
        sample_id for sample_id, count in sample_counts.items() if count > 1
    )
    record_counts = Counter(
        (row["dataset_source"], row["image_path"], row["identity_key"])
        for row in all_records
    )
    duplicate_records = sorted(
        "|".join(key) for key, count in record_counts.items() if count > 1
    )
    msmt_contamination = [
        row["sample_id"] for row in all_records
        if row["dataset_source"] == MSMT_SOURCE
        and (
            row["split"] != "train"
            or row.get("source_split") != "train"
            or not row.get("source_list", "").endswith("/list_train.txt")
            or "/MSMT17_V1/train/" not in f"/{row['image_path']}"
        )
    ]
    market_test_records = [
        row["sample_id"] for row in all_records
        if row["dataset_source"] == MARKET_SOURCE
        and (
            row["split"] != "train"
            or row["usage"] != "supplemental_training_only"
            or row.get("source_split") != "bounding_box_train"
            or row["image_path"].replace("\\", "/").startswith(
                "bounding_box_test/"
            )
        )
    ]
    expected_prefixes = {
        PEOPLE_SOURCE: "peoplelocation:",
        MARKET_SOURCE: "market1501:",
        MSMT_SOURCE: "msmt17:",
    }
    namespace_violations = [
        row["sample_id"] for row in all_records
        if row["dataset_source"] not in expected_prefixes
        or not row["identity_key"].startswith(
            expected_prefixes.get(row["dataset_source"], "\0")
        )
    ]
    invalid_labels = [
        row["sample_id"] for row in all_records
        if not row["identity_key"]
        or row.get("original_person_id") is None
        or (
            row["dataset_source"] in (MARKET_SOURCE, MSMT_SOURCE)
            and (
                not isinstance(row["original_person_id"], int)
                or row["original_person_id"] < 0
            )
        )
    ]
    return {
        "train_validation_test_identity_overlaps": overlaps,
        "identity_leakage_count": sum(len(items) for items in overlaps.values()),
        "identity_leakage": any(overlaps.values()),
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_record_count": len(duplicate_records),
        "duplicate_records": duplicate_records,
        "msmt17_validation_test_contamination_count": len(msmt_contamination),
        "msmt17_contaminating_sample_ids": msmt_contamination,
        "market1501_test_contamination_count": len(market_test_records),
        "market1501_test_sample_ids": market_test_records,
        "namespace_violation_count": len(namespace_violations),
        "namespace_violation_sample_ids": namespace_violations,
        "invalid_label_count": len(invalid_labels),
        "invalid_label_sample_ids": invalid_labels,
    }


def jsonl_text(records):
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )


def render_report(stats):
    people = stats["peoplelocation_splits"]
    market = stats["market1501_training_source"]
    msmt = stats["msmt17_training_source"]
    combined = stats["combined_train"]
    lines = [
        "# Fine-Tuning Dataset Preparation Report",
        "",
        f"Dataset version: `{stats['dataset_version']}`  ",
        f"Split seed: `{stats['split_seed']}`  ",
        "PeopleLocation crop policy: raw ground-truth bbox with no margin.  ",
        "External images are pre-cropped and are not cropped again.",
        "",
        "## Identity-disjoint PeopleLocation splits",
        "",
        "| Split | Identities | Images | Cross-camera identities | Purpose |",
        "|---|---:|---:|---:|---|",
        f"| train | {people['train']['identities']} | {people['train']['images']} | "
        f"{people['train']['cross_camera_identities']} | training |",
        f"| val | {people['val']['identities']} | {people['val']['images']} | "
        f"{people['val']['cross_camera_identities']} | model/threshold selection |",
        f"| test | {people['test']['identities']} | {people['test']['images']} | "
        f"{people['test']['cross_camera_identities']} | locked final evaluation |",
        "",
        "## Training sources",
        "",
        "| Source | Identities | Images | Namespace |",
        "|---|---:|---:|---|",
        f"| PeopleLocation train | {people['train']['identities']} | "
        f"{people['train']['images']} | `peoplelocation:*` |",
        f"| Market-1501 train supplement | {market['identities']} | "
        f"{market['images']} | `market1501:*` |",
        f"| MSMT17 train supplement | {msmt['identities']} | "
        f"{msmt['images']} | `msmt17:*` |",
        f"| **Combined train** | **{combined['identities']}** | "
        f"**{combined['images']}** | dataset-scoped |",
        "",
        "## Validation",
        "",
        f"- Identity leakage: `{stats['leakage_audit']['identity_leakage_count']}`",
        f"- Duplicate sample IDs: `{stats['leakage_audit']['duplicate_sample_id_count']}`",
        f"- Duplicate records: `{stats['leakage_audit']['duplicate_record_count']}`",
        f"- MSMT17 validation/test contamination: "
        f"`{stats['leakage_audit']['msmt17_validation_test_contamination_count']}`",
        f"- Market-1501 test contamination: "
        f"`{stats['leakage_audit']['market1501_test_contamination_count']}`",
        f"- Namespace violations: `{stats['leakage_audit']['namespace_violation_count']}`",
        f"- Invalid labels: `{stats['leakage_audit']['invalid_label_count']}`",
        f"- Missing image paths: `{stats['image_validation']['missing_image_path_count']}`",
        f"- Unreadable images: `{stats['image_validation']['unreadable_image_count']}`",
        f"- Empty crops: `{stats['image_validation']['empty_crop_count']}`",
        f"- Out-of-bounds crops: "
        f"`{stats['image_validation']['out_of_bounds_crop_count']}`",
        f"- Upstream rejected PeopleLocation GT samples: "
        f"`{stats['upstream_rejected_peoplelocation_samples']}`",
        "",
        "## Readiness",
        "",
        "**READY FOR PROMPT 04 WITH DATA-SIZE CONSTRAINTS.** The manifests pass "
        "leakage and contamination checks. PeopleLocation has only seven identities, "
        "so validation and held-out results are suitable for a controlled experiment "
        "but not for strong generalization claims. The held-out test must remain locked "
        "until all checkpoint, threshold, crop-policy, hyperparameter, and epoch choices "
        "are final.",
        "",
        "No fine-tuning was run. Production and camera/video/calibration code were not "
        "modified.",
        "",
    ]
    return "\n".join(lines)


def build_finetuning_dataset(
    people_manifest, people_summary_path, external_manifest,
    external_summary_path, msmt17_root, output_root, project_root,
    overwrite=False,
):
    targets = [
        output_root / split / "manifest.jsonl" for split in ("train", "val", "test")
    ] + [
        output_root / "dataset_stats.json",
        output_root / "DATASET_PREPARATION_REPORT.md",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite outputs: " + ", ".join(map(str, existing))
        )

    people_summary = json.loads(people_summary_path.read_text(encoding="utf-8"))
    external_summary = json.loads(external_summary_path.read_text(encoding="utf-8"))
    people_hash = sha256_file(people_manifest)
    external_hash = sha256_file(external_manifest)
    if people_hash != people_summary["manifest_sha256"]:
        raise ValueError("PeopleLocation manifest SHA-256 does not match its summary")
    if external_hash != external_summary["manifest_sha256"]:
        raise ValueError("External manifest SHA-256 does not match its summary")

    people_records = read_jsonl(people_manifest)
    external_records = read_jsonl(external_manifest)
    msmt_list = msmt17_root / "list_train.txt"
    msmt_train_root = msmt17_root / "train"
    if not msmt_list.is_file() or not msmt_train_root.is_dir():
        raise FileNotFoundError(
            "MSMT17 requires MSMT17_V1/list_train.txt and MSMT17_V1/train/"
        )
    msmt_records, msmt_source_audit = parse_msmt17_train_list(
        msmt_list, msmt_train_root, project_root
    )
    if any(msmt_source_audit[key] for key in (
        "invalid_label_count", "missing_image_count", "escaped_path_count",
        "duplicate_source_record_count",
    )):
        raise ValueError("MSMT17 list_train.txt source validation failed")
    assignments = choose_peoplelocation_splits(people_records)
    identity_to_split = {
        identity: split for split, identities in assignments.items()
        for identity in identities
    }
    split_records = {"train": [], "val": [], "test": []}
    for record in people_records:
        split = identity_to_split[record["dataset_identity_key"]]
        split_records[split].append(normalize_people_record(record, split))
    split_records["train"].extend(
        normalize_market1501_record(record) for record in external_records
    )
    split_records["train"].extend(msmt_records)
    for records in split_records.values():
        records.sort(key=lambda row: (
            row["dataset_source"], row["identity_key"], row["sample_id"]
        ))

    leakage = leakage_audit(split_records)
    image_validation = validate_images(
        [row for rows in split_records.values() for row in rows], project_root
    )
    validation_failure_count = (
        leakage["identity_leakage_count"]
        + leakage["duplicate_sample_id_count"]
        + leakage["duplicate_record_count"]
        + leakage["msmt17_validation_test_contamination_count"]
        + leakage["market1501_test_contamination_count"]
        + leakage["namespace_violation_count"]
        + leakage["invalid_label_count"]
        + image_validation["missing_image_path_count"]
        + image_validation["unreadable_image_count"]
        + image_validation["empty_crop_count"]
        + image_validation["out_of_bounds_crop_count"]
    )
    if validation_failure_count:
        raise ValueError("Fine-tuning dataset validation failed")

    manifest_hashes = {}
    for split, records in split_records.items():
        path = output_root / split / "manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = jsonl_text(records)
        path.write_text(content, encoding="utf-8", newline="\n")
        manifest_hashes[split] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    people_stats = {
        split: {
            **split_statistics(split_records[split], dataset_source=PEOPLE_SOURCE),
            "identity_keys": [
                f"peoplelocation:{identity}" for identity in assignments[split]
            ],
            "source_identity_keys": list(assignments[split]),
        }
        for split in ("train", "val", "test")
    }
    market_train = [
        row for row in split_records["train"]
        if row["dataset_source"] == MARKET_SOURCE
    ]
    msmt_train = [
        row for row in split_records["train"]
        if row["dataset_source"] == MSMT_SOURCE
    ]
    stats = {
        "dataset_version": DATASET_VERSION,
        "manifest_schema_version": SCHEMA_VERSION,
        "split_seed": SPLIT_SEED,
        "source_manifests": {
            "peoplelocation": {
                "path": people_manifest.resolve().relative_to(
                    project_root.resolve()
                ).as_posix(),
                "sha256": people_hash,
            },
            "market1501_train_only": {
                "path": external_manifest.resolve().relative_to(
                    project_root.resolve()
                ).as_posix(),
                "sha256": external_hash,
            },
            "msmt17_train_only": {
                "path": msmt_list.resolve().relative_to(
                    project_root.resolve()
                ).as_posix(),
                "sha256": msmt_source_audit["source_list_sha256"],
            },
        },
        "crop_policies": {
            "peoplelocation": "raw_gt_bbox_no_margin",
            "market1501": "pre_cropped_no_recrop",
            "msmt17": "pre_cropped_no_recrop",
        },
        "peoplelocation_splits": people_stats,
        "market1501_training_source": split_statistics(market_train),
        "msmt17_training_source": split_statistics(msmt_train),
        "msmt17_source_audit": msmt_source_audit,
        "combined_train": split_statistics(split_records["train"]),
        "manifest_record_counts": {
            split: len(records) for split, records in split_records.items()
        },
        "manifest_sha256": manifest_hashes,
        "leakage_audit": leakage,
        "image_validation": image_validation,
        "wrong_gt_mapping_count": len(
            people_summary["identity_collision_audit"][
                "namespace_identity_key_collisions"
            ]
        ),
        "upstream_rejected_peoplelocation_samples": people_summary[
            "rejected_samples"
        ],
        "hard_negative_policy": (
            "Natural different-identity samples only; no automatic relabeling or "
            "synthetic hard-negative generation"
        ),
        "validation_usage": (
            "checkpoint, threshold, crop-policy, hyperparameter, and epoch selection"
        ),
        "held_out_test_usage": "final evaluation only after all model choices are locked",
        "market1501_test_records_in_any_split": 0,
        "msmt17_non_train_records_in_any_split": 0,
        "fine_tuning_run": False,
        "production_modified": False,
        "camera_video_calibration_modified": False,
        "prompt_04_ready": True,
        "readiness_constraint": (
            "Only seven PeopleLocation identities; do not make strong generalization "
            "claims from validation or held-out test metrics"
        ),
    }
    stats_path = output_root / "dataset_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "DATASET_PREPARATION_REPORT.md").write_text(
        render_report(stats), encoding="utf-8", newline="\n"
    )
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--people-manifest", type=Path,
        default=Path("reid_dataset/master_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--people-summary", type=Path,
        default=Path("reid_dataset/master_manifest_v1_summary.json"),
    )
    parser.add_argument(
        "--external-manifest", type=Path,
        default=Path("reid_dataset/external_reid_train_manifest_v1.jsonl"),
    )
    parser.add_argument(
        "--external-summary", type=Path,
        default=Path("reid_dataset/external_reid_train_manifest_v1_summary.json"),
    )
    parser.add_argument(
        "--msmt17-root", type=Path,
        default=Path("datasets/msmt17/MSMT17_V1"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("datasets/peoplelocation_reid_v1"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build_finetuning_dataset(
        arguments.people_manifest,
        arguments.people_summary,
        arguments.external_manifest,
        arguments.external_summary,
        arguments.msmt17_root,
        arguments.output_root,
        arguments.project_root,
        overwrite=arguments.overwrite,
    )
    print(json.dumps({
        "peoplelocation_splits": result["peoplelocation_splits"],
        "market1501_training_source": result["market1501_training_source"],
        "msmt17_training_source": result["msmt17_training_source"],
        "combined_train": result["combined_train"],
        "leakage_audit": result["leakage_audit"],
        "prompt_04_ready": result["prompt_04_ready"],
    }, indent=2, sort_keys=True))
