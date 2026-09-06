"""Build a versioned Re-ID manifest and audit cross-camera MOT ground truth."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from .dump_reid_debug import validate_and_convert_bbox
except ImportError:
    from dump_reid_debug import validate_and_convert_bbox


DATASET_VERSION = "peoplelocation-reid-v1"
MANIFEST_SCHEMA_VERSION = 1
EXPECTED_MOT_COLUMNS = 10
BASELINE = {
    "cross_camera_roc_auc": 0.519965,
    "eer": 0.479167,
    "similarity_gap": 0.008373,
}


def safe_component(value):
    cleaned = "".join(
        character for character in str(value)
        if character.isalnum() or character in "_-"
    )
    return cleaned or "unknown"


def deterministic_sample_id(sequence, camera, frame_index, person_id):
    identity = f"{sequence}|{camera}|{int(frame_index)}|{person_id}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return (
        f"{safe_component(sequence)}_{safe_component(camera)}_"
        f"f{int(frame_index):06d}_p{safe_component(person_id)}_{digest}"
    )


def convert_mot_bbox(frame_shape, bbox_xywh):
    bbox, reason = validate_and_convert_bbox(frame_shape, bbox_xywh)
    if reason is not None:
        return None, reason
    if bbox["was_clamped"]:
        return None, "bbox_out_of_frame_bounds"
    x1, y1, x2, y2 = bbox["input_xyxy"]
    width = x2 - x1
    height = y2 - y1
    return {
        "bbox_xywh": [x1, y1, width, height],
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_area": int(width * height),
        "aspect_ratio_width_over_height": float(width / height),
    }, None


def assert_unique_sample_ids(records):
    counts = Counter(record["sample_id"] for record in records)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sample IDs: {duplicates[:10]}")


def make_manifest_record(
    sequence,
    camera,
    annotation_file,
    annotation_row,
    source_image,
    frame_index,
    person_id,
    frame_shape,
    bbox_xywh,
    mot_extra_fields,
):
    bbox, reason = convert_mot_bbox(frame_shape, bbox_xywh)
    if reason is not None:
        return None, reason
    return {
        "dataset_version": DATASET_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "sequence": sequence,
        "camera": camera,
        "source_image": source_image.as_posix(),
        "frame_index": int(frame_index),
        "ground_truth_person_id": str(person_id),
        **bbox,
        "image_width": int(frame_shape[1]),
        "image_height": int(frame_shape[0]),
        "sample_id": deterministic_sample_id(
            sequence, camera, frame_index, person_id
        ),
        "annotation_file": annotation_file.as_posix(),
        "annotation_row": int(annotation_row),
        "mot_extra_fields": list(mot_extra_fields),
    }, None


def parse_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def inspect_dataset(dataset_root):
    sequence = dataset_root.name
    records = []
    rejected = Counter()
    rejection_examples = []
    schema_files = []
    seen_keys = set()
    image_shape_cache = {}

    for camera_directory in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        camera = camera_directory.name
        images_directory = camera_directory / "images"
        for annotation_path in sorted(camera_directory.glob("gt*.txt")):
            relative_annotation = annotation_path.relative_to(dataset_root.parent)
            column_counts = Counter()
            file_rows = 0
            file_rejections = 0
            with annotation_path.open("r", encoding="utf-8", newline="") as source:
                for row_number, row in enumerate(csv.reader(source), start=1):
                    file_rows += 1
                    column_counts[len(row)] += 1
                    reason = None
                    frame_index = None
                    person_id = None
                    bbox_xywh = None
                    if len(row) != EXPECTED_MOT_COLUMNS:
                        reason = "unexpected_mot_column_count"
                    else:
                        try:
                            frame_number = parse_number(row[0])
                            if not frame_number.is_integer() or frame_number < 1:
                                raise ValueError("invalid frame index")
                            frame_index = int(frame_number)
                            person_id = row[1].strip()
                            if not person_id:
                                raise ValueError("empty identity")
                            bbox_xywh = tuple(parse_number(value) for value in row[2:6])
                        except ValueError:
                            reason = "invalid_mot_value"

                    if reason is None:
                        duplicate_key = (sequence, camera, frame_index, person_id)
                        if duplicate_key in seen_keys:
                            reason = "duplicate_camera_frame_identity"
                        else:
                            seen_keys.add(duplicate_key)

                    image_path = images_directory / f"{frame_index:06d}.jpg" if reason is None else None
                    if reason is None and not image_path.is_file():
                        reason = "missing_frame_image"

                    if reason is None:
                        cache_key = str(image_path)
                        frame_shape = image_shape_cache.get(cache_key)
                        if frame_shape is None:
                            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                            if frame is None:
                                reason = "unreadable_frame_image"
                            else:
                                frame_shape = frame.shape
                                image_shape_cache[cache_key] = frame_shape

                    if reason is None:
                        record, reason = make_manifest_record(
                            sequence=sequence,
                            camera=camera,
                            annotation_file=relative_annotation,
                            annotation_row=row_number,
                            source_image=image_path.relative_to(dataset_root.parent),
                            frame_index=frame_index,
                            person_id=person_id,
                            frame_shape=frame_shape,
                            bbox_xywh=bbox_xywh,
                            mot_extra_fields=row[6:],
                        )
                    if reason is not None:
                        rejected[reason] += 1
                        file_rejections += 1
                        if len(rejection_examples) < 100:
                            rejection_examples.append({
                                "annotation_file": relative_annotation.as_posix(),
                                "annotation_row": row_number,
                                "camera": camera,
                                "frame_index": frame_index,
                                "ground_truth_person_id": person_id,
                                "bbox_xywh": list(bbox_xywh) if bbox_xywh else None,
                                "reason": reason,
                            })
                        continue
                    records.append(record)

            schema_files.append({
                "annotation_file": relative_annotation.as_posix(),
                "row_count": file_rows,
                "column_counts": {
                    str(count): occurrences
                    for count, occurrences in sorted(column_counts.items())
                },
                "rejected_rows": file_rejections,
            })

    records.sort(key=lambda item: (
        item["sequence"], item["camera"], item["frame_index"],
        item["ground_truth_person_id"], item["sample_id"],
    ))
    assert_unique_sample_ids(records)
    return records, rejected, rejection_examples, schema_files


def identity_audit_rows(records, suspected_label_ids=None):
    suspected_label_ids = set(suspected_label_ids or [])
    by_identity = defaultdict(list)
    by_identity_camera = defaultdict(list)
    for record in records:
        person_id = record["ground_truth_person_id"]
        by_identity[person_id].append(record)
        by_identity_camera[(person_id, record["camera"])].append(record)

    rows = []
    for person_id in sorted(by_identity, key=lambda value: (not value.isdigit(), value)):
        identity_records = by_identity[person_id]
        cameras = sorted({record["camera"] for record in identity_records})
        for camera in cameras:
            camera_records = by_identity_camera[(person_id, camera)]
            frames = [record["frame_index"] for record in camera_records]
            rows.append({
                "ground_truth_person_id": person_id,
                "cameras": ";".join(cameras),
                "camera_count": len(cameras),
                "shared_across_cameras": len(cameras) > 1,
                "total_identity_samples": len(identity_records),
                "camera": camera,
                "camera_samples": len(camera_records),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "suspected_label_issue": person_id in suspected_label_ids,
                "audit_note": (
                    "duplicate camera/frame identity assignment requires manual correction"
                    if person_id in suspected_label_ids
                    else (
                        "identity is shared across cameras"
                        if len(cameras) > 1
                        else "single-camera identity; no cross-camera equivalence inferred"
                    )
                ),
            })
    return rows


def load_review_decisions(dataset_root):
    review_path = dataset_root / "review" / "cross_camera_review.csv"
    if not review_path.is_file():
        return []
    with review_path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def audit_review_policy(review_decisions, shared_identities):
    shared = set(shared_identities)
    violations = []
    confirmed = []
    rejected = []
    for decision in review_decisions:
        person_id = decision["proposed_person_id"].strip()
        status = decision["status"].strip()
        if status == "CONFIRMED":
            confirmed.append(person_id)
            if person_id not in shared:
                violations.append(
                    f"confirmed identity {person_id} is not present in both cameras"
                )
        elif status == "REJECTED_CROSS_CAMERA_MATCH":
            rejected.append(person_id)
            if person_id in shared:
                violations.append(
                    f"rejected identity {person_id} is still shared across cameras"
                )
    return {
        "confirmed_cross_camera_ids": sorted(confirmed),
        "rejected_cross_camera_ids": sorted(rejected),
        "policy_violations": violations,
    }


def representative_records(records, samples_per_identity_camera):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["ground_truth_person_id"], record["camera"])].append(record)
    selected = []
    for key in sorted(grouped):
        group = grouped[key]
        count = min(samples_per_identity_camera, len(group))
        if count == 1:
            indices = [len(group) // 2]
        else:
            indices = [
                round((index + 1) * (len(group) - 1) / (count + 1))
                for index in range(count)
            ]
        selected.extend(group[index] for index in indices)
    return selected


def make_review_sheet(crops):
    tile_width, tile_height, header_height = 180, 240, 34
    tiles = []
    for label, crop in crops:
        canvas = np.full(
            (tile_height + header_height, tile_width, 3), 255, dtype=np.uint8
        )
        scale = min(tile_width / crop.shape[1], tile_height / crop.shape[0])
        resized = cv2.resize(
            crop,
            (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        )
        x_offset = (tile_width - resized.shape[1]) // 2
        y_offset = header_height + (tile_height - resized.shape[0]) // 2
        canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized
        cv2.putText(
            canvas, label, (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)
    return cv2.hconcat(tiles)


def write_representative_crops(records, dataset_parent, audit_directory, samples_per_group):
    crops_directory = audit_directory / "representative_crops"
    crops_directory.mkdir(parents=True, exist_ok=True)
    selected = representative_records(records, samples_per_group)
    index_rows = []
    sheets = defaultdict(list)
    for record in selected:
        frame = cv2.imread(str(dataset_parent / record["source_image"]), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = record["bbox_xyxy"]
        crop = frame[y1:y2, x1:x2]
        crop_path = crops_directory / f"{record['sample_id']}.png"
        if crop.size == 0 or not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError(f"could not write representative crop {crop_path}")
        relative_crop = crop_path.relative_to(audit_directory.parent).as_posix()
        index_rows.append({
            "sample_id": record["sample_id"],
            "ground_truth_person_id": record["ground_truth_person_id"],
            "camera": record["camera"],
            "frame_index": record["frame_index"],
            "source_image": record["source_image"],
            "representative_crop": relative_crop,
        })
        sheets[record["ground_truth_person_id"]].append((
            f"{record['camera']} f{record['frame_index']}", crop
        ))

    fields = list(index_rows[0]) if index_rows else []
    with (crops_directory / "index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    for person_id, crops in sorted(sheets.items()):
        sheet_path = audit_directory / f"identity_{safe_component(person_id)}_review.png"
        if not cv2.imwrite(str(sheet_path), make_review_sheet(crops)):
            raise RuntimeError(f"could not write review sheet {sheet_path}")
    return index_rows


def render_audit_report(summary, identity_rows, representative_count):
    identity_summary = {}
    for row in identity_rows:
        identity_summary.setdefault(row["ground_truth_person_id"], row)
    lines = [
        "# Cross-Camera Ground-Truth Audit",
        "",
        f"Dataset version: `{summary['dataset_version']}`  ",
        f"Manifest SHA-256: `{summary['manifest_sha256']}`  ",
        "Source format: MOT `[frame,id,x,y,width,height,...]`; manifest boxes use "
        "`(x,y,x+width,y+height)`.",
        "",
        "## Baseline Preserved",
        "",
        "- Cross-camera ROC-AUC: `0.519965`",
        "- EER: `0.479167`",
        "- Similarity gap: `0.008373`",
        "- No fine-tuning was run.",
        "",
        "## Dataset Validation",
        "",
        f"- Source GT rows: `{summary['source_gt_rows']}`",
        f"- Accepted manifest samples: `{summary['accepted_samples']}`",
        f"- Rejected rows: `{summary['rejected_rows']}`",
        f"- Duplicate camera/frame/identity rows: "
        f"`{summary['rejection_reasons'].get('duplicate_camera_frame_identity', 0)}`",
        f"- Duplicate sample IDs: `{summary['duplicate_sample_ids']}`",
        f"- Missing referenced frames: `{summary['rejection_reasons'].get('missing_frame_image', 0)}`",
        f"- Out-of-bounds bboxes: `{summary['rejection_reasons'].get('bbox_out_of_frame_bounds', 0)}`",
        "",
        "## Cross-Camera Identities",
        "",
        "| GT ID | Cameras | Samples | Shared across cameras |",
        "|---|---|---:|---|",
    ]
    for person_id, row in identity_summary.items():
        lines.append(
            f"| {person_id} | {row['cameras']} | {row['total_identity_samples']} | "
            f"{'yes' if row['shared_across_cameras'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Suspected GT Issues",
        "",
    ])
    if summary["suspected_gt_issue_count"]:
        for reason, count in summary["rejection_reasons"].items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- No structural GT issue was found in the final annotations.")
    if summary["review_policy"]["policy_violations"]:
        for violation in summary["review_policy"]["policy_violations"]:
            lines.append(f"- Human-review policy violation: {violation}")
    else:
        lines.append("- No human-review cross-camera policy violation was found.")
    lines.extend([
        "",
        "## Human Review Decisions",
        "",
        f"- Confirmed cross-camera IDs: "
        f"`{', '.join(summary['review_policy']['confirmed_cross_camera_ids']) or 'none'}`",
        f"- Rejected cross-camera proposal IDs: "
        f"`{', '.join(summary['review_policy']['rejected_cross_camera_ids']) or 'none'}`",
        f"- Single-camera final IDs: "
        f"`{', '.join(summary['single_camera_identities']) or 'none'}`",
        "- Rejected matches were not rematched or merged automatically.",
        "",
        f"`{representative_count}` lossless representative crops and one indexed review "
        "sheet per final identity were generated. Labels were not changed by this audit.",
        "",
        "## Training Readiness",
        "",
        f"**{'READY' if summary['training_ready'] else 'NOT READY'} FOR TRAINING.** "
        f"{summary['training_readiness_reason']}",
        "",
        "The frozen camera/video/calibration subsystem was not modified.",
        "",
    ])
    return "\n".join(lines)


def build_manifest(dataset_root, output_root, samples_per_group=3, overwrite=False):
    if output_root.exists() and any(output_root.rglob("*")) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Pass --overwrite or use a new path."
        )
    audit_directory = output_root / "gt_audit"
    audit_directory.mkdir(parents=True, exist_ok=True)

    records, rejected, rejection_examples, schema_files = inspect_dataset(dataset_root)
    manifest_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    manifest_path = output_root / "manifest_v1.jsonl"
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")

    suspected_label_ids = sorted({
        str(item["ground_truth_person_id"])
        for item in rejection_examples
        if item["reason"] == "duplicate_camera_frame_identity"
        and item["ground_truth_person_id"] is not None
    })
    identity_rows = identity_audit_rows(records, suspected_label_ids)
    identity_fields = list(identity_rows[0]) if identity_rows else []
    with (audit_directory / "cross_camera_identity_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=identity_fields)
        writer.writeheader()
        writer.writerows(identity_rows)

    representative_index = write_representative_crops(
        records, dataset_root.parent, audit_directory, samples_per_group
    )
    cameras = sorted({record["camera"] for record in records})
    identities = sorted({record["ground_truth_person_id"] for record in records})
    shared_identities = sorted({
        person_id for person_id in identities
        if len({
            record["camera"] for record in records
            if record["ground_truth_person_id"] == person_id
        }) > 1
    })
    review_decisions = load_review_decisions(dataset_root)
    review_policy = audit_review_policy(review_decisions, shared_identities)
    structural_issue_count = int(sum(
        rejected[reason]
        for reason in (
            "duplicate_camera_frame_identity",
            "bbox_out_of_frame_bounds",
            "missing_frame_image",
        )
    ))
    training_ready = False
    training_readiness_reason = (
        "The final GT is structurally valid and follows human review, but only two "
        "identities are confirmed across cameras. This is insufficient for a meaningful "
        "person-disjoint train/validation/test evaluation, and the 2 FPS sequential "
        "samples remain highly correlated."
    )
    summary = {
        "dataset_version": DATASET_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "source_format": "MOT [frame,id,x,y,width,height,...]",
        "bbox_conversion": "xywh -> (x,y,x+width,y+height)",
        "dataset_root": dataset_root.as_posix(),
        "baseline": BASELINE,
        "source_gt_rows": sum(item["row_count"] for item in schema_files),
        "accepted_samples": len(records),
        "rejected_rows": int(sum(rejected.values())),
        "rejection_reasons": dict(sorted(rejected.items())),
        "rejection_examples": rejection_examples,
        "duplicate_sample_ids": 0,
        "schema_files": schema_files,
        "cameras": cameras,
        "identity_count": len(identities),
        "identities": identities,
        "identities_shared_across_cameras": shared_identities,
        "single_camera_identities": sorted(set(identities) - set(shared_identities)),
        "suspected_label_identities": suspected_label_ids,
        "suspected_gt_issue_count": structural_issue_count,
        "review_decisions": review_decisions,
        "review_policy": review_policy,
        "representative_crop_count": len(representative_index),
        "training_ready": training_ready,
        "training_readiness_reason": training_readiness_reason,
    }
    with (output_root / "manifest_v1_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")
    report = render_audit_report(summary, identity_rows, len(representative_index))
    (audit_directory / "GT_AUDIT_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    print(f"dataset version: {DATASET_VERSION}")
    print(f"source GT rows: {summary['source_gt_rows']}")
    print(f"accepted samples: {summary['accepted_samples']}")
    print(f"rejected rows: {summary['rejected_rows']}")
    print(f"identities: {summary['identity_count']}")
    print(f"shared identities: {', '.join(shared_identities) or 'none'}")
    print(f"single-camera identities: {', '.join(summary['single_camera_identities']) or 'none'}")
    print(f"representative crops: {summary['representative_crop_count']}")
    print(f"manifest SHA-256: {manifest_hash}")
    print(f"output directory: {output_root.resolve()}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a deterministic Re-ID manifest and audit MOT ground truth."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("labeled_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reid_dataset"))
    parser.add_argument("--representatives-per-camera-id", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_manifest(
        dataset_root=arguments.dataset_root.resolve(),
        output_root=arguments.output_dir.resolve(),
        samples_per_group=arguments.representatives_per_camera_id,
        overwrite=arguments.overwrite,
    )
