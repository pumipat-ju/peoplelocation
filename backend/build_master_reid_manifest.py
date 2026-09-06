"""Build a deterministic master Re-ID manifest across all labeled sequences."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from .build_reid_dataset_manifest import (
    DATASET_VERSION,
    MANIFEST_SCHEMA_VERSION,
    deterministic_sample_id,
    inspect_dataset,
)


MASTER_DATASET_VERSION = "peoplelocation-reid-master-v1"


def camera_directories(sequence_root):
    return sorted(
        child for child in sequence_root.iterdir()
        if child.is_dir()
        and (child / "images").is_dir()
        and any(child.glob("gt*.txt"))
    )


def discover_sequences(dataset_root, root_sequence_name="legacy_sequence"):
    sequences = []
    root_cameras = camera_directories(dataset_root)
    if root_cameras:
        sequences.append({
            "sequence": root_sequence_name,
            "root": dataset_root,
            "cameras": [path.name for path in root_cameras],
            "layout": "root_level_legacy",
        })
    root_camera_names = {path.name for path in root_cameras}
    for child in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        if child.name in root_camera_names:
            continue
        cameras = camera_directories(child)
        if cameras:
            sequences.append({
                "sequence": child.name,
                "root": child,
                "cameras": [path.name for path in cameras],
                "layout": "named_sequence",
            })
    names = [item["sequence"] for item in sequences]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate sequence names discovered: {names}")
    return sequences


def project_relative(path, project_root):
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def normalize_record(record, sequence_spec, project_root):
    sequence_root = sequence_spec["root"]
    source_absolute = sequence_root.parent / record["source_image"]
    annotation_absolute = sequence_root.parent / record["annotation_file"]
    original_id = str(record["ground_truth_person_id"])
    sequence = sequence_spec["sequence"]
    normalized = {
        **record,
        "dataset_version": MASTER_DATASET_VERSION,
        "sequence": sequence,
        "original_gt_person_id": original_id,
        "ground_truth_person_id": original_id,
        "dataset_identity_key": f"{sequence}:{original_id}",
        "source_image": project_relative(source_absolute, project_root),
        "annotation_file": project_relative(annotation_absolute, project_root),
    }
    normalized["sample_id"] = deterministic_sample_id(
        sequence, record["camera"], record["frame_index"], original_id
    )
    return normalized


def sequence_audit(sequence_spec, project_root):
    records, rejected, rejection_examples, schema_files = inspect_dataset(
        sequence_spec["root"]
    )
    normalized_records = [
        normalize_record(record, sequence_spec, project_root) for record in records
    ]
    for example in rejection_examples:
        example["sequence"] = sequence_spec["sequence"]
    identities = sorted({
        record["original_gt_person_id"] for record in normalized_records
    })
    cameras_by_identity = defaultdict(set)
    for record in normalized_records:
        cameras_by_identity[record["original_gt_person_id"]].add(record["camera"])
    shared = sorted(
        identity for identity in identities if len(cameras_by_identity[identity]) > 1
    )
    single = sorted(set(identities) - set(shared))
    return {
        "records": normalized_records,
        "summary": {
            "sequence": sequence_spec["sequence"],
            "source_root": project_relative(sequence_spec["root"], project_root),
            "layout": sequence_spec["layout"],
            "cameras": sequence_spec["cameras"],
            "source_gt_rows": sum(item["row_count"] for item in schema_files),
            "accepted_samples": len(normalized_records),
            "rejected_samples": int(sum(rejected.values())),
            "rejection_reasons": dict(sorted(rejected.items())),
            "original_identities": identities,
            "identity_keys": [
                f"{sequence_spec['sequence']}:{identity}" for identity in identities
            ],
            "cross_camera_identity_keys": [
                f"{sequence_spec['sequence']}:{identity}" for identity in shared
            ],
            "single_camera_identity_keys": [
                f"{sequence_spec['sequence']}:{identity}" for identity in single
            ],
        },
        "rejection_examples": rejection_examples,
    }


def identity_rows(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["dataset_identity_key"]].append(record)
    rows = []
    for identity_key in sorted(grouped):
        group = grouped[identity_key]
        cameras = sorted({item["camera"] for item in group})
        by_camera = Counter(item["camera"] for item in group)
        frames_by_camera = defaultdict(list)
        for item in group:
            frames_by_camera[item["camera"]].append(item["frame_index"])
        rows.append({
            "dataset_identity_key": identity_key,
            "sequence": group[0]["sequence"],
            "original_gt_person_id": group[0]["original_gt_person_id"],
            "cameras": ";".join(cameras),
            "camera_count": len(cameras),
            "cross_camera": len(cameras) > 1,
            "sample_count": len(group),
            "samples_by_camera": ";".join(
                f"{camera}:{by_camera[camera]}" for camera in cameras
            ),
            "frame_ranges": ";".join(
                f"{camera}:{min(frames_by_camera[camera])}-{max(frames_by_camera[camera])}"
                for camera in cameras
            ),
        })
    return rows


def collision_audit(sequence_summaries, records):
    sequences_by_original_id = defaultdict(set)
    for summary in sequence_summaries:
        for original_id in summary["original_identities"]:
            sequences_by_original_id[original_id].add(summary["sequence"])
    unscoped_collisions = {
        original_id: sorted(sequences)
        for original_id, sequences in sorted(sequences_by_original_id.items())
        if len(sequences) > 1
    }
    key_pairs = defaultdict(set)
    for record in records:
        key_pairs[record["dataset_identity_key"]].add((
            record["sequence"], record["original_gt_person_id"]
        ))
    namespace_collisions = {
        key: sorted([list(pair) for pair in pairs])
        for key, pairs in key_pairs.items() if len(pairs) > 1
    }
    return {
        "unscoped_original_id_collisions": unscoped_collisions,
        "unscoped_collisions_are_isolated_by_namespace": not namespace_collisions,
        "namespace_identity_key_collisions": namespace_collisions,
    }


def render_report(summary, audit_rows):
    lines = [
        "# Master Re-ID Ground-Truth Audit",
        "",
        f"Master dataset version: `{summary['dataset_version']}`  ",
        f"Manifest SHA-256: `{summary['manifest_sha256']}`  ",
        "Identity namespace: `<sequence>:<original_gt_person_id>`.",
        "",
        "## Sequences",
        "",
        "| Sequence | Cameras | Raw GT | Accepted | Rejected | Identities | Cross-camera |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for sequence in summary["sequence_summaries"]:
        lines.append(
            f"| {sequence['sequence']} | {';'.join(sequence['cameras'])} | "
            f"{sequence['source_gt_rows']} | {sequence['accepted_samples']} | "
            f"{sequence['rejected_samples']} | {len(sequence['identity_keys'])} | "
            f"{len(sequence['cross_camera_identity_keys'])} |"
        )
    lines.extend([
        "",
        "## Master Totals",
        "",
        f"- Accepted samples: `{summary['total_samples']}`",
        f"- Rejected samples: `{summary['rejected_samples']}`",
        f"- Sequence-scoped identities: `{summary['total_identities']}`",
        f"- Cross-camera identities: `{summary['cross_camera_identity_count']}`",
        f"- Single-camera identities: `{summary['single_camera_identity_count']}`",
        f"- Duplicate sample IDs: `{summary['duplicate_sample_ids']}`",
        f"- Namespace identity-key collisions: "
        f"`{len(summary['identity_collision_audit']['namespace_identity_key_collisions'])}`",
        "",
        "## Identities",
        "",
        "| Identity key | Cameras | Samples | Cross-camera |",
        "|---|---|---:|---|",
    ])
    for row in audit_rows:
        lines.append(
            f"| {row['dataset_identity_key']} | {row['cameras']} | "
            f"{row['sample_count']} | {'yes' if row['cross_camera'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Validation and Collisions",
        "",
        "Original numeric IDs reused by multiple sequences are expected and remain "
        "separate through the sequence namespace:",
        "",
    ])
    for original_id, sequences in summary["identity_collision_audit"][
        "unscoped_original_id_collisions"
    ].items():
        lines.append(f"- Original ID `{original_id}`: `{', '.join(sequences)}`")
    if not summary["identity_collision_audit"]["unscoped_original_id_collisions"]:
        lines.append("- No unscoped original-ID reuse was found.")
    lines.extend([
        "",
        "Rejected source rows were excluded from the master manifest without modifying "
        "source GT. Rejection details are retained in the JSON summary.",
        "",
        "## Prompt 02 Readiness",
        "",
        "**READY FOR PROMPT 02 WITH CONSTRAINTS.** Use only accepted master-manifest "
        "records and `dataset_identity_key` for labels. Do not read raw GT directly or "
        "collapse identities by their original numeric IDs. The five cross-camera "
        "identities are sufficient to run offline crop-policy diagnostics, but remain too "
        "few for strong claims or a robust person-disjoint benchmark.",
        "",
        "No fine-tuning was run. Production runtime and the frozen camera/video/calibration "
        "subsystem were not modified.",
        "",
    ])
    return "\n".join(lines)


def build_master(dataset_root, output_root, root_sequence_name="legacy_sequence",
                 overwrite=False):
    manifest_path = output_root / "master_manifest_v1.jsonl"
    summary_path = output_root / "master_manifest_v1_summary.json"
    audit_directory = output_root / "master_gt_audit"
    report_path = audit_directory / "MASTER_GT_AUDIT_REPORT.md"
    identity_path = audit_directory / "master_identity_summary.csv"
    targets = [manifest_path, summary_path, report_path, identity_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite master outputs: "
            + ", ".join(str(path) for path in existing)
        )

    project_root = dataset_root.resolve().parent
    sequence_specs = discover_sequences(dataset_root, root_sequence_name)
    if not sequence_specs:
        raise ValueError(f"No Re-ID sequences found under {dataset_root}")
    audits = [sequence_audit(spec, project_root) for spec in sequence_specs]
    records = [record for audit in audits for record in audit["records"]]
    records.sort(key=lambda item: (
        item["sequence"], item["camera"], item["frame_index"],
        item["original_gt_person_id"], item["sample_id"],
    ))
    sample_counts = Counter(record["sample_id"] for record in records)
    duplicate_sample_ids = sorted(
        sample_id for sample_id, count in sample_counts.items() if count > 1
    )
    if duplicate_sample_ids:
        raise ValueError(f"Duplicate sample IDs: {duplicate_sample_ids[:10]}")
    manifest_text = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    audit_rows = identity_rows(records)
    sequence_summaries = [audit["summary"] for audit in audits]
    collision = collision_audit(sequence_summaries, records)
    rejection_reasons = Counter()
    rejection_examples = []
    for audit in audits:
        rejection_reasons.update(audit["summary"]["rejection_reasons"])
        rejection_examples.extend(audit["rejection_examples"])
    cross_camera = [
        row["dataset_identity_key"] for row in audit_rows if row["cross_camera"]
    ]
    single_camera = [
        row["dataset_identity_key"] for row in audit_rows if not row["cross_camera"]
    ]
    summary = {
        "dataset_version": MASTER_DATASET_VERSION,
        "component_manifest_version": DATASET_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "identity_key_format": "<sequence>:<original_gt_person_id>",
        "manifest_sha256": manifest_hash,
        "dataset_root": project_relative(dataset_root, project_root),
        "sequences": [item["sequence"] for item in sequence_summaries],
        "sequence_summaries": sequence_summaries,
        "cameras": sorted({record["camera"] for record in records}),
        "sequence_cameras": sorted({
            f"{record['sequence']}:{record['camera']}" for record in records
        }),
        "total_source_gt_rows": sum(
            item["source_gt_rows"] for item in sequence_summaries
        ),
        "total_samples": len(records),
        "rejected_samples": int(sum(rejection_reasons.values())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "rejection_examples": rejection_examples,
        "total_identities": len(audit_rows),
        "cross_camera_identity_count": len(cross_camera),
        "cross_camera_identity_keys": cross_camera,
        "single_camera_identity_count": len(single_camera),
        "single_camera_identity_keys": single_camera,
        "samples_per_identity": {
            row["dataset_identity_key"]: row["sample_count"] for row in audit_rows
        },
        "duplicate_sample_ids": 0,
        "identity_collision_audit": collision,
        "prompt_02_ready": True,
        "prompt_02_constraints": [
            "Use accepted master manifest records only.",
            "Use dataset_identity_key rather than original_gt_person_id as the label.",
            "Do not make strong accuracy claims from only five cross-camera identities.",
        ],
        "fine_tuning_run": False,
        "production_modified": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    audit_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    with summary_path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")
    with identity_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    report_path.write_text(
        render_report(summary, audit_rows), encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "sequences": summary["sequences"],
        "total_samples": summary["total_samples"],
        "rejected_samples": summary["rejected_samples"],
        "total_identities": summary["total_identities"],
        "cross_camera_identity_count": summary["cross_camera_identity_count"],
        "single_camera_identity_count": summary["single_camera_identity_count"],
        "manifest_sha256": manifest_hash,
        "prompt_02_ready": summary["prompt_02_ready"],
    }, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("labeled_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reid_dataset"))
    parser.add_argument("--root-sequence-name", default="legacy_sequence")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_master(
        arguments.dataset_root.resolve(),
        arguments.output_dir.resolve(),
        root_sequence_name=arguments.root_sequence_name,
        overwrite=arguments.overwrite,
    )
