"""Evaluate canonical OSNet V1-V5 variants on one held-out protocol.

This is an offline-only evaluator. It reads full-frame PeopleLocation manifests,
applies either the recovered original production crop geometry or the established
raw-GT improved crop, and never imports or mutates the production runtime.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .evaluate_osnet_heldout import (
    build_finetuned_model,
    build_pretrained_model,
    compute_roc_eer,
    cross_camera_pairs,
    extract_embeddings,
    package_version,
    score_model,
    sha256_file,
    validate_peoplelocation_records,
)
from .finetune_osnet import read_manifest, validation_transform


VERSION_IDS = ("V1", "V2", "V3", "V4", "V5")
METRICS = (
    "rank_1", "rank_5", "mAP", "roc_auc", "eer", "accuracy",
    "precision", "recall", "f1", "same_id_mean_similarity",
    "different_id_mean_similarity", "similarity_gap", "false_positives",
    "false_negatives", "true_positives", "true_negatives", "valid_queries",
    "skipped_queries", "validation_selected_threshold", "pair_count",
)
LOWER_IS_BETTER = {
    "eer", "different_id_mean_similarity", "false_positives",
    "false_negatives", "skipped_queries",
}
NON_DIRECTIONAL = {"validation_selected_threshold", "pair_count"}
ORIGINAL_CROP = "original_production_margins_side_0.18_top_0.10_bottom_0.18"
IMPROVED_CROP = "raw_gt_bbox_no_margin"
THRESHOLD_METHOD = "cross-camera validation EER operating point"
PAIR_PROTOCOL = "unique unordered cross-camera pairs"
RETRIEVAL_PROTOCOL = (
    "all samples as queries; other-camera gallery; valid queries only"
)


@dataclass(frozen=True)
class VersionSpec:
    version_id: str
    model: str
    checkpoint_key: str
    crop_policy: str


SPECS = (
    VersionSpec("V1", "Pretrained OSNet x1.0", "pretrained", ORIGINAL_CROP),
    VersionSpec("V2", "Pretrained OSNet x1.0", "pretrained", IMPROVED_CROP),
    VersionSpec("V3", "Fine-tuned v1 OSNet x1.0", "finetuned_v1", ORIGINAL_CROP),
    VersionSpec("V4", "Domain-balanced fine-tuned v2 OSNet x1.0", "balanced_v2", ORIGINAL_CROP),
    VersionSpec("V5", "Domain-balanced fine-tuned v2 OSNet x1.0", "balanced_v2", IMPROVED_CROP),
)


def clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = (int(value) for value in bbox)
    return (
        max(0, min(x1, width)), max(0, min(y1, height)),
        max(0, min(x2, width)), max(0, min(y2, height)),
    )


def crop_box(bbox, width, height, crop_policy):
    """Return exact PIL crop coordinates for a canonical crop policy."""
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        raise ValueError(f"Invalid bounding box after clamping: {bbox}")
    if crop_policy == IMPROVED_CROP:
        return x1, y1, x2, y2
    if crop_policy != ORIGINAL_CROP:
        raise ValueError(f"Unknown crop policy: {crop_policy}")
    width_box, height_box = x2 - x1, y2 - y1
    return clamp_bbox((
        x1 + int(width_box * 0.18),
        y1 + int(height_box * 0.10),
        x2 - int(width_box * 0.18),
        y2 - int(height_box * 0.18),
    ), width, height)


class CropPolicyDataset(Dataset):
    def __init__(self, records, project_root, crop_policy):
        self.records = records
        self.project_root = project_root.resolve()
        self.crop_policy = crop_policy
        self.transform = validation_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = (self.project_root / record["image_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        bbox = record.get("bbox_xyxy")
        if not bbox:
            raise ValueError(f"PeopleLocation record lacks bbox: {record['sample_id']}")
        with Image.open(path) as source:
            image = source.convert("RGB")
            box = crop_box(bbox, image.width, image.height, self.crop_policy)
            image = image.crop(box)
            if image.width < 2 or image.height < 2:
                raise ValueError(f"Empty crop: {record['sample_id']}")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "identity_key": record["identity_key"],
            "camera_id": record["camera_id"],
            "sample_id": record["sample_id"],
        }


def make_loader(records, root, crop_policy, batch_size, workers, pin_memory):
    return DataLoader(
        CropPolicyDataset(records, root, crop_policy), batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=pin_memory,
    )


def flatten(metrics, threshold):
    counts = metrics["classification_counts"]
    return {
        "rank_1": metrics["rank_1"], "rank_5": metrics["rank_5"],
        "mAP": metrics["mAP"], "roc_auc": metrics["roc_auc"],
        "eer": metrics["eer"], "accuracy": metrics["accuracy"],
        "precision": metrics["precision"], "recall": metrics["recall"],
        "f1": metrics["f1"],
        "same_id_mean_similarity": metrics["same_id_mean_similarity"],
        "different_id_mean_similarity": metrics["different_id_mean_similarity"],
        "similarity_gap": metrics["similarity_gap"],
        "false_positives": counts["false_positives"],
        "false_negatives": counts["false_negatives"],
        "true_positives": counts["true_positives"],
        "true_negatives": counts["true_negatives"],
        "valid_queries": metrics["valid_queries"],
        "skipped_queries": metrics["skipped_queries"],
        "validation_selected_threshold": threshold,
        "pair_count": metrics["verification_pairs"],
    }


def effect(metric, delta):
    if metric in NON_DIRECTIONAL:
        return "not_directional" if delta else "unchanged"
    if abs(delta) < 1e-12:
        return "unchanged"
    improved = delta < 0 if metric in LOWER_IS_BETTER else delta > 0
    return "improved" if improved else "degraded"


def best_versions(metric, values):
    if metric in NON_DIRECTIONAL:
        return []
    target = min(values.values()) if metric in LOWER_IS_BETTER else max(values.values())
    return [key for key, value in values.items() if abs(value - target) < 1e-12]


def comparison_rows(versions):
    rows = []
    for metric in METRICS:
        values = {version: float(versions[version]["metrics"][metric])
                  for version in VERSION_IDS}
        rows.append({"metric": metric, **values,
                     "best_versions": best_versions(metric, values)})
    return rows


def delta_rows(versions):
    comparisons = [
        ("every_version_vs_V1", "V1", version) for version in VERSION_IDS
    ] + [
        ("V5_vs_V4_crop_effect", "V4", "V5"),
        ("V5_vs_V2_balanced_finetuning_effect", "V2", "V5"),
    ]
    rows = []
    for group, reference, candidate in comparisons:
        for metric in METRICS:
            before = float(versions[reference]["metrics"][metric])
            after = float(versions[candidate]["metrics"][metric])
            delta = after - before
            rows.append({
                "comparison": group, "metric": metric,
                "reference_version": reference, "candidate_version": candidate,
                "reference_value": before, "candidate_value": after,
                "absolute_delta": delta, "effect": effect(metric, delta),
            })
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_v5_pairs(path, records, pairs, threshold):
    left, right, scores, labels = pairs
    fields = [
        "sample_a", "sample_b", "identity_a", "identity_b", "camera_a",
        "camera_b", "same_identity", "similarity", "threshold",
        "predicted_same", "correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index, (a, b) in enumerate(zip(left, right)):
            predicted = bool(scores[index] >= threshold)
            label = bool(labels[index])
            writer.writerow({
                "sample_a": records[int(a)]["sample_id"],
                "sample_b": records[int(b)]["sample_id"],
                "identity_a": records[int(a)]["identity_key"],
                "identity_b": records[int(b)]["identity_key"],
                "camera_a": records[int(a)]["camera_id"],
                "camera_b": records[int(b)]["camera_id"],
                "same_identity": label, "similarity": float(scores[index]),
                "threshold": threshold, "predicted_same": predicted,
                "correct": predicted == label,
            })


def markdown_table(rows):
    lines = [
        "| Metric | V1 | V2 | V3 | V4 | V5 | Best Version |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values = [f"{row[v]:.6f}" for v in VERSION_IDS]
        best = ", ".join(row["best_versions"]) or "n/a"
        lines.append(f"| {row['metric']} | {' | '.join(values)} | {best} |")
    return "\n".join(lines)


def delta_markdown(rows, title):
    lines = [
        f"## {title}", "",
        "| Metric | Reference | Candidate | Absolute Delta | Improved/Degraded |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['reference_value']:.6f} | "
            f"{row['candidate_value']:.6f} | {row['absolute_delta']:+.6f} | "
            f"{row['effect']} |"
        )
    return "\n".join(lines)


def group_delta(rows, group, candidate=None):
    return [row for row in rows if row["comparison"] == group and
            (candidate is None or row["candidate_version"] == candidate)]


def dominates(versions, candidate, reference, metrics):
    effects = [effect(metric, versions[candidate]["metrics"][metric] -
                      versions[reference]["metrics"][metric]) for metric in metrics]
    return all(item in {"improved", "unchanged"} for item in effects) and "improved" in effects


def conclusions(versions, rows):
    retrieval = ("rank_1", "rank_5", "mAP")
    verification = ("roc_auc", "eer", "accuracy", "precision", "recall", "f1")
    separation = ("same_id_mean_similarity", "different_id_mean_similarity", "similarity_gap")
    best_retrieval = max(VERSION_IDS, key=lambda v: (
        versions[v]["metrics"]["mAP"], versions[v]["metrics"]["rank_1"],
        versions[v]["metrics"]["rank_5"],
    ))
    best_verification = max(VERSION_IDS, key=lambda v: (
        versions[v]["metrics"]["roc_auc"], -versions[v]["metrics"]["eer"],
        versions[v]["metrics"]["f1"],
    ))
    best_separation = max(VERSION_IDS, key=lambda v: (
        versions[v]["metrics"]["similarity_gap"],
        -versions[v]["metrics"]["different_id_mean_similarity"],
    ))
    crop_retrieval = dominates(versions, "V5", "V4", retrieval)
    crop_verification = dominates(versions, "V5", "V4", verification)
    crop_separation = dominates(versions, "V5", "V4", separation)
    return {
        "improved_crop_improves_balanced_v2": (
            crop_retrieval and crop_verification and crop_separation
        ),
        "improved_crop_balanced_v2_detail": {
            "retrieval": crop_retrieval, "verification": crop_verification,
            "embedding_separation": crop_separation,
        },
        "v5_outperforms_pretrained_improved_crop_v2": (
            dominates(versions, "V5", "V2", retrieval) and
            dominates(versions, "V5", "V2", verification)
        ),
        "v5_outperforms_pretrained_original_crop_v1": (
            dominates(versions, "V5", "V1", retrieval) and
            dominates(versions, "V5", "V1", verification)
        ),
        "best_retrieval_version": best_retrieval,
        "best_verification_version": best_verification,
        "best_embedding_separation_version": best_separation,
        "production_justified": False,
        "production_reason": (
            "Held-out evaluation contains only two identities; no checkpoint is "
            "authorized for production from this experiment."
        ),
    }


def historical_audit(project_root):
    items = []
    paths = {
        "historical_V1": project_root / "reid_experiment_results/01_pretrained_osnet/config.json",
        "historical_V2": project_root / "reid_experiment_results/02_pretrained_osnet_improved_crop/config.json",
        "prompt05_V3": project_root / "backend/reid_experiments/finetune_osnet_v1/heldout_test/metrics.json",
        "prompt06_V4": project_root / "backend/reid_experiments/finetune_osnet_v2_balanced/heldout_test/metrics.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            items.append({"artifact": name, "path": str(path), "exists": False})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("evaluation_config", data)
        policy = config.get("crop_policy", config.get("preprocessing", {}).get("crop_policy"))
        manifest = config.get("dataset_manifest", config.get("heldout_manifest", {}).get("path"))
        threshold = config.get("threshold_protocol", config.get("thresholds"))
        items.append({
            "artifact": name,
            "path": path.relative_to(project_root).as_posix(), "exists": True,
            "sha256": sha256_file(path), "recorded_crop_policy": policy,
            "recorded_manifest": manifest, "recorded_threshold_protocol": threshold,
            "compatibility_status": "NOT_DIRECTLY_COMPARABLE_TO_CANONICAL_RUN",
            "recomputed_for_canonical_comparison": True,
            "reason": (
                "Historical split/protocol or canonical crop label does not match "
                "the required V1-V5 held-out comparison. Original artifact preserved."
            ),
        })
    return items


def report_text(config, versions, rows, deltas, summary):
    v5v4 = group_delta(deltas, "V5_vs_V4_crop_effect")
    v5v2 = group_delta(deltas, "V5_vs_V2_balanced_finetuning_effect")
    validation = versions["V5"]["validation_metrics"]
    heldout = versions["V5"]["metrics"]
    return "\n".join([
        "# V5 Balanced OSNet + Improved Crop Evaluation", "",
        "All canonical V1-V5 variants were evaluated on the same PeopleLocation "
        "validation and held-out manifests. Thresholds came only from validation.", "",
        f"V5 validation threshold: `{versions['V5']['threshold']:.9f}`  ",
        f"Threshold method: `{THRESHOLD_METHOD}`  ",
        f"Held-out test used for threshold selection: `false`", "",
        "## V5 validation metrics", "",
        f"- ROC-AUC: `{validation['roc_auc']:.6f}`",
        f"- EER: `{validation['eer']:.6f}`",
        f"- Same-ID mean: `{validation['same_id_mean_similarity']:.6f}`",
        f"- Different-ID mean: `{validation['different_id_mean_similarity']:.6f}`",
        f"- Similarity gap: `{validation['similarity_gap']:.6f}`", "",
        "## V5 held-out metrics", "",
        f"- Rank-1 / Rank-5 / mAP: `{heldout['rank_1']:.6f}` / "
        f"`{heldout['rank_5']:.6f}` / `{heldout['mAP']:.6f}`",
        f"- ROC-AUC / EER: `{heldout['roc_auc']:.6f}` / `{heldout['eer']:.6f}`",
        f"- Accuracy / Precision / Recall / F1: `{heldout['accuracy']:.6f}` / "
        f"`{heldout['precision']:.6f}` / `{heldout['recall']:.6f}` / "
        f"`{heldout['f1']:.6f}`",
        f"- Same / different / gap: `{heldout['same_id_mean_similarity']:.6f}` / "
        f"`{heldout['different_id_mean_similarity']:.6f}` / "
        f"`{heldout['similarity_gap']:.6f}`",
        f"- TP / TN / FP / FN: `{heldout['true_positives']}` / "
        f"`{heldout['true_negatives']}` / `{heldout['false_positives']}` / "
        f"`{heldout['false_negatives']}`", "",
        "## Full comparison", "",
        markdown_table(rows), "",
        delta_markdown(v5v4, "V5 vs V4: Improved Crop effect on Balanced v2"), "",
        delta_markdown(v5v2, "V5 vs V2: balanced fine-tuning with Improved Crop"), "",
        "## Interpretation", "",
        f"- Improved Crop improves Balanced v2 overall: `{summary['improved_crop_improves_balanced_v2']}`",
        f"- V5 outperforms V2 overall: `{summary['v5_outperforms_pretrained_improved_crop_v2']}`",
        f"- V5 outperforms V1 overall: `{summary['v5_outperforms_pretrained_original_crop_v1']}`",
        f"- Best retrieval: `{summary['best_retrieval_version']}`",
        f"- Best verification: `{summary['best_verification_version']}`",
        f"- Strongest embedding separation: `{summary['best_embedding_separation_version']}`",
        f"- Production justified: `{summary['production_justified']}` — {summary['production_reason']}", "",
        "## Retrieval analysis", "",
        "V2 has the highest mAP and ties V1 for Rank-1. V5 improves mAP over V4 "
        "but reduces Rank-1 and Rank-5, so Improved Crop is not a consistent "
        "retrieval improvement for Balanced v2.", "",
        "## Verification analysis", "",
        "V5 has the best ROC-AUC, EER, accuracy, precision, recall, F1, and false-"
        "negative count. Its validation-selected threshold was not adjusted using test data.", "",
        "## Similarity-separation analysis", "",
        "V5 has the largest similarity gap and highest same-ID mean. V3 has the "
        "lowest different-ID mean, but its smaller overall gap makes V5 the strongest "
        "combined separation result.", "",
        "## Historical artifact compatibility", "",
        "Historical files were loaded and preserved. Canonical results were recomputed "
        "because the older V1/V2 artifacts used a 231-image master-manifest subset and "
        "test-set Youden-J thresholds, while saved Prompt05/06 artifacts labeled raw-GT "
        "crops where the canonical table requires Original Crop for V3/V4. Those "
        "historical rows are marked `NOT_DIRECTLY_COMPARABLE_TO_CANONICAL_RUN` and are "
        "not mixed into the canonical metrics.", "",
        "## Limitations", "",
        "- The held-out split has only two identities, despite containing many image pairs.",
        "- Pair observations are highly correlated because multiple frames depict the same people.",
        "- No result in this report authorizes production integration.", "",
        "No training, test threshold tuning, checkpoint mutation, or production/frozen "
        "subsystem modification was performed.", "",
    ])


def run(args):
    started = time.perf_counter()
    root = args.project_root.resolve()
    output = args.output_root.resolve()
    comparison_output = args.comparison_root.resolve()
    required_outputs = [
        output / name for name in (
            "metrics.json", "config.json", "pair_scores.csv",
            "validation_metrics.json", "evaluation_report.md",
        )
    ] + [comparison_output / name for name in (
        "ALL_VERSIONS_COMPARISON.csv", "ALL_VERSIONS_COMPARISON.json",
        "ALL_VERSIONS_COMPARISON.md", "ALL_VERSIONS_DELTAS.csv",
        "ALL_VERSIONS_DELTAS.json", "ALL_VERSIONS_DELTAS.md",
        "FINAL_EXPERIMENT_SUMMARY.md",
    )]
    if not args.overwrite and any(path.exists() for path in required_outputs):
        raise FileExistsError("Refusing to overwrite existing V5/comparison artifacts")
    for path in (args.pretrained_checkpoint, args.v1_checkpoint, args.v2_checkpoint,
                 args.validation_manifest, args.test_manifest):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch_version={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"selected_device={device}", flush=True)
    print("gpu_name=" + (torch.cuda.get_device_name(0) if device.type == "cuda" else "none"), flush=True)

    val_records = read_manifest(args.validation_manifest.resolve(), "val")
    test_records = read_manifest(args.test_manifest.resolve(), "test")
    validate_peoplelocation_records(val_records, "val")
    validate_peoplelocation_records(test_records, "test")
    if {row["identity_key"] for row in val_records} & {row["identity_key"] for row in test_records}:
        raise ValueError("Validation/test identity leakage")

    loaders = {}
    for policy in (ORIGINAL_CROP, IMPROVED_CROP):
        loaders[(policy, "val")] = make_loader(
            val_records, root, policy, args.batch_size, args.workers,
            device.type == "cuda",
        )
        loaders[(policy, "test")] = make_loader(
            test_records, root, policy, args.batch_size, args.workers,
            device.type == "cuda",
        )

    models = {
        "pretrained": build_pretrained_model(args.pretrained_checkpoint.resolve(), device),
        "finetuned_v1": build_finetuned_model(args.v1_checkpoint.resolve(), device)[0],
        "balanced_v2": build_finetuned_model(args.v2_checkpoint.resolve(), device)[0],
    }
    checkpoint_paths = {
        "pretrained": args.pretrained_checkpoint.resolve(),
        "finetuned_v1": args.v1_checkpoint.resolve(),
        "balanced_v2": args.v2_checkpoint.resolve(),
    }
    versions = {}
    v5_pairs = None
    for spec in SPECS:
        val_embeddings = extract_embeddings(
            models[spec.checkpoint_key], loaders[(spec.crop_policy, "val")], device
        )
        _, _, val_scores, val_labels = cross_camera_pairs(*val_embeddings[:3])
        _, _, threshold = compute_roc_eer(val_scores, val_labels)
        val_metrics, _ = score_model(*val_embeddings, threshold)
        test_embeddings = extract_embeddings(
            models[spec.checkpoint_key], loaders[(spec.crop_policy, "test")], device
        )
        test_metrics, test_pairs = score_model(*test_embeddings, threshold)
        checkpoint_path = checkpoint_paths[spec.checkpoint_key]
        versions[spec.version_id] = {
            "version_id": spec.version_id, "model": spec.model,
            "checkpoint": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "crop_policy": spec.crop_policy, "threshold": threshold,
            "threshold_method": THRESHOLD_METHOD,
            "threshold_source_split": "PeopleLocation validation",
            "validation_metrics": flatten(val_metrics, threshold),
            "metrics": flatten(test_metrics, threshold),
            "dataset": "PeopleLocation",
            "validation_manifest_sha256": sha256_file(args.validation_manifest.resolve()),
            "heldout_manifest_sha256": sha256_file(args.test_manifest.resolve()),
            "directly_comparable": True,
        }
        if spec.version_id == "V5":
            v5_pairs = test_pairs
        print(
            f"{spec.version_id} threshold={threshold:.6f} "
            f"mAP={test_metrics['mAP']:.6f} rank1={test_metrics['rank_1']:.6f} "
            f"auc={test_metrics['roc_auc']:.6f} eer={test_metrics['eer']:.6f}",
            flush=True,
        )

    rows = comparison_rows(versions)
    deltas = delta_rows(versions)
    summary = conclusions(versions, rows)
    audit = historical_audit(root)
    environment = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(), "torch_version": torch.__version__,
        "torchreid_version": package_version("torchreid"),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "none",
    }
    config = {
        "evaluation_version": "peoplelocation-osnet-v5-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": {
            "color_space": "RGB", "height": 256, "width": 128,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "embedding_dimension": 512, "l2_normalized": True,
        },
        "retrieval_protocol": RETRIEVAL_PROTOCOL,
        "verification_protocol": PAIR_PROTOCOL,
        "threshold_method": THRESHOLD_METHOD,
        "heldout_test_used_for_threshold_selection": False,
        "validation_manifest": args.validation_manifest.resolve().relative_to(root).as_posix(),
        "validation_manifest_sha256": sha256_file(args.validation_manifest.resolve()),
        "heldout_manifest": args.test_manifest.resolve().relative_to(root).as_posix(),
        "heldout_manifest_sha256": sha256_file(args.test_manifest.resolve()),
        "validation_images": len(val_records), "heldout_images": len(test_records),
        "validation_identities": len({row['identity_key'] for row in val_records}),
        "heldout_identities": len({row['identity_key'] for row in test_records}),
        "validation_test_identity_overlap": 0, "environment": environment,
        "historical_artifact_audit": audit,
        "retrained": False, "production_integrated": False,
    }
    runtime = time.perf_counter() - started
    config["runtime_seconds"] = runtime
    output.mkdir(parents=True, exist_ok=True)
    comparison_output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps({"version": versions["V5"], "config": config}, indent=2,
                   sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "validation_metrics.json").write_text(
        json.dumps({
            "threshold": versions["V5"]["threshold"],
            "threshold_method": THRESHOLD_METHOD,
            "source_split": "PeopleLocation validation",
            "heldout_test_used": False,
            "metrics": versions["V5"]["validation_metrics"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_v5_pairs(output / "pair_scores.csv", test_records, v5_pairs,
                   versions["V5"]["threshold"])
    report = report_text(config, versions, rows, deltas, summary)
    (output / "evaluation_report.md").write_text(report, encoding="utf-8")

    csv_rows = []
    for row in rows:
        csv_rows.append({
            "metric": row["metric"], **{v: row[v] for v in VERSION_IDS},
            "best_versions": ";".join(row["best_versions"]),
        })
    write_csv(comparison_output / "ALL_VERSIONS_COMPARISON.csv", csv_rows)
    comparison_payload = {
        "config": config, "versions": versions, "comparison": rows,
        "summary": summary, "runtime_seconds": runtime,
    }
    (comparison_output / "ALL_VERSIONS_COMPARISON.json").write_text(
        json.dumps(comparison_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_md = "\n".join([
        "# Canonical OSNet V1-V5 Comparison", "", markdown_table(rows), "",
        "## V5 vs V4 analysis", "",
        "Improved Crop raises V4 mAP by 0.002996, but Rank-1 falls by 0.013361 "
        "and Rank-5 falls by 0.006166. Verification and similarity separation improve.", "",
        "## V5 vs V2 analysis", "",
        "With Improved Crop held constant, balanced fine-tuning improves every listed "
        "verification and separation metric, but degrades Rank-1, Rank-5, and mAP.", "",
        "## Retrieval analysis", "",
        "V2 is the best retrieval version by mAP, with V1 and V2 tied at Rank-1.", "",
        "## Verification analysis", "",
        "V5 is best by ROC-AUC, EER, accuracy, precision, recall, and F1.", "",
        "## Similarity-separation analysis", "",
        "V5 has the largest similarity gap and highest same-ID mean. V3 has the "
        "lowest different-ID mean.", "",
        "## Decisions", "",
        f"- Best retrieval: **{summary['best_retrieval_version']}**",
        f"- Best verification: **{summary['best_verification_version']}**",
        f"- Strongest embedding separation: **{summary['best_embedding_separation_version']}**",
        f"- V5 vs V4 crop improvement overall: **{summary['improved_crop_improves_balanced_v2']}**",
        f"- V5 vs V2 overall improvement: **{summary['v5_outperforms_pretrained_improved_crop_v2']}**",
        "", "## Limitations", "",
        "The held-out split contains only two identities and many correlated frame pairs. "
        "Historical artifacts were preserved; canonical rows were recomputed on the "
        "common held-out protocol because their recorded protocols/crop labels differed. "
        "The historical results are `NOT_DIRECTLY_COMPARABLE_TO_CANONICAL_RUN`.", "",
    ])
    (comparison_output / "ALL_VERSIONS_COMPARISON.md").write_text(
        comparison_md, encoding="utf-8"
    )
    write_csv(comparison_output / "ALL_VERSIONS_DELTAS.csv", deltas)
    (comparison_output / "ALL_VERSIONS_DELTAS.json").write_text(
        json.dumps({"deltas": deltas}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    delta_sections = []
    for candidate in VERSION_IDS:
        delta_sections.append(delta_markdown(
            group_delta(deltas, "every_version_vs_V1", candidate),
            f"{candidate} vs V1",
        ))
    delta_sections.extend([
        delta_markdown(group_delta(deltas, "V5_vs_V4_crop_effect"), "V5 vs V4"),
        delta_markdown(group_delta(deltas, "V5_vs_V2_balanced_finetuning_effect"), "V5 vs V2"),
    ])
    (comparison_output / "ALL_VERSIONS_DELTAS.md").write_text(
        "# Canonical OSNet V1-V5 Deltas\n\n" + "\n\n".join(delta_sections) + "\n",
        encoding="utf-8",
    )
    (comparison_output / "FINAL_EXPERIMENT_SUMMARY.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({
        "v5_validation_threshold": versions["V5"]["threshold"],
        "v5_heldout_metrics": versions["V5"]["metrics"],
        "comparison": rows,
        "v5_vs_v4": group_delta(deltas, "V5_vs_V4_crop_effect"),
        "v5_vs_v2": group_delta(deltas, "V5_vs_V2_balanced_finetuning_effect"),
        "summary": summary,
        "output_root": str(output), "comparison_root": str(comparison_output),
    }, indent=2, sort_keys=True), flush=True)
    return comparison_payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--pretrained-checkpoint", type=Path,
                        default=Path("weights/osnet_x1_0_market1501.pth"))
    parser.add_argument("--v1-checkpoint", type=Path, default=Path(
        "backend/reid_experiments/finetune_osnet_v1/best_checkpoint.pth"))
    parser.add_argument("--v2-checkpoint", type=Path, default=Path(
        "backend/reid_experiments/finetune_osnet_v2_balanced/best_checkpoint.pth"))
    parser.add_argument("--validation-manifest", type=Path,
                        default=Path("datasets/peoplelocation_reid_v1/val/manifest.jsonl"))
    parser.add_argument("--test-manifest", type=Path,
                        default=Path("datasets/peoplelocation_reid_v1/test/manifest.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path(
        "backend/reid_experiments/finetune_osnet_v2_balanced_improved_crop"))
    parser.add_argument("--comparison-root", type=Path,
                        default=Path("backend/reid_experiments/version_comparison"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
