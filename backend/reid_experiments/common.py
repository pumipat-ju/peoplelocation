"""Shared protocol and artifact writer for the four offline Re-ID experiments.

This module deliberately consumes frames from the master manifest.  It does not
open videos, start capture workers, or alter the production version registry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from backend.baseline_metrics import classification_metrics, confusion_at_threshold
from backend.dump_reid_debug import close_runtime, load_production_runtime
from backend.evaluate_reid_verification import build_pair_rows, evaluate_pairs
from backend.reid_config import (
    OSNET_ARCHITECTURE,
    OSNET_DEFAULT_CHECKPOINT_NAME,
    osnet_preprocessing_metadata,
)
from backend.reid_similarity_matrix import l2_normalize_rows
from backend.run_reid_crop_ablation import (
    extract_embeddings,
    load_manifest,
    select_samples,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "reid_dataset" / "master_manifest_v1.jsonl"
RESULTS_ROOT = PROJECT_ROOT / "reid_experiment_results"
PRETRAINED_CHECKPOINT = PROJECT_ROOT / "weights" / OSNET_DEFAULT_CHECKPOINT_NAME
FINETUNED_CHECKPOINT = PROJECT_ROOT / "weights" / "osnet_x1_0_finetuned.pth"
EXPERIMENT_SCHEMA_VERSION = "peoplelocation-reid-experiment-v1"
PAIR_RULE = "unique unordered cross-camera pairs; self-pairs excluded"
THRESHOLD_PROTOCOL = "Youden J maximum on the evaluated cross-camera pair set"
SAMPLES_PER_IDENTITY_CAMERA = 20

ORIGINAL_CROP = "Original/current production crop margins"
IMPROVED_CROP = "Raw GT bbox (no margin), selected by offline crop ablation"


class FinetunedCheckpointNotAvailable(RuntimeError):
    """Raised instead of silently falling back to pre-trained weights."""


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    display_name: str
    output_directory_name: str
    checkpoint_kind: str
    checkpoint_path: Path
    crop_policy: str

    @property
    def uses_improved_crop(self) -> bool:
        return self.crop_policy == IMPROVED_CROP

    @property
    def is_finetuned(self) -> bool:
        return self.checkpoint_kind == "fine-tuned"


SPECS = {
    "01_pretrained_osnet": ExperimentSpec(
        "01_pretrained_osnet",
        "Pre-trained OSNet x1.0",
        "01_pretrained_osnet",
        "pre-trained",
        PRETRAINED_CHECKPOINT,
        ORIGINAL_CROP,
    ),
    "02_pretrained_osnet_improved_crop": ExperimentSpec(
        "02_pretrained_osnet_improved_crop",
        "Pre-trained OSNet x1.0 + Improved Crop",
        "02_pretrained_osnet_improved_crop",
        "pre-trained",
        PRETRAINED_CHECKPOINT,
        IMPROVED_CROP,
    ),
    "03_finetuned_osnet": ExperimentSpec(
        "03_finetuned_osnet",
        "Fine-tuned OSNet x1.0",
        "03_finetuned_osnet",
        "fine-tuned",
        FINETUNED_CHECKPOINT,
        ORIGINAL_CROP,
    ),
    "04_finetuned_osnet_improved_crop": ExperimentSpec(
        "04_finetuned_osnet_improved_crop",
        "Fine-tuned OSNet x1.0 + Improved Crop",
        "04_finetuned_osnet_improved_crop",
        "fine-tuned",
        FINETUNED_CHECKPOINT,
        IMPROVED_CROP,
    ),
}


def raw_gt_crop(frame, record):
    x1, y1, x2, y2 = (int(value) for value in record["bbox_xyxy"])
    crop = frame[y1:y2, x1:x2]
    return crop if crop is not None and crop.size else None


def production_crop(runtime, frame, record):
    return runtime.extract_person_crop(
        frame, *(int(value) for value in record["bbox_xyxy"])
    )


def validate_checkpoint(spec, checkpoint):
    checkpoint = checkpoint.resolve()
    if spec.is_finetuned and not checkpoint.is_file():
        raise FinetunedCheckpointNotAvailable(
            "FINETUNED_CHECKPOINT_NOT_AVAILABLE: "
            f"expected a real fine-tuned osnet_x1_0 checkpoint at {checkpoint} "
            "(or pass --checkpoint); pre-trained fallback is forbidden"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CHECKPOINT_NOT_AVAILABLE: {checkpoint}")
    if not spec.is_finetuned and checkpoint != PRETRAINED_CHECKPOINT.resolve():
        raise ValueError(
            "PRETRAINED_CHECKPOINT_REQUIRED: this runner is pinned to "
            f"{PRETRAINED_CHECKPOINT.resolve()}"
        )
    if spec.is_finetuned and checkpoint == PRETRAINED_CHECKPOINT.resolve():
        raise ValueError(
            "FINETUNED_CHECKPOINT_REQUIRED: refusing the pre-trained checkpoint"
        )
    return checkpoint


def prepare_common_samples(manifest_records, runtime):
    """Use the intersection valid for both crop policies to keep comparisons fair."""
    prepared = []
    rejections = {}
    for record in manifest_records:
        frame = cv2.imread(str(PROJECT_ROOT / record["source_image"]), cv2.IMREAD_COLOR)
        reason = None
        baseline = improved = None
        if frame is None:
            reason = "missing_or_unreadable_image"
        else:
            baseline = production_crop(runtime, frame, record)
            improved = raw_gt_crop(frame, record)
            if baseline is None or baseline.size == 0:
                reason = "empty_original_crop"
            elif improved is None or improved.size == 0:
                reason = "empty_improved_crop"
            elif min(baseline.shape[:2]) < runtime.REID_MIN_CROP_SIZE:
                reason = "original_crop_below_minimum"
            elif min(improved.shape[:2]) < runtime.REID_MIN_CROP_SIZE:
                reason = "improved_crop_below_minimum"
        if reason:
            rejections[reason] = rejections.get(reason, 0) + 1
            continue
        prepared.append((record, baseline, improved))
    return prepared, dict(sorted(rejections.items()))


def retrieval_metrics(similarities, records):
    """Standard cross-camera CMC and AP, averaged over valid queries.

    Every sample is a query; samples from other cameras form its gallery. Queries
    without an identity match in that gallery are excluded.
    """
    reciprocal_ranks = []
    average_precisions = []
    for query_index, query in enumerate(records):
        gallery = [
            index for index, candidate in enumerate(records)
            if candidate["camera"] != query["camera"]
        ]
        ranked = sorted(
            gallery,
            key=lambda index: (-float(similarities[query_index, index]),
                               str(records[index]["sample_id"])),
        )
        relevant = [
            index for index in ranked
            if records[index]["dataset_identity_key"] == query["dataset_identity_key"]
        ]
        if not relevant:
            continue
        hits = 0
        precision_sum = 0.0
        first_rank = None
        relevant_set = set(relevant)
        for rank, index in enumerate(ranked, start=1):
            if index not in relevant_set:
                continue
            hits += 1
            first_rank = rank if first_rank is None else first_rank
            precision_sum += hits / rank
        reciprocal_ranks.append(first_rank)
        average_precisions.append(precision_sum / len(relevant))
    if not reciprocal_ranks:
        return {
            "rank_1": "NOT_AVAILABLE",
            "rank_5": "NOT_AVAILABLE",
            "mAP": "NOT_AVAILABLE",
            "valid_query_count": 0,
            "reason": "no query has a same-identity sample in another camera",
        }
    ranks = np.asarray(reciprocal_ranks)
    return {
        "rank_1": float(np.mean(ranks <= 1)),
        "rank_5": float(np.mean(ranks <= 5)),
        "mAP": float(np.mean(average_precisions)),
        "valid_query_count": len(reciprocal_ranks),
        "protocol": "all samples as queries; other-camera gallery; valid queries only",
    }


def _pair_fieldnames():
    return [
        "sample_a", "sample_b", "gt_person_a", "gt_person_b", "camera_a",
        "camera_b", "frame_a", "frame_b", "same_gt_identity",
        "camera_relation", "similarity", "threshold", "predicted_same_identity",
        "correct",
    ]


def write_pair_scores(path, pairs, threshold):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=_pair_fieldnames())
        writer.writeheader()
        for pair in pairs:
            predicted = float(pair["similarity"]) >= threshold
            writer.writerow({
                **pair,
                "threshold": threshold,
                "predicted_same_identity": predicted,
                "correct": predicted == bool(pair["same_gt_identity"]),
            })


def render_report(config, metrics):
    def value(name):
        item = metrics[name]
        return item if isinstance(item, str) else f"{item:.6f}"

    return "\n".join([
        f"# {config['display_name']}",
        "",
        f"Status: **{config['status']}**  ",
        f"Architecture: `{config['architecture']}`  ",
        f"Checkpoint: `{config['checkpoint']}`  ",
        f"Checkpoint SHA-256: `{config['checkpoint_sha256']}`  ",
        f"Crop policy: `{config['crop_policy']}`  ",
        f"Master manifest SHA-256: `{config['dataset_manifest_sha256']}`  ",
        f"Samples: `{config['sample_count']}`  ",
        f"Threshold: `{config['threshold']:.9f}` ({config['threshold_protocol']})",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *[f"| {label} | {value(key)} |" for label, key in (
            ("Same-ID Mean", "same_id_mean"),
            ("Different-ID Mean", "different_id_mean"),
            ("Similarity Gap", "similarity_gap"),
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
            ("ROC-AUC", "roc_auc"),
            ("EER", "eer"),
            ("Rank-1", "rank_1"),
            ("Rank-5", "rank_5"),
            ("mAP", "mAP"),
        )],
        "",
        "## Protocol",
        "",
        f"- Pairs: {PAIR_RULE}.",
        "- Labels: `dataset_identity_key` from the shared master manifest.",
        "- Embeddings are L2-normalized and scored by cosine similarity.",
        "- The sample set is the deterministic intersection accepted by both crop policies.",
        "- Global-ID metrics: `NOT_AVAILABLE` in this embedding-only phase; no full-system replay was run.",
        "",
    ])


def evaluate_experiment(spec, manifest_path, checkpoint, output_directory,
                        device="cpu", batch_size=32, overwrite=False):
    checkpoint = validate_checkpoint(spec, checkpoint)
    manifest_path = manifest_path.resolve()
    targets = tuple(output_directory / name for name in (
        "metrics.json", "config.json", "pair_scores.csv", "evaluation_report.md"
    ))
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment artifacts in {output_directory}"
        )

    manifest_records = load_manifest(manifest_path)
    selected_records = select_samples(
        manifest_records, SAMPLES_PER_IDENTITY_CAMERA
    )
    runtime = load_production_runtime(checkpoint, device)
    try:
        runtime_status = runtime.REID_RUNTIME_STATUS
        if not runtime_status.get("checkpoint_loaded"):
            raise RuntimeError(
                "OSNET_RUNTIME_NOT_AVAILABLE: the requested OSNet checkpoint was "
                "not loaded; refusing lightweight fallback; "
                f"reason={runtime_status.get('error')}"
            )
        prepared, rejections = prepare_common_samples(selected_records, runtime)
        records = [item[0] for item in prepared]
        crops = [item[2] if spec.uses_improved_crop else item[1] for item in prepared]
        if not records:
            raise ValueError("shared crop protocol produced no evaluable samples")
        embeddings, input_norms = extract_embeddings(
            runtime.appearance_extractor, crops, batch_size
        )
    finally:
        close_runtime(runtime)

    embeddings, _ = l2_normalize_rows(embeddings)
    similarities = embeddings @ embeddings.T
    pair_records = [
        {**record, "ground_truth_person_id": record["dataset_identity_key"]}
        for record in records
    ]
    all_pairs = build_pair_rows(similarities, pair_records)
    pairs = [pair for pair in all_pairs if pair["camera_relation"] == "cross_camera"]
    verification = evaluate_pairs(pairs)
    threshold = verification["best_threshold"]
    scores = np.asarray([pair["similarity"] for pair in pairs])
    labels = np.asarray([int(pair["same_gt_identity"]) for pair in pairs])
    confusion = confusion_at_threshold(scores, labels, threshold)
    classification = classification_metrics(confusion)
    retrieval = retrieval_metrics(similarities, records)

    metrics = {
        "same_id_mean": verification["same_id"]["mean"],
        "different_id_mean": verification["different_id"]["mean"],
        "similarity_gap": verification["similarity_gap"],
        **classification,
        "roc_auc": verification["roc_auc"],
        "eer": verification["eer"],
        "rank_1": retrieval["rank_1"],
        "rank_5": retrieval["rank_5"],
        "mAP": retrieval["mAP"],
        "confusion_matrix": confusion,
        "positive_pair_count": verification["same_id"]["count"],
        "negative_pair_count": verification["different_id"]["count"],
        "retrieval": retrieval,
        "global_id_metrics": {
            "status": "NOT_AVAILABLE",
            "reason": "embedding-only evaluation; full-system replay deferred",
        },
    }
    config = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": spec.experiment_id,
        "display_name": spec.display_name,
        "output_directory_name": spec.output_directory_name,
        "checkpoint_kind": spec.checkpoint_kind,
        "status": "AVAILABLE",
        "architecture": OSNET_ARCHITECTURE,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "crop_policy": spec.crop_policy,
        "preprocessing": osnet_preprocessing_metadata(),
        "l2_normalization": True,
        "similarity": "cosine (dot product after L2 normalization)",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "evaluation_split": "deterministic master-manifest evaluation subset",
        "samples_per_identity_camera": SAMPLES_PER_IDENTITY_CAMERA,
        "identity_label": "dataset_identity_key",
        "pair_rule": PAIR_RULE,
        "threshold": threshold,
        "threshold_protocol": THRESHOLD_PROTOCOL,
        "sample_count": len(records),
        "manifest_record_count": len(manifest_records),
        "rejected_sample_count": sum(rejections.values()),
        "rejection_reasons": rejections,
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(record["sample_id"] for record in records).encode("utf-8")
        ).hexdigest(),
        "embedding_input_norm_min": float(np.min(input_norms)),
        "embedding_input_norm_max": float(np.max(input_norms)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_version": EXPERIMENT_SCHEMA_VERSION,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_pair_scores(output_directory / "pair_scores.csv", pairs, threshold)
    (output_directory / "evaluation_report.md").write_text(
        render_report(config, metrics), encoding="utf-8"
    )
    return config, metrics


def run_cli(spec):
    parser = argparse.ArgumentParser(description=spec.display_name)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=spec.checkpoint_path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=RESULTS_ROOT / spec.output_directory_name,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    try:
        config, _ = evaluate_experiment(
            spec,
            arguments.manifest,
            arguments.checkpoint,
            arguments.output_dir,
            arguments.device,
            arguments.batch_size,
            arguments.overwrite,
        )
    except FinetunedCheckpointNotAvailable as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    print(f"{spec.display_name}: completed {config['sample_count']} samples")
    print(f"output: {arguments.output_dir.resolve()}")


def runner_main(experiment_id):
    run_cli(SPECS[experiment_id])
