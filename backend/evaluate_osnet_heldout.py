"""Final offline held-out comparison of pretrained and fine-tuned OSNet x1.0."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchreid

from .finetune_osnet import (
    ManifestReidDataset,
    compute_roc_eer,
    read_manifest,
    validation_metrics,
    validation_transform,
)


EVALUATION_VERSION = "peoplelocation-heldout-evaluation-v1"
METRICS = (
    "rank_1", "rank_5", "mAP", "roc_auc", "eer", "accuracy",
    "precision", "recall", "f1", "same_id_mean_similarity",
    "different_id_mean_similarity", "similarity_gap",
)
HIGHER_IS_BETTER = set(METRICS) - {"eer", "different_id_mean_similarity"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def validate_peoplelocation_records(records, split):
    invalid = [
        record["sample_id"] for record in records
        if record.get("dataset_source") != "peoplelocation"
        or record.get("split") != split
        or not record.get("identity_key", "").startswith("peoplelocation:")
    ]
    if invalid:
        raise ValueError(
            f"{split} evaluation must contain only PeopleLocation: {invalid[:5]}"
        )


def build_finetuned_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "osnet_x1_0":
        raise ValueError("Fine-tuned checkpoint is not OSNet x1.0")
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=int(checkpoint["num_classes"]),
        loss="triplet", pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def build_pretrained_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=int(state["classifier.weight"].shape[0]),
        loss="triplet", pretrained=False,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def extract_embeddings(model, loader, device):
    features, identities, cameras, sample_ids = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = F.normalize(
                model(batch["image"].to(device, non_blocking=True)), p=2, dim=1
            )
            if output.shape[1] != 512 or not bool(torch.isfinite(output).all()):
                raise ValueError("Expected finite L2-normalized 512-D embeddings")
            features.append(output.cpu().numpy())
            identities.extend(batch["identity_key"])
            cameras.extend(batch["camera_id"])
            sample_ids.extend(batch["sample_id"])
    embeddings = np.concatenate(features)
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("Embedding L2 normalization validation failed")
    return embeddings, identities, cameras, sample_ids


def cross_camera_pairs(embeddings, identities, cameras):
    similarities = embeddings @ embeddings.T
    identities = np.asarray(identities)
    cameras = np.asarray(cameras)
    mask = np.triu(cameras[:, None] != cameras[None, :], k=1)
    left, right = np.nonzero(mask)
    return left, right, similarities[left, right], identities[left] == identities[right]


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def threshold_metrics(scores, labels, threshold):
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum(predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": safe_ratio(tp + tn, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn,
    }


def distribution(scores):
    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "std": float(values.std()), "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def hard_pair_examples(left, right, scores, labels, sample_ids, count=20):
    labels = np.asarray(labels, dtype=bool)
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    hard_positive = positives[np.argsort(scores[positives])[:count]]
    hard_negative = negatives[np.argsort(-scores[negatives])[:count]]

    def rows(indices):
        return [{
            "sample_a": sample_ids[int(left[index])],
            "sample_b": sample_ids[int(right[index])],
            "similarity": float(scores[index]),
        } for index in indices]

    return {
        "hard_positive_lowest_similarity": rows(hard_positive),
        "hard_negative_highest_similarity": rows(hard_negative),
    }


def score_model(embeddings, identities, cameras, sample_ids, threshold):
    retrieval = validation_metrics(embeddings, identities, cameras)
    left, right, scores, labels = cross_camera_pairs(
        embeddings, identities, cameras
    )
    roc_auc, eer, reporting_threshold = compute_roc_eer(scores, labels)
    same = distribution(scores[labels])
    different = distribution(scores[~labels])
    classification = threshold_metrics(scores, labels, threshold)
    metrics = {
        **retrieval,
        "roc_auc": roc_auc, "eer": eer,
        "test_eer_threshold_reporting_only_not_used_for_classification": (
            reporting_threshold
        ),
        "accuracy": classification["accuracy"],
        "precision": classification["precision"],
        "recall": classification["recall"], "f1": classification["f1"],
        "classification_counts": classification,
        "same_id": same, "different_id": different,
        "same_id_mean_similarity": same["mean"],
        "different_id_mean_similarity": different["mean"],
        "similarity_gap": same["mean"] - different["mean"],
        "skipped_queries": len(identities) - retrieval["valid_queries"],
        "hard_pairs": hard_pair_examples(left, right, scores, labels, sample_ids),
    }
    return metrics, (left, right, scores, labels)


def comparison_rows(baseline, finetuned):
    rows = []
    for metric in METRICS:
        before, after = float(baseline[metric]), float(finetuned[metric])
        delta = after - before
        if abs(delta) < 1e-12:
            effect = "unchanged"
        elif metric in HIGHER_IS_BETTER:
            effect = "improved" if delta > 0 else "degraded"
        else:
            effect = "improved" if delta < 0 else "degraded"
        rows.append({
            "metric": metric, "pretrained": before, "finetuned": after,
            "difference_finetuned_minus_pretrained": delta, "effect": effect,
        })
    return rows


def write_pair_scores(path, records, baseline_pairs, fine_pairs,
                      baseline_threshold, fine_threshold):
    left, right, baseline_scores, labels = baseline_pairs
    fine_left, fine_right, fine_scores, fine_labels = fine_pairs
    if not (np.array_equal(left, fine_left) and np.array_equal(right, fine_right)
            and np.array_equal(labels, fine_labels)):
        raise ValueError("Baseline/fine-tuned pair protocols differ")
    fields = [
        "sample_a", "sample_b", "identity_a", "identity_b", "camera_a",
        "camera_b", "same_identity", "pretrained_similarity",
        "finetuned_similarity", "pretrained_predicted_same",
        "finetuned_predicted_same", "pretrained_correct", "finetuned_correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index, (left_index, right_index) in enumerate(zip(left, right)):
            label = bool(labels[index])
            baseline_prediction = bool(baseline_scores[index] >= baseline_threshold)
            fine_prediction = bool(fine_scores[index] >= fine_threshold)
            writer.writerow({
                "sample_a": records[int(left_index)]["sample_id"],
                "sample_b": records[int(right_index)]["sample_id"],
                "identity_a": records[int(left_index)]["identity_key"],
                "identity_b": records[int(right_index)]["identity_key"],
                "camera_a": records[int(left_index)]["camera_id"],
                "camera_b": records[int(right_index)]["camera_id"],
                "same_identity": label,
                "pretrained_similarity": float(baseline_scores[index]),
                "finetuned_similarity": float(fine_scores[index]),
                "pretrained_predicted_same": baseline_prediction,
                "finetuned_predicted_same": fine_prediction,
                "pretrained_correct": baseline_prediction == label,
                "finetuned_correct": fine_prediction == label,
            })


def render_comparison_table(rows):
    lines = [
        "| Metric | Pretrained OSNet | Fine-tuned OSNet | Difference | Effect |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['pretrained']:.6f} | "
            f"{row['finetuned']:.6f} | "
            f"{row['difference_finetuned_minus_pretrained']:+.6f} | "
            f"{row['effect']} |"
        )
    return "\n".join(lines)


def make_conclusion(comparison):
    effects = {row["metric"]: row["effect"] for row in comparison}
    supporting = sum(
        effects[key] == "improved"
        for key in ("rank_1", "roc_auc", "eer", "similarity_gap")
    )
    if effects["mAP"] == "improved" and supporting >= 2:
        return (
            "Fine-tuning improved held-out PeopleLocation Re-ID on primary mAP "
            "and a majority of supporting identity-separation metrics. The held-out "
            "set has only two identities, so this supports continued guarded offline "
            "validation, not automatic production integration."
        )
    return (
        "Fine-tuning did not show sufficiently consistent held-out improvement across "
        "mAP and supporting identity-separation metrics. Do not integrate it into "
        "production based on this result."
    )


def report_text(config, baseline, fine, comparison, conclusion):
    threshold = config["thresholds"]
    return "\n".join([
        "# Final Held-Out OSNet Evaluation", "",
        f"Held-out manifest: {config['heldout_manifest']['path']}",
        f"Fine-tuned checkpoint: {config['finetuned_checkpoint']['path']}",
        "Protocol: all held-out samples are queries; other-camera samples are gallery.",
        "", "## Threshold policy", "",
        f"- Fine-tuned threshold: {threshold['finetuned']['value']:.9f}",
        f"- Pretrained threshold: {threshold['pretrained']['value']:.9f}",
        f"- Method: {threshold['finetuned']['method']}",
        f"- Source: {threshold['finetuned']['source_split']}",
        "- The held-out test was not used to tune either threshold.",
        "", "## Comparison", "", render_comparison_table(comparison), "",
        f"Valid/skipped queries: {fine['valid_queries']} / {fine['skipped_queries']}.",
        f"Fine-tuned false positives/false negatives: "
        f"{fine['classification_counts']['false_positives']} / "
        f"{fine['classification_counts']['false_negatives']}.",
        f"Pretrained false positives/false negatives: "
        f"{baseline['classification_counts']['false_positives']} / "
        f"{baseline['classification_counts']['false_negatives']}.",
        "", "## Conclusion", "", conclusion, "",
        "No retraining, test-set threshold tuning, checkpoint selection, or production "
        "integration was performed.", "",
    ])


def run(args):
    started = time.perf_counter()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    filenames = (
        "metrics.json", "pair_scores.csv", "evaluation_report.md",
        "baseline_comparison.json", "baseline_comparison.md",
    )
    existing = [output_root / name for name in filenames
                if (output_root / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite held-out outputs: "
            + ", ".join(str(path) for path in existing)
        )
    required = (
        args.finetuned_checkpoint, args.pretrained_checkpoint,
        args.test_manifest, args.validation_manifest, args.training_summary,
    )
    for path in required:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch_version={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"selected_device={device}", flush=True)
    print(f"gpu_name={gpu_name}", flush=True)

    fine_model, fine_checkpoint = build_finetuned_model(
        args.finetuned_checkpoint.resolve(), device
    )
    best_epoch = json.loads(
        args.training_summary.read_text(encoding="utf-8")
    )["best_epoch"]
    if fine_checkpoint["epoch"] != best_epoch:
        raise ValueError("Checkpoint is not the validation-selected best epoch")
    fine_threshold = float(fine_checkpoint["validation_metrics"]["eer_threshold"])

    test_records = read_manifest(args.test_manifest.resolve(), "test")
    validation_records = read_manifest(args.validation_manifest.resolve(), "val")
    validate_peoplelocation_records(test_records, "test")
    validate_peoplelocation_records(validation_records, "val")
    test_loader = DataLoader(
        ManifestReidDataset(test_records, project_root, validation_transform()),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ManifestReidDataset(
            validation_records, project_root, validation_transform()
        ),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    baseline_model = build_pretrained_model(
        args.pretrained_checkpoint.resolve(), device
    )
    baseline_val = extract_embeddings(baseline_model, validation_loader, device)
    _, _, baseline_val_scores, baseline_val_labels = cross_camera_pairs(
        baseline_val[0], baseline_val[1], baseline_val[2]
    )
    _, _, baseline_threshold = compute_roc_eer(
        baseline_val_scores, baseline_val_labels
    )
    baseline_test = extract_embeddings(baseline_model, test_loader, device)
    baseline_metrics, baseline_pairs = score_model(
        *baseline_test, baseline_threshold
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fine_test = extract_embeddings(fine_model, test_loader, device)
    fine_metrics, fine_pairs = score_model(*fine_test, fine_threshold)
    comparison = comparison_rows(baseline_metrics, fine_metrics)
    final_conclusion = make_conclusion(comparison)

    threshold_method = "cross-camera validation EER operating point"
    environment = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(), "torch_version": torch.__version__,
        "torchreid_version": package_version("torchreid"),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(device), "gpu_name": gpu_name,
    }
    config = {
        "evaluation_version": EVALUATION_VERSION,
        "architecture": "osnet_x1_0",
        "preprocessing": {
            "color_space": "RGB", "height": 256, "width": 128,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "embedding_dimension": 512, "l2_normalized": True,
            "crop_policy": "raw_gt_bbox_no_margin",
        },
        "retrieval_protocol": (
            "all held-out samples as queries; other-camera gallery; valid queries only"
        ),
        "verification_protocol": "unique unordered cross-camera pairs",
        "heldout_manifest": {
            "path": args.test_manifest.resolve().relative_to(project_root).as_posix(),
            "sha256": sha256_file(args.test_manifest.resolve()),
            "images": len(test_records),
            "identities": len({row["identity_key"] for row in test_records}),
        },
        "validation_manifest": {
            "path": args.validation_manifest.resolve().relative_to(
                project_root
            ).as_posix(),
            "sha256": sha256_file(args.validation_manifest.resolve()),
            "used_only_for_threshold_selection": True,
        },
        "finetuned_checkpoint": {
            "path": args.finetuned_checkpoint.resolve().relative_to(
                project_root
            ).as_posix(),
            "sha256": sha256_file(args.finetuned_checkpoint.resolve()),
            "epoch": fine_checkpoint["epoch"],
            "selected_on_validation_only": True,
        },
        "pretrained_checkpoint": {
            "path": args.pretrained_checkpoint.resolve().relative_to(
                project_root
            ).as_posix(),
            "sha256": sha256_file(args.pretrained_checkpoint.resolve()),
        },
        "thresholds": {
            "finetuned": {
                "value": fine_threshold, "method": threshold_method,
                "source_split": "PeopleLocation validation",
                "source": "best checkpoint saved validation metrics",
            },
            "pretrained": {
                "value": baseline_threshold, "method": threshold_method,
                "source_split": "PeopleLocation validation",
                "source": "pretrained model evaluated on validation split",
            },
            "heldout_test_used_for_threshold_selection": False,
        },
        "environment": environment,
    }
    result = {
        "evaluation_config": config, "finetuned": fine_metrics,
        "pretrained": baseline_metrics, "comparison": comparison,
        "conclusion": final_conclusion,
        "evaluation_runtime_seconds": time.perf_counter() - started,
        "retrained": False, "production_integrated": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(
        json.dumps({
            "evaluation_config": config,
            "finetuned_heldout_metrics": fine_metrics,
            "evaluation_runtime_seconds": result["evaluation_runtime_seconds"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_pair_scores(
        output_root / "pair_scores.csv", test_records,
        baseline_pairs, fine_pairs, baseline_threshold, fine_threshold,
    )
    (output_root / "baseline_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = report_text(
        config, baseline_metrics, fine_metrics, comparison, final_conclusion
    )
    (output_root / "evaluation_report.md").write_text(
        report, encoding="utf-8"
    )
    (output_root / "baseline_comparison.md").write_text(
        "# Pretrained vs Fine-Tuned OSNet\n\n"
        + render_comparison_table(comparison) + "\n\n" + final_conclusion + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "finetuned": {key: fine_metrics[key] for key in METRICS},
        "pretrained": {key: baseline_metrics[key] for key in METRICS},
        "thresholds": config["thresholds"], "comparison": comparison,
        "conclusion": final_conclusion,
        "valid_queries": fine_metrics["valid_queries"],
        "skipped_queries": fine_metrics["skipped_queries"],
    }, indent=2, sort_keys=True), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetuned-checkpoint", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v1/best_checkpoint.pth"
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint", type=Path,
        default=Path("weights/osnet_x1_0_market1501.pth"),
    )
    parser.add_argument(
        "--test-manifest", type=Path,
        default=Path("datasets/peoplelocation_reid_v1/test/manifest.jsonl"),
    )
    parser.add_argument(
        "--validation-manifest", type=Path,
        default=Path("datasets/peoplelocation_reid_v1/val/manifest.jsonl"),
    )
    parser.add_argument(
        "--training-summary", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v1/training_summary.json"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v1/heldout_test"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
