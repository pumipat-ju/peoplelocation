"""Offline three-model held-out evaluation for domain-balanced OSNet."""

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_osnet_heldout import (
    METRICS,
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
from .finetune_osnet import (
    ManifestReidDataset,
    read_manifest,
    validation_transform,
)


MODEL_KEYS = ("pretrained", "finetuned_v1", "balanced_v2")
DISPLAY_METRICS = METRICS + (
    "false_positives", "false_negatives", "valid_queries", "skipped_queries",
)
LOWER_IS_BETTER = {
    "eer", "different_id_mean_similarity", "false_positives",
    "false_negatives", "skipped_queries",
}


def flat_metrics(metrics):
    return {
        **{key: metrics[key] for key in METRICS},
        "false_positives": metrics["classification_counts"]["false_positives"],
        "false_negatives": metrics["classification_counts"]["false_negatives"],
        "valid_queries": metrics["valid_queries"],
        "skipped_queries": metrics["skipped_queries"],
    }


def comparison_rows(results):
    flattened = {model: flat_metrics(metrics) for model, metrics in results.items()}
    rows = []
    for metric in DISPLAY_METRICS:
        values = {model: float(flattened[model][metric]) for model in MODEL_KEYS}
        target = min(values.values()) if metric in LOWER_IS_BETTER else max(values.values())
        best = [model for model, value in values.items()
                if abs(value - target) < 1e-12]
        rows.append({
            "metric": metric,
            **values,
            "best": best,
            "balanced_v2_minus_pretrained": (
                values["balanced_v2"] - values["pretrained"]
            ),
            "balanced_v2_minus_finetuned_v1": (
                values["balanced_v2"] - values["finetuned_v1"]
            ),
        })
    return rows


def table(rows):
    lines = [
        "| Metric | Pretrained | Fine-tuned v1 | Balanced v2 | Best |",
        "|---|---:|---:|---:|---|",
    ]
    labels = {
        "pretrained": "Pretrained", "finetuned_v1": "Fine-tuned v1",
        "balanced_v2": "Balanced v2",
    }
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['pretrained']:.6f} | "
            f"{row['finetuned_v1']:.6f} | {row['balanced_v2']:.6f} | "
            f"{', '.join(labels[key] for key in row['best'])} |"
        )
    return "\n".join(lines)


def conclusion(rows):
    by_metric = {row["metric"]: row for row in rows}
    retrieval = all(
        "balanced_v2" in by_metric[key]["best"]
        for key in ("rank_1", "rank_5", "mAP")
    )
    verification = all(
        "balanced_v2" in by_metric[key]["best"]
        for key in ("roc_auc", "eer", "similarity_gap")
    )
    if retrieval and verification:
        return (
            "Balanced v2 improved both retrieval and core verification performance "
            "on this held-out split. Because the test contains only two identities, "
            "the result is evidence for further guarded evaluation, not authorization "
            "for production integration."
        ), True, True
    return (
        "Balanced v2 did not improve both retrieval and core verification performance "
        "consistently on this held-out split. Do not integrate it into production."
    ), retrieval, verification


def write_pair_scores(path, records, pairs_by_model, thresholds):
    reference = pairs_by_model["pretrained"]
    left, right, _, labels = reference
    for pairs in pairs_by_model.values():
        if not (np.array_equal(left, pairs[0]) and np.array_equal(right, pairs[1])
                and np.array_equal(labels, pairs[3])):
            raise ValueError("Pair protocols differ between models")
    fields = [
        "sample_a", "sample_b", "identity_a", "identity_b", "camera_a",
        "camera_b", "same_identity",
    ]
    for model in MODEL_KEYS:
        fields.extend([
            f"{model}_similarity", f"{model}_predicted_same",
            f"{model}_correct",
        ])
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index, (left_index, right_index) in enumerate(zip(left, right)):
            label = bool(labels[index])
            row = {
                "sample_a": records[int(left_index)]["sample_id"],
                "sample_b": records[int(right_index)]["sample_id"],
                "identity_a": records[int(left_index)]["identity_key"],
                "identity_b": records[int(right_index)]["identity_key"],
                "camera_a": records[int(left_index)]["camera_id"],
                "camera_b": records[int(right_index)]["camera_id"],
                "same_identity": label,
            }
            for model in MODEL_KEYS:
                score = float(pairs_by_model[model][2][index])
                predicted = score >= thresholds[model]["value"]
                row.update({
                    f"{model}_similarity": score,
                    f"{model}_predicted_same": predicted,
                    f"{model}_correct": predicted == label,
                })
            writer.writerow(row)


def report(config, rows, final_conclusion, retrieval_improved,
           verification_improved):
    thresholds = config["thresholds"]
    return "\n".join([
        "# Balanced v2 Final Held-Out Evaluation", "",
        "Protocol and preprocessing are identical to Prompt 05.",
        "Thresholds were selected on PeopleLocation validation only:",
        f"- Pretrained: {thresholds['pretrained']['value']:.9f}",
        f"- Fine-tuned v1: {thresholds['finetuned_v1']['value']:.9f}",
        f"- Balanced v2: {thresholds['balanced_v2']['value']:.9f}",
        "- Method: cross-camera validation EER operating point", "",
        table(rows), "", "## Conclusion", "", final_conclusion,
        f"Retrieval improved: {retrieval_improved}.  ",
        f"Core verification improved: {verification_improved}.", "",
        "No retraining, test threshold tuning, checkpoint selection, or production "
        "integration was performed.", "",
    ])


def run(args):
    started = time.perf_counter()
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_names = (
        "metrics.json", "pair_scores.csv", "evaluation_report.md",
        "model_comparison.json", "model_comparison.md",
    )
    existing = [output_root / name for name in output_names
                if (output_root / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite outputs: " + ", ".join(map(str, existing))
        )
    required = (
        args.pretrained_checkpoint, args.v1_checkpoint, args.v2_checkpoint,
        args.v1_summary, args.v2_summary, args.test_manifest,
        args.validation_manifest,
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

    test_records = read_manifest(args.test_manifest.resolve(), "test")
    val_records = read_manifest(args.validation_manifest.resolve(), "val")
    validate_peoplelocation_records(test_records, "test")
    validate_peoplelocation_records(val_records, "val")
    test_loader = DataLoader(
        ManifestReidDataset(test_records, project_root, validation_transform()),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ManifestReidDataset(val_records, project_root, validation_transform()),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    v1_model, v1_checkpoint = build_finetuned_model(
        args.v1_checkpoint.resolve(), device
    )
    v2_model, v2_checkpoint = build_finetuned_model(
        args.v2_checkpoint.resolve(), device
    )
    v1_best_epoch = json.loads(args.v1_summary.read_text(encoding="utf-8"))[
        "best_epoch"
    ]
    v2_best_epoch = json.loads(args.v2_summary.read_text(encoding="utf-8"))[
        "best_epoch"
    ]
    if v1_checkpoint["epoch"] != v1_best_epoch:
        raise ValueError("v1 checkpoint is not its validation-selected best")
    if v2_checkpoint["epoch"] != v2_best_epoch:
        raise ValueError("v2 checkpoint is not its validation-selected best")
    thresholds = {
        "finetuned_v1": {
            "value": float(v1_checkpoint["validation_metrics"]["eer_threshold"]),
            "method": "cross-camera validation EER operating point",
            "source_split": "PeopleLocation validation",
        },
        "balanced_v2": {
            "value": float(v2_checkpoint["validation_metrics"]["eer_threshold"]),
            "method": "cross-camera validation EER operating point",
            "source_split": "PeopleLocation validation",
        },
    }

    pretrained_model = build_pretrained_model(
        args.pretrained_checkpoint.resolve(), device
    )
    pretrained_val = extract_embeddings(pretrained_model, val_loader, device)
    _, _, val_scores, val_labels = cross_camera_pairs(
        pretrained_val[0], pretrained_val[1], pretrained_val[2]
    )
    _, _, pretrained_threshold = compute_roc_eer(val_scores, val_labels)
    thresholds["pretrained"] = {
        "value": pretrained_threshold,
        "method": "cross-camera validation EER operating point",
        "source_split": "PeopleLocation validation",
    }

    models = {
        "pretrained": pretrained_model,
        "finetuned_v1": v1_model,
        "balanced_v2": v2_model,
    }
    results = {}
    pairs_by_model = {}
    for model_key in MODEL_KEYS:
        embeddings = extract_embeddings(models[model_key], test_loader, device)
        results[model_key], pairs_by_model[model_key] = score_model(
            *embeddings, thresholds[model_key]["value"]
        )
        print(
            f"model={model_key} mAP={results[model_key]['mAP']:.6f} "
            f"rank1={results[model_key]['rank_1']:.6f} "
            f"auc={results[model_key]['roc_auc']:.6f} "
            f"eer={results[model_key]['eer']:.6f}", flush=True,
        )
    rows = comparison_rows(results)
    final_conclusion, retrieval_improved, verification_improved = conclusion(rows)
    config = {
        "evaluation_version": "peoplelocation-balanced-v2-heldout-v1",
        "protocol_reference": "Prompt 05 exact metric and preprocessing code",
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
        "checkpoints": {
            "pretrained": {
                "path": args.pretrained_checkpoint.resolve().relative_to(
                    project_root
                ).as_posix(),
                "sha256": sha256_file(args.pretrained_checkpoint.resolve()),
            },
            "finetuned_v1": {
                "path": args.v1_checkpoint.resolve().relative_to(
                    project_root
                ).as_posix(),
                "sha256": sha256_file(args.v1_checkpoint.resolve()),
                "epoch": v1_checkpoint["epoch"],
            },
            "balanced_v2": {
                "path": args.v2_checkpoint.resolve().relative_to(
                    project_root
                ).as_posix(),
                "sha256": sha256_file(args.v2_checkpoint.resolve()),
                "epoch": v2_checkpoint["epoch"],
            },
        },
        "thresholds": {
            **thresholds, "heldout_test_used_for_threshold_selection": False,
        },
        "environment": {
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(), "torch_version": torch.__version__,
            "torchreid_version": package_version("torchreid"),
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "selected_device": str(device), "gpu_name": gpu_name,
        },
    }
    payload = {
        "evaluation_config": config, "models": results,
        "comparison": rows, "conclusion": final_conclusion,
        "retrieval_improved": retrieval_improved,
        "verification_improved": verification_improved,
        "evaluation_runtime_seconds": time.perf_counter() - started,
        "retrained": False, "production_integrated": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(
        json.dumps({
            "evaluation_config": config,
            "balanced_v2_heldout_metrics": results["balanced_v2"],
            "evaluation_runtime_seconds": payload["evaluation_runtime_seconds"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_pair_scores(
        output_root / "pair_scores.csv", test_records,
        pairs_by_model, thresholds,
    )
    (output_root / "model_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_text = report(
        config, rows, final_conclusion, retrieval_improved,
        verification_improved,
    )
    (output_root / "evaluation_report.md").write_text(
        report_text, encoding="utf-8"
    )
    (output_root / "model_comparison.md").write_text(
        "# Three-Model Held-Out Comparison\n\n" + table(rows) + "\n\n"
        + final_conclusion + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "models": {key: flat_metrics(results[key]) for key in MODEL_KEYS},
        "thresholds": thresholds, "comparison": rows,
        "retrieval_improved": retrieval_improved,
        "verification_improved": verification_improved,
        "conclusion": final_conclusion,
    }, indent=2, sort_keys=True), flush=True)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained-checkpoint", type=Path,
        default=Path("weights/osnet_x1_0_market1501.pth"),
    )
    parser.add_argument(
        "--v1-checkpoint", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v1/best_checkpoint.pth"
        ),
    )
    parser.add_argument(
        "--v2-checkpoint", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v2_balanced/"
            "best_checkpoint.pth"
        ),
    )
    parser.add_argument(
        "--v1-summary", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v1/training_summary.json"
        ),
    )
    parser.add_argument(
        "--v2-summary", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v2_balanced/"
            "training_summary.json"
        ),
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
        "--output-root", type=Path,
        default=Path(
            "backend/reid_experiments/finetune_osnet_v2_balanced/heldout_test"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
