"""Offline ROC-AUC/EER evaluation for canonical dumped Re-ID embeddings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .reid_similarity_matrix import (
        distribution_stats,
        l2_normalize_rows,
        load_dump_records,
        select_records,
    )
except ImportError:
    from reid_similarity_matrix import (
        distribution_stats,
        l2_normalize_rows,
        load_dump_records,
        select_records,
    )


BASELINE = {
    "same_id_mean": 0.8017,
    "different_id_mean": 0.7915,
    "similarity_gap": 0.0102,
    "roc_auc": 0.5350,
    "eer": 0.4648,
}


def build_pair_rows(similarity_matrix, records):
    matrix = np.asarray(similarity_matrix, dtype=np.float64)
    if matrix.shape != (len(records), len(records)):
        raise ValueError("similarity matrix shape does not match records")
    pairs = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left = records[left_index]
            right = records[right_index]
            same_identity = (
                str(left["ground_truth_person_id"])
                == str(right["ground_truth_person_id"])
            )
            pairs.append({
                "sample_a": left["sample_id"],
                "sample_b": right["sample_id"],
                "gt_person_a": str(left["ground_truth_person_id"]),
                "gt_person_b": str(right["ground_truth_person_id"]),
                "camera_a": left["camera"],
                "camera_b": right["camera"],
                "frame_a": left.get("frame_index"),
                "frame_b": right.get("frame_index"),
                "same_gt_identity": same_identity,
                "camera_relation": (
                    "same_camera" if left["camera"] == right["camera"]
                    else "cross_camera"
                ),
                "similarity": float(matrix[left_index, right_index]),
            })
    return pairs


def compute_roc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if scores.ndim != 1 or labels.ndim != 1 or scores.size != labels.size:
        raise ValueError("scores and labels must be same-length 1-D arrays")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires positive and negative pairs")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    thresholds = [float("inf")]
    true_positive_rates = [0.0]
    false_positive_rates = [0.0]
    true_positives = 0
    false_positives = 0
    index = 0
    while index < scores.size:
        threshold = sorted_scores[index]
        next_index = index
        while next_index < scores.size and sorted_scores[next_index] == threshold:
            if sorted_labels[next_index] == 1:
                true_positives += 1
            else:
                false_positives += 1
            next_index += 1
        thresholds.append(float(threshold))
        true_positive_rates.append(true_positives / positives)
        false_positive_rates.append(false_positives / negatives)
        index = next_index

    fpr = np.asarray(false_positive_rates, dtype=np.float64)
    tpr = np.asarray(true_positive_rates, dtype=np.float64)
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    auc = float(trapezoid(tpr, fpr))
    return fpr, tpr, threshold_array, auc


def compute_eer(fpr, tpr, thresholds):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    false_negative_rates = 1.0 - tpr
    difference = fpr - false_negative_rates

    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-12))
    if exact.size:
        index = int(exact[0])
        return float(fpr[index]), float(thresholds[index])

    for index in range(len(difference) - 1):
        if difference[index] < 0.0 < difference[index + 1]:
            fraction = -difference[index] / (difference[index + 1] - difference[index])
            eer = fpr[index] + fraction * (fpr[index + 1] - fpr[index])
            left_threshold = thresholds[index]
            right_threshold = thresholds[index + 1]
            if np.isfinite(left_threshold) and np.isfinite(right_threshold):
                threshold = left_threshold + fraction * (right_threshold - left_threshold)
            else:
                threshold = right_threshold
            return float(eer), float(threshold)

    index = int(np.argmin(np.abs(difference)))
    eer = (fpr[index] + false_negative_rates[index]) / 2.0
    return float(eer), float(thresholds[index])


def evaluate_pairs(pairs):
    if not pairs:
        raise ValueError("no verification pairs selected")
    scores = np.asarray([pair["similarity"] for pair in pairs], dtype=np.float64)
    labels = np.asarray([int(pair["same_gt_identity"]) for pair in pairs], dtype=np.int8)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    fpr, tpr, thresholds, auc = compute_roc(scores, labels)
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)
    youden = tpr - fpr
    best_index = int(np.argmax(youden))
    same_stats = distribution_stats(positive_scores)
    different_stats = distribution_stats(negative_scores)
    return {
        "same_id": same_stats,
        "different_id": different_stats,
        "similarity_gap": same_stats["mean"] - different_stats["mean"],
        "roc_auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eer_threshold_nearest_roc_point": float(
            thresholds[np.argmin(np.abs(fpr - (1.0 - tpr)))]
        ),
        "best_threshold_method": "Youden J = TPR - FPR",
        "best_threshold": float(thresholds[best_index]),
        "best_youden_j": float(youden[best_index]),
        "tpr_at_best_threshold": float(tpr[best_index]),
        "fpr_at_best_threshold": float(fpr[best_index]),
        "roc_points": int(len(thresholds)),
    }


def compare_with_baseline(current):
    current_values = {
        "same_id_mean": current["same_id"]["mean"],
        "different_id_mean": current["different_id"]["mean"],
        "similarity_gap": current["similarity_gap"],
        "roc_auc": current["roc_auc"],
        "eer": current["eer"],
    }
    return {
        key: {
            "before": BASELINE[key],
            "after": current_values[key],
            "delta": current_values[key] - BASELINE[key],
        }
        for key in BASELINE
    }


def write_pair_scores(path, pairs, primary_rule):
    fields = [
        "sample_a", "sample_b", "gt_person_a", "gt_person_b",
        "camera_a", "camera_b", "frame_a", "frame_b",
        "same_gt_identity", "camera_relation", "included_in_primary",
        "similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            row = dict(pair)
            row["included_in_primary"] = (
                primary_rule == "all-pairs" or pair["camera_relation"] == "cross_camera"
            )
            writer.writerow(row)


def consistent_configuration(records):
    fields = ("model_name", "checkpoint_path", "device", "preprocessing")
    configuration = {field: records[0].get(field) for field in fields}
    for record in records[1:]:
        for field in fields:
            if record.get(field) != configuration[field]:
                raise ValueError(f"inconsistent {field} across dumped samples")
    return configuration


def print_comparison(comparison):
    labels = {
        "same_id_mean": "Same-ID mean",
        "different_id_mean": "Different-ID mean",
        "similarity_gap": "Gap",
        "roc_auc": "ROC-AUC",
        "eer": "EER",
    }
    print(f"{'Metric':<24}{'Before':>12}{'After':>12}{'Delta':>12}")
    for key in BASELINE:
        row = comparison[key]
        print(
            f"{labels[key]:<24}{row['before']:>12.4f}"
            f"{row['after']:>12.4f}{row['delta']:>+12.4f}"
        )


def run(input_root, output_directory, primary_rule, max_samples=None, sample_ids=None):
    records = select_records(load_dump_records(input_root), max_samples, sample_ids)
    configuration = consistent_configuration(records)
    embeddings = np.stack([
        np.load(record["resolved_embedding_file"]).reshape(-1) for record in records
    ]).astype(np.float32)
    normalized, original_norms = l2_normalize_rows(embeddings)
    normalized_pairs = build_pair_rows(normalized @ normalized.T, records)
    raw_pairs = build_pair_rows(embeddings @ embeddings.T, records)

    cross_pairs = [
        pair for pair in normalized_pairs if pair["camera_relation"] == "cross_camera"
    ]
    primary_pairs = cross_pairs if primary_rule == "cross-camera" else normalized_pairs
    raw_primary_pairs = [
        pair for pair in raw_pairs
        if primary_rule == "all-pairs" or pair["camera_relation"] == "cross_camera"
    ]

    primary = evaluate_pairs(primary_pairs)
    all_pairs = evaluate_pairs(normalized_pairs)
    cross_camera = evaluate_pairs(cross_pairs)
    raw_primary = evaluate_pairs(raw_primary_pairs)
    comparison = compare_with_baseline(primary)
    report = {
        "evaluation_name": "canonical_dumped_embedding_verification",
        "evaluation_split": (
            "Prompt 03 deterministic diagnostic sample; no train/test split claim"
        ),
        "sample_count": len(records),
        "identity_count": len({str(item["ground_truth_person_id"]) for item in records}),
        "cameras": sorted({item["camera"] for item in records}),
        "embedding_dimension": int(normalized.shape[1]),
        "configuration": configuration,
        "validation": {
            "embeddings_finite": bool(np.isfinite(embeddings).all()),
            "input_norms": distribution_stats(original_norms),
            "input_norms_approximately_one": bool(
                np.allclose(original_norms, 1.0, rtol=1e-5, atol=1e-5)
            ),
            "label_source": "ground_truth_person_id only",
            "self_pairs_excluded": True,
            "duplicate_pair_policy": "unique unordered pairs only (i < j)",
            "primary_pair_rule": primary_rule,
        },
        "primary_metrics": primary,
        "all_pairs_metrics": all_pairs,
        "cross_camera_metrics": cross_camera,
        "baseline_comparison": comparison,
        "l2_normalization_effect": {
            "raw_dot_roc_auc": raw_primary["roc_auc"],
            "normalized_cosine_roc_auc": primary["roc_auc"],
            "roc_auc_delta": primary["roc_auc"] - raw_primary["roc_auc"],
            "raw_dot_eer": raw_primary["eer"],
            "normalized_cosine_eer": primary["eer"],
            "eer_delta": primary["eer"] - raw_primary["eer"],
            "material": (
                abs(primary["roc_auc"] - raw_primary["roc_auc"]) > 1e-4
                or abs(primary["eer"] - raw_primary["eer"]) > 1e-4
            ),
        },
        "baseline_pair_rule_warning": (
            "The supplied baseline did not specify all-pairs versus cross-camera; "
            "the comparison uses the selected primary rule and may not be apples-to-apples."
        ),
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "reid_eval_results.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
    write_pair_scores(
        output_directory / "reid_pair_scores.csv", normalized_pairs, primary_rule
    )

    print(f"primary pair rule: {primary_rule}")
    print(f"samples: {len(records)} | identities: {report['identity_count']}")
    print(
        f"positive pairs: {primary['same_id']['count']} | "
        f"negative pairs: {primary['different_id']['count']}"
    )
    print(f"same-ID stats: {primary['same_id']}")
    print(f"different-ID stats: {primary['different_id']}")
    print(f"similarity gap: {primary['similarity_gap']:.6f}")
    print(f"ROC-AUC: {primary['roc_auc']:.6f}")
    print(f"EER: {primary['eer']:.6f}")
    print(f"EER threshold (interpolated): {primary['eer_threshold']:.6f}")
    print(
        f"Youden threshold: {primary['best_threshold']:.6f} | "
        f"TPR={primary['tpr_at_best_threshold']:.6f} | "
        f"FPR={primary['fpr_at_best_threshold']:.6f}"
    )
    print_comparison(comparison)
    print(f"L2 normalization materially changed metrics: {report['l2_normalization_effect']['material']}")
    print(f"output directory: {output_directory.resolve()}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ROC-AUC and EER from canonical dumped Re-ID embeddings."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("debug_reid"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--primary-rule",
        choices=("cross-camera", "all-pairs"),
        default="cross-camera",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    input_directory = arguments.input_dir.resolve()
    output_directory = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else input_directory / "evaluation"
    )
    run(
        input_root=input_directory,
        output_directory=output_directory,
        primary_rule=arguments.primary_rule,
        max_samples=arguments.max_samples,
        sample_ids=arguments.sample_id,
    )
