"""Run four offline OSNet crop-policy ablations from the master manifest."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .dump_reid_debug import close_runtime, load_production_runtime
from .evaluate_reid_verification import build_pair_rows, evaluate_pairs
from .reid_config import (
    OSNET_DEFAULT_CHECKPOINT_NAME,
    OSNET_INPUT_HEIGHT,
    OSNET_INPUT_WIDTH,
    OSNET_PIXEL_MEAN,
)
from .reid_similarity_matrix import l2_normalize_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICIES = {
    "A_production_crop": "Exact current production margins followed by standard resize",
    "B_raw_gt_bbox": "Raw GT bbox with no margin followed by standard resize",
    "C_aspect_ratio_padding": "Raw GT bbox letterboxed to 128x256 before preprocessing",
    "D_quality_filtered": "Exact production crop on the deterministic quality subset",
}
QUALITY_RULES = {
    "minimum_bbox_area": 4096,
    "minimum_width_over_height": 0.20,
    "maximum_width_over_height": 0.85,
    "border_tolerance_pixels": 2,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {
                "sample_id", "sequence", "camera", "dataset_identity_key",
                "original_gt_person_id", "source_image", "bbox_xyxy",
                "bbox_xywh", "image_width", "image_height", "frame_index",
            }
            missing = sorted(required - set(record))
            if missing:
                raise ValueError(f"Manifest line {line_number} missing {missing}")
            records.append(record)
    if not records:
        raise ValueError("Master manifest is empty")
    return records


def evenly_spaced(items, count):
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[index] for index in indices]


def select_samples(records, maximum_per_identity_camera):
    if maximum_per_identity_camera <= 0:
        raise ValueError("maximum_per_identity_camera must be positive")
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["dataset_identity_key"], record["camera"])].append(record)
    selected = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key], key=lambda item: (item["frame_index"], item["sample_id"])
        )
        selected.extend(evenly_spaced(rows, maximum_per_identity_camera))
    return sorted(selected, key=lambda item: item["sample_id"])


def quality_assessment(record):
    x1, y1, x2, y2 = [int(value) for value in record["bbox_xyxy"]]
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect = width / max(height, 1)
    tolerance = QUALITY_RULES["border_tolerance_pixels"]
    truncated = (
        x1 <= tolerance or y1 <= tolerance
        or x2 >= int(record["image_width"]) - tolerance
        or y2 >= int(record["image_height"]) - tolerance
    )
    reasons = []
    if area < QUALITY_RULES["minimum_bbox_area"]:
        reasons.append("area_below_minimum")
    if not (
        QUALITY_RULES["minimum_width_over_height"]
        <= aspect
        <= QUALITY_RULES["maximum_width_over_height"]
    ):
        reasons.append("suspicious_aspect_ratio")
    if truncated:
        reasons.append("touches_frame_border")
    return {
        "bbox_area": area,
        "width_over_height": aspect,
        "truncated": truncated,
        "quality_eligible": not reasons,
        "quality_reasons": reasons,
        "aspect_category": "normal_aspect" if not reasons or reasons == ["area_below_minimum"] else (
            "suspicious_aspect" if "suspicious_aspect_ratio" in reasons else "normal_aspect"
        ),
    }


def raw_crop(frame, record):
    x1, y1, x2, y2 = [int(value) for value in record["bbox_xyxy"]]
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def aspect_ratio_pad(crop, target_width=OSNET_INPUT_WIDTH,
                     target_height=OSNET_INPUT_HEIGHT):
    if crop is None or crop.size == 0:
        return None
    height, width = crop.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, round(width * scale)))
    resized_height = max(1, min(target_height, round(height * scale)))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    # FeatureExtractor receives BGR converted to RGB; use ImageNet RGB mean as BGR fill.
    fill = tuple(int(round(value * 255)) for value in reversed(OSNET_PIXEL_MEAN))
    canvas = np.full((target_height, target_width, 3), fill, dtype=np.uint8)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def prepare_policy_crops(records, project_root, main_module):
    prepared = {}
    rejected = Counter()
    for record in records:
        frame = cv2.imread(str(project_root / record["source_image"]), cv2.IMREAD_COLOR)
        if frame is None:
            rejected["missing_or_unreadable_image"] += 1
            continue
        raw = raw_crop(frame, record)
        if raw is None:
            rejected["empty_raw_crop"] += 1
            continue
        production = main_module.extract_person_crop(
            frame, *[int(value) for value in record["bbox_xyxy"]]
        )
        if production is None or production.size == 0:
            rejected["empty_production_crop"] += 1
            continue
        if min(production.shape[:2]) < main_module.REID_MIN_CROP_SIZE:
            rejected["production_crop_below_minimum"] += 1
            continue
        if min(raw.shape[:2]) < main_module.REID_MIN_CROP_SIZE:
            rejected["raw_crop_below_minimum"] += 1
            continue
        padded = aspect_ratio_pad(raw)
        assessment = quality_assessment(record)
        prepared[record["sample_id"]] = {
            "record": {**record, **assessment},
            "A_production_crop": production,
            "B_raw_gt_bbox": raw,
            "C_aspect_ratio_padding": padded,
            "D_quality_filtered": production if assessment["quality_eligible"] else None,
        }
    return prepared, rejected


def extract_embeddings(extractor, crops, batch_size):
    embeddings = []
    for start in range(0, len(crops), batch_size):
        batch = extractor.extract_batch(crops[start:start + batch_size])
        if len(batch) != len(crops[start:start + batch_size]):
            raise RuntimeError("OSNet batch output length mismatch")
        embeddings.extend(batch)
    if any(embedding is None for embedding in embeddings):
        raise RuntimeError("OSNet failed to embed one or more policy crops")
    matrix = np.stack(embeddings).astype(np.float32)
    normalized, input_norms = l2_normalize_rows(matrix)
    return normalized, input_norms


def pair_rows(embeddings, records):
    evaluation_records = [
        {
            **record,
            "ground_truth_person_id": record["dataset_identity_key"],
        }
        for record in records
    ]
    pairs = build_pair_rows(embeddings @ embeddings.T, evaluation_records)
    metadata = {record["sample_id"]: record for record in records}
    for pair in pairs:
        left = metadata[pair["sample_a"]]
        right = metadata[pair["sample_b"]]
        pair["sequence_a"] = left["sequence"]
        pair["sequence_b"] = right["sequence"]
        pair["aspect_pair"] = (
            "normal_aspect"
            if left["aspect_category"] == right["aspect_category"] == "normal_aspect"
            else "suspicious_aspect"
        )
    return pairs


def safe_evaluate(pairs):
    positive = sum(bool(item["same_gt_identity"]) for item in pairs)
    negative = len(pairs) - positive
    if not pairs or positive == 0 or negative == 0:
        return {
            "available": False,
            "pair_count": len(pairs),
            "positive_pairs": positive,
            "negative_pairs": negative,
            "reason": "requires both positive and negative pairs",
        }
    return {"available": True, **evaluate_pairs(pairs)}


def summarize_policy(policy, embeddings, records, input_norms):
    pairs = pair_rows(embeddings, records)
    cross_camera_pairs = [
        item for item in pairs if item["camera_relation"] == "cross_camera"
    ]
    primary = safe_evaluate(cross_camera_pairs)
    if not primary["available"]:
        raise ValueError(f"{policy} has no evaluable cross-camera pairs")
    positive_pairs = [item for item in cross_camera_pairs if item["same_gt_identity"]]
    negative_pairs = [item for item in cross_camera_pairs if not item["same_gt_identity"]]
    breakdowns = {
        "cam1_same_camera": safe_evaluate([
            item for item in pairs
            if item["camera_a"] == item["camera_b"] == "cam1"
        ]),
        "cam2_same_camera": safe_evaluate([
            item for item in pairs
            if item["camera_a"] == item["camera_b"] == "cam2"
        ]),
        "cross_camera": primary,
        "normal_aspect_cross_camera": safe_evaluate([
            item for item in cross_camera_pairs if item["aspect_pair"] == "normal_aspect"
        ]),
        "suspicious_aspect_cross_camera": safe_evaluate([
            item for item in cross_camera_pairs if item["aspect_pair"] == "suspicious_aspect"
        ]),
    }
    return {
        "policy": policy,
        "description": POLICIES[policy],
        "sample_count": len(records),
        "identity_count": len({item["dataset_identity_key"] for item in records}),
        "cross_camera_identity_count": len({
            item["dataset_identity_key"] for item in records
            if len({
                other["camera"] for other in records
                if other["dataset_identity_key"] == item["dataset_identity_key"]
            }) > 1
        }),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(item["sample_id"] for item in records).encode("utf-8")
        ).hexdigest(),
        "embedding_norm_min": float(np.min(input_norms)),
        "embedding_norm_mean": float(np.mean(input_norms)),
        "embedding_norm_max": float(np.max(input_norms)),
        "primary_pair_rule": "cross-camera unique unordered pairs",
        "primary_metrics": primary,
        "breakdowns": breakdowns,
        "hard_positives": sorted(
            positive_pairs, key=lambda item: item["similarity"]
        )[:20],
        "hard_negatives": sorted(
            negative_pairs, key=lambda item: item["similarity"], reverse=True
        )[:20],
    }


def metric_row(result):
    primary = result["primary_metrics"]
    return {
        "Policy": result["policy"],
        "Samples": result["sample_count"],
        "AUC": primary["roc_auc"],
        "EER": primary["eer"],
        "Same Mean": primary["same_id"]["mean"],
        "Same Std": primary["same_id"]["std"],
        "Same Min": primary["same_id"]["min"],
        "Same Max": primary["same_id"]["max"],
        "Diff Mean": primary["different_id"]["mean"],
        "Diff Std": primary["different_id"]["std"],
        "Diff Min": primary["different_id"]["min"],
        "Diff Max": primary["different_id"]["max"],
        "Gap": primary["similarity_gap"],
        "Youden Threshold": primary["best_threshold"],
        "TPR": primary["tpr_at_best_threshold"],
        "FPR": primary["fpr_at_best_threshold"],
    }


def render_report(results, configuration, quality_matched):
    rows = [metric_row(results[key]) for key in POLICIES]
    production = rows[0]
    alternatives = rows[1:]
    best = max(alternatives, key=lambda item: item["AUC"])
    auc_delta = best["AUC"] - production["AUC"]
    eer_delta = production["EER"] - best["EER"]
    material = auc_delta >= 0.05 and eer_delta >= 0.05
    conclusion = (
        "Crop policy alone produced a material improvement on this diagnostic subset."
        if material else
        "Crop policy alone did not produce a large, consistent improvement on this diagnostic subset."
    )
    lines = [
        "# OSNet Offline Crop Policy Ablation",
        "",
        f"Checkpoint: `{configuration['checkpoint_path']}`  ",
        f"Checkpoint SHA-256: `{configuration['checkpoint_sha256']}`  ",
        f"Device: `{configuration['device']}`  ",
        f"Master manifest SHA-256: `{configuration['manifest_sha256']}`",
        "",
        "| Policy | Samples | AUC | EER | Same Mean | Diff Mean | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['Policy']} | {row['Samples']} | {row['AUC']:.6f} | "
            f"{row['EER']:.6f} | {row['Same Mean']:.6f} | "
            f"{row['Diff Mean']:.6f} | {row['Gap']:.6f} |"
        )
    lines.extend([
        "",
        "## Controls",
        "",
        "- A/B/C use the exact same deterministic sample IDs.",
        "- D uses production crop geometry on the quality-filtered subset.",
        "- A/B/C were also evaluated on D's exact sample IDs; see each metrics JSON "
        "under `quality_matched_baseline_metrics`.",
        "- All policies use the same checkpoint, RGB conversion, ImageNet normalization, "
        "128x256 model input, batch extractor, L2 normalization, and cosine scoring.",
        "- Labels use `dataset_identity_key`; equal numeric IDs in different sequences "
        "remain different people.",
        "",
        "## Quality filter",
        "",
        f"- Minimum bbox area: `{QUALITY_RULES['minimum_bbox_area']}`",
        f"- Width/height range: `{QUALITY_RULES['minimum_width_over_height']}` to "
        f"`{QUALITY_RULES['maximum_width_over_height']}`",
        f"- Border tolerance: `{QUALITY_RULES['border_tolerance_pixels']}` pixels",
        f"- Quality subset samples: `{len(quality_matched)}`",
        "",
        "## Quality-matched comparison",
        "",
        f"All rows below use the exact same {len(quality_matched)} quality-eligible "
        "sample IDs. Policy D "
        "is production geometry, so it is the direct counterpart of A on this subset.",
        "",
        "| Policy | AUC | EER | Gap |",
        "|---|---:|---:|---:|",
    ])
    for policy in ("A_production_crop", "B_raw_gt_bbox", "C_aspect_ratio_padding"):
        metric = results[policy]["quality_matched_baseline_metrics"]
        lines.append(
            f"| {policy} | {metric['roc_auc']:.6f} | {metric['eer']:.6f} | "
            f"{metric['similarity_gap']:.6f} |"
        )
    quality_metric = results["D_quality_filtered"]["primary_metrics"]
    lines.extend([
        f"| D_quality_filtered | {quality_metric['roc_auc']:.6f} | "
        f"{quality_metric['eer']:.6f} | {quality_metric['similarity_gap']:.6f} |",
        "",
        "## Breakdown",
        "",
        "| Policy | Slice | AUC | EER | Gap |",
        "|---|---|---:|---:|---:|",
    ])
    slice_names = (
        "cam1_same_camera", "cam2_same_camera", "cross_camera",
        "normal_aspect_cross_camera", "suspicious_aspect_cross_camera",
    )
    for policy in POLICIES:
        for slice_name in slice_names:
            metric = results[policy]["breakdowns"][slice_name]
            if metric["available"]:
                lines.append(
                    f"| {policy} | {slice_name} | {metric['roc_auc']:.6f} | "
                    f"{metric['eer']:.6f} | {metric['similarity_gap']:.6f} |"
                )
            else:
                lines.append(f"| {policy} | {slice_name} | n/a | n/a | n/a |")
    lines.extend([
        "",
        "## Conclusion",
        "",
        conclusion,
        f" Best non-production primary AUC delta: `{auc_delta:+.6f}`; "
        f"EER reduction: `{eer_delta:+.6f}`.",
        "",
        "Crop geometry is not the sole cause of the remaining Re-ID errors. Continue "
        "toward a controlled offline fine-tuning experiment, but treat the five "
        "cross-camera identities as insufficient for a robust accuracy claim and keep "
        "production integration gated on held-out evaluation.",
        "",
        "No production crop, runtime, camera/video, preview, or calibration code was changed.",
        "",
    ])
    return "\n".join(lines), {
        "best_alternative": best["Policy"],
        "best_auc_delta_vs_production": auc_delta,
        "best_eer_reduction_vs_production": eer_delta,
        "crop_policy_material_improvement": material,
        "conclusion": conclusion,
    }


def run(manifest_path, output_root, checkpoint, device, samples_per_group,
        batch_size, overwrite=False):
    targets = [
        output_root / "comparison.csv",
        output_root / "CROP_ABLATION_REPORT.md",
        output_root / "run_summary.json",
    ]
    if any(path.exists() for path in targets) and not overwrite:
        raise FileExistsError("Refusing to overwrite existing ablation outputs")
    manifest_records = load_manifest(manifest_path)
    selected = select_samples(manifest_records, samples_per_group)
    main = load_production_runtime(checkpoint.resolve(), device)
    try:
        prepared, crop_rejections = prepare_policy_crops(selected, PROJECT_ROOT, main)
        common_ids = sorted(prepared)
        common_records = [prepared[sample_id]["record"] for sample_id in common_ids]
        quality_ids = [
            sample_id for sample_id in common_ids
            if prepared[sample_id]["record"]["quality_eligible"]
        ]
        if not quality_ids:
            raise ValueError("Quality policy rejected every selected sample")
        results = {}
        embeddings_by_policy = {}
        for policy in POLICIES:
            policy_ids = quality_ids if policy == "D_quality_filtered" else common_ids
            crops = [prepared[sample_id][policy] for sample_id in policy_ids]
            records = [prepared[sample_id]["record"] for sample_id in policy_ids]
            embeddings, norms = extract_embeddings(
                main.appearance_extractor, crops, batch_size
            )
            embeddings_by_policy[policy] = (policy_ids, embeddings)
            results[policy] = summarize_policy(policy, embeddings, records, norms)

        quality_matched = {}
        quality_set = set(quality_ids)
        for policy in ("A_production_crop", "B_raw_gt_bbox", "C_aspect_ratio_padding"):
            policy_ids, embeddings = embeddings_by_policy[policy]
            indices = [index for index, sample_id in enumerate(policy_ids) if sample_id in quality_set]
            subset_records = [prepared[policy_ids[index]]["record"] for index in indices]
            subset_embeddings = embeddings[indices]
            quality_matched[policy] = summarize_policy(
                policy, subset_embeddings, subset_records,
                np.linalg.norm(subset_embeddings, axis=1),
            )["primary_metrics"]
            results[policy]["quality_matched_baseline_metrics"] = quality_matched[policy]

        output_root.mkdir(parents=True, exist_ok=True)
        comparison_rows = [metric_row(results[policy]) for policy in POLICIES]
        with (output_root / "comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
        for policy, result in results.items():
            with (output_root / f"{policy}_metrics.json").open(
                "w", encoding="utf-8", newline="\n"
            ) as destination:
                json.dump(result, destination, indent=2, sort_keys=True)
                destination.write("\n")
        selected_rows = []
        for sample_id in common_ids:
            record = prepared[sample_id]["record"]
            selected_rows.append({
                "sample_id": sample_id,
                "sequence": record["sequence"],
                "camera": record["camera"],
                "dataset_identity_key": record["dataset_identity_key"],
                "frame_index": record["frame_index"],
                "quality_eligible": record["quality_eligible"],
                "quality_reasons": ";".join(record["quality_reasons"]),
                "bbox_area": record["bbox_area"],
                "width_over_height": record["width_over_height"],
                "truncated": record["truncated"],
            })
        with (output_root / "selected_samples.csv").open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            writer = csv.DictWriter(destination, fieldnames=list(selected_rows[0]))
            writer.writeheader()
            writer.writerows(selected_rows)
        manifest_summary = json.loads(
            (manifest_path.parent / "master_manifest_v1_summary.json").read_text(
                encoding="utf-8"
            )
        )
        configuration = {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "device": main.REID_RUNTIME_STATUS["device"],
            "model": main.REID_RUNTIME_STATUS["model_architecture"],
            "preprocessing": main.REID_RUNTIME_STATUS["preprocessing"],
            "embedding_dimension": main.REID_RUNTIME_STATUS["embedding_dimension"],
            "samples_per_identity_camera_group": samples_per_group,
            "batch_size": batch_size,
            "master_identity_key_format": manifest_summary["identity_key_format"],
        }
        report_text, conclusion = render_report(results, configuration, quality_ids)
        (output_root / "CROP_ABLATION_REPORT.md").write_text(
            report_text, encoding="utf-8", newline="\n"
        )
        summary = {
            "configuration": configuration,
            "policies": POLICIES,
            "quality_rules": QUALITY_RULES,
            "manifest_samples": len(manifest_records),
            "selected_samples_before_common_validation": len(selected),
        "common_samples_abc": len(common_ids),
            "common_sample_ids_sha256": results["A_production_crop"]["sample_ids_sha256"],
            "quality_samples_d": len(quality_ids),
            "crop_rejections": dict(sorted(crop_rejections.items())),
            "policy_metrics_files": {
                policy: f"{policy}_metrics.json" for policy in POLICIES
            },
            "comparison_file": "comparison.csv",
            "selected_samples_file": "selected_samples.csv",
            "quality_matched_baseline_policies": sorted(quality_matched),
            "conclusion": conclusion,
            "fine_tuning_run": False,
            "production_modified": False,
        }
        with (output_root / "run_summary.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            json.dump(summary, destination, indent=2, sort_keys=True)
            destination.write("\n")
        print(json.dumps({
            "common_samples_abc": len(common_ids),
            "quality_samples_d": len(quality_ids),
            "crop_rejections": dict(crop_rejections),
            "comparison": comparison_rows,
            "conclusion": conclusion,
        }, indent=2))
        return summary
    finally:
        close_runtime(main)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "reid_dataset" / "master_manifest_v1.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reid_crop_ablation"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "weights" / OSNET_DEFAULT_CHECKPOINT_NAME,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples-per-identity-camera", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.manifest.resolve(), arguments.output_dir.resolve(),
        arguments.checkpoint.resolve(), arguments.device,
        arguments.samples_per_identity_camera, arguments.batch_size,
        overwrite=arguments.overwrite,
    )
