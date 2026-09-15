"""Create offline OSNet verification and production Global-ID baseline artifacts.

This diagnostic imports the production backend with an in-memory identity store,
feeds labeled image sequences downstream of acquisition, and never starts a
camera/video worker.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .baseline_metrics import (
    classification_metrics,
    confusion_at_threshold,
    evaluate_global_id_records,
    match_frame_observations,
)
from .dump_reid_debug import close_runtime, load_production_runtime
from .evaluate_reid_verification import evaluate_pairs
from .reid_config import OSNET_DEFAULT_CHECKPOINT_NAME
from .reid_similarity_matrix import l2_normalize_rows
from .run_reid_crop_ablation import extract_embeddings, load_manifest, raw_crop, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_SCHEMA_VERSION = "peoplelocation-baseline-metrics-v1"
GLOBAL_MATCH_MINIMUM_IOU = 0.5
SEQUENCE_FPS = {"legacy_sequence": 30.0, "m_sequence": 2.0}


def _cross_camera_pairs(embeddings, records):
    similarities = embeddings @ embeddings.T
    scores = []
    labels = []
    sample_count = len(records)
    for left in range(sample_count):
        for right in range(left + 1, sample_count):
            if records[left]["camera"] == records[right]["camera"]:
                continue
            scores.append(float(similarities[left, right]))
            labels.append(int(
                records[left]["dataset_identity_key"]
                == records[right]["dataset_identity_key"]
            ))
    if not scores:
        raise ValueError("master manifest has no cross-camera verification pairs")
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8)


def evaluate_reid(main, manifest_records, checkpoint, manifest_path, batch_size):
    crops = []
    records = []
    rejections = Counter()
    for record in manifest_records:
        frame = cv2.imread(
            str(PROJECT_ROOT / record["source_image"]), cv2.IMREAD_COLOR
        )
        if frame is None:
            rejections["missing_or_unreadable_image"] += 1
            continue
        crop = raw_crop(frame, record)
        if crop is None or crop.size == 0:
            rejections["empty_raw_gt_bbox"] += 1
            continue
        crops.append(crop)
        records.append(record)

    embeddings, input_norms = extract_embeddings(
        main.appearance_extractor, crops, batch_size
    )
    embeddings, _ = l2_normalize_rows(embeddings)
    scores, labels = _cross_camera_pairs(embeddings, records)
    pairs = [
        {"similarity": float(score), "same_gt_identity": bool(label)}
        for score, label in zip(scores, labels)
    ]
    verification = evaluate_pairs(pairs)
    youden_threshold = verification["best_threshold"]
    eer_threshold = verification["eer_threshold"]
    youden_confusion = confusion_at_threshold(scores, labels, youden_threshold)
    eer_confusion = confusion_at_threshold(scores, labels, eer_threshold)
    youden_classification = classification_metrics(youden_confusion)
    eer_classification = classification_metrics(eer_confusion)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "evaluation": "OSNet cross-camera verification",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "master_manifest": str(manifest_path.resolve()),
        "master_manifest_sha256": sha256_file(manifest_path),
        "crop_policy": "Raw GT bbox (no production margin), then unchanged production preprocessing",
        "label_source": "dataset_identity_key",
        "pair_rule": "unique unordered cross-camera pairs; self-pairs excluded",
        "sample_count": len(records),
        "manifest_record_count": len(manifest_records),
        "rejected_sample_count": int(sum(rejections.values())),
        "rejection_reasons": dict(sorted(rejections.items())),
        "identity_count": len({row["dataset_identity_key"] for row in records}),
        "cross_camera_identity_count": len({
            row["dataset_identity_key"]
            for row in records
            if len({
                candidate["camera"] for candidate in records
                if candidate["dataset_identity_key"] == row["dataset_identity_key"]
            }) > 1
        }),
        "cameras": sorted({row["camera"] for row in records}),
        "positive_pair_count": int(np.sum(labels == 1)),
        "negative_pair_count": int(np.sum(labels == 0)),
        "same_id_mean": verification["same_id"]["mean"],
        "different_id_mean": verification["different_id"]["mean"],
        "similarity_gap": verification["similarity_gap"],
        "roc_auc": verification["roc_auc"],
        "eer": verification["eer"],
        "youden": {
            "threshold_method": verification["best_threshold_method"],
            "threshold": youden_threshold,
            "youden_j": verification["best_youden_j"],
            "confusion_matrix": youden_confusion,
            **youden_classification,
        },
        "eer_operating_point": {
            "threshold": eer_threshold,
            "confusion_matrix": eer_confusion,
            **eer_classification,
        },
        "embedding_input_norm": {
            "minimum": float(np.min(input_norms)),
            "mean": float(np.mean(input_norms)),
            "maximum": float(np.max(input_norms)),
        },
        "preprocessing": main.REID_RUNTIME_STATUS["preprocessing"],
        "model": main.REID_RUNTIME_STATUS["model_architecture"],
        "device": main.REID_RUNTIME_STATUS["device"],
        "production_thresholds_changed": False,
        "fine_tuning_run": False,
    }


class OfflineFlushCoordinator:
    """Production coordinator semantics with deterministic explicit flushes."""

    @staticmethod
    def build(main, manager):
        class Coordinator(main.GlobalAssignmentCoordinator):
            def _start_timer_locked(self):
                # The harness flushes after all cameras at a frame index submit.
                # Suppressing wall-clock scheduling prevents CPU speed from
                # changing batch membership; the production window value and
                # manager assignment logic remain unchanged.
                return None

        return Coordinator(
            lambda: manager,
            window_sec=main.GLOBAL_ASSIGNMENT_WINDOW_SEC,
        )


def _camera_data(main, sequence, camera):
    return {
        "url": f"offline://{sequence}/{camera}",
        "source_type": "offline_labeled_sequence",
        "loop_video": False,
        "processor": None,
        "src_pts": None,
        "dst_pts": None,
        "last_frame": None,
        "prev_assignments": [],
        **main.new_camera_tracker_context(),
    }


def _sequence_image_index(records):
    roots = {}
    for row in records:
        image_path = PROJECT_ROOT / row["source_image"]
        roots[row["camera"]] = image_path.parent
    indexed = {}
    for camera, directory in sorted(roots.items()):
        frame_paths = {}
        for path in sorted(directory.glob("*.jpg")):
            try:
                frame_paths[int(path.stem)] = path
            except ValueError:
                continue
        indexed[camera] = frame_paths
    return indexed


def replay_global_ids(main, manifest_records, minimum_iou):
    predictions = []
    replay_stats = {}
    grouped_sequences = defaultdict(list)
    for row in manifest_records:
        grouped_sequences[row["sequence"]].append(row)

    # Resolve the checked-in detector explicitly. This avoids any download and
    # changes only this diagnostic process.
    detector_path = PROJECT_ROOT / "backend" / "yolov8s.pt"
    if not detector_path.is_file():
        raise FileNotFoundError(
            f"Offline replay requires the existing detector checkpoint: {detector_path}"
        )
    main.YOLO_MODEL_PATH = str(detector_path.resolve())

    for sequence in sorted(grouped_sequences):
        sequence_records = grouped_sequences[sequence]
        image_index = _sequence_image_index(sequence_records)
        if not image_index or any(not frames for frames in image_index.values()):
            replay_stats[sequence] = {"status": "skipped", "reason": "missing image sequence"}
            continue
        manager = main.GlobalIdentityManager()
        coordinator = OfflineFlushCoordinator.build(main, manager)
        captured = []
        original_assign = manager.assign_global_batch

        def recording_assign(camera_detections, **kwargs):
            result = original_assign(camera_detections, **kwargs)
            for camera in sorted(camera_detections):
                for detection, assignment in zip(
                    camera_detections[camera], result.get(camera, [])
                ):
                    if assignment is None:
                        continue
                    captured.append({
                        "sequence": sequence,
                        "camera": camera,
                        "frame_index": int(detection["frame_index"]),
                        "predicted_global_id": assignment["gid"],
                        "predicted_global_identity": f"{sequence}:{assignment['gid']}",
                        "local_track_id": int(detection["tid"]),
                        "predicted_bbox_xyxy": list(detection["box"]),
                        "detector_confidence": detection.get("conf"),
                        "assignment_score": assignment.get("score"),
                        "assignment_source": assignment.get("source"),
                    })
            return result

        manager.assign_global_batch = recording_assign
        main.global_identity_manager = manager
        main.global_assignment_coordinator = coordinator
        with main.cameras_lock:
            main.cameras.clear()
            for camera in sorted(image_index):
                main.cameras[camera] = _camera_data(main, sequence, camera)

        frame_indices = sorted({
            frame_index
            for frames in image_index.values()
            for frame_index in frames
        })
        unreadable = 0
        processed = 0
        fps = SEQUENCE_FPS.get(sequence, 30.0)
        for frame_index in frame_indices:
            event_time = float(frame_index) / fps
            for camera in sorted(image_index):
                path = image_index[camera].get(frame_index)
                if path is None:
                    continue
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    unreadable += 1
                    continue
                main.process_camera_frame(
                    camera, frame, frame_index, event_time=event_time
                )
                processed += 1
            coordinator.flush()
        coordinator.flush()
        coordinator.stop()
        predictions.extend(captured)
        replay_stats[sequence] = {
            "status": "evaluated",
            "cameras": sorted(image_index),
            "fps_used_for_event_time": fps,
            "available_image_frames": sum(len(rows) for rows in image_index.values()),
            "processed_image_frames": processed,
            "unreadable_image_frames": unreadable,
            "committed_predictions": len(captured),
        }

    predictions_by_frame = defaultdict(list)
    for prediction in predictions:
        predictions_by_frame[(
            prediction["sequence"], prediction["camera"], prediction["frame_index"]
        )].append(prediction)
    gt_by_frame = defaultdict(list)
    for row in manifest_records:
        gt_by_frame[(row["sequence"], row["camera"], int(row["frame_index"]))].append(row)

    all_records = []
    matched_prediction_count = 0
    for frame_key in sorted(gt_by_frame):
        gt_rows = gt_by_frame[frame_key]
        prediction_rows = predictions_by_frame.get(frame_key, [])
        matched = {
            gt_index: (prediction_rows[prediction_index], iou)
            for gt_index, prediction_index, iou in match_frame_observations(
                gt_rows, prediction_rows, minimum_iou
            )
        }
        matched_prediction_count += len(matched)
        for index, gt in enumerate(gt_rows):
            base = {
                "sample_id": gt["sample_id"],
                "sequence": gt["sequence"],
                "camera": gt["camera"],
                "frame_index": int(gt["frame_index"]),
                "dataset_identity_key": gt["dataset_identity_key"],
                "gt_bbox_xyxy": gt["bbox_xyxy"],
                "match_iou_threshold": float(minimum_iou),
            }
            if index not in matched:
                all_records.append({
                    **base,
                    "evaluation_status": "no_evaluable_global_id_prediction",
                    "match_iou": None,
                    "predicted_global_id": None,
                    "predicted_global_identity": None,
                    "local_track_id": None,
                    "predicted_bbox_xyxy": None,
                    "detector_confidence": None,
                    "assignment_score": None,
                    "assignment_source": None,
                })
                continue
            prediction, iou = matched[index]
            all_records.append({
                **base,
                "evaluation_status": "valid",
                "match_iou": iou,
                **{key: prediction.get(key) for key in (
                    "predicted_global_id", "predicted_global_identity",
                    "local_track_id", "predicted_bbox_xyxy", "detector_confidence",
                    "assignment_score", "assignment_source",
                )},
            })

    evaluated = evaluate_global_id_records(all_records)
    evaluated_by_sample = {row["sample_id"]: row for row in evaluated["records"]}
    for row in all_records:
        evaluated_row = evaluated_by_sample.get(row["sample_id"])
        row["mapped_gt_identity"] = (
            evaluated_row.get("mapped_gt_identity") if evaluated_row else None
        )
        row["global_id_correct"] = (
            evaluated_row.get("global_id_correct") if evaluated_row else None
        )
    return all_records, evaluated, replay_stats, {
        "committed_prediction_count": len(predictions),
        "spatially_matched_prediction_count": matched_prediction_count,
        "unmatched_committed_prediction_count": len(predictions) - matched_prediction_count,
    }


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            writer.writerow(row)


def _format(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_report(reid, global_report):
    global_metrics = global_report["metrics"]
    primary = reid["youden"]
    table = [
        ("Same-ID Mean", reid["same_id_mean"]),
        ("Different-ID Mean", reid["different_id_mean"]),
        ("Similarity Gap", reid["similarity_gap"]),
        ("Accuracy", primary["accuracy"]),
        ("Precision", primary["precision"]),
        ("Recall", primary["recall"]),
        ("F1", primary["f1"]),
        ("ROC-AUC", reid["roc_auc"]),
        ("EER", reid["eer"]),
        ("Valid Global ID Records", global_metrics["valid_global_id_records"]),
        ("Correct Global IDs", global_metrics["correct_global_ids"]),
        ("Global ID Accuracy", global_metrics["global_id_accuracy"]),
        ("ID Switches", global_metrics["id_switches"]),
        ("Identity Splits", global_metrics["identity_splits"]),
        ("Identity Merges", global_metrics["identity_merges"]),
    ]
    lines = [
        "# Current Re-ID + Global ID Baseline Metrics",
        "",
        "| Metric | Current Baseline |",
        "|---|---:|",
        *[f"| {name} | {_format(value)} |" for name, value in table],
        "",
        "## Re-ID verification protocol",
        "",
        f"- Checkpoint: `{reid['checkpoint']}`",
        f"- Checkpoint SHA-256: `{reid['checkpoint_sha256']}`",
        f"- Master manifest: `{reid['master_manifest']}`",
        f"- Master manifest SHA-256: `{reid['master_manifest_sha256']}`",
        f"- Crop policy: {reid['crop_policy']}",
        f"- Samples: `{reid['sample_count']}`; positive pairs: `{reid['positive_pair_count']}`; negative pairs: `{reid['negative_pair_count']}`",
        f"- Primary classification threshold: `{primary['threshold']:.9f}` from Youden J on the ROC (no manual threshold selection).",
        f"- Primary confusion matrix: TP={primary['confusion_matrix']['tp']}, TN={primary['confusion_matrix']['tn']}, FP={primary['confusion_matrix']['fp']}, FN={primary['confusion_matrix']['fn']}.",
        f"- EER threshold: `{reid['eer_operating_point']['threshold']:.9f}`; secondary accuracy/F1: `{reid['eer_operating_point']['accuracy']:.6f}` / `{reid['eer_operating_point']['f1']:.6f}`.",
        "",
        "## Global identity protocol",
        "",
        "GT observations are matched one-to-one to committed detector/global-assignment rows in the same sequence, camera, and frame by Hungarian maximum IoU with IoU >= 0.5. BoT-SORT local track IDs are retained only as diagnostics and never used as GT identity.",
        "",
        "Predicted identities are namespaced as `<sequence>:<global_id>`. A deterministic Hungarian maximum-overlap mapping between sorted predicted identities and sorted `dataset_identity_key` values defines correctness. Therefore equal numeric IDs in different sequences cannot merge.",
        "",
        "ID switches are changes between consecutive valid predicted Global IDs within one `(sequence, camera, dataset_identity_key)` trajectory ordered by frame. A split is one GT identity using more than one namespaced predicted ID; a merge is one namespaced predicted ID used by more than one GT identity.",
        "",
        f"- Sequences: `{', '.join(global_report['sequences_used'])}`",
        f"- GT records: `{global_report['gt_record_count']}`; valid records: `{global_metrics['valid_global_id_records']}`",
        f"- Split excess predicted IDs: `{global_metrics['identity_split_excess_predicted_ids']}`",
        f"- Merge-affected GT identities: `{global_metrics['identity_merge_affected_gt_identities']}`",
        "",
        "## Limitations",
        "",
        "- Verification pairs share temporally adjacent observations; this is a current-dataset diagnostic baseline, not a person-disjoint generalization estimate.",
        "- The master dataset contains only five cross-camera identities, so uncertainty is high and the numbers must not be treated as broad accuracy claims.",
        "- Offline Global-ID replay starts a fresh identity namespace per sequence and uses extracted labeled image sequences; it does not exercise acquisition, decoding, preview, reconnect, Docker device access, or calibration.",
        "- Global-ID validity depends on detector/BoT-SORT confirmation, OSNet availability, and IoU matching; unmatched GT observations remain in `global_id_records.csv` but are excluded from the Global ID accuracy denominator.",
        "- `m_sequence:1001` and `m_sequence:2001` remain camera-local GT identities and are never asserted to be the same cross-camera person.",
        "",
        "No fine-tuning, model integration, GT edits, production threshold changes, or frozen camera/video/calibration changes were performed.",
        "",
    ]
    return "\n".join(lines)


def run(manifest_path, checkpoint, output_directory, device, batch_size, minimum_iou, overwrite):
    targets = [
        output_directory / name for name in (
            "reid_verification_metrics.json", "reid_confusion_matrix.csv",
            "global_id_metrics.json", "global_id_records.csv",
            "identity_switches.csv", "identity_splits.csv", "identity_merges.csv",
            "BASELINE_METRICS_REPORT.md",
        )
    ]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite existing baseline artifacts")
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_records = load_manifest(manifest_path)
    main = load_production_runtime(checkpoint.resolve(), device)
    try:
        reid = evaluate_reid(main, manifest_records, checkpoint, manifest_path, batch_size)
        all_records, evaluated, replay_stats, prediction_stats = replay_global_ids(
            main, manifest_records, minimum_iou
        )
    finally:
        coordinator = getattr(main, "global_assignment_coordinator", None)
        if coordinator is not None:
            coordinator.stop()
        with main.cameras_lock:
            main.cameras.clear()
        close_runtime(main)

    global_report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "evaluation": "offline production Global-ID replay",
        "master_manifest": str(manifest_path.resolve()),
        "master_manifest_sha256": sha256_file(manifest_path),
        "checkpoint": str(checkpoint.resolve()),
        "sequences_used": [
            sequence for sequence, status in sorted(replay_stats.items())
            if status["status"] == "evaluated"
        ],
        "gt_record_count": len(manifest_records),
        "matching": {
            "observation_matching": "Hungarian maximum IoU within sequence/camera/frame",
            "minimum_iou": float(minimum_iou),
            "identity_mapping": "Hungarian maximum valid-record overlap; sorted IDs make ties deterministic",
            "predicted_identity_namespace": "<sequence>:<global_id>",
            "gt_identity_source": "dataset_identity_key only",
            "local_track_id_is_gt": False,
        },
        "id_switch_rule": "change between consecutive valid GIDs in one sequence/camera/GT trajectory",
        "split_rule": "GT identity has more than one distinct namespaced predicted GID",
        "merge_rule": "namespaced predicted GID has more than one distinct dataset_identity_key",
        "identity_mapping": evaluated["mapping"],
        "metrics": evaluated["metrics"],
        "replay": replay_stats,
        "prediction_counts": prediction_stats,
        "production_thresholds_changed": False,
        "fine_tuning_run": False,
    }

    (output_directory / "reid_verification_metrics.json").write_text(
        json.dumps(reid, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    confusion_rows = []
    for operating_point in ("youden", "eer_operating_point"):
        item = reid[operating_point]
        confusion_rows.append({
            "operating_point": operating_point,
            "threshold": item["threshold"],
            **item["confusion_matrix"],
            **{key: item[key] for key in ("accuracy", "precision", "recall", "f1")},
        })
    _write_csv(
        output_directory / "reid_confusion_matrix.csv", confusion_rows,
        ["operating_point", "threshold", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"],
    )
    (output_directory / "global_id_metrics.json").write_text(
        json.dumps(global_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_directory / "global_id_records.csv", all_records, [
        "sample_id", "sequence", "camera", "frame_index", "dataset_identity_key",
        "gt_bbox_xyxy", "evaluation_status", "match_iou_threshold", "match_iou",
        "predicted_global_id", "predicted_global_identity", "mapped_gt_identity",
        "global_id_correct", "local_track_id", "predicted_bbox_xyxy",
        "detector_confidence", "assignment_score", "assignment_source",
    ])
    _write_csv(output_directory / "identity_switches.csv", evaluated["switches"], [
        "sequence", "camera", "dataset_identity_key", "from_frame", "to_frame",
        "from_predicted_global_identity", "to_predicted_global_identity",
        "from_sample_id", "to_sample_id",
    ])
    _write_csv(output_directory / "identity_splits.csv", evaluated["splits"], [
        "dataset_identity_key", "predicted_global_id_count", "excess_predicted_ids",
        "predicted_global_identities",
    ])
    _write_csv(output_directory / "identity_merges.csv", evaluated["merges"], [
        "predicted_global_identity", "gt_identity_count", "dataset_identity_keys",
    ])
    (output_directory / "BASELINE_METRICS_REPORT.md").write_text(
        render_report(reid, global_report), encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "output_directory": str(output_directory.resolve()),
        "reid": {
            "samples": reid["sample_count"], "roc_auc": reid["roc_auc"],
            "eer": reid["eer"], "accuracy": reid["youden"]["accuracy"],
        },
        "global_id": global_report["metrics"],
    }, indent=2))
    return reid, global_report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reid_dataset" / "master_manifest_v1.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "weights" / OSNET_DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reid_baseline_metrics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-iou", type=float, default=GLOBAL_MATCH_MINIMUM_IOU)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.manifest.resolve(), arguments.checkpoint.resolve(),
        arguments.output_dir.resolve(), arguments.device, arguments.batch_size,
        arguments.minimum_iou, arguments.overwrite,
    )
