"""Pure metric helpers for offline Re-ID and Global-ID baselines."""

from collections import Counter, defaultdict

import numpy as np


def confusion_at_threshold(scores, labels, threshold):
    """Return TP/TN/FP/FN for the rule cosine similarity >= threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if scores.ndim != 1 or labels.ndim != 1 or scores.size != labels.size:
        raise ValueError("scores and labels must be same-length 1-D arrays")
    predicted = scores >= float(threshold)
    positive = labels == 1
    return {
        "tp": int(np.sum(predicted & positive)),
        "tn": int(np.sum(~predicted & ~positive)),
        "fp": int(np.sum(predicted & ~positive)),
        "fn": int(np.sum(~predicted & positive)),
    }


def classification_metrics(confusion):
    tp = int(confusion["tp"])
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def bbox_iou(left, right):
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_frame_observations(gt_rows, prediction_rows, minimum_iou=0.5):
    """Deterministic one-to-one IoU matching, independent of tracker IDs."""
    if not gt_rows or not prediction_rows:
        return []
    from scipy.optimize import linear_sum_assignment

    gt_order = sorted(
        range(len(gt_rows)),
        key=lambda index: (str(gt_rows[index]["sample_id"]), index),
    )
    prediction_order = sorted(
        range(len(prediction_rows)),
        key=lambda index: (
            str(prediction_rows[index].get("predicted_global_id")),
            str(prediction_rows[index].get("local_track_id")),
            tuple(prediction_rows[index]["predicted_bbox_xyxy"]),
            index,
        ),
    )
    matrix = np.asarray([
        [
            bbox_iou(
                gt_rows[gt_index]["bbox_xyxy"],
                prediction_rows[prediction_index]["predicted_bbox_xyxy"],
            )
            for prediction_index in prediction_order
        ]
        for gt_index in gt_order
    ], dtype=np.float64)
    # Sorted inputs plus stable, tiny column preference make tied assignments
    # reproducible while IoU remains the primary objective.
    tie = np.arange(matrix.shape[1], dtype=np.float64) * 1e-12
    row_indices, column_indices = linear_sum_assignment(-(matrix - tie))
    matches = []
    for row_index, column_index in zip(row_indices, column_indices):
        iou = float(matrix[row_index, column_index])
        if iou >= float(minimum_iou):
            matches.append((
                gt_order[int(row_index)],
                prediction_order[int(column_index)],
                iou,
            ))
    return sorted(matches)


def deterministic_identity_mapping(records):
    """Map predicted IDs one-to-one to GT IDs by maximum record overlap."""
    if not records:
        return {}
    from scipy.optimize import linear_sum_assignment

    gt_ids = sorted({str(row["dataset_identity_key"]) for row in records})
    predicted_ids = sorted({str(row["predicted_global_identity"]) for row in records})
    counts = Counter(
        (str(row["dataset_identity_key"]), str(row["predicted_global_identity"]))
        for row in records
    )
    overlap = np.asarray([
        [counts[(gt_id, predicted_id)] for predicted_id in predicted_ids]
        for gt_id in gt_ids
    ], dtype=np.int64)
    row_indices, column_indices = linear_sum_assignment(-overlap)
    return {
        predicted_ids[int(column_index)]: gt_ids[int(row_index)]
        for row_index, column_index in zip(row_indices, column_indices)
        if overlap[int(row_index), int(column_index)] > 0
    }


def evaluate_global_id_records(records):
    """Calculate accuracy, switches, splits and merges from valid records."""
    valid = [
        dict(row) for row in records
        if row.get("predicted_global_identity") not in (None, "")
    ]
    mapping = deterministic_identity_mapping(valid)
    for row in valid:
        mapped = mapping.get(str(row["predicted_global_identity"]))
        row["mapped_gt_identity"] = mapped
        row["global_id_correct"] = mapped == str(row["dataset_identity_key"])

    trajectories = defaultdict(list)
    for row in valid:
        trajectories[(
            str(row["sequence"]),
            str(row["camera"]),
            str(row["dataset_identity_key"]),
        )].append(row)
    switches = []
    for (sequence, camera, identity), rows in sorted(trajectories.items()):
        ordered = sorted(
            rows,
            key=lambda row: (int(row["frame_index"]), str(row["sample_id"])),
        )
        previous = None
        for row in ordered:
            current = str(row["predicted_global_identity"])
            if previous is not None and current != previous[0]:
                switches.append({
                    "sequence": sequence,
                    "camera": camera,
                    "dataset_identity_key": identity,
                    "from_frame": int(previous[1]["frame_index"]),
                    "to_frame": int(row["frame_index"]),
                    "from_predicted_global_identity": previous[0],
                    "to_predicted_global_identity": current,
                    "from_sample_id": previous[1]["sample_id"],
                    "to_sample_id": row["sample_id"],
                })
            previous = (current, row)

    ids_by_gt = defaultdict(set)
    gt_by_id = defaultdict(set)
    for row in valid:
        gt_id = str(row["dataset_identity_key"])
        predicted_id = str(row["predicted_global_identity"])
        ids_by_gt[gt_id].add(predicted_id)
        gt_by_id[predicted_id].add(gt_id)
    splits = [
        {
            "dataset_identity_key": gt_id,
            "predicted_global_id_count": len(predicted_ids),
            "excess_predicted_ids": len(predicted_ids) - 1,
            "predicted_global_identities": sorted(predicted_ids),
        }
        for gt_id, predicted_ids in sorted(ids_by_gt.items())
        if len(predicted_ids) > 1
    ]
    merges = [
        {
            "predicted_global_identity": predicted_id,
            "gt_identity_count": len(gt_ids_for_prediction),
            "dataset_identity_keys": sorted(gt_ids_for_prediction),
        }
        for predicted_id, gt_ids_for_prediction in sorted(gt_by_id.items())
        if len(gt_ids_for_prediction) > 1
    ]
    correct = sum(bool(row["global_id_correct"]) for row in valid)
    return {
        "records": valid,
        "mapping": mapping,
        "switches": switches,
        "splits": splits,
        "merges": merges,
        "metrics": {
            "valid_global_id_records": len(valid),
            "correct_global_ids": correct,
            "global_id_accuracy": correct / len(valid) if valid else 0.0,
            "id_switches": len(switches),
            "identity_splits": len(splits),
            "identity_split_excess_predicted_ids": sum(
                row["excess_predicted_ids"] for row in splits
            ),
            "identity_merges": len(merges),
            "identity_merge_affected_gt_identities": len({
                identity
                for row in merges
                for identity in row["dataset_identity_keys"]
            }),
        },
    }
