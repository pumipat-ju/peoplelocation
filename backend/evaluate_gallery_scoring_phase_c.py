"""Offline, appearance-only gallery-scoring ablation. Production code is untouched.

Run with ``python -m backend.evaluate_gallery_scoring_phase_c``. The fixed
selection, checkpoint, threshold, and episode protocol are shared by variants.
This is not a full camera/GlobalIdentityManager replay.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate_events
from .evaluate_reid_verification import compute_eer, compute_roc
from .reid.gallery import (
    gallery_similarity,
    normalize_embedding_candidate,
    robust_identity_prototype,
)


VARIANTS = ("bottom-3", "top-3", "mean-all", "median", "top-k-mean-margin")
DIVERSITY_THRESHOLD = 0.985
PROTOTYPE_WEIGHT = 0.75
SUPPORT_WEIGHT = 0.25
MIN_CONSENSUS = 0.70
APPEARANCE_THRESHOLD = 0.55  # Existing conservative cross-camera gate.
MARGIN_GUARD = 0.08  # Existing cross-camera assignment margin.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced(records, limit):
    if len(records) <= limit:
        return list(records)
    return [records[int(index)] for index in np.linspace(0, len(records) - 1, limit)]


def select_protocol(manifest, gallery_limit=12, query_limit=8):
    groups = defaultdict(list)
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[(row["dataset_identity_key"], row["camera"])].append(row)
    identities = sorted({key for key, _camera in groups})
    galleries, queries, query_relation = {}, [], {}
    for identity in identities:
        cameras = sorted(camera for key, camera in groups if key == identity)
        gallery_camera = cameras[0]
        source = sorted(groups[(identity, gallery_camera)],
                        key=lambda row: (row["frame_index"], row["sample_id"]))
        galleries[identity] = evenly_spaced(source[:max(gallery_limit, len(source) // 2)],
                                             gallery_limit)
        if len(cameras) > 1:
            query_camera = cameras[1]
            pool = sorted(groups[(identity, query_camera)],
                          key=lambda row: (row["frame_index"], row["sample_id"]))
            query_relation[identity] = "cross_camera"
        else:
            pool = source[len(source) // 2:]
            query_relation[identity] = "same_camera_reentry_proxy"
        chosen = evenly_spaced(pool, query_limit)
        if len(galleries[identity]) < 3 or len(chosen) < 2:
            raise ValueError(f"Insufficient samples for {identity}")
        queries.extend(chosen)
    ids = [row["sample_id"] for rows in galleries.values() for row in rows]
    ids.extend(row["sample_id"] for row in queries)
    if len(ids) != len(set(ids)):
        raise ValueError("Gallery/query sample leakage")
    return galleries, queries, query_relation


def deduplicated_candidates(identity, query_size):
    raw = list(identity.get("gallery", []) or [])
    if identity.get("embedding") is not None:
        raw.append(identity["embedding"])
    candidates = []
    for value in raw:
        candidate = normalize_embedding_candidate(value, query_size)
        if candidate is None:
            continue
        if any(float(np.dot(candidate, old)) >= DIVERSITY_THRESHOLD for old in candidates):
            continue
        candidates.append(candidate)
    return candidates


def support_score(raw_scores, variant):
    values = sorted(float(score) for score in raw_scores)
    if not values:
        raise ValueError("support_score needs at least one candidate")
    k = min(3, len(values))
    if variant == "bottom-3":
        return float(np.median(values[:k]))  # Exact production support statistic.
    if variant == "top-3":
        return float(np.median(values[-k:]))
    if variant == "mean-all":
        return float(np.mean(values))
    if variant == "median":
        return float(np.median(values))
    if variant == "top-k-mean-margin":
        return float(np.mean(values[-k:]))
    raise ValueError(f"Unknown variant: {variant}")


def score_gallery(query_value, identity, variant):
    query = normalize_embedding_candidate(query_value)
    if query is None:
        return -1.0
    candidates = deduplicated_candidates(identity, query.size)
    if not candidates:
        return -1.0
    raw_scores = [float(np.clip(np.dot(query, item), -1, 1)) for item in candidates]
    support = support_score(raw_scores, variant)
    if len(candidates) < 2:
        return support
    prototype, consensus, _ = robust_identity_prototype(
        candidates, min_samples=2, min_consensus=MIN_CONSENSUS)
    if prototype is None:
        return support
    prototype_score = float(np.clip(np.dot(query, prototype), -1, 1))
    if consensus < MIN_CONSENSUS:
        return min(prototype_score, support)
    return float(np.clip(PROTOTYPE_WEIGHT * prototype_score
                         + SUPPORT_WEIGHT * support, -1, 1))


def build_identities(galleries, features):
    identities = {}
    for key, rows in galleries.items():
        vectors = [features[row["sample_id"]] for row in rows]
        aggregate = normalize_embedding_candidate(np.mean(vectors, axis=0))
        identities[key] = {"gallery": vectors, "embedding": aggregate}
    return identities


def score_matrix(queries, identities, features, variant):
    keys = sorted(identities)
    matrix = np.array([
        [score_gallery(features[row["sample_id"]], identities[key], variant)
         for key in keys]
        for row in queries
    ], dtype=np.float64)
    return keys, matrix


def decide(scores, variant):
    order = np.argsort(-np.asarray(scores), kind="stable")
    top1, top2 = float(scores[order[0]]), float(scores[order[1]])
    if top1 < APPEARANCE_THRESHOLD:
        return None
    if variant == "top-k-mean-margin" and top1 - top2 < MARGIN_GUARD:
        return None
    return int(order[0])


def evaluate_variant(queries, keys, matrix, variant, relation):
    labels = np.array([[row["dataset_identity_key"] == key for key in keys]
                       for row in queries], dtype=bool)
    ranks = []
    for index, scores in enumerate(matrix):
        ordering = np.argsort(-scores, kind="stable")
        ranks.append(int(np.flatnonzero(labels[index, ordering])[0]) + 1)
    fpr, tpr, thresholds, auc = compute_roc(matrix.ravel(), labels.ravel())
    eer, _ = compute_eer(fpr, tpr, thresholds)

    by_person = defaultdict(list)
    for index, row in enumerate(queries):
        by_person[row["dataset_identity_key"]].append(index)
    truth, predictions = [], []
    next_gid = len(keys) + 1
    handoff_total = handoff_recovered = 0
    for key in keys:
        base_gid = keys.index(key) + 1
        event_id = f"{key}:gallery"
        truth.append({"event_id": event_id, "person_id": key, "event_time": -1.0})
        predictions.append({"event_id": event_id, "global_id": base_gid})
        indices = by_person[key]
        midpoint = len(indices) // 2
        for episode, chunk in enumerate((indices[:midpoint], indices[midpoint:]), start=1):
            episode_scores = np.mean(matrix[chunk], axis=0)
            decision = decide(episode_scores, variant)
            gid = decision + 1 if decision is not None else next_gid
            if decision is None:
                next_gid += 1
            event_id = f"{key}:episode-{episode}"
            handoff = relation[key] == "cross_camera"
            handoff_total += int(handoff)
            handoff_recovered += int(handoff and gid == base_gid)
            truth.append({"event_id": event_id, "person_id": key,
                          "event_time": float(episode), "handoff": handoff,
                          "expected_gid": base_gid})
            predictions.append({"event_id": event_id, "global_id": gid})
    event_report = evaluate_events(truth, predictions)
    counts = event_report["counts"]
    return {
        "id_fragmentation_false_splits": counts["false_splits"],
        "false_merges": counts["false_merges"],
        "id_switches": counts["temporal_id_switches"],
        "handoff_recovered": handoff_recovered,
        "handoff_total": handoff_total,
        "handoff_recovery": handoff_recovered / handoff_total,
        "rank_1": float(np.mean(np.array(ranks) <= 1)),
        "rank_5": float(np.mean(np.array(ranks) <= 5)),
        "mAP": float(np.mean(1.0 / np.array(ranks))),
        "roc_auc": auc,
        "eer": eer,
        "queries": len(queries), "gallery_identities": len(keys),
        "verification_pairs": int(labels.size),
        "positive_pairs": int(labels.sum()),
        "negative_pairs": int((~labels).sum()),
        "episode_events": len(keys) * 2,
    }


def extract_features(records, root, checkpoint, batch_size=32):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from .evaluate_osnet_heldout import build_finetuned_model
    from .finetune_osnet import validation_transform

    class Crops(Dataset):
        def __init__(self):
            self.transform = validation_transform()

        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            row = records[index]
            path = root / row["source_image"]
            with Image.open(path) as image:
                crop = image.convert("RGB").crop(tuple(row["bbox_xyxy"]))
            if crop.width <= 1 or crop.height <= 1:
                raise ValueError(f"Invalid crop: {row['sample_id']}")
            return self.transform(crop), row["sample_id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_finetuned_model(checkpoint, device)
    loader = DataLoader(Crops(), batch_size=batch_size, shuffle=False, num_workers=0)
    features = {}
    with torch.inference_mode():
        for images, sample_ids in loader:
            output = model(images.to(device))
            output = torch.nn.functional.normalize(output, p=2, dim=1)
            for sample_id, vector in zip(sample_ids, output.cpu().numpy()):
                features[sample_id] = vector.astype(np.float32)
    return features, str(device)


def run(manifest, checkpoint, root, gallery_limit=12, query_limit=8):
    galleries, queries, relation = select_protocol(manifest, gallery_limit, query_limit)
    records = [row for rows in galleries.values() for row in rows] + queries
    features, device = extract_features(records, root, checkpoint)
    identities = build_identities(galleries, features)
    candidate_counts = {
        key: len(deduplicated_candidates(identity, 512))
        for key, identity in identities.items()
    }
    results = {}
    for variant in VARIANTS:
        keys, matrix = score_matrix(queries, identities, features, variant)
        results[variant] = evaluate_variant(queries, keys, matrix, variant, relation)
    return {
        "protocol": "fixed GT-crop gallery and query; appearance-only episode proxy",
        "limitations": [
            "Not a full GlobalIdentityManager/camera replay; motion, topology, overlap and Hungarian assignment are excluded.",
            "Gallery crops are GT crops, not quality-approved online tracklet prototypes.",
            "Seven identities only; some were in OSNet fine-tuning data. Exploratory, not held-out generalization.",
            "Rank-5 has only seven gallery identities and is weakly discriminative.",
            "Runtime two-person trace has no per-observation embeddings, so exact replay is unavailable.",
        ],
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "device": device, "gallery_limit": gallery_limit, "query_limit": query_limit,
        "crop_policy": "raw GT person bbox; OSNet validation resize/normalization",
        "appearance_threshold": APPEARANCE_THRESHOLD,
        "top_k_margin_guard": MARGIN_GUARD,
        "gallery_candidate_counts": candidate_counts,
        "query_relation": relation,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("reid_dataset/master_manifest_v1.jsonl"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("weights/osnet_x1_0_peoplelocation_balanced_v2.pth"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("backend/reid_experiments/phase_c_gallery_ablation.json"))
    args = parser.parse_args()
    report = run(args.manifest, args.checkpoint, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for variant, metrics in report["results"].items():
        print(variant, json.dumps(metrics, sort_keys=True))
    print("report:", args.output)


if __name__ == "__main__":
    main()
