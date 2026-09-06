"""Offline cosine-similarity matrix analysis for dumped Re-ID embeddings."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


def l2_normalize_rows(embeddings, eps=1e-12):
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain NaN or Inf")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= eps):
        raise ValueError("embeddings contain a zero or near-zero row")
    return matrix / norms[:, None], norms


def distribution_stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def pair_payload(similarity, left, right, label):
    return {
        "similarity": float(similarity),
        "label": label,
        "camera_relation": (
            "same_camera" if left["camera"] == right["camera"] else "cross_camera"
        ),
        "sample_a": {
            "sample_id": left["sample_id"],
            "ground_truth_person_id": left["ground_truth_person_id"],
            "camera": left["camera"],
            "source_image": left.get("source_image"),
            "frame_index": left.get("frame_index"),
        },
        "sample_b": {
            "sample_id": right["sample_id"],
            "ground_truth_person_id": right["ground_truth_person_id"],
            "camera": right["camera"],
            "source_image": right.get("source_image"),
            "frame_index": right.get("frame_index"),
        },
    }


def analyze_similarity(matrix, records, top_k=10):
    matrix = np.asarray(matrix, dtype=np.float64)
    sample_count = len(records)
    if matrix.shape != (sample_count, sample_count):
        raise ValueError("similarity matrix shape does not match metadata")
    if not np.isfinite(matrix).all():
        raise ValueError("similarity matrix contains NaN or Inf")
    if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-6):
        raise ValueError("similarity matrix is not symmetric")

    diagonal = np.diag(matrix)
    same_pairs = []
    different_pairs = []
    for left_index in range(sample_count):
        for right_index in range(left_index + 1, sample_count):
            left = records[left_index]
            right = records[right_index]
            same_identity = (
                str(left["ground_truth_person_id"])
                == str(right["ground_truth_person_id"])
            )
            payload = pair_payload(
                matrix[left_index, right_index],
                left,
                right,
                "same_gt" if same_identity else "different_gt",
            )
            (same_pairs if same_identity else different_pairs).append(payload)

    same_values = [item["similarity"] for item in same_pairs]
    different_values = [item["similarity"] for item in different_pairs]
    same_stats = distribution_stats(same_values)
    different_stats = distribution_stats(different_values)
    gap = None
    if same_stats["mean"] is not None and different_stats["mean"] is not None:
        gap = same_stats["mean"] - different_stats["mean"]

    stratified = {}
    for relation in ("same_camera", "cross_camera"):
        relation_same = [
            item["similarity"] for item in same_pairs
            if item["camera_relation"] == relation
        ]
        relation_different = [
            item["similarity"] for item in different_pairs
            if item["camera_relation"] == relation
        ]
        relation_same_stats = distribution_stats(relation_same)
        relation_different_stats = distribution_stats(relation_different)
        relation_gap = None
        if (
            relation_same_stats["mean"] is not None
            and relation_different_stats["mean"] is not None
        ):
            relation_gap = (
                relation_same_stats["mean"] - relation_different_stats["mean"]
            )
        stratified[relation] = {
            "same_id": relation_same_stats,
            "different_id": relation_different_stats,
            "similarity_gap": relation_gap,
        }

    cross_same = [item for item in same_pairs if item["camera_relation"] == "cross_camera"]
    cross_different = [
        item for item in different_pairs if item["camera_relation"] == "cross_camera"
    ]

    return {
        "diagonal": distribution_stats(diagonal),
        "same_id_off_diagonal": same_stats,
        "different_id": different_stats,
        "similarity_gap": gap,
        "stratified_by_camera_relation": stratified,
        "hard_positives": sorted(same_pairs, key=lambda item: item["similarity"])[:top_k],
        "hard_negatives": sorted(
            different_pairs, key=lambda item: item["similarity"], reverse=True
        )[:top_k],
        "cross_camera_hard_positives": sorted(
            cross_same, key=lambda item: item["similarity"]
        )[:top_k],
        "cross_camera_hard_negatives": sorted(
            cross_different, key=lambda item: item["similarity"], reverse=True
        )[:top_k],
        "pair_counting": "unique unordered pairs only (i < j); diagonal excluded",
        "label_source": "ground_truth_person_id only",
    }


def load_dump_records(input_root):
    records = []
    metadata_directory = input_root / "metadata"
    embedding_directory = input_root / "embeddings"
    for metadata_path in sorted(metadata_directory.glob("*.json")):
        if metadata_path.name == "summary.json":
            continue
        with metadata_path.open("r", encoding="utf-8") as source:
            metadata = json.load(source)
        for key in ("sample_id", "ground_truth_person_id", "camera"):
            if metadata.get(key) in (None, ""):
                raise ValueError(f"{metadata_path} is missing {key}")
        embedding_path = embedding_directory / f"{metadata['sample_id']}.npy"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Missing embedding for {metadata['sample_id']}")
        metadata["resolved_embedding_file"] = str(embedding_path.resolve())
        records.append(metadata)
    if not records:
        raise ValueError(f"No sample metadata found under {metadata_directory}")
    return records


def select_records(records, max_samples=None, sample_ids=None):
    by_id = {record["sample_id"]: record for record in records}
    if sample_ids:
        missing = sorted(set(sample_ids) - set(by_id))
        if missing:
            raise ValueError(f"Unknown sample IDs: {missing}")
        return [by_id[sample_id] for sample_id in sample_ids]
    if max_samples is None or max_samples >= len(records):
        return list(records)
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")

    grouped = defaultdict(list)
    for record in records:
        grouped[(str(record["ground_truth_person_id"]), record["camera"])].append(record)
    for group in grouped.values():
        group.sort(key=lambda item: (item.get("frame_index", 0), item["sample_id"]))

    selected = []
    depth = 0
    keys = sorted(grouped)
    while len(selected) < max_samples:
        added = False
        for key in keys:
            group = grouped[key]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        depth += 1
    return selected


def write_matrix_csv(path, matrix, records):
    sample_ids = [record["sample_id"] for record in records]
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["sample_id", *sample_ids])
        for sample_id, row in zip(sample_ids, matrix):
            writer.writerow([sample_id, *[f"{float(value):.9f}" for value in row]])


def write_metadata_outputs(output_directory, records):
    fields = [
        "matrix_index", "sample_id", "ground_truth_person_id", "camera",
        "frame_index", "source_image", "resolved_embedding_file",
    ]
    rows = []
    for index, record in enumerate(records):
        rows.append({field: record.get(field) for field in fields})
        rows[-1]["matrix_index"] = index
    with (output_directory / "matrix_metadata.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output_directory / "matrix_metadata.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(rows, destination, indent=2, sort_keys=True)


def format_pair(pair):
    left = pair["sample_a"]
    right = pair["sample_b"]
    return (
        f"{pair['similarity']:.6f} | {pair['label']} | "
        f"A={left['sample_id']} GT={left['ground_truth_person_id']} "
        f"cam={left['camera']} frame={left['frame_index']} | "
        f"B={right['sample_id']} GT={right['ground_truth_person_id']} "
        f"cam={right['camera']} frame={right['frame_index']}"
    )


def run(input_root, output_directory, max_samples=None, sample_ids=None, top_k=10):
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    records = select_records(load_dump_records(input_root), max_samples, sample_ids)
    embeddings = np.stack([
        np.load(record["resolved_embedding_file"]).reshape(-1) for record in records
    ])
    normalized, original_norms = l2_normalize_rows(embeddings)
    matrix = normalized @ normalized.T

    checks = {
        "all_input_embeddings_finite": bool(np.isfinite(embeddings).all()),
        "all_input_embeddings_unit_norm": bool(
            np.allclose(original_norms, 1.0, rtol=1e-5, atol=1e-5)
        ),
        "matrix_finite": bool(np.isfinite(matrix).all()),
        "matrix_symmetric": bool(np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-6)),
        "diagonal_approximately_one": bool(
            np.allclose(np.diag(matrix), 1.0, rtol=1e-5, atol=1e-5)
        ),
    }
    if not checks["matrix_finite"] or not checks["matrix_symmetric"]:
        raise RuntimeError(f"Similarity correctness check failed: {checks}")

    analysis = analyze_similarity(matrix, records, top_k=top_k)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(output_directory / "similarity_matrix.csv", matrix, records)
    write_metadata_outputs(output_directory, records)
    report = {
        "matrix_shape": list(matrix.shape),
        "embedding_dimension": int(normalized.shape[1]),
        "identities": sorted({str(item["ground_truth_person_id"]) for item in records}),
        "cameras": sorted({item["camera"] for item in records}),
        "checks": checks,
        "input_norms": distribution_stats(original_norms),
        **analysis,
    }
    with (output_directory / "similarity_summary.json").open(
        "w", encoding="utf-8"
    ) as destination:
        json.dump(report, destination, indent=2, sort_keys=True)

    print(f"matrix size: {matrix.shape[0]} x {matrix.shape[1]}")
    print(f"identities: {', '.join(report['identities'])}")
    print(f"cameras: {', '.join(report['cameras'])}")
    print(f"diagonal stats: {report['diagonal']}")
    print(f"same-ID off-diagonal stats: {report['same_id_off_diagonal']}")
    print(f"different-ID stats: {report['different_id']}")
    print(f"similarity gap: {report['similarity_gap']}")
    print(f"camera-relation strata: {report['stratified_by_camera_relation']}")
    print(f"checks: {checks}")
    print("hard positives (lowest same-ID):")
    for pair in report["hard_positives"]:
        print(f"  {format_pair(pair)}")
    print("hard negatives (highest different-ID):")
    for pair in report["hard_negatives"]:
        print(f"  {format_pair(pair)}")
    print(f"output directory: {output_directory.resolve()}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an offline cosine-similarity matrix from Prompt 03 dumps."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("debug_reid"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    input_directory = arguments.input_dir.resolve()
    output_directory = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else input_directory / "similarity"
    )
    run(
        input_root=input_directory,
        output_directory=output_directory,
        max_samples=arguments.max_samples,
        sample_ids=arguments.sample_id,
        top_k=arguments.top_k,
    )
