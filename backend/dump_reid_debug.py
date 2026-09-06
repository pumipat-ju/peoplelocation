"""Dump exact production Re-ID crops and embeddings from offline MOT data."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from .osnet_pipeline_audit import summarize_embedding
    from .reid_config import OSNET_DEFAULT_CHECKPOINT_NAME
except ImportError:
    from osnet_pipeline_audit import summarize_embedding
    from reid_config import OSNET_DEFAULT_CHECKPOINT_NAME


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Observation:
    row_index: int
    frame_index: int
    person_id: str
    camera: str
    image_path: Path
    annotation_path: Path
    bbox_xywh: tuple

    @property
    def sample_id(self):
        person = "".join(char for char in self.person_id if char.isalnum() or char in "_-")
        camera = "".join(char for char in self.camera if char.isalnum() or char in "_-")
        return (
            f"{camera}_p{person}_f{self.frame_index:06d}_"
            f"r{self.row_index:06d}"
        )


def validate_and_convert_bbox(frame_shape, bbox_xywh):
    frame_height, frame_width = frame_shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return None, "invalid_frame_shape"

    try:
        x, y, width, height = [float(value) for value in bbox_xywh]
    except (TypeError, ValueError):
        return None, "non_numeric_bbox"

    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None, "non_finite_bbox"
    if width <= 0 or height <= 0:
        return None, "non_positive_bbox_size"

    # Production truncates detector coordinates to integers before cropping.
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + width), int(y + height)
    if x2 <= x1 or y2 <= y1:
        return None, "invalid_xyxy_order"
    if x2 <= 0 or y2 <= 0 or x1 >= frame_width or y1 >= frame_height:
        return None, "bbox_outside_frame"

    clamped = (
        max(0, min(x1, frame_width - 1)),
        max(0, min(y1, frame_height - 1)),
        max(0, min(x2, frame_width)),
        max(0, min(y2, frame_height)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None, "empty_bbox_after_clamp"

    return {
        "input_xywh": [x1, y1, int(width), int(height)],
        "input_xyxy": [x1, y1, x2, y2],
        "frame_clamped_xyxy": list(clamped),
        "was_clamped": clamped != (x1, y1, x2, y2),
        "format_conversion": "MOT xywh -> production xyxy",
    }, None


def crop_quality_flags(crop):
    height, width = crop.shape[:2]
    aspect_ratio = width / max(height, 1)
    flags = []
    if aspect_ratio >= 1.0:
        flags.append("wide_or_square_crop")
    if aspect_ratio >= 2.0 or aspect_ratio <= 0.2:
        flags.append("extreme_aspect_ratio")
    return aspect_ratio, flags


def discover_observations(dataset_root):
    observations = []
    malformed = Counter()
    for camera_directory in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        annotation_files = sorted(camera_directory.glob("gt*.txt"))
        images_directory = camera_directory / "images"
        for annotation_path in annotation_files:
            with annotation_path.open("r", encoding="utf-8", newline="") as source:
                for row_index, row in enumerate(csv.reader(source), start=1):
                    if len(row) < 6:
                        malformed["annotation_has_fewer_than_6_columns"] += 1
                        continue
                    try:
                        frame_index = int(float(row[0]))
                        bbox_xywh = tuple(float(value) for value in row[2:6])
                    except ValueError:
                        malformed["annotation_has_non_numeric_fields"] += 1
                        continue
                    observations.append(Observation(
                        row_index=row_index,
                        frame_index=frame_index,
                        person_id=str(row[1]).strip(),
                        camera=camera_directory.name,
                        image_path=images_directory / f"{frame_index:06d}.jpg",
                        annotation_path=annotation_path,
                        bbox_xywh=bbox_xywh,
                    ))
    return observations, malformed


def select_observations(observations, max_samples):
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")

    grouped = defaultdict(list)
    for observation in observations:
        grouped[(observation.camera, observation.person_id)].append(observation)

    groups = []
    target_per_group = max(1, math.ceil(max_samples / max(1, len(grouped))))
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: (item.frame_index, item.row_index))
        sample_count = min(target_per_group, len(rows))
        if sample_count == 1:
            indices = [len(rows) // 2]
        else:
            indices = [round(index * (len(rows) - 1) / (sample_count - 1)) for index in range(sample_count)]
        groups.append([rows[index] for index in indices])

    selected = []
    depth = 0
    while len(selected) < max_samples:
        added = False
        for group in groups:
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        depth += 1
    return selected


def prepare_output_directories(output_root, overwrite=False):
    if output_root.exists() and any(output_root.rglob("*")) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. "
            "Choose another directory or pass --overwrite."
        )
    directories = {
        "crops": output_root / "crops",
        "embeddings": output_root / "embeddings",
        "metadata": output_root / "metadata",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def write_sample_artifacts(directories, sample_id, crop, embedding, metadata):
    crop_path = directories["crops"] / f"{sample_id}.png"
    embedding_path = directories["embeddings"] / f"{sample_id}.npy"
    metadata_path = directories["metadata"] / f"{sample_id}.json"

    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError(f"Could not write crop: {crop_path}")
    np.save(embedding_path, np.asarray(embedding, dtype=np.float32))

    payload = dict(metadata)
    payload["sample_id"] = sample_id
    payload["crop_file"] = str(crop_path)
    payload["embedding_file"] = str(embedding_path)
    with metadata_path.open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
    return crop_path, embedding_path, metadata_path


def load_production_runtime(checkpoint, device):
    os.environ["IDENTITY_DB_PATH"] = ":memory:"
    os.environ["REID_ENABLED"] = "true"
    os.environ["REID_CHECKPOINT_PATH"] = str(checkpoint)
    os.environ["REID_DEVICE"] = device

    backend_directory = str(Path(__file__).resolve().parent)
    if backend_directory not in sys.path:
        sys.path.insert(0, backend_directory)
    import main
    return main


def close_runtime(main_module):
    manager = getattr(main_module, "global_identity_manager", None)
    store = getattr(manager, "identity_store", None)
    if store is not None:
        store.close()


def dump_dataset(dataset_root, output_root, checkpoint, device, max_samples, overwrite):
    observations, malformed = discover_observations(dataset_root)
    selected = select_observations(observations, max_samples)
    directories = prepare_output_directories(output_root, overwrite=overwrite)
    rejected = Counter()
    pending = []
    main = load_production_runtime(checkpoint, device)

    try:
        for observation in selected:
            frame = cv2.imread(str(observation.image_path), cv2.IMREAD_COLOR)
            if frame is None:
                rejected["missing_or_unreadable_image"] += 1
                continue

            bbox, reason = validate_and_convert_bbox(frame.shape, observation.bbox_xywh)
            if reason is not None:
                rejected[reason] += 1
                continue

            crop = main.extract_person_crop(frame, *bbox["input_xyxy"])
            if crop is None or crop.size == 0:
                rejected["empty_production_crop"] += 1
                continue
            crop_height, crop_width = crop.shape[:2]
            if crop_height < main.REID_MIN_CROP_SIZE or crop_width < main.REID_MIN_CROP_SIZE:
                rejected["production_crop_below_minimum_size"] += 1
                continue
            pending.append((observation, bbox, crop))

        embeddings = main.appearance_extractor.extract_batch(
            [item[2] for item in pending]
        )
        dumped_metadata = []
        dumped_embeddings = []
        for (observation, bbox, crop), embedding in zip(pending, embeddings):
            if embedding is None:
                rejected["embedding_extraction_failed"] += 1
                continue

            stats = summarize_embedding(embedding)
            crop_aspect_ratio, crop_flags = crop_quality_flags(crop)
            metadata = {
                "sample_id": observation.sample_id,
                "source_image": str(observation.image_path.resolve()),
                "source_annotations": str(observation.annotation_path.resolve()),
                "frame_index": observation.frame_index,
                "timestamp": None,
                "camera": observation.camera,
                "ground_truth_person_id": observation.person_id,
                "local_track_id": None,
                "global_id": None,
                "bbox": bbox,
                "crop_width": int(crop.shape[1]),
                "crop_height": int(crop.shape[0]),
                "crop_aspect_ratio": crop_aspect_ratio,
                "crop_quality_flags": crop_flags,
                "crop_color_order": "BGR (exact pre-preprocessing production crop)",
                "embedding_dimension": int(np.asarray(embedding).size),
                "embedding_l2_norm": stats["L2 norm"],
                "embedding_min": stats["min"],
                "embedding_max": stats["max"],
                "embedding_mean": stats["mean"],
                "embedding_std": stats["std"],
                "embedding_contains_nan": stats["contains_nan"],
                "embedding_contains_inf": stats["contains_inf"],
                "checkpoint_path": main.REID_RUNTIME_STATUS["checkpoint_path"],
                "model_name": main.REID_RUNTIME_STATUS["model_architecture"],
                "device": main.REID_RUNTIME_STATUS["device"],
                "preprocessing": main.REID_RUNTIME_STATUS["preprocessing"],
            }
            write_sample_artifacts(
                directories, observation.sample_id, crop, embedding, metadata
            )
            dumped_metadata.append(metadata)
            dumped_embeddings.append(np.asarray(embedding, dtype=np.float32))

        norms = [float(np.linalg.norm(item)) for item in dumped_embeddings]
        summary = {
            "samples_requested": len(selected),
            "max_samples": max_samples,
            "available_observations": len(observations),
            "valid_crops_dumped": len(dumped_embeddings),
            "invalid_crops_rejected": int(sum(rejected.values())),
            "rejection_reasons": dict(sorted(rejected.items())),
            "annotation_discovery_issues": dict(sorted(malformed.items())),
            "unique_gt_identities": sorted({item[0].person_id for item in pending}),
            "unique_cameras": sorted({item[0].camera for item in pending}),
            "suspicious_crop_flag_counts": dict(sorted(Counter(
                flag
                for metadata in dumped_metadata
                for flag in metadata["crop_quality_flags"]
            ).items())),
            "suspicious_crop_samples": [
                metadata["sample_id"]
                for metadata in dumped_metadata
                if metadata["crop_quality_flags"]
            ],
            "embedding_norm_min": min(norms) if norms else None,
            "embedding_norm_mean": float(np.mean(norms)) if norms else None,
            "embedding_norm_max": max(norms) if norms else None,
            "nan_count": int(sum(np.isnan(item).sum() for item in dumped_embeddings)),
            "inf_count": int(sum(np.isinf(item).sum() for item in dumped_embeddings)),
            "checkpoint_path": main.REID_RUNTIME_STATUS["checkpoint_path"],
            "model_name": main.REID_RUNTIME_STATUS["model_architecture"],
        }
        summary_path = directories["metadata"] / "summary.json"
        with summary_path.open("w", encoding="utf-8") as destination:
            json.dump(summary, destination, indent=2, sort_keys=True)

        print(f"samples requested: {summary['samples_requested']}")
        print(f"valid crops dumped: {summary['valid_crops_dumped']}")
        print(f"invalid crops rejected: {summary['invalid_crops_rejected']}")
        print(f"unique GT identities: {len(summary['unique_gt_identities'])}")
        print(f"unique cameras: {len(summary['unique_cameras'])}")
        print(
            "embedding norm min/mean/max: "
            f"{summary['embedding_norm_min']}/"
            f"{summary['embedding_norm_mean']}/"
            f"{summary['embedding_norm_max']}"
        )
        print(f"NaN count: {summary['nan_count']}")
        print(f"Inf count: {summary['inf_count']}")
        print(f"output directory: {output_root.resolve()}")
        return summary
    finally:
        close_runtime(main)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump exact OSNet inputs and embeddings from offline MOT annotations."
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "labeled_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug_reid")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights" / OSNET_DEFAULT_CHECKPOINT_NAME,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    dump_dataset(
        dataset_root=arguments.dataset_root.resolve(),
        output_root=arguments.output_dir.resolve(),
        checkpoint=arguments.checkpoint.resolve(),
        device=arguments.device,
        max_samples=arguments.max_samples,
        overwrite=arguments.overwrite,
    )
