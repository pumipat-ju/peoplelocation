"""Prepare the m1/m2 videos as an offline, review-first Re-ID dataset."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


MOT_COLUMNS = 10
REVIEW_STATUS = "NEEDS_HUMAN_REVIEW"


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return {
        "path": path.as_posix(),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extraction_plan(metadata, source_offset_seconds, sample_fps, overlap_seconds):
    sample_count = int(math.floor(overlap_seconds * sample_fps)) + 1
    plan = []
    for dataset_frame in range(1, sample_count + 1):
        sequence_time = (dataset_frame - 1) / sample_fps
        source_time = sequence_time + source_offset_seconds
        source_frame_zero_based = int(round(source_time * metadata["fps"]))
        if source_frame_zero_based >= metadata["frame_count"]:
            break
        plan.append({
            "dataset_frame": dataset_frame,
            "sequence_time_seconds": sequence_time,
            "source_time_seconds": source_time,
            "source_frame_zero_based": source_frame_zero_based,
            "source_frame_one_based": source_frame_zero_based + 1,
        })
    return plan


def extract_camera(video_path, images_directory, plan, jpeg_quality):
    images_directory.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    rows = []
    try:
        for item in plan:
            capture.set(cv2.CAP_PROP_POS_FRAMES, item["source_frame_zero_based"])
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(
                    f"Could not decode {video_path} frame "
                    f"{item['source_frame_zero_based']}"
                )
            output_path = images_directory / f"{item['dataset_frame']:06d}.jpg"
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite {output_path}")
            if not cv2.imwrite(
                str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            ):
                raise RuntimeError(f"Could not write {output_path}")
            rows.append({
                **item,
                "image": output_path.name,
                "image_sha256": sha256_file(output_path),
            })
    finally:
        capture.release()
    return rows


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_and_track(images_directory, weights, confidence, device):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    detections = []
    for image_path in sorted(images_directory.glob("*.jpg")):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        result = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            conf=confidence,
            iou=0.55,
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue
        frame_number = int(image_path.stem)
        for xyxy, track_id, score in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.id.int().cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):
            x1, y1, x2, y2 = xyxy
            x1 = max(0, min(frame.shape[1] - 1, int(math.floor(x1))))
            y1 = max(0, min(frame.shape[0] - 1, int(math.floor(y1))))
            x2 = max(x1 + 1, min(frame.shape[1], int(math.ceil(x2))))
            y2 = max(y1 + 1, min(frame.shape[0], int(math.ceil(y2))))
            detections.append({
                "frame": frame_number,
                "local_track_id": int(track_id),
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(score),
            })
    return detections


def write_candidate_mot(path, detections):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        for item in detections:
            writer.writerow([
                item["frame"], item["local_track_id"], item["x"], item["y"],
                item["width"], item["height"], f"{item['confidence']:.6f}",
                1, 1.0, -1,
            ])


def evenly_spaced(items, count):
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[index] for index in indices]


def crop_tile(frame, detection, camera, tile_width=220, tile_height=300):
    x = detection["x"]
    y = detection["y"]
    crop = frame[y:y + detection["height"], x:x + detection["width"]]
    if crop.size == 0:
        raise ValueError("empty crop")
    header = 50
    tile = np.full((tile_height + header, tile_width, 3), 245, dtype=np.uint8)
    scale = min(tile_width / crop.shape[1], tile_height / crop.shape[0])
    resized = cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)),
         max(1, round(crop.shape[0] * scale))),
    )
    left = (tile_width - resized.shape[1]) // 2
    top = header + (tile_height - resized.shape[0]) // 2
    tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    label1 = f"{camera} frame {detection['frame']} track {detection['local_track_id']}"
    label2 = (
        f"xywh {detection['x']},{detection['y']},"
        f"{detection['width']},{detection['height']}"
    )
    cv2.putText(tile, label1, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(tile, label2, (5, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def make_track_sheets(camera, images_directory, detections, review_directory,
                      max_tracks=8, samples_per_track=5):
    grouped = defaultdict(list)
    for detection in detections:
        grouped[detection["local_track_id"]].append(detection)
    ranked = sorted(grouped, key=lambda value: (-len(grouped[value]), value))[:max_tracks]
    index_rows = []
    for track_id in ranked:
        selected = evenly_spaced(grouped[track_id], samples_per_track)
        tiles = []
        for detection in selected:
            frame = cv2.imread(
                str(images_directory / f"{detection['frame']:06d}.jpg"),
                cv2.IMREAD_COLOR,
            )
            tiles.append(crop_tile(frame, detection, camera))
        sheet = cv2.hconcat(tiles)
        output_path = review_directory / f"{camera}_track_{track_id:04d}.jpg"
        if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {output_path}")
        frames = [item["frame"] for item in grouped[track_id]]
        index_rows.append({
            "camera": camera,
            "local_track_id": track_id,
            "detection_count": len(frames),
            "first_dataset_frame": min(frames),
            "last_dataset_frame": max(frames),
            "review_sheet": output_path.name,
            "cross_camera_status": REVIEW_STATUS,
        })
    return index_rows


def validate_detections(camera, images_directory, detections):
    errors = []
    seen = set()
    shapes = {}
    for row_number, item in enumerate(detections, start=1):
        key = (camera, item["frame"], item["local_track_id"])
        if key in seen:
            errors.append({"row": row_number, "reason": "duplicate_camera_frame_person"})
        seen.add(key)
        image_path = images_directory / f"{item['frame']:06d}.jpg"
        if not image_path.is_file():
            errors.append({"row": row_number, "reason": "missing_frame"})
            continue
        shape = shapes.get(image_path)
        if shape is None:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                errors.append({"row": row_number, "reason": "invalid_frame"})
                continue
            shape = frame.shape
            shapes[image_path] = shape
        x, y, width, height = (
            item["x"], item["y"], item["width"], item["height"]
        )
        if width <= 0 or height <= 0:
            errors.append({"row": row_number, "reason": "invalid_xywh"})
        elif x < 0 or y < 0 or x + width > shape[1] or y + height > shape[0]:
            errors.append({"row": row_number, "reason": "bbox_out_of_bounds"})
        else:
            crop = cv2.imread(str(image_path), cv2.IMREAD_COLOR)[
                y:y + height, x:x + width
            ]
            if crop.size == 0:
                errors.append({"row": row_number, "reason": "empty_crop"})
    return errors


def write_readme(output_root, summary):
    text = f"""# m_sequence Offline Re-ID Dataset

This dataset was extracted offline. It is not connected to production runtime.

## Alignment and sampling

- `cam1`: `{summary['sources']['cam1']['path']}` (`m1.mp4`)
- `cam2`: `{summary['sources']['cam2']['path']}` (`m2.mp4`)
- Evidence from burned-in timestamps: `cam2` starts about 5 seconds before `cam1`.
- Sequence time 0 maps to cam1 source time 0 and cam2 source time 5 seconds.
- Sampling rate: `{summary['sample_fps']}` FPS.
- Extracted frames per camera: `{summary['extracted_frames_per_camera']}`.

## Annotation state

`gt1.txt` and `gt2.txt` are intentionally empty until a human confirms global person IDs.
The `candidate_local_tracks.txt` files use MOT columns
`[frame,local_track_id,x,y,width,height,confidence,class,visibility,unused]` and are
detector/tracker suggestions, not ground truth. Every cross-camera candidate remains
`{REVIEW_STATUS}`. Copy only reviewed rows into GT and replace local IDs with the same
global ID in both cameras.

Validation must be rerun after review. Do not train from candidate files.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8", newline="\n")


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {args.output}")
    if args.sample_fps <= 0 or args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError("Invalid sampling or JPEG settings")
    sources = {"cam1": args.cam1.resolve(), "cam2": args.cam2.resolve()}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = {camera: video_metadata(path) for camera, path in sources.items()}
    overlap = min(
        metadata["cam1"]["duration_seconds"] - args.cam1_offset_seconds,
        metadata["cam2"]["duration_seconds"] - args.cam2_offset_seconds,
    )
    if overlap <= 0:
        raise ValueError("The configured camera offsets have no overlapping duration")

    args.output.mkdir(parents=True, exist_ok=False)
    review_directory = args.output / "review"
    review_directory.mkdir()
    extracted = {}
    detections_by_camera = {}
    validation_errors = []
    track_index = []
    for camera in ("cam1", "cam2"):
        camera_directory = args.output / camera
        camera_directory.mkdir()
        images_directory = camera_directory / "images"
        offset = getattr(args, f"{camera}_offset_seconds")
        plan = extraction_plan(metadata[camera], offset, args.sample_fps, overlap)
        extracted[camera] = extract_camera(
            sources[camera], images_directory, plan, args.jpeg_quality
        )
        write_csv(camera_directory / "frame_index_map.csv", extracted[camera])
        (camera_directory / f"gt{camera[-1]}.txt").write_text("", encoding="utf-8")

        detections = detect_and_track(
            images_directory, args.weights.resolve(), args.confidence, args.device
        )
        detections_by_camera[camera] = detections
        write_candidate_mot(camera_directory / "candidate_local_tracks.txt", detections)
        camera_errors = validate_detections(camera, images_directory, detections)
        for error in camera_errors:
            validation_errors.append({"camera": camera, **error})
        track_index.extend(make_track_sheets(
            camera, images_directory, detections, review_directory,
            max_tracks=args.review_tracks,
            samples_per_track=args.review_samples,
        ))

    write_csv(
        review_directory / "candidate_track_index.csv",
        track_index,
        fieldnames=[
            "camera", "local_track_id", "detection_count",
            "first_dataset_frame", "last_dataset_frame", "review_sheet",
            "cross_camera_status",
        ],
    )
    write_csv(
        review_directory / "cross_camera_review.csv",
        [],
        fieldnames=[
            "proposed_person_id", "cam1_local_track_id", "cam2_local_track_id",
            "status", "reviewer", "evidence", "notes",
        ],
    )
    write_csv(
        args.output / "validation_errors.csv",
        validation_errors,
        fieldnames=["camera", "row", "reason"],
    )
    source_report = {}
    for camera in ("cam1", "cam2"):
        source_report[camera] = {
            **metadata[camera],
            "sha256": sha256_file(sources[camera]),
            "source_offset_seconds": getattr(args, f"{camera}_offset_seconds"),
        }
    track_counts = {
        camera: len({item["local_track_id"] for item in detections})
        for camera, detections in detections_by_camera.items()
    }
    summary = {
        "dataset": "m_sequence",
        "offline_only": True,
        "same_sequence_assessment": (
            "CONFIRMED_TWO_VIEWS_BY_MATCHING_ROOM_EVENT_AND_BURNED_IN_TIMESTAMPS"
        ),
        "alignment_evidence": "cam2 burned-in timestamp is 5 seconds earlier than cam1",
        "sources": source_report,
        "sample_fps": args.sample_fps,
        "sampling_interval_seconds": 1.0 / args.sample_fps,
        "overlap_seconds": overlap,
        "extracted_frames_per_camera": {
            camera: len(rows) for camera, rows in extracted.items()
        },
        "candidate_detection_rows": {
            camera: len(rows) for camera, rows in detections_by_camera.items()
        },
        "candidate_local_tracks": track_counts,
        "candidate_cross_camera_identities": 0,
        "confirmed_cross_camera_identities": 0,
        "human_review_status": REVIEW_STATUS,
        "invalid_or_rejected_annotations": len(validation_errors),
        "validation_error_counts": dict(Counter(
            item["reason"] for item in validation_errors
        )),
        "gt_rows": {"cam1": 0, "cam2": 0},
        "fine_tuning_run": False,
        "production_integration": False,
    }
    with (args.output / "summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")
    write_readme(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam1", type=Path, default=Path("clip_data/m1.mp4"))
    parser.add_argument("--cam2", type=Path, default=Path("clip_data/m2.mp4"))
    parser.add_argument("--output", type=Path, default=Path("labeled_data/m_sequence"))
    parser.add_argument("--weights", type=Path, default=Path("backend/yolov8s.pt"))
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--cam1-offset-seconds", type=float, default=0.0)
    parser.add_argument("--cam2-offset-seconds", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--review-tracks", type=int, default=8)
    parser.add_argument("--review-samples", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
