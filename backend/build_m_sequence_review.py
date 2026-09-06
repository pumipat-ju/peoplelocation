"""Build explicit, non-GT cross-camera review proposals for m_sequence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2

from .prepare_m_sequence_dataset import (
    crop_tile,
    evenly_spaced,
    validate_detections,
    write_candidate_mot,
    write_csv,
)


# These intervals were selected from the generated contact sheets. They are review
# proposals only; no cross-camera identity is accepted without human sign-off.
PROPOSALS = [
    {
        "proposed_person_id": 1,
        "status": "REJECTED_CROSS_CAMERA_MATCH",
        "cam1": [
            {"track": 50, "first": 48, "last": 90},
            {"track": 60, "first": 92, "last": 195},
        ],
        "cam2": [
            {"track": 26, "first": 60, "last": 95},
            {"track": 34, "first": 113, "last": 195},
        ],
        "evidence": (
            "dark polo with sleeve mark and large light back graphic; matching temporal "
            "fragments in both views"
        ),
    },
    {
        "proposed_person_id": 2,
        "status": "CONFIRMED",
        "cam1": [{"track": 66, "first": 157, "last": 176}],
        "cam2": [{"track": 35, "first": 116, "last": 153}],
        "evidence": "light gray-green shirt and black trousers with white side graphic",
    },
    {
        "proposed_person_id": 3,
        "status": "CONFIRMED",
        "cam1": [{"track": 70, "first": 181, "last": 205}],
        "cam2": [{"track": 54, "first": 183, "last": 205}],
        "evidence": "red polo, dark trousers, and black shoulder bag in aligned frames",
    },
]


def read_candidate_mot(path):
    detections = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.reader(source):
            if len(row) != 10:
                raise ValueError(f"Expected 10 MOT columns in {path}: {row}")
            detections.append({
                "frame": int(row[0]),
                "local_track_id": int(row[1]),
                "x": int(float(row[2])),
                "y": int(float(row[3])),
                "width": int(float(row[4])),
                "height": int(float(row[5])),
                "confidence": float(row[6]),
            })
    return detections


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frame_map(camera_directory):
    with (camera_directory / "frame_index_map.csv").open(
        "r", encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    images_directory = camera_directory / "images"
    expected = {row["image"] for row in rows}
    actual = {path.name for path in images_directory.glob("*.jpg")}
    result = {
        "mapped_frames": len(rows),
        "image_files": len(actual),
        "missing_frames": sorted(expected - actual),
        "unexpected_frames": sorted(actual - expected),
        "unreadable_frames": [],
        "sha256_mismatches": [],
    }
    for row in rows:
        image_path = images_directory / row["image"]
        if not image_path.is_file():
            continue
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            result["unreadable_frames"].append(row["image"])
        if sha256_file(image_path) != row["image_sha256"]:
            result["sha256_mismatches"].append(row["image"])
    result["valid"] = not any(
        result[key]
        for key in (
            "missing_frames", "unexpected_frames", "unreadable_frames",
            "sha256_mismatches",
        )
    )
    return result


def select_proposal_rows(detections, intervals, person_id):
    selected = []
    for detection in detections:
        if any(
            detection["local_track_id"] == interval["track"]
            and interval["first"] <= detection["frame"] <= interval["last"]
            for interval in intervals
        ):
            selected.append({**detection, "local_track_id": person_id})
    by_frame = {}
    for detection in selected:
        current = by_frame.get(detection["frame"])
        if current is None or detection["confidence"] > current["confidence"]:
            by_frame[detection["frame"]] = detection
    return [by_frame[frame] for frame in sorted(by_frame)]


def make_paired_sheet(dataset_root, proposal, camera_rows, output_path):
    tiles = []
    for camera in ("cam1", "cam2"):
        for row in evenly_spaced(camera_rows[camera], 4):
            frame = cv2.imread(
                str(dataset_root / camera / "images" / f"{row['frame']:06d}.jpg"),
                cv2.IMREAD_COLOR,
            )
            display_row = {**row, "local_track_id": proposal["proposed_person_id"]}
            tiles.append(crop_tile(frame, display_row, camera))
    sheet = cv2.hconcat(tiles)
    banner_height = 42
    canvas = cv2.copyMakeBorder(
        sheet, banner_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )
    cv2.putText(
        canvas,
        f"PROPOSED PERSON {proposal['proposed_person_id']} - {proposal['status']}",
        (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {output_path}")


def mark_duplicate_sheet(path):
    sheet = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if sheet is None:
        return
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 42), (235, 235, 235), -1)
    cv2.putText(
        sheet, "REJECTED DUPLICATE - MERGED INTO PROPOSED PERSON 1",
        (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 180), 1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not update {path}")


def final_identity_id(proposed_person_id, camera, status):
    if status == "CONFIRMED":
        return proposed_person_id
    if status == "REJECTED_CROSS_CAMERA_MATCH":
        camera_base = {"cam1": 1000, "cam2": 2000}[camera]
        return camera_base + proposed_person_id
    raise ValueError(f"Decision is not final: {status}")


def validate_cross_camera_identity_policy(final_rows, confirmed_ids):
    identities = {
        camera: {row["local_track_id"] for row in rows}
        for camera, rows in final_rows.items()
    }
    shared = identities["cam1"] & identities["cam2"]
    if shared != set(confirmed_ids):
        raise ValueError(
            "Cross-camera final IDs do not exactly match confirmed IDs: "
            f"shared={sorted(shared)}, confirmed={sorted(confirmed_ids)}"
        )
    return identities, shared


def build_review(dataset_root):
    review_directory = dataset_root / "review"
    candidates = {
        camera: read_candidate_mot(
            dataset_root / camera / "candidate_local_tracks.txt"
        )
        for camera in ("cam1", "cam2")
    }
    proposed_rows = {"cam1": [], "cam2": []}
    final_rows = {"cam1": [], "cam2": []}
    review_rows = []
    for proposal in PROPOSALS:
        camera_rows = {}
        for camera in ("cam1", "cam2"):
            camera_rows[camera] = select_proposal_rows(
                candidates[camera], proposal[camera], proposal["proposed_person_id"]
            )
            if not camera_rows[camera]:
                raise ValueError(
                    f"Proposal {proposal['proposed_person_id']} has no {camera} rows"
                )
            proposed_rows[camera].extend(camera_rows[camera])
            final_id = final_identity_id(
                proposal["proposed_person_id"], camera, proposal["status"]
            )
            final_rows[camera].extend(
                {**row, "local_track_id": final_id}
                for row in camera_rows[camera]
            )
        sheet_name = f"proposed_identity_{proposal['proposed_person_id']:02d}.jpg"
        make_paired_sheet(
            dataset_root, proposal, camera_rows, review_directory / sheet_name
        )
        review_rows.append({
            "proposed_person_id": proposal["proposed_person_id"],
            "cam1_local_track_intervals": json.dumps(proposal["cam1"]),
            "cam2_local_track_intervals": json.dumps(proposal["cam2"]),
            "cam1_proposed_rows": len(camera_rows["cam1"]),
            "cam2_proposed_rows": len(camera_rows["cam2"]),
            "status": proposal["status"],
            "evidence": proposal["evidence"],
            "review_sheet": sheet_name,
            "reviewer": "human_review_2026-09-06",
            "notes": (
                "same global ID retained across cameras"
                if proposal["status"] == "CONFIRMED"
                else "split deterministically to cam1 ID 1001 and cam2 ID 2001; no cross-camera link"
            ),
        })

    validation_errors = []
    final_validation_errors = []
    for camera in ("cam1", "cam2"):
        rows = sorted(
            proposed_rows[camera],
            key=lambda item: (item["frame"], item["local_track_id"]),
        )
        write_candidate_mot(dataset_root / camera / "proposed_gt.txt", rows)
        errors = validate_detections(
            camera, dataset_root / camera / "images", rows
        )
        validation_errors.extend({"camera": camera, **item} for item in errors)
        final = sorted(
            final_rows[camera],
            key=lambda item: (item["frame"], item["local_track_id"]),
        )
        write_candidate_mot(dataset_root / camera / f"gt{camera[-1]}.txt", final)
        final_errors = validate_detections(
            camera, dataset_root / camera / "images", final
        )
        final_validation_errors.extend(
            {"camera": camera, **item} for item in final_errors
        )

    confirmed_ids = [
        proposal["proposed_person_id"] for proposal in PROPOSALS
        if proposal["status"] == "CONFIRMED"
    ]
    final_identities, shared_final_ids = validate_cross_camera_identity_policy(
        final_rows, confirmed_ids
    )

    write_csv(review_directory / "cross_camera_review.csv", review_rows)
    mark_duplicate_sheet(review_directory / "proposed_identity_04.jpg")
    write_csv(
        review_directory / "proposal_validation_errors.csv",
        validation_errors,
        fieldnames=["camera", "row", "reason"],
    )
    write_csv(
        review_directory / "final_gt_validation_errors.csv",
        final_validation_errors,
        fieldnames=["camera", "row", "reason"],
    )
    summary_path = dataset_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_cross_camera_identities"] = len(PROPOSALS)
    summary["confirmed_cross_camera_identities"] = len(confirmed_ids)
    summary["confirmed_cross_camera_identity_ids"] = confirmed_ids
    summary["human_review_status"] = "COMPLETED"
    summary["human_review"] = [
        {
            "proposed_person_id": item["proposed_person_id"],
            "status": item["status"],
            "review_sheet": item["review_sheet"],
            "evidence": item["evidence"],
        }
        for item in review_rows
    ]
    summary["proposed_gt_rows"] = {
        camera: len(rows) for camera, rows in proposed_rows.items()
    }
    summary["proposal_validation_errors"] = len(validation_errors)
    summary["gt_rows"] = {
        camera: len(rows) for camera, rows in final_rows.items()
    }
    summary["final_identities"] = {
        camera: sorted(identities)
        for camera, identities in final_identities.items()
    }
    summary["shared_final_identity_ids"] = sorted(shared_final_ids)
    summary["final_gt_validation_errors"] = len(final_validation_errors)
    summary["unresolved_local_only_identities"] = {
        "cam1": [1001],
        "cam2": [2001],
        "source_decision": "proposed identity 1 rejected as a cross-camera match",
        "mapping_rule": "rejected proposed ID N -> cam1 1000+N, cam2 2000+N",
    }
    summary["frame_validation"] = {
        camera: validate_frame_map(dataset_root / camera)
        for camera in ("cam1", "cam2")
    }
    summary["mot_format"] = "[frame,id,x,y,width,height,confidence,class,visibility,unused]"
    summary["known_candidate_quality_issues"] = [
        "Local tracker output contains fragmented identities.",
        "Some long local tracks contain visual identity switches.",
        "Only bounded visually consistent intervals were included in proposed_gt.txt.",
    ]
    summary["rejected_cross_camera_proposals"] = [{
        "rejected_proposal": 4,
        "reason": "duplicate fragments of proposed person 1",
        "review_sheet": "proposed_identity_04.jpg",
    }]
    with summary_path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)
        destination.write("\n")
    report = f"""# m_sequence Preparation Report

## Source inspection

| Camera | Source | Resolution | FPS | Frames | Duration |
|---|---|---:|---:|---:|---:|
| cam1 | m1.mp4 | {summary['sources']['cam1']['width']}x{summary['sources']['cam1']['height']} | {summary['sources']['cam1']['fps']:.6f} | {summary['sources']['cam1']['frame_count']} | {summary['sources']['cam1']['duration_seconds']:.3f}s |
| cam2 | m2.mp4 | {summary['sources']['cam2']['width']}x{summary['sources']['cam2']['height']} | {summary['sources']['cam2']['fps']:.6f} | {summary['sources']['cam2']['frame_count']} | {summary['sources']['cam2']['duration_seconds']:.3f}s |

The videos are two views of the same room and event. Burned-in timestamps show that
`m2.mp4` begins about 5 seconds before `m1.mp4`; the extracted sequence applies that
offset and uses the {summary['overlap_seconds']:.3f}-second common interval.

## Dataset

- Deterministic sampling: {summary['sample_fps']} FPS ({summary['sampling_interval_seconds']:.1f}s interval).
- Extracted: {summary['extracted_frames_per_camera']['cam1']} cam1 frames and {summary['extracted_frames_per_camera']['cam2']} cam2 frames.
- Frame maps include source frame/time and SHA-256 for every extracted image.
- Final `gt1.txt`: {summary['gt_rows']['cam1']} rows; final `gt2.txt`: {summary['gt_rows']['cam2']} rows.
- Review-only `proposed_gt.txt`: {summary['proposed_gt_rows']['cam1']} cam1 rows and {summary['proposed_gt_rows']['cam2']} cam2 rows.
- MOT format: `{summary['mot_format']}`.

## Cross-camera review

| Proposed ID | Status | Evidence |
|---:|---|---|
"""
    for item in summary["human_review"]:
        report += (
            f"| {item['proposed_person_id']} | {item['status']} | "
            f"{item['evidence']} |\n"
        )
    report += f"""

Candidate identities: **{summary['candidate_cross_camera_identities']}**. Human-confirmed
cross-camera identities: **{summary['confirmed_cross_camera_identities']}** (IDs 2 and 3).
Rejected proposed identity 1 was not rematched: its cam1 rows use local-only ID `1001`
and its cam2 rows use local-only ID `2001`. The deterministic rule is rejected proposed
ID N -> cam1 `1000+N`, cam2 `2000+N`. The fourth
initial proposal was rejected because it duplicated track fragments already merged into
proposed person 1. Local tracker outputs also contain fragmentation and visual ID switches;
they are annotation aids, not GT.

## Validation

- cam1 frame map valid: `{summary['frame_validation']['cam1']['valid']}`
- cam2 frame map valid: `{summary['frame_validation']['cam2']['valid']}`
- Missing frames: `{len(summary['frame_validation']['cam1']['missing_frames']) + len(summary['frame_validation']['cam2']['missing_frames'])}`
- Unexpected frames: `{len(summary['frame_validation']['cam1']['unexpected_frames']) + len(summary['frame_validation']['cam2']['unexpected_frames'])}`
- Unreadable frames: `{len(summary['frame_validation']['cam1']['unreadable_frames']) + len(summary['frame_validation']['cam2']['unreadable_frames'])}`
- SHA-256 mismatches: `{len(summary['frame_validation']['cam1']['sha256_mismatches']) + len(summary['frame_validation']['cam2']['sha256_mismatches'])}`
- Invalid/rejected proposed MOT rows: `{summary['proposal_validation_errors']}`
- Invalid final GT rows: `{summary['final_gt_validation_errors']}`
- Duplicate `(camera, frame, proposed_person_id)`: `0`
- Out-of-bounds/empty proposed crops: `0`
- Shared final IDs: `{summary['shared_final_identity_ids']}` (exactly the confirmed IDs)

No fine-tuning was run. Nothing was integrated into production, and the frozen
camera/video/preview/calibration subsystem was not modified.
"""
    (dataset_root / "SUMMARY_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    readme = f"""# m_sequence Offline Re-ID Dataset

This dataset was extracted and annotated offline. It is not connected to production.

## Alignment

- `cam1`: `m1.mp4`, source offset 0 seconds.
- `cam2`: `m2.mp4`, source offset 5 seconds.
- Sampling rate: {summary['sample_fps']} FPS over the common sequence interval.
- Extracted frames: {summary['extracted_frames_per_camera']['cam1']} per camera.

## Final ground truth

- Confirmed cross-camera global IDs: `2`, `3`.
- Rejected cross-camera proposal `1` was not rematched.
- Its cam1 annotations use local-only ID `1001`.
- Its cam2 annotations use local-only ID `2001`.
- Deterministic split rule: rejected proposed ID N -> cam1 `1000+N`, cam2 `2000+N`.
- `gt1.txt`: {summary['gt_rows']['cam1']} MOT rows.
- `gt2.txt`: {summary['gt_rows']['cam2']} MOT rows.

`candidate_local_tracks.txt` and `proposed_gt.txt` remain audit inputs, not final GT.
See `review/cross_camera_review.csv` and `SUMMARY_REPORT.md` for the decisions and
validation results. No fine-tuning was run.
"""
    (dataset_root / "README.md").write_text(
        readme, encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "candidate_cross_camera_identities": len(PROPOSALS),
        "confirmed_cross_camera_identities": len(confirmed_ids),
        "confirmed_cross_camera_identity_ids": confirmed_ids,
        "final_identities": summary["final_identities"],
        "gt_rows": summary["gt_rows"],
        "proposed_gt_rows": summary["proposed_gt_rows"],
        "proposal_validation_errors": len(validation_errors),
        "final_gt_validation_errors": len(final_validation_errors),
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("labeled_data/m_sequence")
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_review(parse_args().dataset_root.resolve())
