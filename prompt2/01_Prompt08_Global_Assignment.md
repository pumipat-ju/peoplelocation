# Prompt 08 — Connect Global Multi-Camera Assignment to Production

## Mandatory project guardrails

Before doing anything, read and obey `AGENTS.md`.

The camera/video acquisition and calibration subsystem is FROZEN and is already confirmed working.

Do NOT refactor, redesign, replace, move, or change behavior related to:
- camera discovery / enumeration
- camera source or device resolution
- `cv2.VideoCapture` or equivalent device opening
- Docker camera/device passthrough
- live-camera capture workers or worker lifecycle
- reconnect logic
- frame grabbing / decoding
- latest-frame cache
- live preview / video feed
- uploaded-video input/playback workers
- existing video synchronization infrastructure
- calibration frame acquisition
- calibration endpoints
- calibration workflow
- APIs used by camera selection, preview, video input, or calibration

Do NOT introduce synchronization barriers, blocking waits, rendezvous ownership, or lifecycle changes into camera/video capture workers.

Known unacceptable regressions:
- Docker cannot detect/open cameras
- live preview/capture stops or freezes
- uploaded-video input breaks
- calibration becomes unusable

All new work must be implemented downstream/around the existing acquisition path:

`existing camera/video acquisition -> existing detection/tracking -> new downstream logic`

If the requested task truly requires changing the frozen subsystem, DO NOT make that change. Stop that part, report the exact dependency/blocker, and continue only with safe in-scope work.

Use `backend/main.py` as the production source of truth.
Do not modify root `main.py` unless explicitly required.

Preserve existing user changes and untracked files.
Do not install/update dependencies, download models, modify real runtime DB data, start/restart/rebuild/deploy services or containers, or perform Git commit/push unless explicitly authorized.

Treat this prompt as an implementation task only for the files and behavior required below. Do not opportunistically fix unrelated findings.


## Goal

Connect the existing global multi-camera assignment logic to the real production Re-ID path so that observations from multiple cameras can be decided in one global one-to-one assignment, while preserving the current working live-camera, uploaded-video, Docker-device, preview, and calibration behavior.

The repository already contains global-assignment logic such as `assign_global_batch()`, but the production path has been observed to still use per-camera assignment in important paths. Verify the current code before changing anything; do not assume line numbers are unchanged.

This task is NOT a camera architecture rewrite.

## Required work

1. Inspect the current production call path in `backend/main.py`.
   - Identify every production caller of per-camera identity assignment.
   - Identify the existing `assign_global_batch()` implementation and its inputs/outputs.
   - Identify how uploaded-video and live-camera observations reach identity assignment.
   - Do not modify capture/input lifecycle during this inspection.

2. Introduce the smallest safe downstream coordination layer needed so production can submit eligible observations from multiple cameras to one global assignment decision.
   - Reuse existing global-assignment logic rather than duplicating it.
   - Keep public function signatures stable where practical.
   - Keep camera workers independent.
   - No camera worker may block waiting for another camera.
   - No unbounded queue.
   - Do not move frame capture, video decoding, preview publishing, reconnect handling, or calibration into the coordinator.

3. Use `GLOBAL_ASSIGNMENT_WINDOW_SEC` (or the existing equivalent configuration) in the real production decision path.
   - If the constant exists but is unused, wire it into the downstream assignment coordinator.
   - The window must apply to identity observations, not camera capture.

4. Enforce one-to-one assignment across all observations participating in the same global decision.
   - A GID already fixed by trusted local evidence must not also be assigned to a conflicting observation in the same global batch.
   - Preserve existing local-track verified assignments and occlusion/anti-ID-swap evidence.
   - Do not weaken existing anti-ID-swap safeguards.

5. Preserve event-time semantics.
   - Use the observation/detection event time already produced by the pipeline.
   - Do not replace it with processing time merely because observations are globally batched.

6. Handle missing/late cameras safely.
   - A camera that has no observation in the current window must not block other cameras.
   - A disconnected or slow camera must not freeze global assignment, live preview, uploaded playback, or calibration.

7. Keep unmatched or ambiguous observations safe.
   - Do not force a bad match just to fill the Hungarian matrix.
   - Preserve/create provisional identity behavior according to existing state rules.
   - Do not redesign the full identity state machine in this prompt.

8. Add production-path diagnostics sufficient to prove the global path is actually used.
   At minimum expose/log in the existing diagnostics style:
   - global batch identifier or timestamp/window
   - cameras represented in the batch
   - number of observations
   - assignment source
   - selected GID or new/provisional result
   - candidate/gate rejection information already available from current scoring

Do not create a new large diagnostics framework.

## Tests

Add or update narrowly scoped tests proving the production call path, not only isolated class behavior.

Required cases:
1. Two cameras submit observations inside one global assignment window and are evaluated in one global batch.
2. Two observations cannot both claim the same non-overlap GID in one decision.
3. A camera with no observation does not block another camera.
4. Existing verified local-track / occlusion-held GID evidence remains protected from conflicting Hungarian reassignment.
5. Uploaded-video identity processing uses the global path without changing uploaded-video capture/playback behavior.
6. Live-camera identity processing uses the global path without changing capture-worker lifecycle.
7. Existing per-camera BoT-SORT state remains isolated per camera.

Tests must not:
- open real cameras
- require Docker
- require real OSNet weights unless a test is explicitly an existing model test
- write to the production SQLite DB
- change calibration data

Use mocks/fakes and temporary state where necessary.

## Acceptance criteria

This prompt is complete only when:

- `assign_global_batch()` or the existing canonical global equivalent has a real production caller.
- Multi-camera observations inside the configured assignment window are decided together.
- One-to-one GID assignment is enforced across that global decision.
- A slow/missing camera cannot block camera capture, preview, uploaded playback, or another camera's identity processing.
- Existing live-camera behavior is unchanged.
- Existing uploaded-video behavior is unchanged.
- Docker camera/device handling code is unchanged unless absolutely unavoidable and explicitly reported instead of modified.
- Calibration code/workflow is unchanged.
- Existing local tracking and anti-ID-swap protections do not regress.
- Relevant syntax/import/tests pass.

## Verification to run

Run only safe local verification that does not start services or touch real devices/data:
- Python syntax/compile checks for modified backend files.
- Relevant assignment/unit/integration tests.
- Existing camera-worker tests only if they are device-free/mocked.
- Report the exact tests run and their results.

Do not claim accuracy improvement from this task. This task corrects production assignment architecture and invariants, not measured Re-ID accuracy.

## Final report

At completion, report:
1. Files changed.
2. Old production assignment flow.
3. New production assignment flow.
4. How the global assignment window is used.
5. How camera workers remain non-blocking and untouched.
6. Tests run and results.
7. Any remaining known limitation or blocker.
