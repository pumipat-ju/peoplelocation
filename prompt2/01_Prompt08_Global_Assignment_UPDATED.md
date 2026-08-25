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

## Goal — Post-Prompt08 regression diagnosis and minimal fix

Prompt 08 global assignment has already been implemented and automated tests have passed.

Current implemented production flow is reported as:

`existing capture/video -> YOLO -> per-camera BoT-SORT -> OSNet -> non-blocking global coordinator -> assign_global_batch()`

Do NOT redo or redesign the completed Prompt 08 architecture unless a confirmed regression in that implementation requires a minimal correction.

A real uploaded-video test exposed a continuity problem:

- a person can disappear from detection/tracking for only about 2–3 source frames
- after the person becomes visible again, BoT-SORT may create a new local track ID
- the same physical person then receives a different Global ID instead of recovering the previous GID
- video processing also feels slow between visible frame updates, but the frozen video playback/capture subsystem must not be modified to address this

The primary objective of this follow-up is to determine exactly where identity continuity is lost and fix the smallest downstream Re-ID/global-assignment cause.

Expected behavior:

```text
before short gap:
camera = camA
local_track_id = 5
gid = 3

person is absent/occluded for 2–3 source frames

after short gap:
camera = camA
local_track_id = 12   # local tracker is allowed to change
gid = 3               # Global Re-ID should recover the same person when evidence passes
```

A changed BoT-SORT local ID by itself is NOT sufficient reason to create a new Global ID.

This task is NOT permission to modify the frozen camera/video/calibration subsystem.

## Required work

### 1. Reproduce the short-gap failure in a deterministic test first

Add a focused regression test using synthetic/mocked observations.

The test must simulate:

1. a person on one camera with an established GID
2. a short absence of approximately 2–3 source frames
3. reappearance on the same camera with a NEW local track ID
4. appearance evidence representing the same physical person
5. valid quality/gating conditions

Expected result:
- the new local track recovers the previous Global GID
- a new GID is NOT created merely because the local track ID changed

Also add the complementary case:
- if appearance/gates clearly reject the previous identity, the system must NOT force reuse of the old GID

Do not use a real camera, real uploaded video, Docker, real OSNet weights, or the production SQLite DB.

### 2. Diagnose which identity layer actually breaks

Before changing production logic, trace the failing synthetic case through the current code.

Capture at minimum:

- source frame index or synthetic sequence index
- observation event time
- camera name
- local track ID
- previous local-to-global mapping if any
- candidate GIDs
- hard-gate result/reason
- appearance score used by existing logic
- top-1/top-2 margin if applicable
- assignment source
- assignment pending/committed state
- batch ID
- generation value
- final GID/state
- reason a new/provisional GID was created

Determine which of these cases is happening:

#### Case A — local track changes and global recovery fails

```text
LID 5 / GID 3
gap
LID 12 / GID 7
```

Fix downstream Re-ID candidate eligibility / recovery / assignment logic so the previous valid identity can be considered and recovered when existing gates and evidence support it.

#### Case B — local track remains the same but GID changes

```text
LID 5 / GID 3
gap/pending
LID 5 / GID 7
```

Fix loss of trusted local-to-global continuity or a coordinator/generation-guard regression.

#### Case C — the old identity is being excluded by an existing lifecycle/idle gate

If the failure is actually caused by the known broader ACTIVE/DORMANT lifecycle bug intended for Prompt 09:
- do not redesign the entire state machine here
- implement only a narrowly scoped correction if it is clearly necessary for this 2–3-frame active short-gap case
- otherwise report the exact blocking gate/reason and leave the full lifecycle redesign for Prompt 09

### 3. Preserve Global Hungarian invariants

Any continuity fix must preserve:

- global one-to-one assignment
- no duplicate non-overlap GID assignment in one batch
- trusted local/occlusion evidence protection
- ambiguity rejection
- hard-gate rejection
- generation safety after tracker reset
- event-time semantics

Do NOT solve continuity by blindly pinning the old GID.

The previous GID may be recovered only when the current Re-ID evidence and existing safety gates legitimately allow it.

### 4. Do not use the display layer as the identity source of truth

If a previous verified GID is temporarily displayed while a new global decision is pending, that may be used only as a rendering continuity hint.

It must NOT:
- bypass global assignment
- update identity/gallery evidence by itself
- become a permanent assignment without a valid commit
- survive a tracker reset/generation change incorrectly

If no display-hold mechanism is needed to fix the actual failure, do not add one.

### 5. Investigate the reported slowness without changing video playback/capture

The user reports that only 2–3 source frames may pass, but those frames feel slow to advance.

Measure/diagnose only from the downstream processing path.

Record timings around the existing processing stages where practical, for example:

- detection/tracking duration
- OSNet/Re-ID feature duration
- coordinator submit duration
- global assignment duration
- total downstream frame-processing duration

Critical requirement:

Submitting an observation to the global coordinator must remain non-blocking.

Do NOT allow `GLOBAL_ASSIGNMENT_WINDOW_SEC = 0.25` to become a per-frame blocking sleep/wait in the uploaded-video or live-camera processing path.

Do NOT arbitrarily change `GLOBAL_ASSIGNMENT_WINDOW_SEC` simply to hide the symptom.

If the 0.25-second window is currently implemented asynchronously/non-blocking as intended, preserve it.

If timing shows that the coordinator incorrectly blocks the producer/processing path, fix only that downstream coordinator behavior.

Do NOT modify:
- video decoding
- playback clock
- uploaded-video synchronization
- capture threads/workers
- frame acquisition
- preview publishing
- Docker device logic
- calibration

### 6. Preserve short-gap identity evidence safely

For a short reappearance where the local track ID changes:

- the previous ACTIVE identity must remain a legitimate candidate while it is still valid under the existing active-idle policy
- the loss of the local track mapping must not automatically make that identity invisible to Re-ID
- same-camera re-entry must still pass appearance and applicable hard gates
- do not seed/update permanent gallery from low-quality or ambiguous evidence just to recover the ID
- do not carry identity across a tracker reset/generation reset without valid evidence

Do not add arbitrary new matching thresholds in this prompt unless a logical bug requires it.
Do not claim improved accuracy from an unmeasured threshold change.

## Tests

At minimum add/update tests for:

1. Same camera, same person, 2–3 frame gap, new local track ID -> same Global GID.
2. Same camera, different person after a short gap -> old GID is not blindly reused.
3. Same local track remains valid across a pending global window -> GID does not unexpectedly change.
4. Generation/tracker reset prevents stale observations from committing.
5. Two-camera global one-to-one invariant still holds after the fix.
6. Ambiguous candidate still rejects rather than force-matching.
7. Coordinator submission remains non-blocking in a mocked timing test.
8. Existing Prompt08 production-path/global-assignment tests continue to pass.

Tests must remain device-free and must not touch production runtime data.

## Acceptance criteria

This follow-up is complete only when:

- a deterministic regression test reproduces the short-gap/new-local-ID problem before the fix or otherwise demonstrates the exact failing condition
- after the fix, a same-person short gap with a new local track ID can recover the previous GID when evidence passes
- a genuinely different/invalid candidate is not forced onto the old GID
- global Hungarian one-to-one behavior remains correct
- generation-reset safety remains correct
- no capture/video/calibration subsystem behavior is changed
- `GLOBAL_ASSIGNMENT_WINDOW_SEC` is not converted into a blocking per-frame wait
- any downstream timing regression found is corrected without modifying frozen acquisition/playback code
- existing Prompt08 tests plus the new regression tests pass

## Verification to run

Run only safe device-free verification:

- Python syntax/compile checks for modified backend files
- existing Prompt08 global-assignment/production-path tests
- new short-gap continuity regression tests
- relevant local-track/generation tests
- mocked non-blocking/timing test if added

Do not:
- open a real camera
- run Docker
- load real OSNet weights unless an existing test explicitly requires it
- play real uploaded videos automatically
- modify calibration data
- write to the production SQLite DB

## Final report

At completion, report clearly:

1. Files changed.
2. Exact root cause of the 2–3-frame continuity failure.
3. Whether BoT-SORT local ID changed in the reproduced case.
4. Why Global Re-ID previously created/recovered the wrong GID.
5. Exact minimal fix made.
6. Whether `GLOBAL_ASSIGNMENT_WINDOW_SEC = 0.25` was blocking or non-blocking.
7. Any measured downstream timing before/after the fix, if measured.
8. Tests run and results.
9. Confirmation that camera discovery, VideoCapture, Docker device access, uploaded-video playback/synchronization, preview, and calibration were not modified.
10. Any remaining blocker that belongs to Prompt 09 rather than this follow-up.
