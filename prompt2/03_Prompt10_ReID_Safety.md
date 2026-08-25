# Prompt 10 — Harden Tracklets, Gallery, and Re-ID Matching Safety

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

Harden downstream Re-ID evidence handling so stale local tracks, poor crops, duplicate embeddings, ambiguity, and simultaneous-presence conflicts do not poison identity decisions.

This prompt must NOT change camera/video acquisition or calibration, and must NOT redesign the global coordinator completed in Prompt 08.

Work downstream of existing detection/tracking outputs and reuse the production global assignment path.

## Required work

### A. Tracklet lifecycle safety

1. Inspect how tracklets are keyed and cleaned up.
2. Prevent a reused `(camera, local_track_id)` from inheriting evidence from a previous person.
3. Use the smallest safe key/generation mechanism consistent with existing code, for example:
   - camera
   - local track ID
   - GID and/or track generation
4. Remove/reset stale tracklet evidence when appropriate:
   - local mapping expires
   - tracker is reset
   - camera is removed through the existing downstream cleanup hook
   - local track is reassigned to a different GID
5. Do not alter how cameras are opened/removed or how tracker workers are scheduled.

### B. Quality evidence

Preserve or improve existing quality gates using metadata already available downstream:
- detection confidence
- crop size
- blur/quality measure if already implemented
- occlusion/overlap
- border clipping
- event time

Do not add heavy new image-processing dependencies.

Poor-quality evidence may support continuity only according to existing safe rules, but must not poison permanent identity galleries.

### C. Near-duplicate semantics

Separate:
- evidence that a tracklet has persisted/matured
from
- permission to append another embedding to the permanent gallery.

Near-identical sequential embeddings may count toward tracklet continuity, but should not fill the gallery with redundant duplicates.

### D. Gallery safety

1. Keep the gallery bounded.
2. Prefer diverse, quality-approved samples.
3. Preserve a stable/global prototype calculation from approved samples.
4. If current code already stores per-camera gallery/prototype information, make it consistent.
5. If per-camera galleries would require a major schema redesign, do not introduce it in this deadline prompt; keep a safe bounded global gallery and report the deferred enhancement.
6. Do not update a permanent identity gallery from insufficiently mature PROVISIONAL evidence.
7. Do not allow low-confidence, heavily occluded, badly clipped, or otherwise rejected samples to update the gallery.

### E. Matching safety

Use the existing global assignment architecture and preserve its one-to-one invariant.

Ensure:
1. trusted local/occlusion evidence remains protected
2. ambiguous matches are rejectable rather than forced
3. top-1/top-2 margin is computed only over candidates that actually passed hard gates
4. impossible simultaneous presence cannot silently assign one non-overlap identity to two cameras
5. overlap exceptions are only allowed if an explicit existing topology/overlap policy authorizes them
6. no appearance score can override a hard impossibility gate
7. matching weights/thresholds are not arbitrarily retuned in this prompt unless required to correct a logical bug

### F. Diagnostics

Extend existing diagnostics minimally to include useful Re-ID evidence such as:
- tracklet maturity/sample count
- gallery size
- quality rejection reason
- candidate GIDs that passed hard gates
- selected candidate
- margin
- assignment source
- hard-gate failure reason

Do not build a new monitoring subsystem.

## Tests

Required cases:
1. Local track ID reuse does not inherit the previous person's tracklet/gallery evidence.
2. Tracker-reset cleanup removes stale downstream tracklet state without changing tracker/camera lifecycle.
3. Near-duplicate embeddings can contribute to maturity without filling the permanent gallery.
4. Blur/occlusion/border/low-confidence samples do not enter the permanent gallery.
5. PROVISIONAL insufficient evidence does not poison an ACTIVE identity gallery.
6. Gallery stays bounded and diverse enough under repeated similar embeddings.
7. Ambiguous candidates are rejected when the margin rule says they are ambiguous.
8. Hard-gated candidates do not participate in margin acceptance.
9. Simultaneous non-overlap conflict is rejected.
10. Existing verified local/occlusion assignment remains stable.

Use synthetic embeddings and mocked observations. No real camera/model/runtime DB.

## Acceptance criteria

- Reused local IDs cannot mix people through stale tracklet evidence.
- Tracklet maturity and permanent gallery insertion are no longer the same decision.
- Low-quality evidence cannot poison galleries.
- Galleries are bounded.
- Ambiguous matching is rejectable.
- Global one-to-one assignment remains intact.
- No camera/video/calibration subsystem change.
- Relevant tests pass.

## Verification to run

Run:
- syntax/compile checks
- tracklet/gallery tests
- global matching safety tests
- relevant existing Re-ID unit tests that do not require real devices/models

Do not claim accuracy gains without evaluation data.

## Final report

Report:
1. Files changed.
2. Tracklet lifetime/key behavior before and after.
3. Gallery admission rules.
4. Ambiguity/hard-gate behavior.
5. Tests run/results.
6. Any deferred gallery/schema improvements.
