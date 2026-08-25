# Prompt 11 — Canonical Event Time and Minimal Topology Hard Gates

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


## Deadline-oriented scope

This prompt intentionally implements the minimum topology/time contract needed for safe Re-ID decisions before the deadline.

Do NOT build or modify a calibration-zone editor in this prompt.
Do NOT change calibration UI/workflow.
Do NOT alter uploaded-video playback synchronization internals unless an existing bug can be fixed purely at the downstream metadata/API boundary.

If richer zone editing requires touching the frozen calibration subsystem, report it as deferred.

## Goal

Make identity decisions use correct canonical event time and a validated minimal camera-transition topology so physically impossible handoffs are rejected before appearance matching.

## Required work

### A. Uploaded-video offset propagation

Inspect the frontend uploader and backend upload API.

If the frontend currently stores per-file offset but does not send it, add the smallest safe request-field fix so the existing backend receives the offset, e.g. the current expected `time_offset_sec` field.

Do not redesign file selection, upload progress, playback workers, synchronization, or decoding.

### B. Canonical event time

Define and consistently propagate one event-time contract:

- uploaded video: existing shared playback/source-time basis + configured per-source offset
- live camera: capture event wall-clock / existing capture timestamp
- processing time: diagnostics only, not identity travel-time truth

Requirements:
1. Detection/tracklet/identity observations carry their own event time.
2. Global assignment uses each observation's event time.
3. Pause/resume/loop/seek/reconnect handling must reuse existing playback/capture semantics; do not rewrite those subsystems.
4. Do not introduce time that moves backward through downstream metadata if current source semantics provide a monotonic basis.
5. Add lag/drift diagnostics only if they can be computed downstream without modifying frozen capture behavior.

### C. Minimal persistent topology contract

Inspect the current camera-transition configuration and extend only as needed for safe matching.

For the deadline, support at minimum:
- source camera
- destination camera
- minimum travel time
- maximum travel time
- overlap permission / simultaneous-presence permission

Optional zone fields may be added to the schema only if they can be represented/configured without changing calibration capture/UI behavior.

Do not require polygon editing to complete this prompt.

### Default topology policy

If an existing topology entry already has configured values, preserve them.

For newly created camera-pair rules where no values exist:
- `min_travel_time_sec`: 0.0
- `max_travel_time_sec`: null / disabled
- `overlap_allowed`: false
- unspecified camera pair: do not invent a transition rule automatically

Do NOT guess physical travel times from camera names, ordering, or floorplan distance.

A missing `max_travel_time_sec` means no upper travel-time hard gate is applied.
A missing/false `overlap_allowed` means simultaneous presence is not allowed.

Existing explicit configuration always takes precedence over these defaults.

### D. Validation

Validate topology configuration:
- schema/version if versioning already exists or can be added minimally
- referenced cameras
- non-negative min/max travel time
- max >= min
- explicit overlap boolean/policy
- malformed configuration must fail safely and visibly rather than silently allowing impossible transitions

Store the topology in the existing persistent data location if practical without migrating user runtime data automatically.

### E. Hard gates before appearance

Before appearance scoring/acceptance:
1. Reject disallowed camera transitions.
2. Reject travel times below the configured minimum.
3. Reject travel times above the configured maximum when such a rule applies.
4. Reject simultaneous presence on non-overlap camera pairs.
5. Allow simultaneous presence only when explicitly permitted by overlap policy.
6. Appearance similarity must never override these hard rejections.

Preserve same-camera motion/local rules already present.

### F. Diagnostics

Expose/log:
- source camera
- destination camera
- event-time delta
- topology rule used
- overlap permission
- hard-gate pass/fail reason

Use the existing diagnostics pattern.

## Tests

Required cases:
1. Frontend uploader includes the configured time offset in the request payload.
2. Deterministic uploaded-video observation time reflects the configured offset.
3. Live observation uses capture/event time, not processing completion time.
4. Allowed camera transition inside min/max passes the topology gate.
5. Too-fast transition is rejected.
6. Too-slow transition is rejected when max time is configured.
7. Disallowed transition is rejected before appearance score acceptance.
8. Simultaneous presence is rejected for non-overlap cameras.
9. Simultaneous presence is allowed only for explicit overlap configuration.
10. Malformed topology config fails safely.
11. No calibration workflow code is modified/tested through real devices.

## Acceptance criteria

- Offset configured in the existing upload UI reaches the backend.
- Canonical observation event time is used in matching.
- Minimal transition topology is validated.
- Impossible routes/travel times are hard rejected before appearance acceptance.
- Explicit overlap policy controls simultaneous presence.
- Calibration remains untouched.
- Camera/video acquisition/playback lifecycle remains untouched.
- Relevant tests pass.

## Verification to run

Run:
- frontend unit/test/static verification relevant to uploader payload, if project tooling exists
- backend syntax/compile checks
- timestamp/topology unit tests
- global matching tests for topology gates

Do not start frontend/backend services or Docker.

## Final report

Report:
1. Files changed.
2. Canonical time definition.
3. Exact upload offset contract.
4. Topology schema supported now.
5. Which richer zone/UI functionality was intentionally deferred.
6. Tests run/results.
