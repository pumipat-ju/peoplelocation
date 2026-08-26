# Read-Only Forensic Audit — Current Re-ID Model and Production Code

## Purpose

Perform a comprehensive READ-ONLY forensic review of the current Multi-camera People Location / Re-ID repository.

The goal is to determine exactly what the current production system does now, especially why cross-camera identity transfer may still create a new GID.

This task is analysis only. DO NOT implement fixes.

## Absolute read-only rules

Before doing anything, read and obey `AGENTS.md`.

Use `backend/main.py` as the production source of truth.

Do NOT modify any file.

Do NOT:
- apply patches or formatting
- edit tests/config/env
- install/update dependencies
- download models
- alter/migrate databases
- delete files
- commit/reset/checkout/clean Git state
- start/stop/restart/rebuild/deploy services or containers
- open real cameras
- access RTSP
- play real uploaded videos
- change calibration data

Preserve working tree exactly as found.

If importing production code may create SQLite files, runtime directories, model caches, or background workers, DO NOT import it.

Prefer static source inspection, grep/search, AST parsing, reading tests/config, `git status`, and `git diff`.

Do not claim behavior exists merely because a function exists. Trace actual production callers.

## A. Audit the production Re-ID model

Report:
1. exact model architecture/name
2. weight/checkpoint resolution and default path
3. env/Docker overrides
4. what happens when weights are missing
5. `loaded`, `fallback`, error behavior
6. preprocessing:
   - crop
   - resize
   - RGB/BGR
   - normalization
   - tensor shape
   - embedding normalization
7. embedding dimension
8. similarity function
9. gallery/prototype generation
10. quality gates and gallery limits
11. whether production cross-camera matching really uses OSNet embeddings

Verify code; do not trust README blindly.

## B. Audit CURRENT thresholds and weights

Find actual current values and where used:

- cross-camera appearance threshold
- same-camera threshold
- total acceptance threshold
- ambiguity/top1-top2 margin
- quality thresholds
- max idle
- dormant TTL
- provisional timeout
- gallery/tracklet maturity
- overlap/occlusion thresholds
- motion thresholds/weights
- appearance weight
- topology min/max travel-time rules
- global assignment window
- fallback/conservative thresholds

For each report:

```text
name
value
file/function
same-camera / cross-camera / both
hard gate / soft score / lifecycle
production-used or unused
```

## C. Trace exact production cross-camera handoff

Trace:

```text
CamA established GID 1
-> leaves CamA
-> CamB gets new BoT-SORT local track
-> embedding
-> candidate GIDs
-> hard gates
-> scoring
-> global Hungarian
-> commit
-> final GID
```

Answer:
1. Does production call `assign_global_batch()`?
2. What submits observations?
3. Is submit non-blocking?
4. How is `GLOBAL_ASSIGNMENT_WINDOW_SEC` used?
5. How are missing/late cameras handled?
6. How are fixed local/occlusion assignments protected?
7. How is one-to-one enforced?
8. What creates provisional/new identity?
9. What assignment source marks successful cross-camera reuse?
10. Why can an existing GID be absent from candidates?

## D. Enumerate every current reason CamA:GID1 -> CamB becomes a new GID

Verify and rank:
- identity idle expired
- DORMANT eligibility/TTL
- appearance below threshold
- score below threshold
- quality-adjusted appearance
- incompatible box size
- incompatible location
- topology rejection
- travel time too short/long
- simultaneous presence conflict
- ambiguity margin
- one-to-one conflict
- generation/stale guard
- fallback extractor
- missing gallery/prototype
- provisional maturity restrictions
- same-camera rules leaking into cross-camera logic

For each include exact diagnostic field/reason.

If a previously known bug is fixed in current code, explicitly mark it FIXED.

## E. Audit lifecycle

Inspect:
- PROVISIONAL
- ACTIVE
- DORMANT
- EXPIRED

Report:
1. allowed transitions
2. trigger for each
3. time basis used
4. whether identity aging uses event time
5. `REID_MAX_IDLE_SEC`
6. `IDENTITY_DORMANT_TTL_SEC`
7. whether DORMANT can recover cross-camera
8. whether active idle gate still blocks DORMANT
9. whether cleanup transitions persist
10. whether EXPIRED can be reused

Answer based on code:

> Can the same person recover old GID after being absent for:
- 5 sec
- 20 sec
- 40 sec
- 200 sec
- beyond dormant TTL

Explain why.

## F. Audit Global Identity vs camera presence

Determine whether current code stores:
- global GID
- last camera
- last seen event time
- `(camera, local_track_id) -> gid`
- per-camera presence
- active/inactive presence
- previous camera
- confirmed handoff history
- overlap-aware multi-camera presence
- gallery/prototype
- state transition audit

Answer directly:

> Does the current architecture explicitly model confirmed CamA -> CamB handoff, or only infer cross-camera reuse through matching?

If missing, explain what is missing. Do NOT implement it.

## G. Audit SQLite persistence

Inspect `backend/identity_store.py`, Compose/config, callers.

Report:
1. local DB path
2. Docker DB path
3. persistent volume coverage
4. tables/schema
5. persisted identity fields
6. transition/audit data
7. presence/handoff persistence
8. transaction boundaries
9. shutdown/connection close
10. restore logic
11. next-GID restore
12. schema/backward compatibility
13. quarantine/rollback methods and callers

Do not open/mutate real production DB.

## H. Audit topology and canonical time

Report:
1. topology schema version(s)
2. `enforce` behavior
3. transition fields
4. v1 compatibility
5. fail-open/fail-closed
6. camera reference validation
7. overlap behavior
8. min/max travel time
9. hard gate order vs appearance
10. uploaded-video event-time formula
11. live-camera event time
12. `time_offset_sec` frontend->backend path
13. non-decreasing clamp
14. any remaining processing wall-time use that affects identity correctness

Separate correctness-related wall time from diagnostics/runtime scheduling only.

## I. Audit tracklet/gallery safety

Report:
- tracklet key
- local-ID reuse protection
- generation/reset guards
- stale cleanup
- gallery update gates
- blur/occlusion/border clipping
- near-duplicate handling
- gallery bounds
- prototype calculation
- provisional restrictions
- per-camera vs global gallery
- poisoning risk

State which Prompt10-style items are already implemented vs missing.

## J. Audit global coordinator performance/concurrency

Report:
1. queue/buffer structure
2. max ready batches
3. bounded/unbounded
4. dispatcher behavior
5. submit blocking behavior
6. assignment execution path
7. stale generation handling
8. exception handling
9. shutdown behavior
10. whether slow assignment can block live capture, uploaded-video processing, preview, or another camera
11. timing diagnostics

Verify whether the old ~203 ms synchronous submit path is gone.

Do not benchmark real models/devices.

## K. Audit tests against production reality

Inventory coverage:

```text
production call path
global Hungarian
short-gap same-camera recovery
cross-camera handoff
one-to-one
lifecycle
dormant recovery
restart/persistence
topology
timestamp/offset
tracklet/gallery
camera worker isolation
uploaded video
live camera
calibration regression guard
```

Identify tests that:
- only test isolated methods
- contain stale constants
- contradict production config
- give false confidence

Do not edit tests.

## L. Current architecture diagram

Produce an ASCII diagram based strictly on current code.

Annotate:
- frozen subsystem boundary
- event-time boundary
- global assignment point
- topology hard-gate point
- SQLite persistence point
- missing presence/handoff layer if absent

## Required final report format

### 1. Executive verdict
Use:
- READY for cross-camera manual validation
- PARTIALLY READY
- NOT READY

### 2. Current production architecture
ASCII diagram + explanation.

### 3. Current Re-ID model
Table: Item | Current production value | Evidence/location

### 4. Current thresholds/weights
Full table.

### 5. Cross-camera handoff trace
Step-by-step production path.

### 6. Reasons a new GID is created
Table: Cause | Exact gate/code | Diagnostic evidence | Risk

### 7. Lifecycle verdict
Include 5s / 20s / 40s / 200s / beyond-TTL.

### 8. Presence/handoff verdict
Explicitly state whether confirmed handoff/presence exists today.

### 9. Persistence verdict

### 10. Time/topology verdict

### 11. Tracklet/gallery verdict

### 12. Concurrency/performance verdict

### 13. Test coverage gaps

### 14. Priority before deadline

Use:
- BLOCKER
- HIGH
- MEDIUM
- DEFER

For each include exact reason, affected files/functions, frozen-subsystem impact, recommended next scope.

### 15. Direct answers

1. Is cross-camera GID transfer implemented in production now?
2. For CamA GID1 -> CamB within 5 sec, what exact conditions must pass?
3. What happens at 40 sec?
4. Is OSNet really loaded by default, or can fallback be active?
5. Does the system explicitly store camera presence/handoff ownership?
6. Is SQLite sufficient for bidirectional CamA <-> CamB continuity after restart?
7. What is the single highest-priority correctness gap now?

## Evidence discipline

Every important finding must include exact file/function references and line numbers where practical.

Clearly distinguish:
- VERIFIED CURRENT CODE
- TEST-ONLY BEHAVIOR
- CONFIGURATION-DEPENDENT
- KNOWN BUT NOT VERIFIED AT RUNTIME

Do not claim real-world accuracy from static code review.

End with:

`READ-ONLY REVIEW COMPLETE — NO FILES MODIFIED`
