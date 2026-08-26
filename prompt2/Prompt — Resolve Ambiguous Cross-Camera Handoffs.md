# Prompt — Resolve Ambiguous Cross-Camera Handoffs

Read and obey `AGENTS.md`.

## Problem

Real-video case:

```text
cam_2 GID2 -> cam_1 incorrectly becomes GID4
cam_2 GID3 -> cam_1 incorrectly becomes GID5
```

Forensic diagnosis confirmed:

- GID2 and GID3 are both valid candidates.
- Hard gates pass.
- Topology is not the cause.
- Lifecycle/timeout is not the cause.
- One-to-one conflict is not the cause.
- Failure reason is `ambiguous_top1_top2`.
- Cross-camera margin requires `0.08`.
- Current top-1 ranking can even swap GID2/GID3, so DO NOT solve this by simply lowering the margin threshold.

## Goal

Change only ambiguous cross-camera handling so the system does **not immediately create a permanent new GID** when a valid cross-camera observation is ambiguous.

Expected behavior:

```text
ambiguous cross-camera observation
        ↓
bounded unresolved/provisional state
        ↓
collect several quality-approved observations
        ↓
jointly resolve competing identities with global one-to-one assignment
        ↓
commit existing GID when evidence becomes sufficient
        ↓
create new GID only if ambiguity remains unresolved after a bounded evidence window
```

## Required changes

1. For `ambiguous_top1_top2` on cross-camera candidates:
   - do not immediately create a permanent new GID
   - keep the local track as unresolved/provisional
   - preserve candidate GIDs and evidence

2. Accumulate a small bounded amount of multi-frame evidence:
   - appearance embeddings/prototype
   - event time
   - candidate scores
   - quality
   - source/destination camera

3. Re-evaluate unresolved tracks using aggregated evidence rather than a single-frame embedding.

4. When multiple ambiguous arrivals compete for multiple existing GIDs, resolve them **jointly using one-to-one assignment**, not independently.

5. Preserve temporal/event-order evidence:
   - departure order from `cam_2`
   - arrival order at `cam_1`
   - canonical event time only
   - do not use processing wall time

6. Once evidence is sufficient:
   - valid existing identity -> commit `global-cross-camera`
   - unresolved after bounded window -> create new/provisional GID according to existing policy

7. Never allow unresolved evidence to:
   - poison permanent gallery
   - deactivate previous camera presence
   - create confirmed handoff history
   - bypass topology/hard gates
   - bypass one-to-one Hungarian

## Do NOT change

- cross-camera similarity threshold
- total score threshold
- `0.08` ambiguity margin unless a separately measured validation proves it necessary
- topology values
- Presence timeout
- Dormant TTL
- camera/video acquisition
- playback/synchronization
- capture workers
- Docker camera handling
- calibration

Keep the frozen camera/video subsystem untouched.

## Mandatory implementation contract

### Unresolved state

For a cross-camera row rejected only by `ambiguous_top1_top2`:

- do not call `_new_identity()` in that batch
- do not allocate or consume a GID as a placeholder
- retain a bounded unresolved record keyed by camera, local track ID, and coordinator/camera generation
- retain canonical event-time bounds, bounded samples, quality, candidate GIDs, scores, and source/destination camera
- clear it deterministically on resolution, expiry, reset, or generation invalidation

Use an existing compatible pending structure if one exists. Do not create a second identity manager, database, or GID namespace.

### Evidence aggregation

Evidence aggregation must accept only samples passing existing crop, blur, confidence, and tracklet-quality rules. Recompute candidate appearance and final scores from an aggregated normalized prototype; do not average stale final scores blindly. Re-run lifecycle, topology, map, size, confirmation, and generation hard gates on every attempt. Use observation event time only. Pending evidence must not update a permanent gallery.

### Joint resolution and event order

When unresolved tracks share candidate GIDs, build one score matrix and resolve them jointly with the existing Hungarian one-to-one mechanism. Departure/arrival event order may be an explicit, deterministic, auditable consistency signal, but it must not bypass thresholds or hard gates and must not invent topology values. If evidence cannot safely distinguish GID2/GID3, do not hard-code IDs or force top-1.

### Commit and fallback semantics

Before resolution, unresolved evidence must not create a committed local mapping, update permanent identity/gallery state, deactivate previous presence, create a confirmed handoff, steal ownership, or be reported as committed. After successful resolution, use the canonical `global-cross-camera` commit path. On bounded expiry, create at most one new GID for the still-current local track, never one per frame.

### Async result propagation

Preserve the non-blocking coordinator. Do not add waits, barriers, or polling to capture/video workers. Publish a later commit through the existing generation-safe result cache/path, discard stale generation results, and do not change preview endpoints or uploaded-video synchronization.

### Diagnostics

Expose bounded diagnostics for the unresolved key, event-time range, sample count, candidates, appearance/final scores, hard gates, topology, aggregate margin, event-order result, pending/resolution reason, final source, and handoff outcome. Do not log embeddings, frames, or crops.

## Required tests

Add deterministic tests for:

1. Two known GIDs leave `cam_2` and enter `cam_1` with initially ambiguous appearance.
2. First ambiguous frame does NOT create GID4/GID5 immediately.
3. Multi-frame evidence resolves:
   - GID2 -> GID2
   - GID3 -> GID3
4. Joint assignment cannot swap/reuse one GID twice.
5. A genuinely unknown person eventually receives a new GID.
6. Ambiguous evidence never updates permanent gallery before resolution.
7. Existing Prompt08, Presence/Handoff, topology, lifecycle and one-to-one tests still pass.
8. Repeated frames of one unresolved local track do not allocate repeated GIDs.
9. A stale async result cannot bind to a restarted/reused local track ID.
10. Pending evidence never updates permanent gallery or camera presence.
11. Successful resolution records exactly one confirmed handoff.
12. Event-time behavior stays deterministic under artificial processing delay.
13. Pending state is bounded and cleared on resolution, reset, or expiry.
14. Rendering/result-generation tests confirm delayed resolution does not break annotated-frame propagation.

## Acceptance criteria

The real failure pattern:

```text
cam_2 GID2 -> cam_1 GID4
cam_2 GID3 -> cam_1 GID5
```

must no longer occur merely because the first cross-camera frame has `ambiguous_top1_top2`.

The system must wait for bounded additional evidence and resolve valid handoffs jointly before creating new identities.

Also require:

- no threshold, topology value, or presence timeout is weakened
- pending state is bounded and generation-safe
- unresolved evidence cannot poison gallery, presence, or handoff state
- successful resolution uses the canonical `global-cross-camera` path
- async coordinator submission remains non-blocking
- frozen camera/video, preview, uploaded playback, Docker, and calibration paths remain unchanged

After implementation report:

- files changed
- root cause confirmed before editing
- unresolved evidence window/bounds used
- unresolved-state key and cleanup rules
- exact resolution logic
- aggregation, joint-assignment, and event-order policy
- async result propagation and stale-generation handling
- commit/gallery/presence/handoff behavior
- unknown/unresolved fallback behavior
- tests/results
- confirmation that thresholds and frozen camera/video/calibration subsystem were not changed
