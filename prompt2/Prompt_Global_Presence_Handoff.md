# Prompt — Global Presence + Confirmed Cross-Camera Handoff

## Mandatory guardrails

Before doing anything, read and obey `AGENTS.md`.

The camera/video acquisition and calibration subsystem is FROZEN and confirmed working.

Do NOT refactor, redesign, replace, move, or change behavior related to:
- camera discovery/enumeration
- camera source/device resolution
- `cv2.VideoCapture` or equivalent opening
- Docker camera/device passthrough
- capture workers / worker lifecycle
- reconnect logic
- frame acquisition/decoding
- latest-frame cache
- live preview/video feed
- uploaded-video input/playback/synchronization
- calibration frame acquisition
- calibration endpoints/UI/workflow

Do NOT add blocking waits, barriers, or cross-camera ownership into capture/video workers.

Use `backend/main.py` as the production source of truth.
Do not modify root `main.py` unless explicitly required.

Do not install/update dependencies, download models, alter real runtime data, start/restart/rebuild/deploy services or containers, or commit/push unless explicitly authorized.

Implement only the scope below. Do not opportunistically fix unrelated findings.

## Problem to solve

The intended identity behavior is:

```text
CamA sees Person X:
local track -> GID 1

Person X leaves CamA and appears on CamB:
CamB must recover GID 1 when Re-ID evidence and hard gates support the handoff.

Later Person X leaves CamB and returns to CamA:
CamA must again recover GID 1.
```

Do NOT physically move/delete the global identity record from one camera database to another.

Use one Global Identity memory/store as the source of truth.
Each camera may keep only local presence/mapping state.

## Goal

Add explicit camera presence and confirmed handoff state around the existing Global Identity Manager so cross-camera transfer is represented correctly and deterministically.

Important distinction:
- remove/deactivate the old camera-local presence
- NEVER delete the Global GID/gallery simply because the person moved to another camera

For a non-overlap handoff:

```text
before:
GID 1 active on CamA

confirmed CamA -> CamB handoff:
CamA presence = inactive
CamB presence = active
GID 1 remains the same global identity
last_camera = CamB
```

For an allowed overlap pair:

```text
GID 1 may temporarily be active on both CamA and CamB
```

only when topology explicitly permits overlap.

## First: inspect the existing implementation

Before editing, determine:
1. How `assign_global_batch()` currently chooses cross-camera GIDs.
2. Where local `(camera, local_track_id) -> gid` mappings are stored.
3. Where identity `last_camera`, last seen event time, state, gallery/prototype are stored.
4. Current SQLite schema in `backend/identity_store.py`.
5. Whether handoff history/presence already exists under another name.
6. How topology `overlap_allowed`, min/max travel time, and event time participate in hard gates.
7. How ACTIVE/DORMANT lifecycle interacts with cross-camera recovery.

Reuse existing structures where possible. Do not create duplicate identity stores.

## Required implementation

### 1. Global identity remains the source of truth

A GID must be independent of camera.

Do NOT create:
- one independent identity DB per camera
- duplicate GID namespaces per camera
- physical row transfer between camera databases

Use the existing SQLite identity store and in-memory identity manager.

### 2. Add/normalize camera presence state

Represent current/recent presence per GID and camera.

Minimal conceptual fields:

```text
gid
camera
local_track_id or local generation when available
first_seen_event_time
last_seen_event_time
active
generation
assignment_source
```

Use an existing equivalent structure if already present.

Presence is downstream identity metadata, not capture-worker state.

### 3. Confirmed handoff behavior

When CamB is validly assigned the same GID previously associated with CamA, record a confirmed handoff:

```text
gid
from_camera
to_camera
exit/last_seen event time on CamA
entry/first_seen event time on CamB
appearance score
final score
assignment source
topology result
reason
```

For non-overlap camera pairs:
- after CamB assignment is committed, deactivate stale CamA presence for that GID
- do not delete the Global Identity
- do not delete gallery/prototype
- do not delete audit/history

For overlap-allowed pairs:
- allow both presences to remain active while simultaneous visibility is valid
- transition to one active presence only after normal evidence indicates one camera no longer sees the person

### 4. Never transfer ownership on an unconfirmed detection

Only commit handoff after:
- hard gates pass
- appearance/matching threshold passes
- ambiguity policy passes
- global Hungarian assigns the GID
- one-to-one invariant passes
- generation/staleness guard passes

Rejected/ambiguous/provisional observations must not steal ownership.

### 5. Cross-camera recovery

Expected:

```text
CamA:
LID 5 -> GID 1

CamA track ends

CamB:
new LID 20
appearance/topology/time valid
-> GID 1
assignment_source = global-cross-camera
```

Later:

```text
CamB:
LID 20 -> GID 1

CamB track ends

CamA:
new LID 31
appearance/topology/time valid
-> GID 1
```

Local track IDs are not global identity.

### 6. Event time, not processing wall time

Presence aging, handoff travel time, identity eligibility, and transition timing must use canonical observation event time.

Do not reintroduce processing wall time into identity correctness logic.

### 7. Topology

Respect existing topology hard gates.

For non-overlap pairs:
- simultaneous active ownership of one GID must not be silently accepted
- confirmed transfer may deactivate previous camera presence

For overlap-allowed pairs:
- temporary simultaneous presence is valid

Do not invent physical travel times or overlap settings.

### 8. Identity lifecycle

Do not delete global identity memory when camera presence becomes inactive.

```text
camera presence inactive != global identity expired
```

If Prompt09 lifecycle fixes already exist, integrate with them.

If the known ACTIVE/DORMANT timeout bug still exists, do not silently redesign the whole state machine. Report the dependency and make only the smallest directly-required correction.

### 9. Persistence

Persist enough state to survive restart:
- global identity
- lifecycle state
- last camera / last seen event time
- camera presence needed for continuity
- confirmed handoff history/audit
- next GID sequence as already supported

Do not persist redundant raw frames/crops.

Do not write to or migrate the user's real DB during tests.

### 10. Diagnostics

Extend existing diagnostics to include:

```text
gid
from_camera
to_camera
previous presence
new presence
event-time delta
candidate appearance
final score
hard-gate result
margin
assignment source
handoff committed true/false
handoff rejection reason
```

Successful cross-camera recovery must be clearly visible as `global-cross-camera` or the existing canonical equivalent.

## Tests

Add device-free tests for:

1. CamA -> CamB successful handoff:
   - CamA LID 5 = GID 1
   - CamA disappears
   - CamB new LID 20
   - valid same-person embedding/gates
   - final GID = 1
   - CamB presence active
   - CamA presence inactive for non-overlap
   - handoff history recorded

2. CamB -> CamA return:
   - new local ID on CamA
   - recover GID 1

3. Different person must not steal GID.

4. Ambiguous candidate does not transfer ownership.

5. Overlap allowed:
   - same GID may temporarily have active presence on both permitted cameras.

6. Overlap not allowed:
   - one confirmed transfer yields only one current active presence.

7. One-to-one:
   - two observations cannot both steal the same non-overlap GID.

8. Restart:
   - identity/presence/handoff metadata restores using a temporary DB.

9. Event time:
   - handoff timing uses observation event time, not processing delay.

10. Existing Prompt08/Prompt08-updated global assignment and short-gap continuity tests still pass.

Tests must not open cameras, run Docker, play real videos, touch calibration, or write production DB.

## Acceptance criteria

Complete only when:
- GID is clearly global rather than owned by one camera DB.
- CamA GID1 -> CamB valid handoff results in CamB GID1.
- CamB GID1 -> CamA valid return results in CamA GID1.
- Old camera presence becomes inactive only after confirmed non-overlap handoff.
- Global identity/gallery is not deleted during handoff.
- Overlap policy allows multi-camera presence only when explicitly configured.
- Rejected/ambiguous detections cannot steal ownership.
- Event time is used.
- One-to-one remains intact.
- Persistence/restart remains correct.
- Frozen camera/video/calibration subsystem is unchanged.
- Relevant tests pass.

## Verification

Run only:
- Python syntax/compile checks
- presence/handoff tests
- Prompt08 global assignment tests
- Prompt08-updated short-gap continuity tests
- relevant lifecycle/store tests with temporary DBs
- topology/time tests

Do not start services, Docker, cameras, RTSP, or real video playback.

## Final report

Report:
1. Files changed.
2. Existing identity/presence architecture before changes.
3. Whether new tables/fields were actually necessary.
4. Final representation of Global Identity vs camera presence.
5. Exact CamA -> CamB handoff commit flow.
6. Exact CamB -> CamA return flow.
7. Non-overlap vs overlap behavior.
8. Persistence behavior.
9. Diagnostics added.
10. Tests run/results.
11. Confirmation camera/video/Docker/calibration paths were not modified.
12. Any remaining reason a valid person may still receive a new GID: appearance, ambiguity, lifecycle expiry, topology, one-to-one, etc.
