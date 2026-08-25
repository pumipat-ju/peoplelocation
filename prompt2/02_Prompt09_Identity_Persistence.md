# Prompt 09 — Fix Identity Lifecycle and SQLite Persistence

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

Fix the production identity lifecycle and persistence bugs without changing camera/video acquisition, preview, uploaded-video playback, Docker device access, or calibration.

Focus on:
- ACTIVE / DORMANT / EXPIRED lifecycle correctness
- dormant recovery
- durable SQLite state
- restart-safe GID sequencing
- transactional state/audit persistence

Do NOT redesign matching architecture or camera processing in this prompt.

## Required work

1. Inspect the existing identity state machine and persistence implementation in:
   - `backend/main.py`
   - `backend/identity_store.py`
   - `docker-compose.yml`
   - `.env.example` if present
   - relevant tests

2. Fix dormant recovery semantics.
   - Verify how `REID_MAX_IDLE_SEC`, `IDENTITY_DORMANT_TTL_SEC`, and state-specific gates currently interact.
   - An identity that has transitioned to DORMANT must remain eligible for dormant recovery until the dormant TTL expires, subject to the existing safe appearance/topology/matching gates.
   - Do not let a generic active-idle gate reject every DORMANT candidate immediately.
   - EXPIRED identities must not be reused.

3. Make state transitions explicit and validated.
   Allowed transitions should be limited to the current intended model, at minimum:
   - PROVISIONAL -> ACTIVE
   - PROVISIONAL -> DORMANT when existing policy legitimately allows it
   - ACTIVE -> DORMANT
   - DORMANT -> ACTIVE
   - DORMANT -> EXPIRED

If the current code supports a slightly different legitimate transition required by existing behavior, preserve it and document why rather than silently breaking it.

4. Persist every meaningful lifecycle transition.
   - State, timestamp, reason, and relevant identity metadata must be written consistently.
   - A cleanup-driven ACTIVE -> DORMANT transition must not exist only in RAM while SQLite still says ACTIVE.
   - Transition persistence and the identity snapshot/update should be atomic where the existing store architecture allows it.

5. Fix the default SQLite location for Docker persistence.
   - Keep host/local development practical.
   - Configure container usage so the identity DB lives under the already persistent data volume, e.g. `/app/data/identity_memory.sqlite3`, or the repository's existing equivalent.
   - Add/use `IDENTITY_DB_PATH` consistently in Compose and `.env.example` if that pattern already exists.
   - Do NOT run a migration against the user's real DB.
   - Do NOT delete or overwrite existing runtime DB files.

6. Ensure restart-safe identity restore.
   - Restore state correctly.
   - Restore/derive the next GID so a restarted process cannot accidentally reuse an existing GID.
   - Preserve schema compatibility where possible.
   - If a schema migration would be required, do not execute it automatically; implement backward-compatible handling or report the blocker.

7. Close persistence resources on normal application shutdown using the existing shutdown/lifespan mechanism.
   - Do not alter camera worker shutdown semantics.

8. Add lightweight diagnostics to the existing status/debug mechanism if safe:
   - counts by state
   - recent transition information
   - DB path/status
Do not expose private embedding data unnecessarily.

## Tests

Use temporary SQLite databases or `:memory:` where compatible.

Required cases:
1. ACTIVE -> DORMANT after idle cleanup is persisted.
2. DORMANT -> ACTIVE recovery works before dormant TTL expiry.
3. DORMANT candidate is not incorrectly rejected by the active idle timeout.
4. DORMANT -> EXPIRED occurs after TTL and EXPIRED is not reused.
5. Restart simulation restores identity state.
6. Restart simulation preserves next-GID monotonicity.
7. Transition audit includes timestamp and reason.
8. Docker/Compose configuration points the container DB into the persistent data volume.
9. Tests never touch the user's production identity DB.

Do not require a real camera, live worker, uploaded-video worker, calibration, or Docker runtime.

## Acceptance criteria

- Dormant recovery is possible during the intended dormant TTL.
- Expired identities are not reused.
- Cleanup-triggered state changes are persisted.
- Restart preserves state and GID sequencing.
- Container DB path is under the persistent volume.
- The real DB is not migrated/modified during verification.
- Camera/video/calibration code paths are unchanged.
- Relevant tests pass.

## Verification to run

Run only:
- syntax/compile checks for modified Python files
- state/store unit tests
- restart simulation tests using temporary DBs
- static validation of Compose/env changes

Do not start Docker or services.

## Final report

Report:
1. Files changed.
2. The dormant-recovery bug found and exact fix.
3. Final state transition rules.
4. Final DB path behavior for local vs Docker.
5. How restart GID sequencing is preserved.
6. Tests run/results.
7. Any migration/backward-compatibility limitation.
