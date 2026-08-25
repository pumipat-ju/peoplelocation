# AGENTS.md

## Project
Multi-camera People Location / Re-ID system.

## Primary backend
Use backend/main.py as the production source of truth.
Do not modify root main.py unless explicitly required.

## Preserve
Do not regress:
- YOLO person detection
- BoT-SORT tracking
- OSNet Re-ID
- Hungarian one-to-one assignment
- occlusion/anti-ID-swap logic
- ACTIVE/DORMANT Re-ID memory
- SQLite persistence
- homography/floorplan mapping
- camera/video acquisition and device discovery
- live camera preview and capture
- uploaded video playback/input
- Docker camera/device access
- camera calibration workflow

## Frozen camera/video subsystem
The current camera and video acquisition pipeline is confirmed stable and MUST be treated as a frozen subsystem.
Do not refactor, redesign, replace, move, or change the behavior of this subsystem unless the user explicitly requests it.
This protection applies to BOTH:
- real-time / live camera input
- uploaded video / file input

Do not modify behavior related to:
- camera discovery or camera enumeration
- camera source/device resolution
- cv2.VideoCapture or equivalent device opening
- Docker camera/device passthrough
- capture workers or threads
- camera worker lifecycle
- reconnect logic
- frame grabbing or decoding
- latest-frame cache
- live preview / video feed
- uploaded-video playback workers
- existing video synchronization infrastructure
- calibration frame acquisition
- calibration endpoints
- calibration workflow
- APIs used by camera selection, preview, video input, or calibration

Known unacceptable regressions include:
- Docker no longer detecting cameras
- camera devices failing to open
- live preview or capture stopping
- uploaded video input breaking
- calibration becoming unusable

New Re-ID, global assignment, topology, persistence, evaluation, or identity logic must be implemented AROUND the existing camera/video pipeline.

Preferred architecture:

existing camera/video acquisition
    -> existing detection/tracking
    -> new downstream logic

Do NOT redesign upstream camera/video acquisition in order to implement downstream features.

If a requested feature appears to require changing the frozen camera/video subsystem, do not make that change automatically. Report the dependency/blocker first and wait for explicit user approval.

Any implementation touching backend camera/video code must verify that existing live-camera, uploaded-video, preview, Docker-device, and calibration behavior is preserved.

Changes to downstream Re-ID logic must not introduce new synchronization,
blocking waits, barriers, or ownership changes into camera/video capture workers.

## Coding rules
- Inspect existing code before adding new abstractions.
- Avoid duplicate logic.
- Keep callers and function signatures consistent.
- Prefer minimal scoped changes over rewrites.
- Run syntax/import/tests after modifications.
- Never claim accuracy improvements without measurements.

## Scope discipline
- Do only the task explicitly requested by the user.
- Do not modify unrelated files or fix unrelated findings without the user's approval.
- Treat review, diagnosis, explanation, and status requests as read-only. Report findings but do not implement fixes unless the user explicitly asks for changes.
- Do not start, stop, restart, rebuild, or deploy services or containers unless the user explicitly requests it.
- Do not install or update dependencies, download models, alter databases or runtime data, delete files, or perform Git commit/push operations unless explicitly authorized.
- For implementation requests, limit edits and verification to what is necessary for the requested change. Do not expand the scope based on inferred improvements.
- If an out-of-scope problem blocks the task, explain the blocker and ask the user before taking additional action.
- Preserve existing user changes and untracked files. Never overwrite or clean them up without explicit permission.
