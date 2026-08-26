# Prompt — Fix Rendering Generation Mismatch

Read and obey `AGENTS.md`.

Fix only the current downstream rendering regression.

## Root cause

`camera_generation` and `coordinator_generation` are different namespaces but are being compared as if they were the same.

This causes `preview_trusted_assignments()` to return `None`, and the rendering loop skips the person entirely.

## Required fix

- Keep `camera_generation` and `coordinator_generation` separate.
- Compare each only with the same generation namespace.
- Ensure trusted preview uses the correct coordinator generation.
- Do not let `res is None` suppress an otherwise valid YOLO/BoT-SORT track.
- If GID is still pending, render at least:
  - bbox
  - local track ID
- Add GID once a trusted assignment is available.
- Preserve one-to-one assignment, stale-generation guards, Presence/Handoff, and Global Hungarian behavior.

## Regression tests

Add tests proving:

1. `camera_generation=1` and `coordinator_generation=0` are not treated as a mismatch.
2. Valid tracked person is rendered even while GID is pending.
3. GID appears after trusted assignment becomes available.
4. Existing Prompt08/Presence/Handoff tests still pass.

## Frozen subsystem

Do NOT modify:
- camera discovery / `VideoCapture`
- capture workers
- Docker camera access
- uploaded-video decoding/playback/synchronization
- preview acquisition/cache architecture
- calibration

Make the smallest downstream fix only.

After implementation, report root cause, files changed, tests run/results, and confirm the frozen subsystem was untouched.