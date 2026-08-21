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

## Coding rules
- Inspect existing code before adding new abstractions.
- Avoid duplicate logic.
- Keep callers and function signatures consistent.
- Prefer minimal scoped changes over rewrites.
- Run syntax/import/tests after modifications.
- Never claim accuracy improvements without measurements.