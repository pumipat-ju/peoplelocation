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

## Scope discipline
- Do only the task explicitly requested by the user.
- Do not modify unrelated files or fix unrelated findings without the user's approval.
- Treat review, diagnosis, explanation, and status requests as read-only. Report findings but do not implement fixes unless the user explicitly asks for changes.
- Do not start, stop, restart, rebuild, or deploy services or containers unless the user explicitly requests it.
- Do not install or update dependencies, download models, alter databases or runtime data, delete files, or perform Git commit/push operations unless explicitly authorized.
- For implementation requests, limit edits and verification to what is necessary for the requested change. Do not expand the scope based on inferred improvements.
- If an out-of-scope problem blocks the task, explain the blocker and ask the user before taking additional action.
- Preserve existing user changes and untracked files. Never overwrite or clean them up without explicit permission.
