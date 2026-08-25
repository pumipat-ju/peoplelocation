# Prompt 12 — Deadline Regression and End-to-End Validation Suite

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

Create a practical regression suite and reproducible validation workflow for the production Re-ID system before the deadline.

This prompt is NOT a request to build a full academic tracking benchmark framework from scratch.

Prioritize tests that prove the production behaviors implemented in Prompts 08–11 and catch regressions in the working camera/video/calibration boundary without opening real devices.

Do not modify production behavior merely to make tests easier unless there is a confirmed bug directly blocking a required acceptance case.

## Required work

### A. Production-path regression tests

Add integration-level tests, using mocks/fakes, for:

1. Multi-camera global assignment
   - two cameras in one decision window
   - one-to-one GID invariant
   - unmatched observation can become safe provisional/new identity

2. Handoff
   - identity moves from camera A to camera B through an allowed transition
   - same GID is recovered when evidence passes gates

3. Dormant recovery
   - ACTIVE -> DORMANT
   - DORMANT -> ACTIVE before TTL
   - DORMANT -> EXPIRED after TTL

4. Persistence/restart
   - state survives store reload
   - next GID remains monotonic
   - expired identity is not reused

5. Tracklet/gallery safety
   - reused local track ID does not inherit stale identity evidence
   - low-quality sample does not update permanent gallery

6. Ambiguity
   - close top candidates with insufficient margin are rejected rather than force-matched

7. Topology/time
   - impossible transition rejected
   - too-fast/too-slow travel rejected
   - overlap simultaneous presence allowed only when explicit

8. Uploaded-video metadata contract
   - configured offset reaches backend event-time logic

### B. Frozen-subsystem regression guards

Create safe tests/static checks that verify downstream changes have not coupled identity assignment to capture-worker blocking.

At minimum:
- global assignment test must not require a real capture worker
- fake/slow/missing camera input must not block another observation's downstream decision
- imported camera/video code must not open real devices during tests
- calibration API/data must not be modified during tests

If the repository already has mocked camera-worker/concurrency tests, run and reuse them rather than rewriting them.

Do NOT create tests that require Docker camera passthrough or a physical webcam as part of the automated suite.

### C. Evaluation metrics — deadline minimum

Extend the existing evaluator only enough to produce clear project-level regression metrics from labeled event sequences.

Include when supported by available ground truth:
- handoff accuracy
- false merge count/rate
- false split / fragmentation count
- temporal ID switch count
- a clearly named project-level identity consistency metric if existing code already has one

If the current repository already calculates approximate/non-standard IDF1, do NOT label it as standard IDF1 unless it is actually implemented according to a recognized definition.

Do not implement HOTA from scratch in this deadline prompt unless a correct implementation/dependency already exists in the repository and can be used without dependency installation.

Do not add Rank-1/mAP here unless a valid Re-ID evaluation dataset/pipeline already exists in the repository.

Clearly separate:
- standard metrics
- project-specific/approximate metrics

### D. Reproducible reports

Provide a simple command/test workflow that outputs:
- machine-readable JSON report
- short human-readable summary

Include metadata such as:
- timestamp
- relevant configuration values
- topology/config version/path where safe
- code/test scenario name
- metric definitions/version where relevant

Do not include secret/private embedding contents.

### E. Scenario coverage

Use synthetic or checked-in lightweight fixture scenarios where possible:
- similar appearance
- crossing/occlusion
- leave and re-enter
- cross-camera handoff
- overlap cameras
- impossible transition
- ambiguous match

Do not create a huge dataset.

### F. Final full regression run

Run the safe automated suite relevant to:
- global assignment
- identity lifecycle/store
- tracklet/gallery safety
- timestamp/topology
- evaluator/report

Also run syntax/import checks that do not initialize real hardware or write runtime data.

Do not start services, containers, webcams, RTSP streams, or real uploaded-video processing automatically.

## Acceptance criteria

- There is a repeatable command/workflow for the deadline regression suite.
- Production global-assignment call path is tested.
- Dormant recovery and restart persistence are tested.
- Tracklet/gallery safety is tested.
- Time/topology hard gates are tested.
- Frozen camera/video/calibration behavior is protected from test-side device access or lifecycle coupling.
- Evaluator outputs reproducible JSON plus a readable summary.
- Non-standard metrics are labeled honestly.
- Test suite does not touch production DB/runtime data.
- Relevant tests pass, or failures are reported precisely without hiding them.

## Final report

Provide a concise deadline-readiness report with:

### PASS
Behaviors proven by automated tests.

### FAIL
Any acceptance criteria currently failing.

### DEFERRED
Important but non-blocking items intentionally not completed before the deadline, such as:
- full OSNet training/threshold pipeline
- full zone editor in calibration UI
- standard HOTA if no correct implementation exists
- large-scale real-world accuracy benchmark

### MANUAL TEST CHECKLIST
List only the manual checks the user should run separately on the real environment, such as:
- Docker detects the real cameras
- live preview opens
- calibration still works
- uploaded videos play
- two real cameras perform a handoff

Do not execute those manual/hardware checks automatically.

Also report:
1. Files changed.
2. Exact commands/tests run.
3. Results.
4. Known limitations.
5. Whether the project is ready for a controlled manual demo/test, without claiming measured real-world accuracy unless such measurements actually exist.
