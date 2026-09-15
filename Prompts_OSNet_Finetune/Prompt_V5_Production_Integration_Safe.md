# Prompt: Safe Production Support for V5 Re-ID

## Objective

Add an optional production Re-ID mode corresponding as closely as possible to:

**V5 = Domain-balanced fine-tuned OSNet x1.0 + Improved Crop**

This task is ONLY for Re-ID inference integration.

Do not change any unrelated production behavior.

---

## 1. STRICT SCOPE

Allowed changes:

1. OSNet checkpoint loading for the fine-tuned model.
2. Re-ID crop selection immediately before the crop is passed into OSNet.
3. Minimal configuration/environment variables required to switch between existing behavior and V5.
4. Small isolated tests for the above behavior.

Do NOT modify:

- camera discovery
- camera device access
- camera opening
- live preview
- uploaded video input
- video decoding
- capture workers
- reconnect logic
- live frame cache
- calibration
- tracker behavior
- detector behavior
- bounding-box generation
- tracker state
- global identity manager
- handoff logic
- topology
- travel-time logic
- frontend
- database schema
- existing matching logic
- existing thresholds unless explicitly configured separately
- any frozen camera/video/calibration subsystem

Do not refactor unrelated code.

---

## 2. BACKWARD COMPATIBILITY

Current production behavior must remain the default.

If no new configuration is provided, the application must behave exactly
as it did before this task.

Add:

`REID_CROP_MODE=original|improved`

Default:

`REID_CROP_MODE=original`

Behavior:

- `original` = preserve the current production crop code exactly
- `improved` = use the raw person bbox for the Re-ID crop without adding
  the existing Re-ID crop margin

Invalid values must safely fall back to `original` and emit a warning.

Do NOT silently change the existing default behavior.

---

## 3. V5 MODEL CHECKPOINT

V5 uses:

`backend/reid_experiments/finetune_osnet_v2_balanced/best_checkpoint.pth`

Continue using the existing:

`REID_CHECKPOINT_PATH`

configuration mechanism if it already exists.

Do not hard-code the V5 checkpoint as the new default.

The existing pretrained checkpoint must remain usable.

---

## 4. CHECKPOINT COMPATIBILITY

Before changing production code, inspect the format of:

`backend/reid_experiments/finetune_osnet_v2_balanced/best_checkpoint.pth`

Determine whether it contains:

- raw state_dict
- model/state_dict wrapper
- classifier weights
- optimizer state
- epoch/training metadata

Production only needs the OSNet feature extractor.

The training classifier contains 1,795 training identities and must NOT
be required for production embedding extraction.

Load the feature-extractor weights correctly while handling classifier
shape mismatch safely.

Do not modify the original checkpoint.

Do not suppress arbitrary missing/unexpected keys.

Only ignore classifier-related keys if necessary and document exactly
which keys were excluded.

After loading, verify:

- model = OSNet x1.0
- model is in eval mode
- output embedding dimension = 512
- output values are finite
- L2 normalization succeeds
- inference works on CUDA when available

If the current production loader already supports the checkpoint,
reuse it and avoid unnecessary changes.

---

## 5. IMPORTANT: MEANING OF "IMPROVED CROP"

The offline V5 experiment used:

**Raw GT bounding box with no additional margin.**

Production does NOT have ground-truth bounding boxes.

Therefore do NOT claim that production uses "Raw GT bbox".

For production, implement the closest runtime equivalent:

**Raw detector/tracker person bbox with no additional Re-ID margin.**

This distinction must be documented clearly.

---

## 6. ORIGINAL CROP MODE

When:

`REID_CROP_MODE=original`

use the exact existing production Re-ID crop implementation.

Do not change:

- current margin
- coordinate expansion
- clipping behavior
- preprocessing
- resize
- detector/tracker bbox
- visualization

This mode must be regression-compatible with the existing system.

---

## 7. IMPROVED CROP MODE

When:

`REID_CROP_MODE=improved`

change ONLY the crop supplied to the OSNet embedding extractor.

Given the existing person bbox:

`x1, y1, x2, y2`

perform:

1. Convert coordinates using the existing bbox convention.
2. Clip coordinates to valid frame boundaries.
3. Do NOT add any extra margin.
4. Do NOT alter the original detector/tracker bbox.
5. Do NOT alter tracking state.
6. Do NOT alter the bbox drawn on the preview.
7. Reject invalid or empty crops using safe existing behavior.
8. Send the resulting crop through the same existing OSNet preprocessing pipeline.

Preserve:

- BGR -> RGB conversion as currently verified
- resize to 256x128
- ImageNet normalization
- OSNet feature extraction
- 512-D embedding
- L2 normalization

Do not modify preprocessing beyond crop selection.

---

## 8. KEEP THE CHANGE ISOLATED

Prefer a small helper or isolated branch such as:

```python
def get_reid_crop(frame, bbox, crop_mode):
    if crop_mode == "improved":
        return crop_raw_bbox_without_margin(frame, bbox)

    return existing_production_crop(frame, bbox)
```

The example above is conceptual.

Use the project's actual architecture and existing functions.

Do not duplicate the Re-ID pipeline if an existing crop helper can be
safely extended.

---

## 9. DO NOT CHANGE MATCHING THRESHOLDS YET

Do NOT automatically replace the current production Re-ID threshold
with the offline V5 validation threshold.

The offline V5 threshold:

`0.583767414`

was selected from the PeopleLocation validation dataset using a
cross-camera EER operating point.

Record this value for reference only.

Do not apply it to production unless the production threshold is already
explicitly configurable and the user selects it later.

This task is checkpoint + crop integration only.

---

## 10. STARTUP DIAGNOSTICS

At startup, expose enough information to confirm which version is
actually running.

Log at least:

- Re-ID enabled status
- checkpoint path
- checkpoint load result
- crop mode
- model/device
- embedding dimension

Example:

```text
ReID enabled: true
ReID model: OSNet x1.0
Checkpoint: ...\finetune_osnet_v2_balanced\best_checkpoint.pth
Crop mode: improved
Device: cuda
Embedding dimension: 512
```

Do not print camera credentials or other secrets.

---

## 11. TARGETED TESTS

Add tests only for the modified Re-ID behavior.

### Crop tests

Verify:

1. `original` preserves the previous production crop behavior.
2. Missing `REID_CROP_MODE` defaults to `original`.
3. Invalid `REID_CROP_MODE` falls back to `original`.
4. `improved` adds no margin.
5. `improved` clips boxes correctly at frame boundaries.
6. Invalid/empty bboxes are safely rejected.
7. Improved crop affects only the Re-ID image crop and does not mutate
   the source bbox.

### Checkpoint tests

Verify:

1. Balanced V2 checkpoint can be loaded for feature extraction.
2. Classifier mismatch is handled intentionally, not silently.
3. Output embedding shape is `(N, 512)`.
4. Embeddings contain finite values.
5. L2-normalized embeddings have norm approximately 1.
6. Model is in eval mode.

Tests must not require a physical/live camera.

---

## 12. REGRESSION SAFETY

Before finishing, confirm that:

- original crop remains the default
- pretrained checkpoint still loads
- existing application startup still works
- no camera/video/calibration files were modified
- no tracker or global-ID logic was modified
- no production matching thresholds were changed
- no frontend code was modified

List every modified file.

If a change outside the allowed scope appears necessary, STOP and
report it instead of modifying that subsystem.

---

## 13. MANUAL RUN COMMANDS

The project is being run from Windows CMD.

Provide exact CMD commands at completion.

### Existing pretrained + original crop

```cmd
cd C:\PeopleLocation\backend
venv\Scripts\activate

set REID_CHECKPOINT_PATH=C:\PeopleLocation\weights\osnet_x1_0_market1501.pth
set REID_CROP_MODE=original

python main.py
```

### Balanced model + original crop (V4-style runtime)

```cmd
cd C:\PeopleLocation\backend
venv\Scripts\activate

set REID_CHECKPOINT_PATH=C:\PeopleLocation\backend\reid_experiments\finetune_osnet_v2_balanced\best_checkpoint.pth
set REID_CROP_MODE=original

python main.py
```

### Balanced model + improved runtime crop (V5-style runtime)

```cmd
cd C:\PeopleLocation\backend
venv\Scripts\activate

set REID_CHECKPOINT_PATH=C:\PeopleLocation\backend\reid_experiments\finetune_osnet_v2_balanced\best_checkpoint.pth
set REID_CROP_MODE=improved

python main.py
```

Important:

Call this `V5-style runtime` or
`Balanced v2 + improved production crop`.

Do not claim it is mathematically identical to offline V5,
because offline V5 used ground-truth bounding boxes while production
uses detector/tracker bounding boxes.

---

## 14. FINAL REPORT

At completion report:

1. Files modified.
2. Exact function(s) involved in Re-ID cropping.
3. Exact checkpoint-loading behavior.
4. Whether classifier keys had to be excluded.
5. Whether the balanced checkpoint loaded successfully.
6. Embedding smoke-test result.
7. Whether `original` remains the default.
8. Number of tests passed/failed.
9. Confirmation that frozen subsystems were untouched.
10. Exact Windows CMD command for testing the V5-style runtime.

Do NOT automatically start cameras.
Do NOT automatically change production defaults.
Do NOT integrate or tune any additional Re-ID logic.
