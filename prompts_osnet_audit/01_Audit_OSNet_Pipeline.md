# Prompt 01 — Audit OSNet Embedding Pipeline Only

You are working on the PeopleLocation project.

## Goal
Audit the OSNet Re-ID embedding pipeline end-to-end and identify any correctness issues before changing model training or dataset size.

## CRITICAL FROZEN SUBSYSTEM — DO NOT MODIFY
The camera/video acquisition subsystem is frozen. Do not edit, refactor, replace, or "clean up" anything related to:

- camera discovery or device access
- Docker/WSL camera passthrough
- live camera opening/capture
- capture workers/threads
- reconnect logic
- uploaded video decoding/playback
- frame cache / last_frame / preview snapshot path
- live preview endpoints
- calibration flow or calibration UI
- homography calibration behavior
- video input registration/removal
- any code whose change could alter camera/video/calibration behavior

If Re-ID code is mixed into a frozen file, make the smallest possible isolated change only when absolutely necessary. Prefer adding helper modules/tests/scripts outside the frozen subsystem.

## Preserve Existing Re-ID Architecture
Do not remove or redesign:

- YOLO detector
- BoT-SORT tracker
- OSNet Re-ID model
- Hungarian/global assignment
- anti-ID-swap logic
- ACTIVE/DORMANT identity memory
- SQLite identity persistence
- topology/travel-time logic
- global presence/handoff behavior

This task is an audit, not an architecture rewrite.

## Current Symptom
Evaluation currently reports approximately:

- Same-ID mean similarity: 0.8017
- Different-ID mean similarity: 0.7915
- Similarity gap: 0.0102
- ROC-AUC: ~53.5%
- EER: ~46.48%

These values are close to random discrimination and may indicate a bug in the embedding/evaluation pipeline.

## Audit Requirements
Trace the full OSNet embedding path and document exactly:

1. Where person crops are created from detector/tracker bounding boxes.
2. Image color order at every step:
   - OpenCV BGR
   - RGB conversion
3. Resize shape and argument order.
   - Verify OSNet input is effectively H=256, W=128.
   - For cv2.resize, verify `(width, height)` is used correctly.
4. Pixel scaling.
   - Is input divided by 255.0?
5. Input normalization.
   - Verify mean/std values.
   - Verify they match the model/training implementation being used.
6. Tensor layout.
   - HWC -> CHW
   - batch dimension
7. Model inference mode.
   - model.eval()
   - torch.no_grad() or inference_mode
8. Actual model checkpoint loaded.
   - exact checkpoint path
   - model architecture name
   - device
   - embedding dimension
   - missing/unexpected keys
9. Feature extraction output.
   - exact tensor/array used as embedding
   - confirm classifier logits are not accidentally used
10. Existing embedding normalization.
    - determine whether L2 normalization already happens
    - determine whether it happens once, multiple times, or not at all
11. Similarity computation.
    - dot product, cosine similarity, scipy/sklearn/pytorch helper, etc.
    - identify whether that function already performs normalization
12. Evaluation pair construction.
    - verify same/different labels are based on ground-truth person IDs, not local tracker IDs
    - verify camera IDs are interpreted correctly
    - verify no query/gallery pairing bug
13. Verify an embedding from one image cannot accidentally be reused for another image/frame/person.
14. Verify crop coordinates are bounded and not swapped/misaligned.

## Required Diagnostics
Add or create a safe diagnostic script/test that can be run without touching live camera behavior.

It should print at minimum:

```text
OSNet architecture: ...
Checkpoint: ...
Device: ...
Embedding dim: ...
Existing normalization: yes/no
Similarity implementation: ...
Preprocess size: ...
Color conversion: ...
Mean: ...
Std: ...
```

For a few sample embeddings, also print:

```text
shape
min
max
mean
std
L2 norm
contains_nan
contains_inf
```

## Deliverables
1. Audit findings grouped as:
   - Correct
   - Suspicious
   - Confirmed bug
2. Exact files/functions involved.
3. Minimal recommended changes only.
4. Do not implement broad changes yet unless required for the diagnostic itself.
5. Add focused tests where practical.

## Validation
Run the relevant existing tests plus any new diagnostic/test.

At the end report:

- files inspected
- files modified
- confirmed preprocessing path
- whether embeddings are currently L2-normalized
- whether similarity is mathematically cosine similarity
- whether evaluation labels/pairs are correct
- any issue likely to explain AUC ~53.5%
- confirmation that the frozen camera/video/calibration subsystem was not modified

Do not proceed to fine-tuning or dataset expansion in this prompt.
