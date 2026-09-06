# Prompt 03 — Dump Re-ID Crops and Embeddings for Forensic Inspection

Continue after Prompt 01 and Prompt 02.

## Goal
Create a reproducible offline/debug mechanism to inspect exactly what image crop and embedding OSNet receives/produces for selected samples.

## CRITICAL FROZEN SUBSYSTEM — DO NOT MODIFY
Do not change live camera/video/calibration acquisition behavior. In particular, do not refactor capture workers, reconnect logic, preview/frame cache, video decoding, device access, calibration, or input handling.

Prefer a standalone diagnostic/evaluation script that consumes existing files/dataset/evaluation samples.

## Output Structure
Create a debug output directory such as:

```text
debug_reid/
  crops/
  embeddings/
  metadata/
```

For each sampled observation, save:

1. The exact person crop that is fed into preprocessing, before tensor normalization.
2. Optional preprocessed visualization only if useful and reversible.
3. Embedding as `.npy`.
4. Metadata as JSON or CSV.

Metadata should include when available:

```text
sample_id
source file/video/image
frame index or timestamp
camera id/name
ground-truth person id
local track id (if applicable)
global id (if applicable)
bbox x1,y1,x2,y2
crop width/height
embedding dimension
embedding L2 norm
embedding min/max/mean/std
checkpoint path/model name
```

Do not place credentials or RTSP URLs with secrets in debug output.

## Crop Validation
Add checks for:

- bbox inside frame bounds
- x2 > x1 and y2 > y1
- crop not empty
- minimum reasonable crop dimensions
- no accidental x/y or width/height swap

Log or count rejected/invalid crops rather than silently producing bad embeddings.

## Sampling
Support a bounded sample count so diagnostics do not generate thousands of files by default.

Example CLI behavior:

```bash
python ... --dump-reid-debug --max-samples 100
```

Exact CLI names may follow project conventions.

Try to include:

- same identity across different cameras
- different identities with similar clothing
- close/far views
- front/back/side poses
- samples currently responsible for false matches if the evaluation script can identify them

## Required Summary
At the end print:

```text
samples requested
valid crops dumped
invalid crops rejected
unique GT identities
unique cameras
embedding norm min/mean/max
NaN count
Inf count
```

## Tests
Add tests for:

- invalid bbox rejection/clamping behavior
- generated embedding file shape
- metadata consistency
- normalized embedding norm
- debug path does not alter production assignment behavior

## Validation
Run this only through an offline/evaluation path unless the project already has a safe diagnostic hook.

At the end report:

- how to run the dump
- output directory
- number of crops/embeddings produced
- any suspicious crops found
- any label/crop/embedding mismatches found
- confirmation no frozen camera/video/calibration subsystem behavior was changed
