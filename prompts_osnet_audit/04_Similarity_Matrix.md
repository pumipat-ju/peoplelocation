# Prompt 04 — Build OSNet Similarity Matrix Diagnostic

Continue after Prompts 01–03.

## Goal
Create an offline similarity-matrix diagnostic using the exact production/evaluation OSNet embedding pipeline.

## CRITICAL FROZEN SUBSYSTEM — DO NOT MODIFY
Do not modify camera/video acquisition, preview, capture workers, reconnect logic, decoding, calibration, homography flow, or input handling.

## Requirements
Use normalized embeddings produced by the canonical Re-ID extraction path.

For N selected samples, construct:

```python
S = E @ E.T
```

where every row of `E` is L2-normalized.

The diagnostic must preserve metadata for each row/column:

```text
sample id
GT person id
camera
source/frame
```

## Selection Modes
Support at least one practical selection mode, preferably both:

1. Explicit dumped embeddings from Prompt 03.
2. Automatic representative selection from the evaluation dataset.

Try to include multiple samples per identity and cross-camera samples.

## Outputs
Generate:

1. Numeric similarity matrix as CSV.
2. Row/column metadata CSV/JSON.
3. Human-readable console summary.
4. Optional heatmap image if matplotlib is already an acceptable project dependency; otherwise CSV is sufficient.

Do not add a heavy new dependency solely for visualization.

## Analysis
Compute and report:

- diagonal similarity statistics
- same-ID off-diagonal similarity statistics
- different-ID similarity statistics
- same-ID mean/std/min/max
- different-ID mean/std/min/max
- similarity gap = same_mean - different_mean
- top-K highest different-ID similarities (hard negatives / likely false matches)
- bottom-K lowest same-ID similarities (hard positives / likely false rejects)

For each suspicious pair print:

```text
similarity
sample A metadata
sample B metadata
same/different GT label
```

## Correctness Checks
Assert or warn if:

- diagonal is not approximately 1.0
- embeddings are not approximately unit norm
- matrix is not symmetric within floating-point tolerance
- any NaN/Inf exists
- identical sample compared to itself is incorrectly counted as a same-ID pair for distribution statistics

## Ground-Truth Safety
Same/different labels must be based on ground-truth identity only.

Do NOT infer same person from:

- local tracker ID equality
- global ID equality produced by the system under evaluation
- file ordering

## Tests
Add focused tests with synthetic embeddings/labels that verify:

- same/different pair extraction
- diagonal exclusion
- symmetry
- hard positive/hard negative ranking
- similarity gap calculation

## Validation
Run against the current evaluation dataset/debug dump.

At the end report:

- matrix size
- identities/cameras represented
- same-ID stats
- different-ID stats
- similarity gap
- worst hard positives
- worst hard negatives
- whether the matrix supports a model problem, preprocessing problem, crop problem, or label problem
- confirmation frozen camera/video/calibration code was not modified
