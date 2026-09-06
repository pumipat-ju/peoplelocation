# Prompt 02 — Add/Standardize L2 Normalization for OSNet Embeddings

Continue from Prompt 01 audit findings.

## Goal
Make OSNet embeddings use one explicit, numerically safe L2-normalization convention before similarity/matching/evaluation.

## CRITICAL FROZEN SUBSYSTEM — DO NOT MODIFY
Do not modify camera/video/calibration/input behavior, including:

- camera discovery/device access
- Docker/WSL camera passthrough
- live capture/open/reconnect
- capture workers
- uploaded video decoding/playback
- live preview/frame cache
- calibration/homography flow
- preview endpoints
- input registration/removal

Preserve all existing tracker/global identity/topology architecture.

## Requirements
1. Locate the canonical feature-extraction point identified in Prompt 01.
2. Normalize the final feature vector using L2 normalization once at the canonical boundary.
3. Use numerically safe normalization.

Preferred conceptual behavior:

```python
embedding = embedding.astype(np.float32, copy=False)
norm = np.linalg.norm(embedding)
embedding = embedding / max(norm, 1e-12)
```

or the equivalent PyTorch operation:

```python
embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1, eps=1e-12)
```

4. Avoid accidental double normalization spread across unrelated modules.
5. Make stored/gallery/dormant/prototype embeddings follow the same convention where they originate from this extraction path.
6. Do not redesign matching thresholds in this prompt.
7. Do not tune threshold values just because similarity values change.
8. If cosine similarity library functions already normalize internally, still establish a single explicit embedding contract if safe, and document whether numerical outputs should remain nearly identical.

## Similarity Contract
After this change, for two valid embeddings `a` and `b`:

```python
np.dot(a, b)
```

should be equivalent to cosine similarity up to floating-point tolerance because:

```text
||a||2 ~= 1
||b||2 ~= 1
```

## Tests
Add focused tests covering:

1. Extracted embedding L2 norm is approximately 1.0.
2. No NaN/Inf after normalization.
3. Zero/near-zero vector handling is safe.
4. Dot product of normalized embeddings matches cosine similarity within tolerance.
5. Existing model feature dimension is unchanged.
6. Batch behavior remains correct if batch extraction exists.

Suggested tolerance:

```text
abs(norm - 1.0) < 1e-5
```

for normal non-zero embeddings.

## Diagnostics
Add a temporary or reusable debug option that can print:

```text
raw_norm=<...>
normalized_norm=<...>
```

for sampled embeddings without flooding normal runtime logs.

## Validation
Run all relevant Re-ID tests.

At the end report:

- exact normalization location
- files changed
- whether any previous normalization was removed/consolidated
- before/after sample L2 norms
- whether dot product now has cosine meaning
- whether any thresholds were changed (expected: no)
- confirmation frozen camera/video/calibration code was not modified
