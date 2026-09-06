# Prompt 05 — Re-run ROC-AUC / EER After OSNet Pipeline Fixes

Continue after Prompts 01–04.

## Goal
Re-run the Re-ID verification evaluation using the corrected/canonical embedding pipeline and compare results with the previous baseline.

## CRITICAL FROZEN SUBSYSTEM — DO NOT MODIFY
Do not modify any camera/video/calibration/input subsystem behavior.

Do not change matching/topology/presence architecture in this prompt.

## Baseline to Compare Against
Previous approximate metrics:

```text
Same-ID mean     = 0.8017
Different-ID mean = 0.7915
Gap              = 0.0102
ROC-AUC          = 0.535
EER              = 0.4648
```

## Evaluation Requirements
Use the exact canonical feature extraction and normalization path established by previous prompts.

Verify before computing metrics:

1. embeddings are finite
2. embedding norms are approximately 1
3. same/different labels use GT identity
4. self-pairs are excluded
5. duplicate pair counting is understood and documented
6. if evaluation is meant to be cross-camera, enforce/report cross-camera rules explicitly
7. number of positive and negative pairs is reported

## Metrics
Compute at minimum:

```text
Same-ID count
Different-ID count
Same-ID mean/std/min/max
Different-ID mean/std/min/max
Similarity gap
ROC-AUC
EER
EER threshold
Best threshold by Youden J or another clearly documented criterion
TPR/FPR at selected threshold
```

If the project already computes Rank-1/mAP for retrieval evaluation, keep that evaluation separate and do not confuse it with verification AUC/EER.

## Comparison
Print a before/after table:

```text
Metric                 Before        After        Delta
Same-ID mean           0.8017        ...          ...
Different-ID mean      0.7915        ...          ...
Gap                    0.0102        ...          ...
ROC-AUC                0.5350        ...          ...
EER                    0.4648        ...          ...
```

## Interpretation Rules
Do not automatically fine-tune the model.

Interpret results roughly as follows:

### Case A — Large jump after normalization/pipeline fix
Example:

```text
AUC: ~0.53 -> 0.70–0.85+
```

Conclusion: previous metric/preprocessing/embedding pipeline was materially wrong.

### Case B — Metrics barely change
Conclusion: investigate crop quality, preprocessing/checkpoint correctness, label generation, dataset domain, or model suitability.

### Case C — Same-ID and Different-ID remain heavily overlapped
Do not hide this with threshold tuning. Report that representation quality is insufficient.

## Target Checkpoints
Treat these as diagnostic goals, not guaranteed pass/fail requirements:

```text
Similarity gap > 0.15 preferred
ROC-AUC > 0.80 preferred
EER < 0.20 preferred
```

The key requirement is clear separation between same-ID and different-ID distributions.

## Artifact/Output Files
Save machine-readable results, for example:

```text
reid_eval_results.json
reid_pair_scores.csv
```

Include model/checkpoint/preprocessing configuration in the JSON so the experiment is reproducible.

## Tests
Add/update focused tests for:

- AUC computation
- EER computation and threshold extraction
- GT pair labels
- cross-camera filtering if used
- normalized embeddings

## Final Report
At the end report:

1. Exact command used.
2. Dataset/evaluation split used.
3. Model/checkpoint/device.
4. Preprocessing summary.
5. Before/after metrics.
6. Whether L2 normalization materially changed results.
7. Remaining suspected root cause ranked by likelihood.
8. Whether adding/fine-tuning dataset is justified now.
9. Confirmation that frozen camera/video/calibration subsystem was not modified.

Do not fine-tune OSNet in this prompt.
