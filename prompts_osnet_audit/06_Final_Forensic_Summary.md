# Prompt 06 — Final OSNet Forensic Summary and Go/No-Go for Fine-Tuning

Use the outputs from Prompts 01–05.

## Goal
Produce a concise engineering conclusion about whether the current weak Re-ID separation is caused primarily by:

- missing/incorrect L2 normalization
- similarity implementation
- preprocessing
- crop quality
- checkpoint/model loading
- pair/label generation
- dataset/domain gap
- insufficient OSNet representation quality

## CRITICAL FROZEN SUBSYSTEM
Do not modify any code in this prompt unless a tiny test/documentation fix is necessary.

Absolutely do not modify camera/video acquisition, live preview, capture workers, reconnect logic, uploaded video decoding, frame cache, calibration, homography flow, device access, or input registration.

## Required Evidence Table
Create a table with columns:

```text
Area | Status | Evidence | Impact | Recommended action
```

Cover at least:

- BGR/RGB
- resize H/W
- pixel scaling
- mean/std normalization
- model.eval/inference mode
- checkpoint loaded
- embedding dimension
- L2 normalization
- similarity function
- crop validity
- GT labels
- pair generation
- similarity matrix
- AUC/EER before vs after

## Decision
Choose one:

### GO — Fine-tuning justified
Use only if the pipeline is verified correct but same/different distributions remain insufficiently separated.

### NO-GO — Fix pipeline/data issue first
Use if any material correctness issue remains.

## If GO
Do not fine-tune yet. Instead specify what the next fine-tuning dataset should contain:

- correct person labels
- cross-camera views
- multiple angles/distances/lighting
- clean person crops
- same-ID variation
- hard negatives with similar clothes/appearance

Also recommend what metrics should be captured before and after fine-tuning.

## If NO-GO
List the exact remaining blocker(s) in priority order and the smallest next fix/test.

## Final Output
Report:

- final root-cause ranking
- verified pipeline contract
- current metrics
- whether results improved from baseline
- GO/NO-GO decision for dataset expansion/fine-tuning
- next recommended prompt/task
- explicit statement that frozen camera/video/calibration subsystem remained untouched
