# Prompt V5: Balanced Fine-tuned OSNet + Improved Crop and Full Version Comparison

```text
Proceed with the next offline Re-ID experiment:

V5 = Domain-balanced fine-tuned OSNet x1.0 + Improved Crop

IMPORTANT:
This is an offline evaluation experiment only.

Do NOT:
- retrain or fine-tune
- modify the model checkpoint
- tune anything using held-out test data
- modify production runtime
- modify camera/video input
- modify tracker
- modify calibration
- modify frontend
- modify any frozen subsystem

==================================================
MODEL
==================================================

Use the best checkpoint from Prompt 06:

backend/reid_experiments/finetune_osnet_v2_balanced/best_checkpoint.pth

==================================================
CROP POLICY
==================================================

Use the EXACT SAME Improved Crop policy used by the existing
Pretrained + Improved Crop experiment.

Improved Crop definition:

- Raw GT bounding box
- no extra margin
- no aspect-ratio padding unless the existing V2 implementation
  explicitly used it
- preserve the exact existing V2 crop implementation
- RGB
- resize 256x128
- ImageNet normalization
- 512-D L2-normalized embedding

Do not implement a new interpretation of "Improved Crop".
Reuse the existing V2 crop/evaluation code wherever possible.

==================================================
VALIDATION THRESHOLD
==================================================

Before held-out evaluation:

1. Evaluate Balanced v2 + Improved Crop on the existing
   PeopleLocation validation split.

2. Select the verification threshold using validation data ONLY.

3. Use the same threshold-selection method used by previous
   experiments:
   cross-camera validation EER operating point.

4. Record:
   - selected threshold
   - validation ROC-AUC
   - validation EER
   - same-ID mean
   - different-ID mean
   - similarity gap

5. Do NOT use held-out test data to select or modify the threshold.

==================================================
HELD-OUT TEST
==================================================

Evaluate on the exact same PeopleLocation held-out test used by
the previous experiments.

Use the same:
- identities
- queries
- gallery protocol
- metric implementation
- pair generation logic
- cross-camera filtering
- ranking implementation

Do not modify the held-out dataset or evaluation protocol.

==================================================
METRICS
==================================================

Report ALL available metrics:

Retrieval:
- Rank-1
- Rank-5
- mAP
- valid queries
- skipped queries

Verification:
- ROC-AUC
- EER
- Accuracy
- Precision
- Recall
- F1

Similarity:
- Same-ID mean similarity
- Different-ID mean similarity
- Similarity gap

Errors:
- False positives
- False negatives
- True positives
- True negatives

Threshold:
- validation-selected threshold
- threshold selection method

Also record pair count if applicable.

==================================================
OUTPUT DIRECTORY
==================================================

Save V5 outputs under:

backend/reid_experiments/finetune_osnet_v2_balanced_improved_crop/

Save at least:

- metrics.json
- config.json
- pair_scores.csv
- validation_metrics.json
- evaluation_report.md

==================================================
FULL VERSION COMPARISON
==================================================

After V5 evaluation completes, create a consolidated comparison
covering ALL existing model/crop versions.

Canonical comparison:

V1:
Pretrained OSNet x1.0 + Original Crop

V2:
Pretrained OSNet x1.0 + Improved Crop

V3:
Fine-tuned v1 + Original Crop

V4:
Domain-balanced Fine-tuned v2 + Original Crop

V5:
Domain-balanced Fine-tuned v2 + Improved Crop

IMPORTANT:
Load the existing saved metrics/artifacts for V1-V4.
Do NOT recompute or silently replace historical results unless
required because protocols are incompatible.

If an existing result uses a different dataset/protocol,
explicitly mark it as NOT DIRECTLY COMPARABLE instead of mixing
numbers.

For directly comparable experiments, verify that they use:
- same held-out split
- same identity set
- same evaluation protocol
- same metric implementation

==================================================
COMPARISON TABLE
==================================================

Create one complete table:

Metric | V1 | V2 | V3 | V4 | V5 | Best Version

Include ALL of:

- Rank-1
- Rank-5
- mAP
- ROC-AUC
- EER
- Accuracy
- Precision
- Recall
- F1
- Same-ID mean similarity
- Different-ID mean similarity
- Similarity gap
- False positives
- False negatives
- True positives
- True negatives
- Valid queries
- Skipped queries
- Validation-selected threshold
- Pair count

For metrics where lower is better:
- EER
- Different-ID similarity
- False positives
- False negatives

mark the smallest value as best.

For other performance metrics, mark the largest value as best
where appropriate.

Do NOT treat threshold itself as "higher is better".

==================================================
DELTA TABLES
==================================================

Also calculate:

1. Delta of every version versus V1 Pretrained Original Crop

2. Delta of V5 versus V4
   This isolates the effect of Improved Crop on Balanced v2.

3. Delta of V5 versus V2
   This compares Balanced fine-tuning against Pretrained
   while both use Improved Crop.

For each metric show:

Metric | Reference | Candidate | Absolute Delta | Improved/Degraded

==================================================
FILES TO CREATE
==================================================

Create a central comparison directory:

backend/reid_experiments/version_comparison/

Create:

1.
ALL_VERSIONS_COMPARISON.csv

2.
ALL_VERSIONS_COMPARISON.json

3.
ALL_VERSIONS_COMPARISON.md

4.
ALL_VERSIONS_DELTAS.csv

5.
ALL_VERSIONS_DELTAS.json

6.
ALL_VERSIONS_DELTAS.md

7.
FINAL_EXPERIMENT_SUMMARY.md

The CSV must contain raw numeric values suitable for plotting or
importing into Excel.

The JSON must contain structured data with:
- version ID
- model
- checkpoint
- crop policy
- threshold
- metrics
- dataset/split
- hashes if available

The Markdown report must include:
- full comparison table
- best version for each metric
- V5 vs V4 analysis
- V5 vs V2 analysis
- retrieval analysis
- verification analysis
- similarity-separation analysis
- limitations

==================================================
IMPORTANT INTERPRETATION
==================================================

Do not decide the final winner using only one metric.

Explicitly distinguish:

Retrieval performance:
- Rank-1
- Rank-5
- mAP

Verification performance:
- ROC-AUC
- EER
- Accuracy
- Precision
- Recall
- F1

Embedding separation:
- same-ID similarity
- different-ID similarity
- similarity gap

At the end answer:

1. Does Improved Crop improve Balanced v2?
2. Does V5 outperform Pretrained + Improved Crop?
3. Does V5 outperform Pretrained + Original Crop?
4. Which version is best for retrieval?
5. Which version is best for verification?
6. Which version has the strongest embedding separation?
7. Is any fine-tuned model currently justified for production?

Do NOT integrate any checkpoint into production.

==================================================
FINAL CONSOLE SUMMARY
==================================================

Print:

- V5 validation threshold
- V5 held-out metrics
- complete V1-V5 comparison
- V5 vs V4 deltas
- V5 vs V2 deltas
- best retrieval version
- best verification version
- best similarity-separation version
- paths to all comparison files
```
