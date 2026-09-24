# Phase C — offline gallery-scoring ablation

## Decision

**Keep production `bottom-3` scoring for now. Do not apply a scoring-policy change.**
The fixed-protocol ablation does not support the hypothesis that bottom-3 is
the cause of fragmentation: it produced the same three episode-level splits
as top-3, mean-all, and median. Top-3 was slightly worse on Rank-1, mAP, and
handoff recovery without reducing false merges. Top-k mean plus a margin guard
reduced false merges but more than doubled splits and reduced handoff recovery.
This is exploratory evidence, not a full Global ID or camera replay.

## Production behavior checked

`backend/reid/gallery.py::gallery_similarity` normalizes and diversity-deduplicates
gallery vectors, sorts cosine scores ascending, then takes the **median of the
lowest three** as support. It blends that support with a robust prototype at
production weights 0.25 / 0.75 when prototype consensus passes 0.70. The
ascending order is explicit and the conservative effect is real; it is not a
sort-direction accident. The repository does not document the original design
intent, so calling it an intentional *policy* is an inference, not a verified
historical fact. The Phase C baseline was checked against the production
helper to numerical tolerance in tests.

## Fixed protocol

- Source: `reid_dataset/master_manifest_v1.jsonl`, SHA-256
  `00ad79659522c45e7709a13d14989f56cd0e6022609603118b71c45d9e8cb634`.
  Seven namespaced identities across legacy and m sequences. Five have both
  cameras; two provide same-camera re-entry proxies.
- Checkpoint: `weights/osnet_x1_0_peoplelocation_balanced_v2.pth`, SHA-256
  `cde5aff18fb3069a8abe8f383f3fc25e9ae9032192a164c9b7c3401f2fb2e043`.
  CUDA was available. OSNet inference was performed once and reused for all
  variants. GT person boxes were cropped and passed through the existing
  validation resize and ImageNet normalization.
- For each identity, 12 evenly spaced gallery crops from the first half of
  its primary camera (or its first 12 samples when the half was shorter) were
  fixed. Eight deterministic query crops came from
  the other camera where available, otherwise the later half of the same
  camera. No gallery/query sample ID overlapped. All variants used the same
  84 gallery and 56 query images, gallery aggregate, diversity threshold
  0.985, prototype parameters, and candidate ordering. Deduplicated gallery
  candidate counts ranged from 9 to 13 (the aggregate is an additional
  candidate).
- `top-3` changes only the subset to the highest three and retains the median
  support statistic. `mean-all` and `median` use all candidate scores.
  `top-k-mean-margin` uses the mean of the highest three and additionally
  rejects an episode when the top-two score margin is below the existing
  cross-camera margin 0.08. No production threshold was changed.
- Retrieval uses seven ranked identity candidates per query. mAP is mean
  reciprocal rank because there is one relevant identity per query.
  ROC-AUC/EER use 56 positive and 336 negative query/identity scores.
- Association numbers are an **appearance-only episode proxy**: two episodes
  per identity (14 total), a fixed 0.55 cross-camera appearance gate, and a
  distinct new GID on rejection. False splits, false merges, and ID switches
  follow `backend/evaluation.py` event definitions. Ten episodes represent
  cross-camera handoffs. They are not full online MOT/Global ID measurements.

## Results

| Variant | Splits | False merges | ID switches | Handoff recovered | Rank-1 | Rank-5 | mAP | ROC-AUC | EER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bottom-3 | 3 | 3 | 4 | 5/10 | 0.6607 | 1.0000 | 0.8095 | 0.9157 | 0.1607 |
| top-3 | 3 | 3 | 3 | 4/10 | 0.6071 | 1.0000 | 0.7827 | 0.9154 | 0.1607 |
| mean-all | 3 | 3 | 3 | 5/10 | 0.6429 | 1.0000 | 0.8006 | 0.9154 | 0.1607 |
| median | 3 | 3 | 3 | 5/10 | 0.6429 | 1.0000 | 0.8006 | 0.9162 | 0.1607 |
| top-k mean + margin guard | 7 | 1 | 7 | 2/10 | 0.6071 | 1.0000 | 0.7827 | 0.9152 | 0.1607 |

The raw reproducible output is `phase_c_gallery_ablation.json` in this
directory. Rank-5 is weakly informative with only seven identities. EER is
identical across variants at this sample size, and small AUC differences
should not be treated as meaningful improvements.

## Safety and fragmentation trade-off

Bottom-3 can lower a same-person score if one or more admitted gallery views
are poor; the regression test demonstrates that support-score mechanism.
However, replacing it with top-3 did **not** reduce fragmentation here and
did not improve false-merge safety. Mean-all and median also did not reduce
splits. The guarded top-k variant improved false-merge safety (3 to 1) but
substantially hurt continuity (3 to 7 splits; 5/10 to 2/10 handoffs). For this
small dataset, retaining bottom-3 is the safer evidence-based recommendation.
This does not establish that its existing three false merges are acceptable.

The real two-person forensic traces cannot be replayed with alternate gallery
scoring: their submissions record boxes, IDs, and summary scores but **not**
per-observation embedding vectors. Re-extracting embeddings from video would
be a different run, not an exact replay of the reported fragmentation.
The new synthetic tests cover score order, a poor gallery sample, multi-view
same-person matching, similar-looking negatives, margin safety, and
determinism, but they do not claim to reproduce that incident.

The checkpoint was trained using some identities in this master manifest.
The sample has only seven identities and lacks production quality-approved
gallery admission, topology, overlap, motion, and Hungarian assignment. A
future production-policy decision needs a held-out, labeled, vector-bearing
multi-camera replay with those gates intact and enough lookalike negatives.
