# Verified GCN design-stage results

Source snapshot: `26a40737a7c86e6a0d6d117e4ac5b2ee5c8fc6b21b96c81a3ac0c14dd3fbea56`.

These are 2018 validation results used for design selection. Final 2019–2020 labels have not been scored.

| Inputs | Update | Selected epoch | Mean validation accuracy | Gain over frozen (pp) | Recorded zero deltas | Training time (s) |
|---|---|---:|---:|---:|---|---:|
| A | head | 0 | 62.891% | +0.000 | True | 355.59 |
| A | lora | 0 | 62.891% | +0.000 | True | 351.62 |
| B | lora | 0 | 62.891% | +0.000 | True | 399.96 |
| C | lora | 0 | 62.891% | +0.000 | True | 418.22 |
| D | head | 0 | 62.891% | +0.000 | True | 413.30 |
| D | lora | 0 | 62.891% | +0.000 | True | 421.43 |

Total recorded generator fitting time: 2360.11 seconds. Job startup, data checks, diagnostics, baselines, replay and packaging are additional. No online speedup follows from these fit times.

Validation selection: inputs **A**, update **head**, using the predeclared tie rules. A tie does not establish that one descriptor or update site is scientifically superior.

## Baseline comparisons

| Method | Cases | Mean validation accuracy | Mean gain (pp) | Harmful cases | Offline practice labels |
|---|---:|---:|---:|---:|---|
| constant_teacher | 42 | 62.652% | -0.238 | 83.3% | True |
| frozen | 42 | 62.891% | +0.000 | 0.0% | False |
| gtrans_feature_only | 42 | 61.153% | -1.737 | 100.0% | False |
| iterative_im_head | 42 | 60.674% | -2.217 | 100.0% | False |
| iterative_im_lora | 42 | 57.230% | -5.661 | 100.0% | False |
| practice_fixed | 42 | 62.953% | +0.062 | 9.5% | True |

Both practice-fixed fitting and constant-teacher construction use permitted offline practice labels; neither uses online target labels. The raw baseline label flag only marks practice-fixed, so this report corrects that incomplete annotation. GTrans is our declared feature-only variant, not a full published-method reproduction. Negative gains remain in the comparison.

## Conditioning sensitivity

| Inputs | Original accuracy | Cyclic-donor accuracy | Prediction disagreement |
|---|---:|---:|---:|
| A | 62.891% | 62.891% | 0.000% |
| B | 62.891% | 62.891% | 0.000% |
| C | 62.891% | 62.891% | 0.000% |
| D | 62.891% | 62.891% | 0.000% |

Changing descriptors for a zero-update checkpoint cannot establish useful graph sensitivity. This diagnostic does not retrain generators or change selection.

Independent saved-update replay verified 420 cases. Recorded argmax differences: 0. The replay uses actual saved factors/deltas rather than regenerating conditioning.

## Interpretation

The completed hypernetwork configurations did not demonstrate adaptation gains on this real-Arxiv validation protocol. Earlier modest synthetic-CSBM development gains therefore have not yet transferred to these tested configurations. This is not a conclusion about every generator family or architecture; the three-architecture, three-seed teacher-hypernetwork and diffusion comparison is separate. Shared graph cases and overlapping bank groups are not independent datasets.

Additional local artifact audit read all 252 selected-fit prediction files: every saved dense update was exactly zero, and every saved logit argmax matched its saved classes. No labels or graph inference were used. Detailed evidence: `../kaggle-gcn-study-20261108/download-version-1/LOCAL_ZERO_UPDATE_AUDIT.json`.

Training histories: [PNG](VALIDATION_HISTORY.png), [SVG](VALIDATION_HISTORY.svg), [raw epoch CSV](TRAINING_HISTORY.csv). These Arxiv generators were trained separately from the earlier synthetic CSBM generators.

## Measured worker costs

| Stage | Seconds |
|---|---:|
| dependencies | 5.86 |
| cuda-components | 14.92 |
| experiments-1-2 | 2523.65 |
| conditioning-control | 517.91 |
| gcn-baselines | 1153.77 |
| independent-replay | 40.88 |
| report | 0.08 |

Total worker elapsed: 4443.42 seconds. Generator fitting (2360.11 seconds) is already included in experiments-1-2; do not add it twice. Service publication and local downloading are outside this worker timer.

Training loss/update-norm diagnostics: [PNG](TRAINING_DIAGNOSTICS.png), [SVG](TRAINING_DIAGNOSTICS.svg). Curves use the30 saved epoch-average records. Detailed predictions/factors/deltas are saved for selected checkpoints; per-case predictions and model weights for every rejected training epoch were not retained. Those missing per-epoch details cannot be reconstructed exactly from the aggregate histories without additional computation, which is not planned.
