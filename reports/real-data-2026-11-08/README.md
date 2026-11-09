# Verified real-data evidence snapshot

This snapshot records completed formal source banks and the completed GCN design stage. Experiments 3–6 are not claimed complete here. All figures and CSVs are derived from saved artifacts; the Git delivery launched no new numerical experiments.

## Results

- [Source-bank report](FORMAL_BANK_REPORT.md): all 270 unique source checkpoints verified, with per-model timings/hashes in [CSV](FORMAL_SOURCE_FITS.csv) and per-protocol diversity/verification JSON in `source-banks/`.
- [Reviewed GCN design report](design/DESIGN_RESULTS.md): four A–D LoRA hypernetworks and A/D head follow-ups; six configurations, 42 validation cases each, five fixed training epochs each. Selected epoch zero for all six. Frozen/selected mean validation accuracy 62.89073299%.
- [Validation history](design/VALIDATION_HISTORY.png), [training loss/update norms](design/TRAINING_DIAGNOSTICS.png), with SVG and raw CSV counterparts.
- [Per-case selected-checkpoint records](design/GENERATOR_VALIDATION_CASES.csv), [baseline records](design/BASELINES-gcn.json), [conditioning control](design/CONDITIONING_CONTROL.json).
- [Numerical replay audit](design/INDEPENDENT_REPLAY.json): 420 cases, zero class mismatches; [local saved-delta audit](design/LOCAL_ZERO_UPDATE_AUDIT.json): all 252 selected-fit dense updates exactly zero.
- `diagnosis/`: read-only evidence for descriptor-scale dominance and source-conditioning ambiguity. These are suspected causes, not confirmed fixes. No changes to the campaign were authorized from these observations.

The generators ran all five epochs; epoch-zero checkpoint selection is not early termination. A/head won declared ties because no trained candidate improved validation; this does not establish A/head as the best scientific recipe. Experiment 3 is conditional on that recipe. The five-epoch budget does not establish convergence. Source GNNs used their separate 200-epoch/30-patience budget.

## Provenance and scope

The immutable fitting ZIP SHA-256 is `26a40737a7c86e6a0d6d117e4ac5b2ee5c8fc6b21b96c81a3ac0c14dd3fbea56`. Source files remain identical; later documentation/operational exports have separate provenance. Training's original dirty/source/software metadata is preserved rather than relabeled by this commit.

Runtime: Tesla T4, Torch2.11.0+cu130, PyG2.8.1; exact records are in the saved JSON. Six generator fits took2360.11s, whole worker4443.42s. Publication/downloading are additional. Constant-teacher and practice-fixed both use permitted offline practice labels; the reviewed report corrects the original baseline flag that tagged only practice-fixed. All methods use no online target labels.

The full checkpoint/prediction tensors, native split masks, dataset archives and large bundles remain in the original workspace and private Kaggle versioned outputs. Paths inside raw JSON/Markdown retain their original execution layout; they are provenance, not guaranteed repository-relative download links. Every included file is listed by SHA-256 in `EVIDENCE_MANIFEST.json`. Its hashes describe this tracked snapshot, not a reassertion that the full excluded bundle can be verified without recovery.

## Private artifact recovery

Relevant version-1 Kaggle jobs, owner `deeptanshukumar`: `afml-formal-validation-20261108`, `afml-formal-resume-20261108`, `afml-gcn-bank-20261108`, `afml-controlled-bank-20261108`, `afml-temporal-bank-20261108`, `afml-country-bank-20261108`, and completed design `afml-gcn-study-20261108`. Current broader comparison: `afml-controlled-extend-20261108`. Private access requires the owner's existing authorized account; credentials are never committed.

The current owner instruction is to complete the existing six-experiment main campaign without extra tuning rounds. E4/E5 sealed final evaluations and E6 diagnostics remain separate stages. See [project state](../../docs/PROJECT_STATE.md) and [operational source/recovery notes](../../tools/kaggle/README.md).
