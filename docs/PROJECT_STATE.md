# Project state

Last updated: 2026-10-25. Stage: CSBM loader, source GCN, and fixed-holdout source-label subset preparation implemented and CPU-verified.

## Current scope and decision

The owner authorized implementation and routine methodological/dependency choices on 2026-10-25 to prepare for the upcoming review. CSBM remains the only dataset. Common graph/label separation, independent development graphs, and stratified 60/20/20 source splits were already approved.

The owner subsequently requested the small source-GCN training and evaluation pipeline using this loader. That milestone is now implemented without new dependencies. Source-validation cross-entropy selects the checkpoint; source-test and target labels are scored only after fitting. See [Decision D2](DECISIONS.md#implemented-decision-d2--small-source-gcn-baseline-2026-10-25).

The owner then requested the remaining work one step at a time. First implemented the five source-label subsets needed before a bank runner. Label fraction is an explicit required argument; 50% of the native training pool was used for development execution only. The final scientific budget and model-bank grouping remain to be settled. See [Decision D3](DECISIONS.md#implemented-decision-d3--fixed-holdout-source-label-subsets-2026-10-25).

Under that delegated authority, selected the released Matcha-code CSBM distribution, exposed through explicit p_in/p_out and a named degree_scale conversion. Exact finite-size expected degree is reported; no silent factor-of-two correction. Fixed-direction Gaussian feature scaling follows the released code. This is not full paper-result reproduction. Details and alternatives are in [DECISIONS.md](DECISIONS.md#implemented-decision-d1--initial-csbm-protocol-2026-10-25).

## Implemented

| Component | Location | Behavior |
| --- | --- | --- |
| Package and dependency constraints | pyproject.toml, .gitignore | Minimal editable package, dev pytest dependency, runtime artifacts excluded |
| Common data contract | src/amortized_gnn_tta/data/api.py | CPU graphs, separate labels/splits, input allowlist/clones, role/schema/edge/split validation |
| CSBM generation and loading | src/amortized_gnn_tta/data/csbm.py | Balanced binary labels, explicit probabilities, Gaussian features, local tagged RNGs, stratified splits, complete provenance |
| Review example | examples/inspect_data.py | Independent source/target generation, offline diagnostics/JSON, real random GCN forward/backward |
| Automated checks | tests/test_data.py | Sampling, boundaries, reproducibility, validation, GCN/GAT/GraphSAGE compatibility |
| Source GCN | src/amortized_gnn_tta/models/gcn.py | Two uncached GCNConv layers, ReLU, hidden dropout, raw logits |
| Shared training | src/amortized_gnn_tta/training.py | Model factory, source-only supervision view, Adam, validation-loss selection/early stopping, restored best weights |
| Shared evaluation | src/amortized_gnn_tta/evaluation.py | Label-free inference; separate explicit-label cross-entropy/accuracy scoring |
| Training example and checks | examples/train_source.py, tests/test_training.py | Reproducible run records, CPU checkpoint reload, learning/selection/leakage/device tests |
| Source-label subset preparation | src/amortized_gnn_tta/source_subsets.py | Stratified selections inside native training pool; saved membership/identity, fixed validation/test, compatible training views |
| Subset example and checks | examples/prepare_source_subsets.py, tests/test_source_subsets.py | Five distinct subsets; persistence, leakage/invariance, budget, and real trainer/checkpoint integration |

Held-out graphs have eval-only indices; training_labels rejects validation/target roles. The loader preserves full graph context and does not add self-loops or model-specific normalization. A persistent cache and paired-shift generator are deliberately absent.

## Reference investigation

Read the local Matcha PDF 2410.06976v2.pdf: main analysis/method/experiments, theoretical and convergence appendices, additional experiments, reproducibility, and limitations. Compared the released CSBM, generation script, dataset construction, and ERM source-training code. Examined assumptions; did not independently verify every proof or reproduce the paper's reported accuracies.

The paper's simplified aggregation/primary GPRGNN architecture differs from a vanilla PyG GCN. Its theoretical Gaussian covariance convention also differs from the released feature scaling. The implemented recipe is tied to the released code explicitly. Unlicensed upstream code was not copied; references remain separate and unchanged.

Earlier dataset/reference investigation and handbook work are preserved in the decision log and handbook. The prior real-dataset-first interpretation was corrected; it is not an approved historical choice.

## Verification actually executed

- OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q: **70 passed** after subset preparation (26 new cases; 44 existing checks retained); upstream PyTorch/PyG deprecation warnings reported.
- New checks include finite GCN gradients/empty-edge execution, learning on a deliberately easy fixture, exact seeded CPU repeatability, best-checkpoint restoration after validation worsens, inference/checkpoint equality, and unchanged training after held-out labels are changed. A spy also verifies source-view extraction reads only permitted source validation labels.
- Python compileall on src/tests/examples: passed.
- Small example: 300 nodes × 32 features per graph; source splits 180/60/60; target eval 300; forward/backward and label-free target input passed.
- Paper-scale example: 5,000 nodes × 2,000 features per graph; source splits 3,000/1,000/1,000; target eval 5,000. Source stored edge entries 12,338, target 12,884; observed mean degrees 2.4676/2.5768. Random GCN outputs [5000,2] with finite source gradients. No training/adaptation accuracy claimed.
- Paper-scale generation plus bundle validation took approximately 1.50 seconds in this local CPU run, excluding Python import/startup and the GCN operation. This is a smoke timing, not a benchmark comparison.
- Reports were written outside Git to /tmp/afml_csbm_smoke.json and /tmp/afml_csbm_paper_scale.json; regenerate via the example commands in the README.
- Direct project dependency requirement checks passed. Whole-environment pip check failed on unrelated inherited packages (e.g. AutoGluon/Pandas/Transformers/TensorBoard constraints), not on the selected project dependencies.
- Git diff/whitespace, documentation consistency, and ignored-artifact checks passed. The owner authorized a checkpoint commit of the completed foundation before the next step; runtime outputs/checkpoints remain excluded and no push is authorized.

Tested: Python 3.13.13, PyTorch 2.13.0+cpu, PyG 2.8.1, NumPy 2.3.5, pytest 9.1.1 on local Linux CPU. .venv inherits system site packages to reuse CPU PyTorch; new dependencies were installed into .venv only. Sandbox network failure was resolved by a permitted install retry. No global environment modifications.

### Source-only end-to-end runs

Both runs used model seed 31, hidden width 32, dropout 0.5, Adam lr 0.01/weight decay 0.0005, one CPU thread, and the existing independent CSBM source/target recipe. Source: degree_scale=5, h=0.8, means −0.03/+0.03, edge/feature/split seeds 11/12/13. Target: degree_scale=5, h=0.2, means −0.01/+0.05, seeds 21/22/23. Noise scale 1; loader feature scaling unchanged.

| Run | Nodes × features per graph | Epoch cap / patience | Epochs run / selected | Source-test accuracy | Target accuracy |
| --- | --- | --- | --- | --- | --- |
| Small review run | 1,000 × 64 | 100 / 20 | 46 / 26 | 71.0% | 53.4% |
| Paper-sized execution check | 5,000 × 2,000 | 20 / 10 | 20 / 20 | 91.0% | 57.1% |

Commands executed: `.venv/bin/python examples/train_source.py --epochs 100 --patience 20 --device cpu --output-dir runs/source-gcn-review` and `.venv/bin/python examples/train_source.py --paper-scale --epochs 20 --patience 10 --device cpu --output-dir runs/source-gcn-paper-smoke`.

Each run saved ignored `best_model.pt` and `metrics.json` under its output directory, then verified predictions after CPU reload. Checkpoints contain CPU weights, source provenance/supervision indices, versions, configs, history, and selection metadata; optimizer/RNG states for exact training resumption are not included. Output directories/checkpoint files refuse overwriting. The small run predates a presentation-only accuracy change from float32 mean to integer correct-count division; its displayed percentages are unchanged.

These are single seeded development runs with different sizes and epoch caps, not a controlled size comparison, a Matcha reproduction, or evidence for adaptation. The paper-sized run reached its epoch cap; no convergence claim is made. No target labels or target metrics enter training/checkpoint selection. Later protocol tuning after viewing these target results must treat these graphs as development data and use fresh held-out targets for final claims.

### Source-subset preparation executed

- Ran `examples/prepare_source_subsets.py --label-fraction 0.5` for both 1,000×64 and 5,000×2,000 source graphs, with seeds 101–105 and explicit outputs `runs/source-subsets-small.json` / `runs/source-subsets-paper.json`. Both created five distinct subsets and verified exact JSON reloads.
- Small case: native 600/200/200 labels unchanged; each subset has 300 train labels (150/class). Paper-sized case: native 3,000/1,000/1,000 unchanged; each subset has 1,500 train labels (750/class). No target graph was needed.
- Tests also executed source fitting from one subset and verified checkpoint membership/provenance after CPU reload. They reject held-out IDs, duplicate selections, changed graph/pool identity, class-budget violations, invalid seeds, and impossible distinct-subset requests. Changing validation/source-test/target labels does not alter subset generation; reordering the same labeled pool does not alter it either.
- An initial round-trip test failed because tuple-valued provenance became lists in JSON. Normalizing provenance to JSON-compatible values at manifest creation resolved the cause; the complete suite then passed. Existing loader and training behavior were preserved.
- Runtime manifests remain ignored. No 30-model bank, generator partitions, or diversity result is claimed.

## Limits and next work

- A trained source baseline exists; no weight bank, adaptation, or generator exists. Repeated-seed estimates and final research evaluation remain outstanding.
- CPU verified only. Clean-isolated installation, macOS, Colab, CUDA, and MPS remain unverified.
- Generation pair-scanning is O(N²), although no dense adjacency is allocated. Do not scale blindly to huge N.
- Independent-draw development examples are not paired attribute-only interventions. Later scientific shift families/severities and holdout lineage require an explicit protocol.
- Source full-graph visibility and fixed-holdout label subsets are implemented. Final label budget, generator meta-training allocation, and model-bank grouping remain scientific decisions.
- Next implementation step: a sequential six-model-seed runner for each of the five subsets, with a bank manifest/checkpoints. Then explicitly choose 18/6/6 grouping and inspect prediction/representation/weight diversity before generator training. These later steps are not yet implemented.
