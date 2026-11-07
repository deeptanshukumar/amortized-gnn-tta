# Project state

Last updated: 2026-11-07. Current milestone: **generator prototype repair and validation**, implemented on the `generator` branch after explicit owner approval. The actual system remains CSBM; generated LoRA, five-candidate diffusion selection and real-graph experiments remain future milestones. CUDA/MPS execution remains unverified.

## Integration in progress

Owner approved a new local integration branch and reviewed commits. `integration/generator-audit-fixes` starts at c137e28 and carries the previously verified repairs without changing the original repair checkout. Its combined repair tree passed 392 CPU tests. Repair checkpoint committed locally as `8c6203a`. Portable shared comparisons are implemented: 24 focused workflow tests and the full 408-test CPU suite passed. The bounded matched compact study is next. Push/merge remain unauthorized.

The c137e28 upload adds five experiment scripts and five result files, with no core changes. Independent arithmetic checks passed 2,205 relationships across 612 full-comparison records. The 44.49× GCN ratio is traceable, but generator timers omit adapted prediction and the compact diffusion timer also omits conditioning. The 64-feature experiment and our 2,000-feature validation study are different workloads.

## Completed repair milestone

Implemented: atomic complete inference checkpoints and reload; numerical/schema guards; restored 57 regression cases; deterministic selection/statistic tests; caller-state/autograd preservation; saved partition/cap/seed handling; validation-default CLI with explicit final-evaluation gate; stable imports; short-linear-schedule restriction; versioned A-D descriptor definitions; bounded-memory edge statistics; shared encoder computation, within-event frozen-feature reuse for generated head predictions and frozen-head teacher reuse; explicit static D fingerprint caching; independent whole-pipeline benchmark and a predeclared three-seed development review.

Verification completed on local CPU: **392 automated tests passed** (including the restored 57 cases, all-architecture/scope/input training matrix, forced selection paths, numerical/schema guards, complete non-default checkpoints, actual CLI workflows and frozen-feature parity). Compileall and complete diff/whitespace review passed. The small complete-review/reuse workflow and independent reconstruction smoke also passed.

The predeclared paper-sized review completed **72 configurations** (3 architectures × A-D × 2 generators × 3 seeds), with **864 saved cases**. Thirty completed compatible training checkpoints were reused after exact generated-delta/configuration/identity checks; all prediction and timing checks reran. Partial training was never resumed. Independent manual named-weight reconstruction verified every source/graph/checkpoint identity, requested/effective delta, final head tensor, prediction, accuracy, per-case aggregate and raw-timing median. Maximum logit replay error and accuracy mismatch count were both zero. The largest requested-versus-effective float32 delta difference was 2.9802322387695312e-08, not an artifact error.

**Research verdict:** the repaired prototype runs correctly on tested inputs, but this workload does not demonstrate an amortized online speedup. Every A-D grouped speed ratio is below 1 against the improved cached IM30 baseline. Mean shifted accuracy improves; unchanged-control harm remains. Hypernetwork CE and diffusion teacher-MSE have different objectives/budgets; no definitive generator ranking or optimal accuracy claim follows.

Fixed D reference (not an input selection for the later Arxiv study); ± is sample SD across three generator seeds:

| Architecture | Frozen | IM30 | Hypernetwork D | Diffusion D |
| --- | ---: | ---: | ---: | ---: |
| gcn | 77.255% | 78.167% | 78.061 ± 0.004% | 78.099 ± 0.011% |
| gat | 77.447% | 77.445% | 77.684 ± 0.015% | 77.679 ± 0.017% |
| graphsage | 79.316% | 80.251% | 81.033 ± 0.031% | 80.724 ± 0.029% |

| Architecture | Hyper D pipeline / speed ratio | Diffusion D pipeline / speed ratio |
| --- | ---: | ---: |
| gcn | 93.89ms / 0.577× | 95.37ms / 0.580× |
| gat | 97.61ms / 0.798× | 100.52ms / 0.799× |
| graphsage | 176.20ms / 0.790× | 178.94ms / 0.786× |

Speed ratio is mean IM latency divided by mean generated latency across the same four declared timed conditions and three seeds; below 1 means slower. Accuracy excludes the control and averages eleven shifted validation conditions. D control losses were hypernetwork/diffusion: GCN −0.307/−0.720pp, GAT −0.927/−1.113pp, GraphSAGE −3.120/−2.093pp. Harm counts across repeated shifted cases were 15/15, 15/15 and 12/13 out of 33 respectively. These are related conditions and repeated seeds, not 33 independent graph draws.

One held-out GNN checkpoint and one validation graph lineage per architecture are used. The three training checkpoints share GNN initialization seed 31 across source-label subsets; only generator training seeds vary. Hypernetwork uses 180 optimizer updates; diffusion 15 denoiser updates plus 1,080 teacher steps. Both use five epochs, which is not equal compute. Current updates are head coordinates, not generated LoRA; diffusion uses one candidate. No final model/environment group was scored.

Evidence (ignored runtime outputs): `runs/generator-repair-review-v5/VERIFIED_REVIEW.md`, `AGGREGATED_RESULTS.csv`, per-seed/per-case/raw timing files, complete checkpoints, actual weight deltas, code snapshots, `PLAN.json`, `HARDWARE.json` and `INDEPENDENT_REPLAY.json`. Reproduce using `examples/run_generator_review.py` and `examples/replay_generator_review.py`; individual checkpoints can be benchmarked with `examples/benchmark_adapters.py`. Hardware: Intel Core Ultra 5 125H, one numerical thread, Python 3.13.13 / Torch 2.13.0+cpu / PyG 2.8.1 / NumPy 2.3.5. CUDA/MPS are unexecuted. CPU sampled RSS includes resident review data/models and is not an exact incremental deployment peak.

A matched one-graph preprocessing probe (5,000 nodes,2,000 features) reduced raw-summary median from 253.17 ms to 80.12 ms (~3.16×), with numerical parity. This is a component measurement, not a pipeline speedup. The unchanged iterative core moved to a normal module with only relative-import changes; its benchmark avoids extra export/prediction bookkeeping. Additional architecture-specific aggregation caching is not implemented, so the baseline is a documented implementation, not an optimal lower bound. The contributor timing recipe is now published at c137e28; its component ratios are internally consistent, while full model/graph replay still requires the matching artifacts and seeds.

The original interrupted optimization runs are preserved with explicit interruption reasons. Existing audit snapshots and main checkout edits remain untouched. The repair was uncommitted before the current integration checkpoint; push/merge remain unauthorized.

The canonical timing protocol and descriptor decisions are in [D12](DECISIONS.md#implemented-decision-d12--generator-reliability-descriptors-and-runtime-2026-11-07). `configs/generator_csbm_review.json` fixes the development study before scores. Existing audit snapshots and all prior numerical evidence remain outside Git and unchanged. The owner now authorizes reviewed local integration commits; no push or merge is authorized.

## Historical implementation milestones

The original generator branch reported 165 tests and +2.34 accuracy gain. That exact run was under-specified and was not independently reproduced. Later audits executed 191/202 branch tests, separately recovered the deleted 57 cases, and replayed 216/288 saved validation updates. These counts describe different revisions and are not the current suite count. The new review is neither a reproduction of that undocumented accuracy claim nor of the historical component-timing experiment, now published at c137e28.

The owner subsequently requested the small source-GCN training and evaluation pipeline using this loader. That milestone is now implemented without new dependencies. Source-validation cross-entropy selects the checkpoint; source-test and target labels are scored only after fitting. See [Decision D2](DECISIONS.md#implemented-decision-d2--small-source-gcn-baseline-2026-10-25).

The owner then requested the remaining work one step at a time. First implemented the five source-label subsets needed before a bank runner. Label fraction is an explicit required argument; 50% of the native training pool was initially a development example. At that earlier milestone budget/grouping were pending; D5 subsequently fixed the Phase1 reference. Final-paper protocols remain separate. See [Decision D3](DECISIONS.md#implemented-decision-d3--fixed-holdout-source-label-subsets-2026-10-25).

The owner authorized committing that foundation and moving on. Local commit **30af4f9** contains it; nothing was pushed. The owner later explicitly approved the separate Phase 1 completion commit covering the bank runner, audit, configuration, tests and docs. See [Decision D4](DECISIONS.md#implemented-decision-d4--sequential-source-only-bank-runner-2026-10-25).

The owner then explicitly requested the remaining task list and completion of Phase 1. The list was provided before further implementation. [Decision D5](DECISIONS.md#implemented-decision-d5--gcncsbm-phase-1-closure-2026-10-25) fixes the phase's reference protocol and whole-subset partition, and records the completed audit. This completes roadmap steps 1–3 for our GCN/CSBM scope, not teammates' architectures, adaptation, or the final research paper.

The owner then explicitly authorized pushing commits 30af4f9/3c7976e and requested shared GAT and GraphSAGE implementations. That expansion is implemented under [Decision D6](DECISIONS.md#implemented-decision-d6--shared-gat-and-graphsage-models-2026-10-25). After verification, the owner explicitly authorized committing and pushing this extension as bc91093. Team experiment ownership is unchanged; later changes require separate Git authorization. The owner subsequently requested all controlled-shift families and environment partitions; [D7](DECISIONS.md#implemented-decision-d7--paired-controlled-shifts-and-environment-partitions-2026-10-25) records the implemented development protocol. After reviewing verification, the owner explicitly authorized committing and pushing this controlled-shift milestone. That approval does not extend to later changes.

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
| Additional source models | src/amortized_gnn_tta/models/{gat,graphsage}.py | GAT: total-width multihead attention/ELU; GraphSAGE: mean/root/ReLU; both expose logits and encode |
| Shared model specification | src/amortized_gnn_tta/models/{config,factory}.py | Common dimension/dropout validation, architecture-specific settings, explicit construction/restoration, legacy GCN compatibility |
| Shared training | src/amortized_gnn_tta/training.py | Model factory, source-only supervision view, Adam, validation-loss selection/early stopping, restored best weights |
| Shared evaluation | src/amortized_gnn_tta/evaluation.py | Label-free inference; separate explicit-label cross-entropy/accuracy scoring |
| Training example and checks | examples/train_source.py, tests/test_training.py | Reproducible run records, CPU checkpoint reload, learning/selection/leakage/device tests |
| Source-label subset preparation | src/amortized_gnn_tta/source_subsets.py | Stratified selections inside native training pool; saved membership/identity, fixed validation/test, compatible training views |
| Subset example and checks | examples/prepare_source_subsets.py, tests/test_source_subsets.py | Five distinct subsets; persistence, leakage/invariance, budget, and real trainer/checkpoint integration |
| Sequential bank runner | src/amortized_gnn_tta/weight_bank.py | Fresh seeded model/optimizer per subset/seed pair, copied supervision manifest, per-model checkpoints/metrics, hashes, CPU reload checks, progress/failure status |
| Bank example and checks | examples/train_weight_bank.py, tests/test_weight_bank.py | Five subsets × six model seeds, source-only reconstruction, Git state recording, job-order reproducibility, partial-failure/no-overwrite verification |
| Artifact audit and model partitions | src/amortized_gnn_tta/bank_artifacts.py | Verified complete-bank loading, restored validation metrics, exact source/config identity, deterministic whole-subset partitions and reload checks |
| Model diversity and explicit offline test evaluation | src/amortized_gnn_tta/bank_diagnostics.py | Label-free source-pool probes, weight/prediction/CKA comparisons, separate source-test scorer with grouped descriptive summaries |
| Phase reference and review command | configs/phase1_csbm_gcn.json, examples/finalize_phase1.py | Frozen phase config, new review directory, persisted partition before scoring, JSON/CSV/Markdown evidence and analysis-code snapshot |
| Phase verification | tests/test_bank_analysis.py | Corruption/mismatch/leak checks, independent CKA arithmetic, hidden-permutation and forward-compatibility tests, real finalization CLI |
| Architecture verification | tests/test_models.py | Manual aggregation/attention semantics, gradients/learning, permutation/label invariance, empty graphs, checkpoints and shared CLI audit |
| Controlled CSBM shifts | src/amortized_gnn_tta/data/shifts.py | Paired feature-center translation, homophily/degree changes and all declared combinations; exact preserved components |
| Environment plans and audit | src/amortized_gnn_tta/data/environments.py | Versioned recipes, whole-lineage role assignment, severity/family holdouts, persistence/validation, lazy loading and duplicate/invariant audit |
| Shift preparation and verification | configs/controlled_csbm_v1.json, examples/prepare_shift_environments.py, tests/test_shift_environments.py | Frozen-before-audit manifest, current code snapshot, CPU small/paper-sized full construction runs and boundary checks |
| Adaptation generator & descriptor | src/amortized_gnn_tta/adaptation/generator.py | Size-invariant GraphShiftDescriptor, HeadAdapterGenerator (MLP), zero-init identity adaptation, functional_call autograd updates |
| Meta-training loop | src/amortized_gnn_tta/adaptation/meta_trainer.py | Multi-pair training over (model, graph) combinations, configurable model/env caps, validation checkpoint selection |
| Human-readable adaptation evaluator | src/amortized_gnn_tta/adaptation/evaluator.py | Strict unlabeled evaluation, net accuracy gain tracking, human-readable Markdown/ASCII tables broken down by shift family and severity |
| Adaptation runner example | examples/train_adapter.py | End-to-end CLI with user-controlled limits, generator checkpoint saving, and Markdown report output |
| Iterative/generator tests | tests/test_adaptation.py, test_generator.py, test_diffusion.py, test_generator_reliability.py, test_generator_cli.py | Restored core adaptation coverage, exact-statistic/selection guards, all-architecture training matrix, real save/reload and benchmark workflows |

Held-out graphs have eval-only indices; training_labels rejects validation/target roles. The loader preserves full graph context and does not add self-loops or model-specific normalization. Paired shifts are a separate explicit API; the original independent loader and source banks are unchanged. No persistent graph cache exists.

## Reference investigation

Read the local Matcha PDF 2410.06976v2.pdf: main analysis/method/experiments, theoretical and convergence appendices, additional experiments, reproducibility, and limitations. Compared the released CSBM, generation script, dataset construction, and ERM source-training code. Examined assumptions; did not independently verify every proof or reproduce the paper's reported accuracies.

The paper's simplified aggregation/primary GPRGNN architecture differs from a vanilla PyG GCN. Its theoretical Gaussian covariance convention also differs from the released feature scaling. The implemented recipe is tied to the released code explicitly. Unlicensed upstream code was not copied; references remain separate and unchanged.

Earlier dataset/reference investigation and handbook work are preserved in the decision log and handbook. The prior real-dataset-first interpretation was corrected; it is not an approved historical choice.

## Verification actually executed

- OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q: **158 passed** after controlled shifts/environment partitions (33 new cases; 125 prior checks retained); upstream PyTorch/PyG deprecation warnings reported.
- New checks include finite GCN gradients/empty-edge execution, learning on a deliberately easy fixture, exact seeded CPU repeatability, best-checkpoint restoration after validation worsens, inference/checkpoint equality, and unchanged training after held-out labels are changed. A spy also verifies source-view extraction reads only permitted source validation labels.
- Python compileall on src/tests/examples: passed.
- Small example: 300 nodes × 32 features per graph; source splits 180/60/60; target eval 300; forward/backward and label-free target input passed.
- Paper-scale example: 5,000 nodes × 2,000 features per graph; source splits 3,000/1,000/1,000; target eval 5,000. Source stored edge entries 12,338, target 12,884; observed mean degrees 2.4676/2.5768. Random GCN outputs [5000,2] with finite source gradients. No training/adaptation accuracy claimed.
- Paper-scale generation plus bundle validation took approximately 1.50 seconds in this local CPU run, excluding Python import/startup and the GCN operation. This is a smoke timing, not a benchmark comparison.
- Reports were written outside Git to /tmp/afml_csbm_smoke.json and /tmp/afml_csbm_paper_scale.json; regenerate via the example commands in the README.
- Direct project dependency requirement checks passed. Whole-environment pip check failed on unrelated inherited packages (e.g. AutoGluon/Pandas/Transformers/TensorBoard constraints), not on the selected project dependencies.
- Git diff/whitespace, documentation consistency, and ignored-artifact checks passed. Runtime outputs/checkpoints remain excluded. Prior commits were explicitly pushed; the owner separately approved committing and pushing the verified controlled-shift milestone.

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

### Sequential banks actually executed

The line above describes the earlier subset-only milestone. Bank construction is now implemented, using model seeds 31–36 for every subset and the same architecture/optimizer/selection rules as the source baseline. Every model uses the saved 50%-pool subset (300/1,500 labels depending on graph size), not the full-pool baseline's budget.

| Development bank | Nodes × features | Epoch cap / patience | Completed models | Selected epochs | CPU reload checks |
| --- | --- | --- | --- | --- | --- |
| runs/gcn-bank-small-review | 1,000 × 64 | 100 / 20 | 30 / 30 | 15–25; ran 35–45 epochs | 30 passed |
| runs/gcn-bank-paper-smoke | 5,000 × 2,000 | 20 / 10 | 30 / 30 | All 20; cap reached | 30 passed |

Commands: `.venv/bin/python examples/train_weight_bank.py --subsets runs/source-subsets-small.json --epochs 100 --patience 20 --device cpu --output-dir runs/gcn-bank-small-review` and the equivalent paper manifest/run with `--epochs 20 --patience 10`.

Each bank has bank.json, a copy of source_subsets.json, and 30 model directories containing best_model.pt/metrics.json. All 60 checkpoint hashes and 60 metric-file hashes were checked against their bank manifests after execution. Both bank statuses are complete, all expected model pairs are present, and all CPU reload verifications succeeded. No target graph was created or source-test/target scores computed by the runner.

The sum of fitting times was approximately 6.41 seconds (small) and 45.71 seconds (paper-sized). These sums exclude reconstruction, checkpoint I/O/reloading, and final scoring, so they are not total wall-clock times or comparative performance benchmarks. The paper-sized bank is deliberately short and does not establish convergence.

Both earlier examples record foundation revision 30af4f9 with dirty=true because the runner had not yet been committed. This is development execution evidence, not a fully frozen paper experiment. Same seeds are reused across subsets as a paired design; thirty models are not thirty independent IID trials. The next closure milestone below fixes the phase's label budget/partition and measures diversity; final-paper evaluation and scientific sufficiency remain separate.

## Phase 1 closure

Reference: configs/phase1_csbm_gcn.json. CSBM source 5,000×2,000 with existing generation seeds/convention; 50% of native training labels (1,500 per model), five subsets, six paired seeds, H32/dropout0.5, Adam0.01/0.0005, 200 epochs maximum/patience30. No new dataset or dependency.

Executed `.venv/bin/python examples/train_weight_bank.py --subsets runs/source-subsets-paper.json --epochs 200 --patience 30 --device cpu --output-dir runs/gcn-bank-phase1`. All 30 jobs completed and CPU reloads passed. Selected epochs were 24–98, training ran 54–128 epochs, and every job stopped after thirty epochs without strict source-validation improvement. Sum of fitting times: approximately 198.73 seconds, excluding reconstruction/checkpoint I/O/reloading/final scoring. The earlier smoke banks were preserved.

Executed `.venv/bin/python examples/finalize_phase1.py --bank runs/gcn-bank-phase1 --output-dir runs/phase1-complete`. Report status complete, issues empty, 30 models, 435 pairs, zero selected checkpoints at the cap. Earlier review directories were preserved during final audit hardening. No training/protocol choices were changed using source-test results.

| Closure check | Observed evidence |
| --- | --- |
| Artifact loader | Complete Cartesian membership, copied-subset/checkpoint/metric hashes, finite/schema-compatible weights, source/config/subset/supervision identity, minimum-CE history selection, and reproduced source-validation metrics all verified |
| 18/6/6 model partition | Seed401; meta_train subsets [1,0,3], validation [4], test [2]; complete/disjoint model coverage, no subset group crossing roles; exact partition reload passed |
| Label-free probe | All 3,000 native source-training nodes, canonical order, no labels used |
| Model variation | 30 distinct hard prediction patterns, zero exact parameter duplicates, zero undefined CKA pairs |
| Pairwise disagreement | Mean 7.3217%, range 0.4333–9.8333%; same-subset mean 1.1867%, same-seed/different-subset mean 8.6000% |
| Probability RMSE / hidden linear CKA | Means 0.141192 / 0.861357; CKA range 0.804388–0.998894 |
| Final source-test accuracy | Mean 88.37%; population SD 0.7091 percentage points; range 86.4–89.5%, on the same 1,000 test nodes |

Per-subset source-test means for subsets 0–4: 89.15%, 88.08%, 88.88%, 87.75%, 87.98%. Those are six-seed descriptive averages, not independent dataset replications or confidence intervals. All settings and model-group assignment were frozen before the final evaluator read test labels; this development source graph/split also has scores viewed in earlier examples, so it is not a sealed paper benchmark.

Review artifacts under runs/phase1-complete: REVIEW.md, phase1_report.json, protocol.json, model_partition.json, diversity.json/diversity_pairs.csv, source_test.json/source_test_models.csv, and hashed analysis_code_snapshot files. They bind the exact bank/config/partition hashes. All generated artifacts are ignored by Git. GCN.encode leaves state keys/weights/forward arithmetic unchanged; automatic tests demonstrate the original dropout formula still gives identical outputs.

The extra CKA reference is a sparse, separate Google Research checkout at ba728e288c51126eac25e316b11dfee94d68c301; README/notebook/Apache2 license inspected, upstream clean. PyTorch feature-space CKA was independently implemented and checked against centered Gram arithmetic. No source copy, dependency installation, or global environment change.

The bank's training Git revision is 30af4f9 with dirty=true. The audit preserves its own code snapshot separately; it is not presented as the exact earlier training source. The owner explicitly authorized committing the completed Phase 1 implementation/config/tests/docs afterward. Historical training provenance remains unchanged; no push is authorized.

## Architecture extension verification

New model files use existing PyG APIs; no package/environment changes. CLI/factory support gcn/gat/graphsage for source training, banks, checkpoint reloads and finalization. configs/phase1_csbm_gat.json and configs/phase1_csbm_graphsage.json retain the GCN source/subset/optimizer/stopping settings with explicit model definitions. GCN's original config/spec fields/state keys remain compatible; runs/gcn-compatibility-after-models audited the original bank under the new code and reproduced its 88.37% mean source accuracy.

Actual single-seed source runs (model31, all native source training labels, no adaptation):

| Model/run | Nodes × features | Epoch cap/patience | Best / epochs run | Source-test | Target | CPU reload |
| --- | --- | --- | --- | --- | --- | --- |
| runs/source-gat-review | 1,000 × 64 | 100/20 | 16 / 36 | 64.5% | 53.4% | Passed |
| runs/source-graphsage-review | 1,000 × 64 | 100/20 | 13 / 33 | 61.5% | 53.5% | Passed |
| runs/source-gat-paper-smoke | 5,000 × 2,000 | 20/10 | 20 / 20 | 90.5% | 56.14% | Passed |
| runs/source-graphsage-paper-smoke | 5,000 × 2,000 | 20/10 | 20 / 20 | 90.0% | 57.54% | Passed |

Each command was examples/train_source.py with explicit --architecture, --device cpu, --epochs/--patience above and a new --output-dir (plus --paper-scale for the larger runs). The larger examples reached their cap. These full-pool single runs are not controlled size comparisons or the half-pool bank evaluation; target scores were computed after source-validation selection and did not change settings.

Reference clones (separate, unchanged): GAT 5af87e7fce2b90ae1cbd621cd58059036a3c7436; GraphSAGE a0fdef95dca7b456dab01cb35034717c8b6dd017. MIT licenses and relevant TensorFlow source inspected. Runtime wrappers independently use PyG; no upstream source/environment was copied.

### Additional architecture banks and audits

Executed examples/train_weight_bank.py with --architecture gat / graphsage, --subsets runs/source-subsets-paper.json, --epochs 200 --patience 30 --device cpu and separate outputs runs/gat-bank-phase1 / runs/graphsage-bank-phase1. Executed examples/finalize_phase1.py with each matching configs/phase1_csbm_<architecture>.json and a new runs/<architecture>-phase1-complete directory. Both reports are complete with empty issue lists.

| Architecture | Models / groups | Parameters | Selected / run epoch ranges | Mean source accuracy / population SD | Mean disagreement / hidden CKA |
| --- | --- | ---: | --- | --- | --- |
| GAT | 30; 18/6/6 | 64,166 | 19–61 / 49–91 | 87.9067% / 1.0139 pp | 7.9007% / 0.853525 |
| GraphSAGE | 30; 18/6/6 | 128,162 | 20–98 / 50–128 | 88.6933% / 0.8485 pp | 6.8962% / 0.815166 |

Both banks use exactly the saved source graph/subsets (1,500 training labels per model) and paired seeds 31–36. All 60 new checkpoint CPU reloads passed; each audit verified complete metadata/artifact consistency, native source supervision, restored validation metrics, the 18/6/6 whole-subset split and 435 pairs. Both have 30 distinct hard prediction patterns, no exact parameter duplicates or undefined CKA, and zero selected checkpoints at the cap. No model was filtered by score or diagnostic distance.

Sum of fitting times: GAT156.69 seconds, GraphSAGE945.51 seconds. These exclude reconstruction/I/O/reloading/final audit; runs overlapped and epoch counts differ, so they are not a controlled architecture-latency benchmark. The slower full-neighborhood SAGE implementation aggregates 2,000-dimensional input features before projection and uses separate root/neighbor maps. No unverified cache, sparse-backend change or sampled approximation was introduced to alter the result.

The historical GCN bank was re-audited at runs/gcn-compatibility-after-models, preserving 88.37% mean source-test accuracy and 7.3217% disagreement. All original GCN settings/checkpoints were retained. New banks record revision 3c7976e with dirty=true; the audit snapshots record the later shared implementation separately. The owner explicitly authorized the subsequent implementation/config/test/doc commit and push. All generated artifacts remain ignored, and historical training provenance is preserved.

These are descriptive initial-model results on a reused development graph. GraphSAGE has about twice the parameters; activation functions and model definitions differ and no architecture-specific search was performed. Do not infer a statistically significant ranking, natural-shift robustness, or adaptation improvement from these values.

## Controlled-shift/environment milestone

Reference: configs/controlled_csbm_v1.json and D7. The existing source distribution is the reference, but all five base draws use fresh generation seeds. Each paired lineage has one control plus selected variants. Defaults cover feature translation, h changes, lower/higher degree scales, and their eleven nonempty combinations. Class separation, feature noise, label identities, dimensions and node membership stay fixed. This is our development extension of referenced CSBM controls, not an official Matcha benchmark reproduction.

| Partition | Independent base lineages | Severity values for shifted variants | Environments including controls |
| --- | ---: | --- | ---: |
| Meta-training | 3 | 0.25 / 0.50 | 69 |
| Validation | 1 | 0.25 / 0.50 | 23 |
| Held-out test | 1 | 0.50 / 0.75 / 1.00 | 34 |

The 126 variants are not independent repetitions. All variants from one base stay in the same role. Test0.50 measures a new realization at a seen severity; test0.75/1.00 uses unseen severity coordinates. Default families are seen; optional broad-family holdout is implemented and tested, not claimed as an executed performance experiment. Environment labels follow the existing role boundary; the source-fitting helper continues to reject meta-training and held-out environments.

Executed commands: `.venv/bin/python examples/prepare_shift_environments.py --output-dir runs/controlled-shifts-small-v1` and the same command with `--paper-scale --output-dir runs/controlled-shifts-paper-v1`, both with OMP_NUM_THREADS=1. Full small300×32 and paper-sized5000×2000 suites completed with status complete, identical69/23/34 counts, all paired invariants verified, distinct base inputs and zero duplicate-variant groups. Audit times were approximately1.10s/36.19s on local CPU, excluding process startup/snapshot writing; the runs overlapped other development checks and are not a controlled timing benchmark.

Each ignored output directory has environment_manifest.json (saved before construction audit), construction_audit.json (fingerprints, invariant checks, offline diagnostics), summary.json, run.json (effective config, software, Git state) and a hashed construction_code_snapshot. The recipe names/severities/parameters and label-derived diagnostics are not model conditioning. No model accuracy, adaptation or generator training was performed. Source banks/configurations/checkpoints were not modified.

Tests cover all eleven definitions, exact feature/edge preservation, class-center translation, nested degree-change edges, finite-size expected-degree drift, roles/splits, source-seed exclusion, whole-lineage exclusivity, severity/family exposure, independent generation/partition RNGs, global RNG isolation, persistence/checksum/re-signed tampering rejection, unknown IDs, duplicate/invalid/no-op conditions, cross-lineage duplicate-input detection, all three real model inference paths, held-out-label invariance and a real preparation CLI. An initial test wrongly assumed the source-only trainer extractor accepted meta_train; its deliberate guard was retained and the test corrected to the existing boundary.

## Parameter-efficient adaptation core

Owner authorized tasks1–5 and the initial-versus-adapted-weight review checkpoint on 2026-10-30. The syntax-corrupting accidental insertion in weight_bank.py was restored to the committed allow_nan=False expression; unrelated handbook formatting is preserved. Separate official LoRA/Tent/SHOT clones were inspected, remain outside the repository, and supply the mechanisms documented in D8/THIRD_PARTY_NOTICES.md. No dependency/global environment changes.

The original adaptation.py (now adaptation/iterative.py under D12) provides explicit head/LoRA sites for all three models, frozen original weights, zero-initialized effective updates, deterministic local factor RNGs, supervised development train/validation selection including step0, and fixed-budget label-free entropy/information-maximization fitting. adaptation_artifacts.py writes actual initial/adapted/delta matrices and factors, validates frozen sites/hashes, and reconstructs ordinary model predictions after CPU reload. No held-out label access enters fitting.

The core checkpoint has39 new core/artifact checks plus the prior158 checks:197 tests, independently verified from the staged source snapshot. A standalone comparison against the actual official LoRA Linear class verified forward and A/B gradients, and SHOT loss arithmetic was independently checked. CPU only; CUDA/MPS remain unverified. The development runner and complete three-architecture experimental evidence are the next delivery checkpoint, not part of this core commit. The owner requested reasonably sized verified commits/pushes, then separately approved Colab support for longer compute runs as another setup milestone.

The following performance checkpoint adds three cross-architecture frozen-head-cache checks (200 core tests total). Exact outputs/gradients and fallback on different or mutated inputs are verified. A paper-sized20-step head-only comparison also preserved histories and final weights exactly. Local uncached/cached fit seconds: GCN0.540/0.154, GAT0.635/0.184, GraphSAGE3.700/0.307. Evidence: ignored runs/head-cache-parity.json. This is one execution check, not a controlled speed benchmark. LoRA and exported models retain complete graph computation. The initial60-step broad sweeps were interrupted and retained with explicit incomplete status, superseded by a prospective uniform40-step development sweep; no pairs were filtered by scores.

### Development review runner checkpoint

Implemented adaptation_review.py, examples/run_adaptation_review.py and seven real integration checks;207 total tests passed. The runner freezes selected IDs/configs/hashes before scoring, accepts meta-training models/environments only, preserves native development-test label separation, and saves every baseline/update with verified initial/final/delta reconstruction. It checks source-reference identity, old bank/partition/checkpoint integrity, frozen parameters, factor consistency and CPU reload predictions. All artifact outputs remain ignored; no held-out model/environment or original bank is modified.

Initial GCN preflight: runs/peft-gcn-preflight-v1, one meta lineage, control plus attribute severity0.5, three source models (subsets1/0/3, shared seed31),20-step budget, head/LoRA × supervised_development/information_maximization. All24 adaptations completed and reloads passed. Attribute-test means: frozen83.53%, supervised head87.93%, supervised LoRA87.00%, unlabeled IM head85.03%, unlabeled IM LoRA86.73%. These are preliminary development observations, not final tests or a tuned-method ranking. LoRA update norms can grow substantially; parameter efficiency is not a small-norm guarantee.

The full v2 sweeps predeclared108 baseline pairs: three meta-training source subsets with shared seed31 × three meta-training environment lineages × twelve conditions (control plus eleven shifts at severity0.5). Each plans648 updates: head/LoRA × supervised/entropy/IM, cap40/patience15, lr.01, factor seed801. GCN/GAT v2 completed; GraphSAGE's completed replacement is v3 below. All share the same source data and environment manifest. Source/model/environment holdouts are preserved. The interrupted60-step v1 sweeps retain431/376/82 completed GCN/GAT/GraphSAGE jobs respectively and explicit interrupted status; they are not declared complete or used to drop unfavorable pairs.

### Portable compute setup checkpoint

Owner separately approved Colab support. Added an unexecuted notebook, fixed-allowlist exporter and backend smoke command. The local CPU device check passed; the final command exercises18 tiny architecture/update/objective cases with actual delta/CPU-reload checks. CPU numerical execution and bundle checks are verified; no authenticated browser/GPU runtime was available to execute Colab/CUDA here. The notebook checks CUDA explicitly before a long run and downloads results rather than mounting Drive. No global/local package installation was performed for this setup.

Actual audited-bank exports were created under ignored runs/colab-gcn-review-bundle.zip (~7.6MB) and runs/colab-graphsage-review-bundle.zip (~14.7MB); they are runtime transfers, not Git artifacts. Export hashes are checked against captured source/artifact bytes. Code-only packaging and notebook syntax/no-saved-output checks pass.213 total tests passed, including the final18-case device probe. Runner plans now record CPU thread count; CLI snapshots record invocation and the code captured before execution. Fitting also handles an enclosing no_grad context, rejects inference_mode explicitly and clears stale source gradients only on its frozen copy, with tests.

GCN/GAT v2 completed all648 updates each. GraphSAGE v2 was stopped after163 jobs following a measured1/2/4-thread compute check; artifacts/progress are preserved with explicit incomplete status. Its completed replacement runs/peft-graphsage-review-v3 uses the identical40-step pair/method/objective grid with4 threads. Separate200-step supervised head checks on control/attribute/homophily/combined conditions also completed. They are additional development references, not label-free results or replacements for the40-step sweep.

## Completed adaptation review checkpoint

All six final studies completed with every saved CPU reload check passing: three broad runs (648 updates/108 baselines each) and three extended head studies (36 updates/36 baselines each). Independent post-run audits loaded all2,052 update artifacts, validated source/bank/manifest/model identities, complete Cartesian job membership, hashes, frozen sites, actual final-minus-initial tensors, factor reconstruction, code snapshots and recomputed native meta-test metrics. One-thread CPU prediction replay also checked the four-thread GraphSAGE run; **zero accuracy differences**. Each final directory contains independent_audit.json bound to its review.json hash.

Broad means below exclude controls:99 shifted pairs per method/architecture, all evaluated on1,000 native meta-test nodes per graph. Three source models share seed31 across distinct source-label subsets; three base lineages and eleven variants are related, so no IID confidence interval or statistical ranking is claimed.

| Update / objective | GCN after (gain pp) | GAT after (gain pp) | GraphSAGE after (gain pp) |
| --- | ---: | ---: | ---: |
| Frozen baseline | 78.36% | 77.94% | 80.30% |
| Head / supervised | 80.80% (+2.44) | 80.08% (+2.14) | 83.34% (+3.04) |
| LoRA / supervised | 80.09% (+1.73) | 79.17% (+1.23) | 84.07% (+3.77) |
| Head / entropy | 67.59% (−10.76) | 73.51% (−4.43) | 72.41% (−7.89) |
| LoRA / entropy | 77.27% (−1.09) | 79.00% (+1.07) | 81.74% (+1.45) |
| Head / information maximization | 79.88% (+1.53) | 78.35% (+0.41) | 81.14% (+0.85) |
| LoRA / information maximization | 79.53% (+1.17) | 79.94% (+2.00) | 83.04% (+2.74) |

Trainable head scalars:66/70/130; LoRA:8,162/8,162/16,324. These counts and sites differ, not matched capacity. GCN head entropy worsened97/99 shifted pairs, while GraphSAGE supervised LoRA worsened0/99; these descriptive development observations do not prove generalization. IM-LoRA mean relative delta norms were4.914/3.951/5.101, demonstrating that parameter efficiency does not enforce a small-norm update. No weight normalization, failed-case filtering or post-hoc objective change was applied.

Extended head references use the same27 shifted pairs for attribute/homophily/combined only, cap200/patience40:

| Model | Frozen | Head40 | Head200 | Selected at200 |
| --- | ---: | ---: | ---: | ---: |
| GCN | 76.00% | 78.92% | 79.01% | 18/27 |
| GAT | 75.53% | 78.07% | 78.77% | 22/27 |
| GraphSAGE | 78.52% | 82.11% | 83.26% | 26/27 |

The narrower coverage differs from the broad table. More budget/patience modestly helps some cases; cap selection does not prove convergence, and validation-loss improvement need not change accuracy. Homophily-only head200 gain was0.18pp for GCN and4.86pp for GraphSAGE; this flags an update/architecture/optimization question, not an identified causal mechanism.

Review assets: runs/peft-review-checkpoint/{summary.csv,summary.json,adaptation_gains.png,head_budget_check.png,analysis_sha256.json} plus captured analysis scripts; weights live in each final study's jobs/job_*/weight_delta.pt. Example: runs/peft-gcn-review-v2/jobs/job_00109/weight_delta.pt is an attribute/head/supervised case with canonical initial/adapted/delta states. Source/reference banks and D7 manifests are unchanged. Initial studies ran from honest dirty working trees with pre-execution code snapshots; later commits do not retroactively change their provenance. Main numerical settings were unchanged by later autograd-context/provenance guards.

The owner's requested items1–5 are implemented and verified. Meaningful recovery and objective failures support continued investigation, not a generator-success claim. Next week: review the teammate's proposed generators and decide update representation/norm control, fair equally informed baselines, and final held-out evaluation before implementing new methods.

## Separate supervised full-fine-tuning diagnostic

The owner approved this diagnostic after questioning limited adaptation gains. Implemented method=full through the shared adaptation/artifact interfaces, optimizing a separate ordinary model copy and preserving an immutable initial copy. Full is supervised-development-only and must run separately from head/LoRA. The matched comparison helper rejects changed optimizer budgets, software/graph/model identities, missing pairs and altered artifact metrics; earlier runs are preserved.

Executed verification: **227 CPU tests passed**. Full-mode gradients and updated weights match an independent plain-PyTorch Adam loop exactly for all three architectures. Tests cover initial-model preservation, dropout-off behavior, changes in both graph layers, held-out/test-label invariance, validation selection, portable delta reconstruction, supervised-only restrictions, complete comparisons and mismatch rejection. Existing upstream deprecation warnings remain visible. No new dependency or GPU result.

All six sweeps completed: separate 40-step/patience 15 full-weight runs on the previous broad development grid (108 episodes each), and200-step/patience 40 runs on the previous narrower attribute/homophily/combined grid (36 each). The suite plan was written before scoring to runs/full-diagnostic-suite-plan-v1.json; every run preserves its pre-execution code snapshot and honest Git state. Earlier PEFT/source artifacts and final holdouts remain unchanged.

Broad means on the **same 99 shifted pairs** per architecture, with controls excluded:

| Architecture | Frozen | Supervised head40 | Supervised LoRA40 | Supervised full40 | Full gain vs frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| GCN | 78.36% | 80.80% | 80.09% | 81.21% | +2.85pp |
| GAT | 77.94% | 80.08% | 79.17% | 80.30% | +2.36pp |
| GraphSAGE | 80.30% | 83.34% | 84.07% | 86.10% | +5.81pp |

Full40 worsened 12/9/0 of 99 shifted pairs and selected step 0 in 9/16/0 for GCN/GAT/GraphSAGE. Largest selected steps were 18/10/7; none reached the cap. Broad mean source-validation accuracy fell by 0.60/0.81/0.58pp after full fitting; those source scores were diagnostics, not selection inputs.

Longer comparisons on the **same 27 shifted pairs** within the narrower grid:

| Architecture | Frozen | Supervised head200 | Supervised full200 |
| --- | ---: | ---: | ---: |
| GCN | 76.00% | 79.01% | 79.03% |
| GAT | 75.53% | 78.77% | 78.01% |
| GraphSAGE | 78.52% | 83.26% | 84.71% |

The selected full200 weight hashes equal full40 hashes on **27/27 shifted cases for every architecture** (with matched cases), not merely equal rounded accuracy. More cap/patience therefore did not change these selected models. This does not establish convergence or an optimum at another learning rate. Homophily-only full-weight gains were-0.10/-0.16/+5.32pp. Under the tested settings, unrestricted updates help GraphSAGE more; GCN/GAT structural recovery still warrants investigation. Do not infer a causal architecture/rank limit or a guaranteed generator remedy.

All **432 new artifacts** passed independent source/bank/manifest/graph/split/hash/parameter/delta/config/metric/prediction replay checks; zero one-thread CPU replay accuracy differences, including four-thread GraphSAGE training. Full/head/LoRA comparisons separately validate identical pair grids/configs/initial weights/evaluation splits and preserve every regression. No IID confidence interval or architecture ranking is claimed from related models/lineages.

Outputs: runs/full-<architecture>-diagnostic-{40,200}-v1, matching full-<architecture>-comparison-{40,200}-v1, and runs/full-diagnostic-checkpoint-v1/{summary.csv,summary.json,supervised_recovery.png,analysis_sha256.json}. Captured analysis/audit/suite/comparison code and input hashes accompany the summary; the chart was visually inspected. Runtime artifacts remain ignored. Method rationale: D10; beginner explanation and results: handbook Section31. A validation-only learning-rate study is a possible next diagnostic, not implemented or selected from final holdouts. These results do not prove an accuracy ceiling or generator success.

## Limits and next work

- Source pretraining, controlled data, PEFT, delta artifacts and development experiment audits are verified for all three architectures on CSBM. Generators and final held-out research evaluation/estimates accounting for the paired design remain outstanding.
- CPU verified only. Clean-isolated installation, macOS, Colab, CUDA, and MPS remain unverified.
- Generation pair-scanning is O(N²), although no dense adjacency is allocated. Do not scale blindly to huge N.
- Original independent-draw examples remain independent; the new paired API explicitly controls realized components. Feature-noise/class-separation shifts, degree-heterogeneity models, concept shifts and natural-shift datasets are not implemented.
- Phase 1 fixes a 50%-pool reference budget and whole-subset model grouping. D7 separately fixes development environment grouping/severities. Final-paper label budgets, independent environment counts, model×environment evaluation allocations, and natural-shift evaluation remain scientific decisions. Do not reinterpret model groups as new graphs or treat126 related variants as126 independent draws; there is only one validation and one test base lineage in the executed development suites.
- Next implementation requires discussion of the teammate's generator proposals. Large relative deltas, supervised label advantages and some weak structural recovery should inform that discussion. Held-out performance, normalization/regularization changes and generator success are not established. No automatic bank redesign or arbitrary weight noise was introduced.
