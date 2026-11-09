# Amortized GNN Test-Time Adaptation

Research project for Advanced Foundations of Machine Learning (UE24AM342AA1), PES University.

We investigate whether an offline-trained generator can predict small parameter updates for a pretrained GNN on an unlabeled shifted target graph, without target-time gradient optimization. The verified CSBM system includes head and LoRA generators alongside iterative adaptation. Real-graph loaders and experiment runners are now being validated on ogbn-arxiv and Pair-Align MAG countries; their numerical studies are in progress. See project state for the distinction between implemented software and completed experiments.

Deeptanshu owns GCN experiments; teammates handle GAT and GraphSAGE. At the owner's request, the shared codebase now implements all three architectures with the same data/training/bank interfaces.

## Current campaign

See the [100-epoch campaign and teammate handoff](docs/CURRENT_CAMPAIGN.md) for the exact live job, source audit, early-stopping rule and replacement queue. The earlier five-epoch evidence remains separately labeled below.

## Verified evidence and operational source

The owner-authorized main checkpoint includes [formal source-bank and GCN design evidence](reports/real-data-2026-11-08/README.md), including CSVs/plots, and [Kaggle operational helper source/restoration notes](tools/kaggle/README.md). All 270 source checkpoints are verified; Experiments 1–2 completed with no generator gain and the broader comparison is in progress. The five-epoch design budget completed normally; A/head is a tie fallback, not an established optimal recipe. This commit does not replace the immutable source ZIP used by running Kaggle jobs. Large datasets, weights, raw predictions and bundles remain outside Git.

## Implemented and verified

- **Completed CSBM integration:** generated head/LoRA for both generators on all three GNNs; 431 full-suite tests plus the expanded CLI checks; 36 generator fits and 780 independently replayed saved update cases. Hyper-LoRA's complete CPU pipeline was 18.09–25.75× faster than matched LoRA IM30, with architecture-dependent accuracy tradeoffs. See [verified results and limitations](docs/PROJECT_STATE.md#completed-generated-lora-comparison).
- **CSBM foundation:** balanced two-class Gaussian graphs with explicit connection probabilities, independent edge/feature/split RNG streams, and recorded generation settings.
- CPU PyG graphs, separate ground truth, named split indices, explicit source/meta-training/validation/target roles.
- Stratified source splits, initially 60/20/20; held-out graphs expose an eval-only split. Full graph context is preserved; split indices select supervision, not induced subgraphs.
- Allowlisted, cloned target inputs without labels or diagnostic metadata. Training-label access rejects held-out environments.
- Paired attribute, structural and combined CSBM shifts; saved whole-lineage environment partitions, severity holdouts, optional family holdouts, and sequential construction audits.
- Head-only/LoRA updates for all three models, explicit supervised-development and label-free objectives, and validated initial/adapted/delta artifacts.
- The historical core milestone had **227 tests**, including the supervised full-fine-tuning diagnostic. Current generator-repair verification and results are recorded in [project state](docs/PROJECT_STATE.md); numerical execution is CPU-verified. A Colab notebook/device probe is provided; Later CUDA verification is documented separately in the formal real-data milestone.
- Small and paper-sized (5,000 nodes × 2,000 features per graph) data/interface examples executed successfully.

**The CSBM pretraining/bank foundation is verified for GCN, GAT and GraphSAGE:** shared loading/training/evaluation, a full-budget 30-model bank per architecture, validated artifact loading, saved 18/6/6 partitions, and weight/prediction/representation diagnostics. PEFT and the CSBM generator prototype are described below; real-data studies are in progress; their verification is recorded separately in project state. These development runs are not final paper benchmarks.

GAT and GraphSAGE implementations now use those same entry points. Architecture-specific definitions and commands are below; matching hidden dimensions does not imply matching parameter budgets or optimal tuning for each model.


## Formal real-data source banks

`configs/*_formal_v1.json` declare 30 models per architecture: five overlapping, source-only supervision groups × six distinct initialization seeds, with 18/6/6 whole-group generator roles. Five-model `*_research_v1.json` profiles remain developmental. Formal planning was checked on all three actual source protocols; all 270 formal source fits are downloaded and independently verified; generator experiments are in progress. [Current evidence and limitations](docs/PROJECT_STATE.md#current-formal-bank-and-research-execution-milestone--2026-11-08).

```bash
# Prepare source manifests and jobs without training; use a fresh directory.
python examples/prepare_formal_bank.py --config configs/arxiv_formal_v1.json --data-root /path/to/ogb --output-dir runs/formal-arxiv
# On CUDA only: run a bounded shard; repeat with identical inputs to recover.
python examples/prepare_formal_bank.py --config configs/arxiv_formal_v1.json --data-root /path/to/ogb --output-dir runs/formal-arxiv --fit --max-models 1
# After the desired source architecture is ready, save explicit probe diagnostics.
python examples/audit_formal_bank.py --config configs/arxiv_formal_v1.json --data-root /path/to/ogb --bank runs/formal-arxiv --probe-nodes 1024 --probe-seed 901
```

The audit command's probe size/seed are explicit example settings, not a validated diversity threshold. Resume verifies recipe, graph, supervision, source, dependency and checkpoint identities; completed fits survive interruptions, while an unfinished fit restarts from its seed. CPU fitting is refused by the CLI. A different GPU software environment needs its own freshly prepared output directory. Private Kaggle validation, nonzero CUDA export/replay and versioned-output recovery are verified. All three complete 90-model source banks are downloaded and checked. The GCN input/site study completed and its downloaded artifacts are verified: six hypernetworks selected exact zero updates, matching the 62.891% frozen validation mean. The practice-fixed baseline gained 0.062pp; the other tested baselines regressed. All 420 generated/control cases passed independent replay with zero class mismatches. The broader three-architecture/method/seed comparison is running; final scoring has not started.

## Real-graph research workflow (in progress)

Install the optional official OGB dependency in the project environment:

```bash
python -m pip install -e '.[dev,ogb,perf]'
```

The native Arxiv loader preserves directed citation edges, raw features, paper IDs, years and official splits. Experiment configurations explicitly choose symmetric graph views. Labels remain separate from model inputs. No dataset or checkpoint belongs in Git.

```bash
python examples/train_arxiv_source.py --help
python examples/prepare_arxiv_bank.py --config configs/arxiv_research_v1.json --data-root /path/to/ogb --output-dir runs/arxiv-bank-v1
python examples/run_arxiv_experiments.py --bank runs/arxiv-bank-v1 --data-root /path/to/ogb --output-dir runs/arxiv-study-v1
```

The ordered runner selects generator inputs (Experiment 1), update sites (Experiment 2), then fits hypernetwork, teacher-matched hypernetwork and five-candidate diffusion configurations (Experiment 3). It saves a frozen validation design. Only the separately gated final runner can score official 2019–2020 labels:

```bash
python examples/evaluate_real_generalization.py --bank runs/arxiv-bank-v1 --data-root /path/to/ogb --output-dir runs/arxiv-study-v1 --final-evaluation --measure-memory
python examples/diagnose_real_adaptation.py --bank runs/arxiv-bank-v1 --data-root /path/to/ogb --output-dir runs/arxiv-study-v1
python examples/summarize_real_study.py --bank runs/arxiv-bank-v1 --study runs/arxiv-study-v1
```

Natural temporal/country banks use their separate JSON configurations. Their generators are trained with `--natural-design /path/to/arxiv-study/FROZEN_DESIGN.json`, using the selected design with fresh dataset-specific fitting. Temporal contexts include only arrived papers. Pair-Align country artifacts have a new documented country/node-split protocol; they are not native heterogeneous ogbn-mag. See [decisions](docs/DECISIONS.md) and [current execution state](docs/PROJECT_STATE.md). Output directories refuse accidental overwrites. These commands describe implemented workflows; consult project state before treating any study as executed or complete.

## Setup

Use Python 3.10 or newer and a virtual environment. Install a suitable PyTorch build for your machine using its [official instructions](https://pytorch.org/get-started/locally/) before installing this package if you need a particular CPU/CUDA build.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/inspect_data.py
~~~

Activation above works in Linux/macOS shells; the Python package itself uses portable paths. Installation constraints are in [pyproject.toml](pyproject.toml); only the following stack has been executed here:

| Component | Tested version |
| --- | --- |
| Python | 3.13.13 |
| PyTorch | 2.13.0+cpu |
| PyG | 2.8.1 |
| NumPy | 2.3.5 |
| pytest | 9.1.1 |

The local .venv was created with --system-site-packages to reuse the existing CPU PyTorch/NumPy installation. New packages were installed only into .venv. Project dependency checks pass; whole-environment pip check also sees unrelated pre-existing package conflicts. A completely clean environment install, macOS, Colab, CUDA, and MPS have not been verified. Upstream PyTorch/PyG deprecation warnings remain visible.

## Compute and Colab

The current CSBM/PEFT pipeline runs on CPU. For longer sweeps, [Google Colab](https://research.google.com/colaboratory/faq.html) and [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks) offer free GPU options subject to availability/account limits; no fixed GPU model or allowance is assumed. A GPU accelerates numerical work, not the implementation itself. Generator compute needs depend on the later selected method.

Open [notebooks/colab_adaptation.ipynb](notebooks/colab_adaptation.ipynb) in Colab. Select a GPU runtime, upload an explicit project bundle, install the package, run CPU tests and the actual CUDA device probe, then run/export the same development protocol. Do not infer GPU compatibility from notebook existence. The notebook has no saved executions or outputs and mounts no Google Drive/account credentials.

Export code only, or explicitly include one complete audited synthetic bank and its matching partitions:

~~~bash
python examples/export_colab_bundle.py --output runs/colab-code.zip
python examples/export_colab_bundle.py --bank runs/phase1-bank --model-partition runs/phase1-review/model_partition.json --environments runs/controlled-shifts-paper/environment_manifest.json --output runs/colab-gcn.zip
python examples/check_adaptation_device.py --device cpu --output-dir runs/my-device-smoke
~~~

Replace artifact paths with your actual paths; choose new output names. The exporter includes a fixed code/config/test/notebook allowlist and, only when all three artifact arguments are supplied, the verified bank's listed checkpoints/metrics. It excludes arbitrary run folders, references, Git state directories, virtual environments and user-home files. Checksums describe the exact captured bytes. ZIPs and models stay outside Git. The same helper supports GAT/GraphSAGE banks.

The device probe exercises all three architectures × head/LoRA × supervised/entropy/IM fitting (18 tiny cases), saves deltas, and verifies CPU restoration. CPU checks have actually executed; CUDA/MPS must be run on available hardware. Explicit unsupported devices fail visibly. Colab regenerates and verifies source identities; a software-version/fingerprint mismatch must be investigated, not bypassed. Download results before the temporary runtime ends. Full optimizer resumption and cloud quota management are not implemented. Experiment plans record CPU thread counts, and CLI provenance records the invocation.

## Inspect data for the review

~~~bash
python examples/inspect_data.py --output runs/csbm-small.json
python examples/inspect_data.py --paper-scale --output runs/csbm-paper-scale.json
~~~

Both generate independent source/target graphs, validate them, and run a random GCN layer with a source-label backward pass. The target's true labels are used only in the offline inspection report. Reports include graph counts, edge homophily, split counts, fingerprints, configurations, versions, and finite-gradient checks.

The examples do not optimize a model or report adaptation accuracy. JSON outputs and generated artifacts are ignored by Git.

## Prepare controlled shifts and environment partitions

~~~bash
python examples/prepare_shift_environments.py --output-dir runs/controlled-shifts-small
python examples/prepare_shift_environments.py --paper-scale --output-dir runs/controlled-shifts-paper
~~~

Use a new output directory. Defaults: 300 nodes × 32 features; --paper-scale uses 5,000 × 2,000 from [controlled_csbm_v1.json](configs/controlled_csbm_v1.json). This is a versioned development protocol, not an official benchmark split or a complete Matcha reproduction. No model is trained or scored.

| Control | Endpoint at severity 1 | Preserved |
| --- | --- | --- |
| Attribute | Add +0.02 to both class-center coefficients | Exact edges; class separation, feature noise and labels |
| Homophily | h: 0.8 → 0.2 | Exact features; degree_scale=5 when used alone |
| Degree decrease / increase | degree_scale: 5 → 2 / 10 | Exact features; h=0.8 when used alone |
| Combinations | All nonempty combinations, with either degree direction | Labels, node identities, dimensions, noise and class separation |

Eleven shift definitions are linearly interpolated by severity. Severity measures movement along the parameter path, not guaranteed difficulty. h is a sampling parameter; neither measured homophily nor observed degree is held exactly fixed. A homophily-only change also has the documented small finite-size expected-degree drift.

Five fresh base draws are assigned as **3 meta-training / 1 validation / 1 test lineages** using partition_seed=501, independently of generation_seed=601. All variants of a base stay together. Training/validation use severity 0.25/0.50; test uses 0.50/0.75/1.00, distinguishing unseen realizations from unseen severity values. Each lineage also has one unshifted control. The resulting 126 environments are **69/23/34**, not 126 independent draws. Default families are all seen during meta-training; optional held_out_families in the config supports a separate unseen-family experiment without exposing that family during validation.

The command saves environment_manifest.json before auditing, then construction_audit.json, summary.json, run.json and a hashed construction-code snapshot. Reports include expected/observed degree, offline diagnostics and exact graph fingerprints. Audit checks preserve labels/splits and the required feature/edge tensors, and reject duplicate inputs across distinct lineages. Runtime artifacts remain ignored. Regeneration is verified on the local CPU software stack; re-audit fingerprints after changing versions.

~~~python
from pathlib import Path
from amortized_gnn_tta.data import load_environment_manifest, load_shift_environment

manifest = load_environment_manifest(Path("runs/controlled-shifts-paper/environment_manifest.json"))
item = next(e for e in manifest["environments"] if e["role"] == "target_test")
bundle = load_shift_environment(manifest, item["id"])  # One graph at a time
target_graph = bundle.unlabeled_input(item["id"])
# Feed only target_graph to models/adaptation; never the manifest or label-bearing bundle.
~~~

Meta-training environments retain train/val/test node splits; validation/test environments have eval-only labels. The original source-training helper continues to reject meta/held-out roles. A future meta-training loop must explicitly consume permitted meta-training supervision. The existing independent load_dataset API and pretrained banks are unchanged. Construction uses latent synthetic classes offline; it is not a label-free real-graph corruption procedure.

Both small and paper-sized full suites passed construction audits. These audits concern data construction; development adaptation results are recorded below. The CSBM generator prototype is documented below; real-data studies remain future work.

## Select an architecture

| CLI name | Two-layer model | Total hidden width | Extra model behavior |
| --- | --- | ---: | --- |
| gcn (default) | GCNConv → ReLU → dropout → GCNConv | 32 | Layer-owned self-loops/symmetric normalization; cached=False |
| gat | GATConv → ELU → dropout → GATConv | 32 = 4 heads × 8 channels | One output head, raw logits; layer-owned self-loops; attention dropout defaults to 0 |
| graphsage | SAGEConv(mean) → ReLU → dropout → SAGEConv(mean) | 32 | Separate root/neighbor transforms; no embedding L2 normalization, neighbor sampling, or added self-loops |

All models implement forward(x, edge_index) → raw [N,C] logits and encode(x, edge_index) → first hidden representations before feature dropout. GAT's representation is post-ELU; GCN/GraphSAGE use post-ReLU. The loader remains unchanged and unweighted. The current full-batch source-graph protocol also applies to GraphSAGE; its sampled original-paper training procedure is not reproduced.

~~~python
from amortized_gnn_tta.models import build_model, make_model_spec

spec = make_model_spec("gat", in_channels=2000, out_channels=2,
                       hidden_channels=32, heads=4, dropout=0.5, attention_dropout=0.0)
model = build_model(spec)
~~~

GAT hidden_channels is the **total after concatenation** and must divide evenly across heads. --heads and --attention-dropout are GAT-only CLI options; other architectures reject them. Hidden-feature dropout and attention-coefficient dropout are separate. GAT uses classic GATConv, not GATv2Conv. The original two_layer_gcn checkpoint tag, configuration fields and state_dict keys remain supported.

At F=2000/H=32/C=2, actual parameter counts are GCN 64,098, GAT 64,166 and GraphSAGE 128,162. They share a width and training protocol, not identical capacity. No input dropout, batch normalization, or architecture-specific hyperparameter search was added.

## Train and evaluate a source model

~~~bash
python examples/train_source.py --output-dir runs/source-gcn
python examples/train_source.py --paper-scale --epochs 20 --patience 10 --device cpu --output-dir runs/source-gcn-paper
python examples/train_source.py --architecture gat --output-dir runs/source-gat
python examples/train_source.py --architecture graphsage --output-dir runs/source-graphsage
~~~

Use a new output directory for each run; existing directories/checkpoints are never overwritten. The default small setting has 1,000 nodes and 64 features per graph. The paper-sized option has 5,000 × 2,000. Both retain the existing data convention/seeds and target attribute/homophily changes.

GCN remains the default. Training for all architectures uses full source graph context, source-training-label cross-entropy, Adam (lr=0.01, weight_decay=5e-4 on all parameters), and source-validation-only checkpoint selection. Default cap/patience are 200/30 epochs; strict improvement resets patience, ties keep the earlier epoch. Model seed 31 is separate from the loader's seeds; the same seed integer does not make differently shaped architectures share initial weights.

The trainer's SourceTrainingData contains only source train/validation supervision. Source-test and target labels are absent. Target inference/scoring happens after restoration of the selected checkpoint. No adaptation, target tuning, or test-based early stopping occurs. The model-neutral loop currently handles unweighted graphs and raw [N,C] logits; unsupported edge attributes/weights are rejected explicitly.

Each run writes best_model.pt and metrics.json, including configs, versions, source splits/provenance, history, chosen epoch, final metrics, and CPU checkpoint reload verification. Checkpoints contain CPU state_dict tensors and plain metadata, loaded with weights_only=True/map_location='cpu'. They support inference restoration, not exact training resumption (optimizer/RNG state is not saved).

~~~python
from pathlib import Path
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint

model, metadata = load_checkpoint(
    Path("runs/source-gcn/best_model.pt"),
    build_model,
    device="cpu",
)
~~~

Auto device selection uses available CUDA, otherwise CPU. MPS requires explicit --device mps and is unverified; unsupported kernels fail visibly rather than silently switching backends. CPU is the only executed backend.

Executed development checks (one model seed, not a benchmark comparison):

| Run | Epoch cap/patience | Chosen epoch | Source-test accuracy | Target accuracy |
| --- | --- | --- | --- | --- |
| 1,000 × 64 | 100/20 | 26 (stopped at 46) | 71.0% | 53.4% |
| 5,000 × 2,000 | 20/10 | 20 (cap reached) | 91.0% | 57.1% |

The graph sizes/features and training caps differ, so this table does not isolate the effect of graph size. Target results were scored after fitting and did not choose settings. Multiple seeds and an agreed experimental protocol are still needed for research conclusions. If these viewed target results inform later tuning, use fresh held-out targets for final claims.

## Prepare five source-label subsets

~~~bash
python examples/prepare_source_subsets.py --label-fraction 0.5 --output runs/source-subsets.json
python examples/prepare_source_subsets.py --label-fraction 0.5 --paper-scale --output runs/source-subsets-paper.json
~~~

The label fraction is required and applies **within the existing training pool**, independently per class, rounding down. At 0.5, the 1,000-node example retains its original 600/200/200 split and selects 300 training labels per subset (150 per class). The 5,000-node example selects 1,500 from its 3,000 training nodes. This 50% reference budget is now fixed for Phase 1; it is not an official CSBM convention or a finalized paper protocol.

Five seeds (101–105 by default) produce distinct selections. Subsets can overlap; no node is sampled twice within one subset. Validation/test membership, graph features, and connectivity stay fixed. Unselected training labels do not move into validation/test. Global/model RNG streams are untouched.

The JSON manifest records exact memberships, seeds, requested fraction/realized class counts, source graph/pool identity, validation indices, provenance, and versions. It refuses overwrites and rejects loading against a different source. Stored memberships are used directly on reload rather than redrawn.

~~~python
from pathlib import Path
from amortized_gnn_tta.training import source_training_data
from amortized_gnn_tta.source_subsets import load_source_subsets, source_subset_view

source_view = source_training_data(bundle, "source")  # Existing bundle from Data API below
manifest = load_source_subsets(Path("runs/source-subsets.json"), source_view)
subset_view = source_subset_view(source_view, manifest, "source_subset_0")
# Pass subset_view to fit_source instead of the full source_view.
~~~

Subset views share the read-only source graph; the trainer clones it before execution. Checkpoints automatically retain the selected IDs and subset/manifest identity. Preparation itself does not train models; the runner below does. Those models use fewer labels than the earlier full-pool single-model baseline, so comparisons must account for the budget.

## Train a source weight bank

~~~bash
python examples/train_weight_bank.py --subsets runs/source-subsets.json --output-dir runs/gcn-bank
python examples/train_weight_bank.py --architecture gat --subsets runs/source-subsets.json --output-dir runs/gat-bank
python examples/train_weight_bank.py --architecture graphsage --subsets runs/source-subsets.json --output-dir runs/graphsage-bank
~~~

The example reconstructs only the source from saved subset provenance, validates its identity, and runs five subsets × six seeds sequentially. Defaults: model seeds 31–36, epoch cap/patience 200/30, the selected architecture and shared Adam defaults above, one CPU thread. Override --epochs/--patience/--device as needed. The same six seeds are reused across subsets within each architecture to pair initializations; the thirty jobs are not thirty IID independent repetitions.

Every job keeps its best source-validation checkpoint, scores selected source-training/source-validation nodes, and verifies source predictions after CPU reload. The runner receives no source-test/target supervision, creates no target graph, retains every job, and does not assign generator partitions or claim model diversity.

| Output | Contents |
| --- | --- |
| bank.json | All planned model IDs, subset/seed pairs, progress/status, configs, versions, device/threads, Git revision/dirty state from the example, artifact hashes, per-model summaries |
| source_subsets.json | Copy of the exact validated supervision manifest |
| models/<model_id>/best_model.pt | CPU state dictionary plus selected-epoch/config/source/subset evidence |
| models/<model_id>/metrics.json | Source-train/validation scores, training history, fit time, CPU reload verification |

Use a new output directory. Ordinary job errors mark the bank failed and preserve completed artifacts; abrupt termination may leave status running. Only a fully finished bank is marked complete. Automatic restart/resumption is not implemented: keep partial evidence and choose a new directory for another run. Checkpoints support inference, not exact optimizer/RNG resumption. Run outputs remain ignored by Git.

Two earlier development banks used 1,000×64 (cap/patience 100/20) and 5,000×2,000 (20/10), each with 50% of the native training pool and 30 checkpoints. All 60 CPU reload checks and saved checkpoint/metric hashes passed. The 20-epoch paper-sized run was a smoke check; the full-budget Phase 1 bank below supersedes it for review.

## Complete Phase 1: audited bank and review report

The current reference settings are in [configs/phase1_csbm_gcn.json](configs/phase1_csbm_gcn.json). They freeze the CSBM/GCN development protocol for this phase, including the 50%-training-pool label budget. They are not an official CSBM split or a final paper experiment protocol.

Matching architecture-specific references are [phase1_csbm_gat.json](configs/phase1_csbm_gat.json) and [phase1_csbm_graphsage.json](configs/phase1_csbm_graphsage.json). Supply the corresponding --config to the same audit command; wrong architecture/config combinations are rejected.

~~~bash
python examples/prepare_source_subsets.py --label-fraction 0.5 --paper-scale --output runs/phase1-subsets.json
python examples/train_weight_bank.py --subsets runs/phase1-subsets.json --epochs 200 --patience 30 --device cpu --output-dir runs/phase1-bank
python examples/finalize_phase1.py --bank runs/phase1-bank --output-dir runs/phase1-review
python examples/train_weight_bank.py --architecture gat --subsets runs/phase1-subsets.json --device cpu --output-dir runs/phase1-gat-bank
python examples/finalize_phase1.py --config configs/phase1_csbm_gat.json --bank runs/phase1-gat-bank --output-dir runs/phase1-gat-review
python examples/train_weight_bank.py --architecture graphsage --subsets runs/phase1-subsets.json --device cpu --output-dir runs/phase1-graphsage-bank
python examples/finalize_phase1.py --config configs/phase1_csbm_graphsage.json --bank runs/phase1-graphsage-bank --output-dir runs/phase1-graphsage-review
~~~

Each output must be new. Finalization checks the bank against the frozen configuration, validates all files/checksums and restored source-validation scores, assigns whole label subsets to 18 meta-train/6 validation/6 test models with seed 401, then measures diversity. Source-test labels enter only the final offline evaluator after the partition is saved. Target graphs/labels are absent. The phase protocol and subset memberships must match; a 20-epoch smoke bank is rejected by this command.

Representation comparison uses first-layer post-activation values (ReLU for GCN/GraphSAGE, ELU for GAT) and centered linear CKA in feature space, based on the [original CKA work](https://proceedings.mlr.press/v97/kornblith19a.html). The fixed probe contains all 3,000 native training nodes, without their labels. Raw weight distance/cosine, prediction disagreement and probability RMSE complement CKA; none alone certifies useful adaptation diversity.

Executed Phase 1 evidence:

| Check | Actual result |
| --- | --- |
| Bank | 30/30 models at 5,000 nodes × 2,000 features; 1,500 labels/model |
| Selected epochs | 24–98; no model selected at the 200-epoch cap |
| Model partition | Meta-train subsets 1/0/3; validation 4; test 2; 18/6/6 models |
| Source probe | 435 pairs; 30 distinct hard-prediction patterns; no exact parameter duplicates or undefined CKA |
| Mean disagreement / hidden CKA | 7.32% / 0.8614 |
| Source-test accuracy | Mean 88.37%; population SD 0.71 percentage points across paired models |

The split holds out label selections, not source graphs or initialization seed values. Individual source training labels can overlap between subsets. Accuracy spread is descriptive, not an IID confidence interval. This source graph/split was used in earlier development examples; viewed scores must not be relabeled as a sealed final benchmark. Validation-based stopping is observed, not a proof of optimization convergence.

Finalization writes REVIEW.md, phase1_report.json, model_partition.json, diversity.json/diversity_pairs.csv, and source_test.json/source_test_models.csv. A hashed analysis-code snapshot is included separately from the bank's training Git state. All artifacts remain ignored by Git. Actual local outputs: runs/gcn-bank-phase1 and runs/phase1-complete; detailed evidence is in [project state](docs/PROJECT_STATE.md#phase-1-closure).

The GAT and GraphSAGE reference banks also completed with cap/patience 200/30 and passed the same full audit. Their review directories are runs/gat-phase1-complete and runs/graphsage-phase1-complete. All three use the same source graph, 1,500 labels/model, subset memberships, six paired seed values and 18/6/6 grouping.

| Architecture | Parameters | Mean source-test accuracy | Mean probe disagreement | Selected epochs |
| --- | ---: | ---: | ---: | --- |
| GCN | 64,098 | 88.37% | 7.32% | 24–98 |
| GAT | 64,166 | 87.91% | 7.90% | 19–61 |
| GraphSAGE (mean, full-neighborhood) | 128,162 | 88.69% | 6.90% | 20–98 |

Each bank has 30 models/435 pairs, no selected checkpoint at its cap, and successful CPU reloads. The historical GCN bank was additionally re-audited through the shared factory without changing its results. These descriptive development values do not establish an architecture ranking: capacities/activations differ, defaults were not separately tuned, and the source test graph was previously viewed. Bank runs are source-only; no new target-label selection or adaptation was performed.


## Parameter-efficient adaptation core

| Update | Trainable sites | Original parameters |
| --- | --- | --- |
| head | All conv2 parameters, including GAT attention/bias and GraphSAGE root/neighbor transforms | conv1 frozen |
| lora | BA updates for the linear transforms in both layers; hidden rank4/alpha4, output rank1/alpha1 | All originals frozen; bias/attention vectors unchanged |

The caller's model is copied, never overwritten. Zero effective initialization preserves its predictions. Hidden and output ranks are explicit and strictly below both weight dimensions. Parameter efficiency does not imply few changed dense entries or a small update norm. Reference mechanisms and MIT attribution are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); no new runtime package is required.

Head-only fitting caches the frozen first-layer representation for the episode's exact, unmodified input tensors. Another graph or an in-place input mutation takes the full path. LoRA recomputes its changing backbone. Cache tests compare outputs and update gradients for all three architectures; a paper-sized 20-step check also produced identical histories/final weights with and without caching. Exported models contain no episode cache.

`fit_supervised_development` accepts explicit train/validation supervision from meta_train environments only and selects minimum validation CE including the initial checkpoint. It is a labeled development reference, not unlabeled deployment. `fit_unlabeled` accepts only a graph and settings, uses fixed-step entropy or information_maximization, and never selects with labels. The latter subtracts marginal prediction entropy to discourage class collapse; its balance preference needs care for imbalanced data. These are PEFT objective baselines, not full Tent/SHOT reproductions. All dropout remains off while update gradients are enabled.

After preparing the paper-sized bank and controlled environments with the commands above:

~~~python
from pathlib import Path
from amortized_gnn_tta.adaptation import (
    AdaptationConfig, PEFTConfig, development_supervision, fit_supervised_development,
)
from amortized_gnn_tta.adaptation_artifacts import save_weight_delta
from amortized_gnn_tta.data import load_environment_manifest, load_shift_environment
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint

manifest = load_environment_manifest(Path("runs/controlled-shifts-paper/environment_manifest.json"))
item = next(e for e in manifest["environments"]
            if e["role"] == "meta_train" and e["shift"] == "attribute" and e["severity"] == 0.5)
bundle = load_shift_environment(manifest, item["id"])
graph = bundle.unlabeled_input(item["id"])
model, metadata = load_checkpoint(
    Path("runs/phase1-bank/models/source_subset_1_seed_31/best_model.pt"), build_model)
result = fit_supervised_development(
    model, graph, development_supervision(bundle, item["id"]),
    PEFTConfig("lora"), AdaptationConfig(steps=60), device="cpu")
save_weight_delta(Path("runs/example-weight-delta.pt"), result, metadata["model_spec"], graph,
                  context={"environment_id": item["id"], "purpose": "supervised development reference"})
~~~

Delta files contain canonical initial/adapted/delta states, A/B factors or head offsets, configs, history, selected step, hashes, predictions and context. Frozen tensors have exact zero deltas. `load_weight_delta` validates consistency and restores an ordinary CPU model. These are inference/update artifacts, not optimizer-state resume checkpoints. CPU is tested; CUDA/MPS require actual device checks.

## Frozen baselines and development adaptation review

~~~bash
python examples/run_adaptation_review.py --bank runs/phase1-bank --model-partition runs/phase1-review/model_partition.json --environments runs/controlled-shifts-paper/environment_manifest.json --output-dir runs/gcn-peft-review --all-meta-lineages --steps 40 --patience 15
~~~

Use the bank and matching model-partition path for GAT/GraphSAGE; architecture is restored from the verified checkpoint. Environment reference settings must match the source bank. Select --baseline-only for a frozen sweep without updates. Output directories must be new. No automatic resume is implemented; partial artifacts are retained and progress.json distinguishes running/failed/complete runs.

The runner validates the full bank and model partition, then selects one model (lowest saved seed) from each of the three meta-training label subsets. Default environment coverage is the first meta-training lineage's control plus eleven shift conditions at severity0.5; --all-meta-lineages includes all three permitted bases. Use --models-per-subset for additional paired seeds, --severity for another declared training severity, --shift-names for an explicit subset, and --methods/--objectives for targeted checks. Validation/test models and environments are rejected. Original source banks and partitions are never changed.

Before scoring, plan.json freezes selected IDs, exact settings and identity hashes. Both methods run fresh episodes for supervised_development, entropy and information_maximization. The supervised reference uses native meta-environment training/validation nodes; all before/after accuracies are scored on its unused native test nodes. Unlabeled fitting sees the whole graph without labels and uses a fixed budget. Source-validation retention is scored separately after adaptation and never enters its objective/selection. Improvements, regressions, zero updates and class collapse are all retained.

Each jobs/job_*/weight_delta.pt contains initial/final/delta weights and factors; metrics.json records source/environment identities, accuracy/loss, prediction entropy/class counts, update norms, parameter counts and fit time. Initial-plus-delta reconstruction, frozen-weight integrity, saved factors and CPU reload predictions are checked. review.json/REVIEW.md aggregate descriptive results by update/objective/family. Related models, base lineages and variants are not independent statistical trials. The CLI preserves a code snapshot captured before execution, including for interrupted runs, separately from the historical bank provenance. Full development sweeps and their results are recorded in PROJECT_STATE.md as they complete; no generator result is implied.

### Review checkpoint: executed development results

All three broad sweeps completed:108 baselines and648 adaptations per architecture, with three source models × three meta-training lineages × twelve conditions (control plus eleven shifts at severity0.5). Two update methods × three objectives use cap40/patience15/lr0.01. The table excludes controls:99 related shifted pairs per method/architecture.

| Architecture | Frozen | Supervised head | Supervised LoRA | Unlabeled IM-LoRA |
| --- | ---: | ---: | ---: | ---: |
| GCN | 78.36% | 80.80% | 80.09% | 79.53% |
| GAT | 77.94% | 80.08% | 79.17% | 79.94% |
| GraphSAGE | 80.30% | 83.34% | 84.07% | 83.04% |

These are descriptive development means, not held-out results, an architecture ranking or generator success. Supervised references have label/selection advantages. Head-only entropy minimization reduced mean accuracy by10.76/4.43/7.89 points respectively; failures are retained. LoRA can produce large update norms despite its smaller trainable count. No confidence interval assumes99 independent samples.

A separate200-step/patience40 supervised head check adds108 adaptations on a narrower matched subset. All **2,052 broad/extended delta artifacts** passed independent state/factor/hash/metric/prediction replay audits, with no CPU replay accuracy differences. Actual outputs: runs/peft-gcn-review-v2, runs/peft-gat-review-v2, runs/peft-graphsage-review-v3, runs/peft-<architecture>-head-budget-check and runs/peft-review-checkpoint (summary CSV/JSON and figures). Runtime files remain ignored. See [project state](docs/PROJECT_STATE.md#completed-adaptation-review-checkpoint) and [the learning handbook](docs/learning/RESEARCH_ENGINEERING_NOTES.md#30-our-working-adaptation-checkpoint-initial-weights-final-weights-and-delta) for interpretation and limitations.

## Separate supervised full-fine-tuning diagnostic

To investigate limited PEFT recovery, allow every original model parameter to change on permitted meta-training labels. Use the existing runner with an explicit full-only supervised method and a new output directory:

~~~bash
python examples/run_adaptation_review.py --bank runs/gcn-bank-phase1 --model-partition runs/phase1-complete/model_partition.json --environments runs/controlled-shifts-paper-v1/environment_manifest.json --output-dir runs/full-gcn-diagnostic-40-v1 --all-meta-lineages --methods full --objectives supervised_development --steps 40 --patience 15
python examples/compare_supervised_reviews.py --full runs/full-gcn-diagnostic-40-v1 --references runs/peft-gcn-review-v2 --output-dir runs/full-gcn-comparison-40-v1
~~~

Paths refer to existing local artifacts; choose new output names. GAT and GraphSAGE use their respective banks and partitions through the same interface. A separate longer check uses `--steps 200 --patience 40 --shift-names attribute homophily attribute_homophily` to match the earlier head-only check. Do not compare means from different case grids.

Full fitting starts from pretrained weights, keeps dropout off, uses Adam lr0.01 without weight decay, and selects minimum development-validation cross-entropy including step 0. Native test labels are offline evaluation only. Every episode preserves initial/adapted/delta states and ordinary-model replay. The historical PEFTConfig container accepts method=full; this method is not parameter-efficient. Full is rejected in unlabeled fitting and mixed-method review runs.

The comparison tool validates complete matching pair grids, plans, bank/manifest/config/software identities, artifact checksums, initial weights and graph/evaluation-split identities. It records per-shift comparisons and regressions, excluding controls from the overall shifted mean. Independent prediction/metric replay is a separate audit. See [D10](docs/DECISIONS.md#implemented-decision-d10--separate-supervised-full-weight-diagnostic-2026-10-30) and [handbook Section31](docs/learning/RESEARCH_ENGINEERING_NOTES.md#31-what-a-training-budget-means-and-why-we-added-full-fine-tuning).

All six separate full-weight sweeps completed: **432 new artifacts independently audited**, with zero CPU replay accuracy differences. On the broad99 shifted pairs, full-weight means were 81.21%/80.30%/86.10% for GCN/GAT/GraphSAGE, gains2.85/2.36/5.81 percentage points over frozen models. The longer check selected identical final weights to the40-step runs on every matched case. See [full comparisons and limitations](docs/PROJECT_STATE.md#separate-supervised-full-fine-tuning-diagnostic).

This is a supervised development diagnostic, not a label-free result or a proven accuracy ceiling. Shared optimizer settings need not be equally optimal for every update method. Full fitting leaves additional recovery over LoRA for GraphSAGE under these settings; it adds little over head-only GCN/GAT and does not resolve their homophily-only weakness.
## Train and benchmark the CSBM generator prototype

Both hypernetwork and diffusion can generate direct deltas for `head`, `backbone` or `all` sites on GCN, GAT and GraphSAGE. These are ordinary coordinate updates, not generated LoRA. The numerical repair review uses head-only updates and one diffusion candidate. Validation chooses checkpoints; final labels are not scored by default.

~~~bash
python examples/train_adapter.py \
  --bank runs/gcn-bank-phase1 \
  --model-partition runs/phase1-complete/model_partition.json \
  --manifest runs/controlled-shifts-paper-v1/environment_manifest.json \
  --output-dir runs/generator-gcn-D \
  --generator hypernetwork --conditioning D --scope head \
  --epochs 5 --max-models 3 --max-environments 12 --seed 202801

python examples/benchmark_adapters.py \
  --checkpoint runs/generator-gcn-D/generator_best.pt \
  --bank runs/gcn-bank-phase1 \
  --model-partition runs/phase1-complete/model_partition.json \
  --manifest runs/controlled-shifts-paper-v1/environment_manifest.json \
  --output-dir runs/generator-gcn-D-timing --steps 30 --repeats 15
~~~

Use `--generator diffusion` to select the cosine x0/DDIM generator; record `--diffusion-steps`, `--sampling-steps`, `--oracle-steps` and `--sampling-seed`. Initialization/training seeds are separate. The saved partition is required; it is never recreated implicitly. Caps apply to every loaded role and are debug subsamples, not guaranteed lineage/severity-balanced studies. Training defaults to validation evaluation. `--final-evaluation` explicitly opens held-out scoring; do not use it during tuning.

The training CLI requires a new output directory; interactive in-memory training remains available through the library APIs. Checkpoints are saved atomically during training, before report writing, with constructor settings, descriptor version, exact schema, sampling settings, hashes, seeds and provenance. `load_generator_checkpoint` reconstructs inference; legacy incomplete files require an explicit migration recipe. This does not resume optimizer state. Run JSON records failures and preserves completed artifacts.

Conditioning v2: A=raw graph; B=A+hidden summaries; C=B+predictive uncertainty; D=C+12 explicit full-model weight statistics. D's weight scope is independent of update sites. Directed isolated nodes have no incoming or outgoing edge. E/N is mean stored outgoing degree, not density. The retained-top-two spectral ratio is a heuristic, not a condition number. Feature/class mismatches fail rather than silently pad/truncate. v1 remains an explicit replay definition; equal vector length does not imply compatible semantics.

The benchmark's main metric is **wall time from a canonical CPU target graph and resident models to usable adapted predictions**. Required target preprocessing, transfer, generator/iterative work, update application and prediction count. Disk loading, benchmark construction, offline training, label scoring and report I/O are separate. A prepared-input generator microbenchmark is diagnostic only. Static D fingerprints may be prepared once per unchanged source checkpoint, with setup time reported; target-specific descriptors are recomputed in every event. Head IM caches frozen features inside each event. Generated head-only predictions also reuse the exact first-layer features already computed for B-D conditioning, within that same event. Broader updates recompute their changed backbone. Raw repetitions, first-event/warm times, separate stage profiles and memory observations are saved. CPU RSS is sampled at 5ms and is a lower bound on peak process memory, not an exact per-operation allocation peak. CUDA/MPS timing paths exist but are unverified here.

The reproducible repair review fixes its configuration before scoring:

~~~bash
python examples/run_generator_review.py \
  --config configs/generator_csbm_review.json \
  --artifact-root /path/to/checkout-containing-existing-runs \
  --output-dir runs/generator-repair-review-v2
~~~

For an inference-equivalent optimization, `--reuse-completed-from /path/to/prior-review` can reuse only completed compatible checkpoints after configuration/identity checks and exact generated-delta agreement; partial training is never resumed. Every prediction/timing measurement is rerun in a new output directory.

It spans three architectures, A-D, both generators and three generator seeds; training uses one seed-31 model from each permitted source-subset group. An independent validation checkpoint and validation lineage are used. Hypernetwork CE and diffusion teacher-MSE have different budgets, so this is development verification rather than a definitive generator ranking. The historical +2.34 run remains under-specified. Commit c137e28 published the timing scripts/results: their arithmetic is verified, but their component timings are not complete-pipeline speedups. Exact replay still needs matching artifacts and recorded initialization/hardware; do not substitute this review for that experiment. Run `python examples/replay_generator_review.py --review-dir runs/generator-repair-review-v2` after completion for independent named-weight reconstruction, metric/timing checks, and `VERIFIED_REVIEW.md`/`AGGREGATED_RESULTS.csv`. See [D12](docs/DECISIONS.md#implemented-decision-d12--generator-reliability-descriptors-and-runtime-2026-11-07).

### Portable comparisons and matched compact baseline

The shared runner supports `conditioning` values `compact`, `structural`, A-D,
and `update_scope` values `head`, `backbone`, `all`. Select architectures and
methods in the JSON plan; use separate scope plans. This covers the experiments
in the contributor's five scripts without adopting their hardcoded paths or
partial timers. The original scripts/results remain in [scratch/](scratch/README.md).

The legacy/default `compact` recipe means hidden means/stds and E/N, plus separate raw update-site weights. The matched head/LoRA configs explicitly set `include_raw_weights=false`, retaining only the shared hidden summary.
`structural` adds hidden edge/dispersion statistics and also retains those weights.
A-D disable the extra raw-weight input. Compact is a reference method alongside
A-D, not another step in the nested A-D ablation. Neither direct `backbone` nor
`all` deltas are generated LoRA.

All scopes report complete generated latency. Broader **direct-coordinate** scopes have no matched optimizer in this benchmark; their head-IM reference has no matched speedup or break-even claim. Explicitly tagged generated-LoRA checkpoints are compared with the matching iterative LoRA adapter.
Training CLI `--threads` defaults to one and is recorded.

The bounded matched study keeps the previous scientific training recipe and
adds only the compact baseline for GCN:

~~~bash
python examples/run_generator_review.py \
  --config configs/generator_compact_matched_gcn.json \
  --artifact-root /path/to/checkout-containing-existing-runs \
  --reuse-completed-from /path/to/completed-generator-repair-review-v5 \
  --output-dir runs/generator-compact-matched-gcn-v1
python examples/replay_generator_review.py \
  --review-dir runs/generator-compact-matched-gcn-v1
~~~

Without the reuse option, the runner trains every selected fit. With it, a
smaller/extended grid or new timing repetitions may reuse a completed fit only
when its training recipe, source/environment identities, method/input/seeds and
regenerated deltas agree. Unknown configuration fields are conservatively part
of the recipe check. Original checkpoints and their training costs/provenance
are preserved; every prediction and timing measurement reruns. Six new compact
fits plus 24 existing A-D fits constitute the planned 30-configuration study.
This does not choose the later Arxiv inputs or establish a diffusion advantage. The study completed: compact accuracies were 78.087% / 77.999% (hypernetwork/diffusion), versus frozen 77.255% and IM30 78.167%. Complete-pipeline ratios were 3.89× / 3.51×; control losses were 0.653pp / 0.427pp. Full A-D comparisons and verification are in [project state](docs/PROJECT_STATE.md#completed-integration-checkpoint).

### Verified repair checkpoint

**392 CPU tests; 72 development configurations; 864 independently replayed saved cases with zero logit/accuracy mismatch.** All tested A-D groups are slower end to end than the documented cached IM30 baseline on this 5,000-node/2,000-feature CPU workload, despite faster preprocessing. Average shifted accuracy improves modestly; unchanged controls can regress. No final-test, generated-LoRA, five-candidate or real-graph result is implied.

Fixed D reference (mean ± generator-seed SD; not selection of the later Arxiv design):

| Architecture | Frozen | IM30 | Hypernetwork D | Diffusion D |
| --- | ---: | ---: | ---: | ---: |
| gcn | 77.255% | 78.167% | 78.061 ± 0.004% | 78.099 ± 0.011% |
| gat | 77.447% | 77.445% | 77.684 ± 0.015% | 77.679 ± 0.017% |
| graphsage | 79.316% | 80.251% | 81.033 ± 0.031% | 80.724 ± 0.029% |

See [project state](docs/PROJECT_STATE.md#current-repair-milestone) for timing, budgets and limitations. Full local evidence is in ignored `runs/generator-repair-review-v5/VERIFIED_REVIEW.md` and `AGGREGATED_RESULTS.csv`. Independent replay uses manual named-weight reconstruction and checks raw timing medians; source/code hashes and all raw cases are preserved. The existing main checkout's unrelated handbook formatting remains untouched.

## Data API

~~~python
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset

source = CSBMConfig.from_matcha(
    num_nodes=1000, num_features=64,
    degree_scale=5, homophily_parameter=0.8,
    edge_seed=11, feature_seed=12, split_seed=13,
)
target = CSBMConfig.from_matcha(
    num_nodes=1000, num_features=64,
    degree_scale=5, homophily_parameter=0.2,
    edge_seed=21, feature_seed=22, split_seed=23,
)
bundle = load_dataset([
    CSBMEnvironment("source", "source_train", source),
    CSBMEnvironment("target", "target_test", target),
])
graph = bundle.unlabeled_input("source")
train_idx = bundle.select("source", "train")
train_labels = bundle.training_labels("source")
target_graph = bundle.unlabeled_input("target")
# Adaptation receives target_graph, never bundle or its label-bearing records.
~~~

The initial loader generates data in memory; there is no disk cache or unnecessary root-directory argument. Configurations and seeds recreate graphs within the tested software stack. Use distinct edge and feature seeds across environments; paired interventions are deliberately not an implicit loader mode.

## CSBM convention

The versioned protocol is csbm-matcha-code-v1, independently implemented from the distribution in [Matcha's released generator](https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/CSBM.py).

- Primitive inputs are p_in and p_out.
- from_matcha converts p_in = degree_scale*h/N and p_out = degree_scale*(1-h)/N.
- The exact expected degree is (N/2−1)*p_in + (N/2)*p_out, approximately degree_scale/2 under that conversion. We do not silently insert a factor of two.
- Features follow x_i = (class_mean[y_i] + noise_scale*z_i)/sqrt(F), with independent standard Gaussian coordinates z_i and a fixed shared feature direction. Default class coefficients are −0.03 and +0.03, noise_scale=1.
- Edges are simple, undirected, stored in both directions, without self-loops. The loader does not normalize features or adjacency. Layer-specific normalization belongs to the model.
- Sampling uses bounded-memory row scans, with O(N²) pair-scanning time and O(N+E) auxiliary edge storage, plus O(NF) features. This is suitable for the verified 5,000-node example, not a claim of large-graph scalability.

This preserves a released-code distribution, not its exact seeded artifacts or the whole paper protocol. Our source-label splits, RNGs, and GCN layer choice differ. See [decisions](docs/DECISIONS.md) for rationale and limitations.

## Project context

- [Project state](docs/PROJECT_STATE.md): verified milestone, evidence, remaining work.
- [Learning handbook](docs/learning/RESEARCH_ENGINEERING_NOTES.md): beginner explanations, important choices, worked examples, and second-opinion prompts.
- [Coding instructions](AGENTS.md): research, software, and Git rules.

The PDFs edit.pdf, Biased_Bozos_OnePager-3-1.pdf, and AFML_Papers_and_Videos_Link_Sheet_Minimal.pdf live in the parent research directory. Upstream checkouts live separately under its references/ directory. Neither is a runtime dependency.

Held-out target labels must never enter adaptation, conditioning, or model selection. Supervised oracles and label-aware offline benchmark construction must be identified separately. No commits or pushes without explicit owner authorization.

## Matched generated head/LoRA comparison

Generated LoRA uses the existing rank/scaling convention and predicts `[dA, B]` around seeded A0, preserving identity initialization. Checkpoints save the explicit factor representation; independent replay reconstructs both factors and effective dense changes. Both generators support GCN/GAT/GraphSAGE. The matched compact-only configurations disable the historical extra raw-weight input. See decision D14 and the learning handbook for initialization/factor-distance assumptions.

Run each study serially to a fresh directory, with `PYTHONPATH=src` when multiple editable checkouts exist:

```bash
python examples/run_generator_review.py --config configs/generator_matched_head.json --artifact-root /path/to/bank-checkout --output-dir runs/matched-head
python examples/replay_generator_review.py --review-dir runs/matched-head
python examples/run_generator_review.py --config configs/generator_matched_lora.json --artifact-root /path/to/bank-checkout --output-dir runs/matched-lora
python examples/replay_generator_review.py --review-dir runs/matched-lora
python examples/compare_adapter_studies.py --head-study runs/matched-head --lora-study runs/matched-lora --output-dir runs/matched-comparison
python examples/compare_full_references.py --comparison-dir runs/matched-comparison --output-dir runs/matched-full-development
```

These are validation development runs: three generator seeds, one candidate, no final-target scoring. Primary latency includes every method-required step through prediction; compare generated LoRA with iterative LoRA, rather than head IM. Full supervised fine-tuning remains a separate labeled reference. The supplied configs measure CPU; CUDA/MPS timing is unverified.

The final command compares every method on the supervised references' identical permitted meta-training checkpoint/graphs, with native test nodes reserved for scoring. Those checkpoint/environment pairs participated in generator fitting: this is an in-sample development diagnostic, kept separate from the held-out validation report. Supervised references retain their public fitting/validation/selection/materialization time; no cross-label-access ratio is presented as a matched unlabeled speedup. Existing output directories are rejected before status logging, preserving completed evidence.
