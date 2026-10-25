# Amortized GNN Test-Time Adaptation

Research project for Advanced Foundations of Machine Learning (UE24AM342AA1), PES University.

We investigate whether an offline-trained generator can predict small parameter updates for a pretrained GNN on an unlabeled shifted target graph, without target-time gradient optimization. This is a research objective; no adaptation result is available yet.

Deeptanshu owns GCN experiments; teammates handle GAT and GraphSAGE. Data interfaces remain architecture-neutral.

## Implemented and verified

- **CSBM only:** balanced two-class Gaussian graphs with explicit connection probabilities, independent edge/feature/split RNG streams, and recorded generation settings.
- CPU PyG graphs, separate ground truth, named split indices, explicit source/meta-training/validation/target roles.
- Stratified source splits, initially 60/20/20; held-out graphs expose an eval-only split. Full graph context is preserved; split indices select supervision, not induced subgraphs.
- Allowlisted, cloned target inputs without labels or diagnostic metadata. Training-label access rejects held-out environments.
- Data, model/training, bank execution, artifact validation, grouped holdouts, and diagnostic math: **104 tests passed** on the tested CPU stack.
- Small and paper-sized (5,000 nodes × 2,000 features per graph) data/interface examples executed successfully.

**GCN/CSBM Phase 1 is complete:** loader, source training/evaluation, a full-budget 30-model bank, validated artifact loading, a saved 18/6/6 model partition, and weight/prediction/representation diagnostics. This is the GCN foundation through pretraining-bank preparation, not the entire team's research roadmap. Adaptation/generators and other datasets remain later work. These development runs are not final paper benchmarks.

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

## Inspect data for the review

~~~bash
python examples/inspect_data.py --output runs/csbm-small.json
python examples/inspect_data.py --paper-scale --output runs/csbm-paper-scale.json
~~~

Both generate independent source/target graphs, validate them, and run a random GCN layer with a source-label backward pass. The target's true labels are used only in the offline inspection report. Reports include graph counts, edge homophily, split counts, fingerprints, configurations, versions, and finite-gradient checks.

The examples do not optimize a model or report adaptation accuracy. JSON outputs and generated artifacts are ignored by Git.

## Train and evaluate a source GCN

~~~bash
python examples/train_source.py --output-dir runs/source-gcn
python examples/train_source.py --paper-scale --epochs 20 --patience 10 --device cpu --output-dir runs/source-gcn-paper
~~~

Use a new output directory for each run; existing directories/checkpoints are never overwritten. The default small setting has 1,000 nodes and 64 features per graph. The paper-sized option has 5,000 × 2,000. Both retain the existing data convention/seeds and target attribute/homophily changes.

The model is GCNConv → ReLU → dropout → GCNConv, with 32 hidden channels, dropout 0.5, and cached=False. Training uses full source graph context, source-training-label cross-entropy, Adam (lr=0.01, weight_decay=5e-4 on all parameters), and source-validation-only checkpoint selection. Default cap/patience are 200/30 epochs; strict improvement resets patience, ties keep the earlier epoch. Model seed 31 is separate from the loader's seeds.

The trainer's SourceTrainingData contains only source train/validation supervision. Source-test and target labels are absent. Target inference/scoring happens after restoration of the selected checkpoint. No adaptation, target tuning, or test-based early stopping occurs. The model-neutral loop currently handles unweighted graphs and raw [N,C] logits; unsupported edge attributes/weights are rejected explicitly.

Each run writes best_model.pt and metrics.json, including configs, versions, source splits/provenance, history, chosen epoch, final metrics, and CPU checkpoint reload verification. Checkpoints contain CPU state_dict tensors and plain metadata, loaded with weights_only=True/map_location='cpu'. They support inference restoration, not exact training resumption (optimizer/RNG state is not saved).

~~~python
from pathlib import Path
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.training import load_checkpoint

model, metadata = load_checkpoint(
    Path("runs/source-gcn/best_model.pt"),
    lambda spec: GCN(GCNConfig(**spec["config"])),
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

## Train a source GCN weight bank

~~~bash
python examples/train_weight_bank.py --subsets runs/source-subsets.json --output-dir runs/gcn-bank
~~~

The example reconstructs only the source from the saved subset provenance, validates its identity, and runs five subsets × six seeds sequentially. Defaults: model seeds 31–36, epoch cap/patience 200/30, the same two-layer GCN/Adam defaults as above, one CPU thread. Override --epochs/--patience/--device as needed. The same six seeds are reused across subsets to compare supervision changes with matched initializations; the thirty jobs are not thirty IID independent repetitions.

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

~~~bash
python examples/prepare_source_subsets.py --label-fraction 0.5 --paper-scale --output runs/phase1-subsets.json
python examples/train_weight_bank.py --subsets runs/phase1-subsets.json --epochs 200 --patience 30 --device cpu --output-dir runs/phase1-bank
python examples/finalize_phase1.py --bank runs/phase1-bank --output-dir runs/phase1-review
~~~

Each output must be new. Finalization checks the bank against the frozen configuration, validates all files/checksums and restored source-validation scores, assigns whole label subsets to 18 meta-train/6 validation/6 test models with seed 401, then measures diversity. Source-test labels enter only the final offline evaluator after the partition is saved. Target graphs/labels are absent. The phase protocol and subset memberships must match; a 20-epoch smoke bank is rejected by this command.

Representation comparison uses first-layer post-ReLU values and centered linear CKA in feature space, based on the [original CKA work](https://proceedings.mlr.press/v97/kornblith19a.html). The fixed probe contains all 3,000 native training nodes, without their labels. Raw weight distance/cosine, prediction disagreement and probability RMSE complement CKA; none alone certifies useful adaptation diversity.

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
