# Amortized GNN Test-Time Adaptation

Research project for Advanced Foundations of Machine Learning (UE24AM342AA1), PES University.

We investigate whether an offline-trained generator can predict small parameter updates for a pretrained GNN on an unlabeled shifted target graph, without target-time gradient optimization. This is a research objective; no adaptation result is available yet.

Deeptanshu owns GCN experiments; teammates handle GAT and GraphSAGE. Data interfaces remain architecture-neutral.

## Implemented and verified

- **CSBM only:** balanced two-class Gaussian graphs with explicit connection probabilities, independent edge/feature/split RNG streams, and recorded generation settings.
- CPU PyG graphs, separate ground truth, named split indices, explicit source/meta-training/validation/target roles.
- Stratified source splits, initially 60/20/20; held-out graphs expose an eval-only split. Full graph context is preserved; split indices select supervision, not induced subgraphs.
- Allowlisted, cloned target inputs without labels or diagnostic metadata. Training-label access rejects held-out environments.
- Structural/statistical tests, layer checks, source-training/checkpoint tests, and fixed-holdout source-subset tests: **70 tests passed** on the tested CPU stack.
- Small and paper-sized (5,000 nodes × 2,000 features per graph) data/interface examples executed successfully.

The two-layer source GCN now trains end to end, selects the minimum source-validation loss checkpoint, and evaluates source-test/target performance afterward. Five source-label subsets can be prepared and reloaded for the upcoming weight bank. The 30-model bank and adaptation generator are not implemented yet. Single-run development results are not paper benchmarks.

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

The label fraction is required and applies **within the existing training pool**, independently per class, rounding down. At 0.5, the 1,000-node example retains its original 600/200/200 split and selects 300 training labels per subset (150 per class). The 5,000-node example selects 1,500 from its 3,000 training nodes. The 50% budget is a development example, not an official CSBM convention or finalized research choice.

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

Subset views share the read-only source graph; the trainer clones it before execution. Checkpoints automatically retain the selected IDs and subset/manifest identity. This step prepares supervision only: it does not train 30 models or assign generator partitions. Those models will use fewer labels than the earlier full-pool single-model baseline, so comparisons must account for the budget.

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
