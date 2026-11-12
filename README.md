# Amortized GNN Test-Time Adaptation

An Advanced Foundations of Machine Learning project at PES University investigating whether learned generators can adapt graph neural networks faster than iterative test-time optimization.

The code supports **GCN, GAT and GraphSAGE**, with direct hypernetworks, teacher-trained hypernetworks and diffusion generators. Generators predict head or LoRA updates from unlabeled target inputs. Permitted practice labels are used offline; final target labels are reserved for evaluation.

## Methods

| Method | Offline training | Target-time work |
|---|---|---|
| Direct hypernetwork | Practice-label cross-entropy through the GNN, with update regularization. | Predict and apply one parameter update. |
| Teacher hypernetwork | Regress supervised practice-teacher updates. | Predict and apply one parameter update. |
| Diffusion | Denoise normalized practice-teacher updates. | Generate five candidates with 10 DDIM steps and select by unlabeled prediction agreement. |
| Iterative IM | Reuse the pretrained source model. | Optimize an information-maximization objective for 30 steps. |

The final controlled design selects rank-4 LoRA for GCN and head updates for GAT/GraphSAGE. Frozen, constant-teacher, practice-fixed and feature-only GTrans controls are included in the numerical exports.

## Datasets

| Dataset | Use |
|---|---|
| [OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) | Controlled graph corruptions and temporal distribution shifts. |
| [Pair-Align MAG country graphs](https://zenodo.org/records/10681285) | US source; CN/DE practice; FR validation; JP/RU final evaluation. Derived from [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag). |
| [Synthetic CSBM](https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/CSBM.py) | Earlier development experiments, generated locally using an independent implementation of Matcha's released distribution. |

Controlled and temporal Arxiv experiments use different protocols on the same underlying dataset. Natural-shift generators are fitted separately for each protocol.

## Results

Final mean accuracy (%), averaged over the declared held-out source checkpoints and environments. Generator results additionally average three training seeds. These are the final real-data campaign results, reviewed **12 November 2026**; earlier CSBM studies are separate.

| Evaluation | Model | Frozen | Direct hypernetwork | Teacher hypernetwork | Diffusion | Practice fixed |
|---|---|---:|---:|---:|---:|---:|
| Controlled Arxiv* | GCN | 60.20 | 59.25 | 60.38 | 60.20 | 61.20 |
| Temporal Arxiv | GCN | 59.41 | 54.27 | 59.63 | 59.41 | 63.20 |
| Temporal Arxiv | GAT | 62.20 | 63.07 | 62.88 | 62.00 | 65.64 |
| Temporal Arxiv | GraphSAGE | 59.97 | 47.61 | 60.10 | 60.38 | 63.85 |
| MAG country | GCN | 65.19 | 67.43 | 66.48 | 68.98 | 70.16 |
| MAG country | GAT | 62.83 | 64.53 | 65.08 | 65.06 | 68.60 |
| MAG country | GraphSAGE | 65.20 | 67.38 | 70.01 | 69.30 | 70.30 |

Diffusion gains on MAG country graphs are **+3.79, +2.23 and +4.10 percentage points** for GCN, GAT and GraphSAGE. Performance varies by shift: direct hypernetworks regress on temporal GCN and GraphSAGE. Practice-fixed models, trained offline on permitted practice labels, outperform all three generator families in the natural-shift final comparisons. Constant-teacher controls explain much of the country GAT/GraphSAGE improvement; their scores are retained in the [full method summary](reports/experiment-summary-20261112/final_method_summary.csv).

Measured online speedups span **3.04–11.07×** for direct hypernetworks and **1.60–5.55×** for diffusion against the corresponding 30-step iterative head/LoRA baseline. Timing includes target preprocessing, required transfer, adaptation and prediction, starting from a CPU graph and resident models. These measurements use generator seed 202801 and exclude offline fitting costs.

*Controlled final evaluation covers GCN only: GAT and GraphSAGE E4 results are absent. Original prediction tensors for four of the six GCN models are not fully locally verified. E5 natural-shift results cover all three architectures. See [evidence and limitations](reports/experiment-summary-20261112/README.md).*

## Experiment Coverage

| Experiment | Question | Status |
|---|---|---|
| E1 | Which inputs should condition the generator? | GCN input ablation complete. |
| E2 | Head or LoRA updates? | Design selection complete for all architectures. |
| E3 | Direct network, teacher network or diffusion? | Controlled validation comparison complete. |
| E4 | Generalization to held-out controlled shifts? | GCN scores available; verification limits above. GAT/GraphSAGE absent. |
| E5 | Generalization under temporal and country shifts? | Final evaluation complete for all architectures. |
| E6 | Can supervised adaptation recover accuracy? | Separate labeled diagnostic complete. |

Each source protocol contains 30 models per architecture, partitioned into 18 generator-training, six validation and six final models by supervision group. E6 uses labels for diagnosis and is reported separately from unlabeled adaptation.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ogb,perf]"
python -m pytest
```

Experiment configurations are in [`configs/`](configs/); training, evaluation and independent replay commands are in [`examples/`](examples/). Use a fresh output directory for each run. Large datasets, checkpoints and prediction tensors are stored outside Git and must be restored before replaying archived experiments.

## Running Experiments

Inspect the available options before preparing a bank or launching a study:

```bash
python examples/prepare_formal_bank.py --help
python examples/run_arxiv_experiments.py --help
python examples/evaluate_real_generalization.py --help
```

The formal protocols are defined in [`arxiv_formal_v1.json`](configs/arxiv_formal_v1.json), [`arxiv_temporal_formal_v1.json`](configs/arxiv_temporal_formal_v1.json) and [`mag_country_formal_v1.json`](configs/mag_country_formal_v1.json). Fit generators and select designs on permitted practice/validation data before running the separately gated final evaluation.

## Repository Layout

| Directory | Contents |
|---|---|
| [`src/amortized_gnn_tta/`](src/amortized_gnn_tta/) | Data interfaces, GNNs, adaptation methods and evaluation. |
| [`configs/`](configs/) | Versioned dataset protocols and experiment settings. |
| [`examples/`](examples/) | Source-bank, training, benchmarking and replay entry points. |
| [`tests/`](tests/) | Software regression and scientific-boundary checks. |
| [`reports/`](reports/) | Compact numerical evidence and verification summaries. |
| [`docs/`](docs/) | Design decisions and research engineering notes. |

## Documentation

- [Results, validation summaries and timing ratios](reports/experiment-summary-20261112/README.md)
- [Design decisions](docs/DECISIONS.md)
- [Research engineering notes](docs/learning/RESEARCH_ENGINEERING_NOTES.md)
- [Earlier milestones and development evidence](docs/PROJECT_STATE.md)
