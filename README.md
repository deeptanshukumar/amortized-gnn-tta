# Amortized GNN Test-Time Adaptation

An Advanced Foundations of Machine Learning project at PES University investigating whether learned generators can adapt graph neural networks faster than iterative test-time optimization.

The code supports **GCN, GAT and GraphSAGE**, with direct hypernetworks, teacher-trained hypernetworks and diffusion generators. Generators predict head or LoRA updates from unlabeled target inputs. Permitted practice labels are used offline; final target labels are reserved for evaluation.

## Methods

| Method | Offline training | Target-time work |
|---|---|---|
| Direct hypernetwork | Practice-label cross-entropy through the GNN, with update regularization. | Predict and apply one parameter update. |
| Teacher hypernetwork | Regress supervised practice-teacher updates. | Predict and apply one parameter update. |
| Diffusion | Denoise normalized practice-teacher updates. | Generate five candidates with 10 DDIM steps and select by unlabeled prediction agreement. |
| Iterative IM | Reuse the pretrained source model. | Optimize an information-maximization objective; the final PDF compares 1, 5, 20 and 40 steps. |

The final controlled design selects rank-4 LoRA for GCN and head updates for GAT/GraphSAGE. Frozen, constant-teacher, practice-fixed and feature-only GTrans controls are included in the numerical exports.

## Datasets

| Dataset | Use |
|---|---|
| [OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) | Controlled graph corruptions and temporal distribution shifts. |
| [Pair-Align MAG country graphs](https://zenodo.org/records/10681285) | US source; CN/DE practice; FR validation; JP/RU final evaluation. Derived from [OGBN-MAG](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag). |
| [Synthetic CSBM](https://github.com/baowenxuan/Matcha/blob/f0e7104a84d118a8dabb7ad210915c43722b94af/src/dataset/CSBM.py) | Earlier development experiments, generated locally using an independent implementation of Matcha's released distribution. |

Controlled and temporal Arxiv experiments use different protocols on the same underlying dataset. Natural-shift generators are fitted separately for each protocol.

## Final submission results

**The final result source is Arvind's `AFML Research Project.pdf`, as selected by the project owner.** The tables below supersede the November 12 campaign summary. E4 values come from PDF page 21 and E5 final-test values from page 22. These are PDF-reported results; the newer run artifacts have not been independently replayed here. The older reports remain historical evidence, not the final submission numbers.

Frozen accuracy is a percentage. Gains and the reported ± dispersion are **percentage points**, not relative percentages. The PDF's dispersion values are reproduced without assuming an independently verified statistical definition.

| Evaluation | Model | Frozen accuracy % | Hypernetwork gain pp | Diffusion gain pp |
|---|---|---:|---:|---:|
| Controlled Arxiv — all cases | GCN | 60.198 | +0.419 ± 0.059 | +0.245 ± 0.279 |
| Controlled Arxiv — all cases | GraphSAGE | 58.499 | +0.198 ± 0.053 | +0.175 ± 0.180 |
| Controlled Arxiv — all cases | GAT | 60.845 | −0.018 ± 0.035 | −0.335 ± 0.135 |
| Temporal Arxiv — final test | GCN | 59.406 | −0.535 ± 2.150 | −21.444 ± 1.069 |
| Temporal Arxiv — final test | GraphSAGE | 59.965 | +1.684 ± 0.126 | −34.946 ± 4.190 |
| Temporal Arxiv — final test | GAT | 62.196 | +0.922 ± 0.159 | −2.575 ± 0.765 |
| MAG country — final test | GCN | 65.195 | +2.142 ± 0.617 | +3.626 ± 2.978 |
| MAG country — final test | GraphSAGE | 65.199 | +1.282 ± 1.126 | +2.453 ± 3.996 |
| MAG country — final test | GAT | 62.834 | +1.671 ± 1.217 | +4.602 ± 2.674 |

Hypernetworks give modest gains on country graphs and on temporal GAT/GraphSAGE, with a small temporal GCN regression. Diffusion improves the reported country means but fails severely on temporal GCN and GraphSAGE. Controlled gains are small and architecture dependent. The earlier teacher-hypernetwork and practice-fixed scores are not mixed into this revised table because the PDF's final E4/E5 tables do not report those columns.

### Efficiency study

PDF pages 23–25 report the following speedup ranges across architectures, relative to **40-step IM-LoRA**. These replace the earlier README's 30-step-baseline ratios.

| Protocol | Hypernetwork speedup | Diffusion speedup |
|---|---:|---:|
| Controlled Arxiv | 7.4–16.3× | 4.3–6.1× |
| Temporal Arxiv | 8.8–27.3× | 4.6–8.1× |
| MAG countries | 12.3–24.0× | 2.9–3.9× |

E6 has its own accuracy-gain tables, which differ from E4/E5. Do not substitute E6 gains into the final-test table or claim that its speedups and E5 accuracies were measured on identical cases without the corresponding manifests. Speed must be interpreted alongside accuracy, particularly for temporal diffusion. The PDF alone does not establish a complete timing-boundary or offline-cost audit.

## Experiment Coverage

| Experiment | Question | Final submission source |
|---|---|---|
| E1 | Which inputs should condition the generator? | PDF pages 16–17 report the input comparison. |
| E2 | Head or LoRA updates? | PDF pages 18–19 report the update-type comparison. |
| E3 | Hypernetwork or diffusion? | PDF page 20 reports the diffusion comparison. |
| E4 | Generalization to held-out controlled shifts? | PDF page 21 reports GCN, GAT and GraphSAGE, including all cases and shift subgroups. |
| E5 | Generalization under temporal and country shifts? | PDF page 22 reports validation and final-test results for all three architectures. |
| E6 | Failure analysis and efficiency? | PDF pages 23–25 report iterative-step comparisons, gains and runtimes. |

The older local campaign's missing GAT/GraphSAGE E4 artifacts remain an archival verification limitation; they do not mean the current PDF lacks those results. Reporting from the PDF does not establish independent verification of its newer runs.

Each original source protocol contains 30 models per architecture, partitioned into 18 generator-training, six validation and six final models by supervision group. These describe the original bank protocol; exact newer-run configurations should be checked against Arvind's manifests before reproducing the final submission.

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

- Final submission results: Arvind's `AFML Research Project.pdf`, pages 16–25 (source document supplied outside Git).
- [Historical November 12 results, validation summaries and timing ratios](reports/experiment-summary-20261112/README.md)
- [Design decisions](docs/DECISIONS.md)
- [Research engineering notes](docs/learning/RESEARCH_ENGINEERING_NOTES.md)
- [Earlier milestones and development evidence](docs/PROJECT_STATE.md)
