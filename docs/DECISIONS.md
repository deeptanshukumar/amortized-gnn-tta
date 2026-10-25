# Decisions and dataset investigation

Investigation began 2026-10-23; data and source-GCN milestones implemented 2026-10-25. Follow-up authority, decisions, and evidence are recorded below.

## Owner-approved decisions and remaining choices

- **Accepted:** the common interface described below, including separate labels, named native splits, and explicit environment selections.
- **Accepted for CSBM development:** configurable stratified source splits, initially 60/20/20, and independent source/target graphs. Preserve existing official splits if real benchmarks are considered in a separately approved milestone.
- **Explicit owner clarification:** use CSBM now; no other datasets in the current implementation scope. The assistant previously recorded a real-benchmark-first preference incorrectly. That record is corrected here, not treated as an approved historical decision.
- **Delegated on 2026-10-25:** owner authorized choosing a defensible recipe after studying the local Matcha paper, implementing the CSBM milestone, and necessary project-local setup. Decision D1 below records the selected convention; this is a delegated engineering/methodological choice, not a claim that the owner separately approved every formula.
- **Still future work:** final scientific shift protocol, generator meta-environment allocation, weight-bank grouping, and backend expansion. Dataset choice is settled; do not ask for a real dataset selection now.
- **Accepted workflow addition:** maintain `docs/learning/RESEARCH_ENGINEERING_NOTES.md` and actively explain/check understanding of difficult concepts. Notes are preparation, not evidence that the owner has mastered a topic.

## Implemented decision D1 — initial CSBM protocol (2026-10-25)

Authority: owner delegated routine choices and implementation after examining Matcha. Selected **csbm-matcha-code-v1**; independent implementation from the released sampling distribution, not copied code or a full paper-result reproduction.

- **Edges:** explicit p_in/p_out. from_matcha maps p_in=degree_scale*h/N and p_out=degree_scale*(1-h)/N. Exact expected degree is (N/2−1)*p_in + (N/2)*p_out; never label degree_scale as average degree.
- **Features:** balanced positive-then-negative classes, fixed shared direction, x_i=(class_mean[y_i]+noise_scale*z_i)/sqrt(F), independent standard Gaussian coordinates. Defaults −0.03/+0.03, noise_scale=1; the example target translates both centers by 0.02.
- **Why released code:** concrete audited reference and easy comparison with its graph distributions. The paper's approximate mean-degree interpretation and theoretical unit-covariance feature model do not exactly match the implementation. Prefer explicit compatibility to a silent correction. Paper-style degree or original latent-direction CSBM remain valid alternatives, but changing to them would require a separately named protocol.
- **Differences:** new tagged PCG64 streams, vectorized row sampling, no seeded-artifact equivalence, approved stratified 60/20/20 source label splits instead of full-source-label ERM. Dimensions are configurable; small and 5000×2000 graphs were executed.
- **Visibility:** retain each full graph; source training selects only training labels. Target graphs are separate and become visible only to the target phase. Full unlabeled target input is consistent with the graph-TTA setting described in Matcha §3.1. Validation/target labels stay evaluation-only.
- **Environment identity:** independent development draws with distinct edge and feature seeds. Reject seed reuse in this loader rather than silently introduce paired graph lineages. Same class schema/feature dimension across a bundle.
- **Input boundary:** label store, generation parameters, and label-derived diagnostics stay outside model inputs. Target training-label access is rejected. This prevents accidental leakage; it is not a security boundary against malicious bundle access.
- **Complexity:** O(N²) pair scans with O(N+E) auxiliary connectivity storage and O(NF) features. No dense adjacency, persistent cache, or speculative registry. Model normalization stays in PyG layers.
- **Setup:** torch/NumPy reused through a project venv with system-site-packages; PyG/pytest installed only there. Full environment pip check sees unrelated inherited conflicts. Project direct dependency checks and CPU runtime tests passed; clean-environment and non-CPU backends remain unverified.
- **Limits:** pair-scanning is not a scalable million-node sampler. Independent distributions do not establish fixed-realization interventions. Thirty-model bank, adaptation, and performance claims are not part of this milestone.

## Implemented decision D2 — small source-GCN baseline (2026-10-25)

Authority: owner explicitly requested a small source GCN training/evaluation pipeline using the verified loader. No new dependencies or dataset changes.

- **Model:** two GCNConv layers, 32 hidden channels, ReLU and 0.5 hidden dropout; raw class logits, no batch normalization, no graph-normalization cache. This is a conventional small baseline, not a Matcha/GPRGNN or original-paper hyperparameter reproduction. The output layer is a second graph convolution; a future head-only adaptation definition must specify what it updates rather than assume a separate linear classifier exists.
- **Training:** model-neutral factory/loop, full source graph, CE on training labels only, Adam lr=0.01 and weight_decay=5e-4 on all trainable parameters. Model seed initializes Python/NumPy/PyTorch before constructing the model and controls dropout; loader seeds remain independent.
- **Selection:** minimum source-validation cross-entropy after each step; strict improvement resets patience, ties keep the earliest checkpoint. Defaults cap=200/patience=30. Copy selected state tensors, then restore before final scoring. No source-test or target metrics are computed inside fitting.
- **Boundary:** SourceTrainingData contains only the label-free source graph and train/val supervision. Extraction never traverses target records or reads source-test labels. Tests flip held-out labels and confirm identical histories/weights/selection.
- **Evaluation:** eval-mode label-free prediction first; explicit labels only for offline cross-entropy/accuracy. Training flags are restored, inputs are cloned, model weights remain unchanged. Current runner is for unweighted graphs; do not silently ignore optional edge data.
- **Checkpoints:** CPU state_dict plus model spec, training config, chosen epoch/history, source provenance/train+val indices, class schema, versions and training device. Restore via caller-supplied model construction using weights_only=True and CPU map; fail rather than overwrite existing files. Exact optimizer/RNG resumption is outside scope.
- **Devices:** auto CUDA-if-available else CPU; MPS explicit opt-in, no silent backend fallback. Only CPU executed. Cross-backend numerical equivalence is not claimed.
- **Verification:** 44 tests passed; short and paper-sized source runs completed with reload checks. One seeded result is development evidence only. No adaptation, weight bank, architecture search, or target-driven model selection.

References inspected: existing original GCN TensorFlow training/masked metrics and OGB's PyG model/training example in separate reference checkouts; [PyG GCNConv](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GCNConv.html) and [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). We do not adopt OGB's fixed-graph cache or symmetrization because our graphs change and our loader already defines edge semantics. Code is independently implemented.

## Implemented decision D3 — fixed-holdout source-label subsets (2026-10-25)

Authority: after the remaining weight-bank tasks were explained, the owner requested implementation one step at a time. This is the first preparation step, not approval of every later research protocol. The owner was asked about a 50% development label budget; the code requires an explicit fraction and does not assume that this is the final scientific budget.

- **Fixed base:** use the already generated source graph and native train/val/test split. Sample only from its permitted training pool; keep graph/features and validation/test membership fixed. Unselected training nodes retain their pool membership but contribute no supervised training loss for that model.
- **Stratification:** select floor(label_fraction × class pool size) nodes without replacement per class, using only permitted source training labels. Reject a fraction that leaves any class empty. The class fraction preserves approximate pool proportions rather than oversampling minority classes. Exactly balanced CSBM pools give exactly balanced subsets.
- **Randomness:** explicit distinct subset seeds, local PCG64 streams from SeedSequence([subset_seed, class_id]), canonical sorted pool IDs. No consumption of global/model RNG states and no native split regeneration. Seeds 101–105 are example settings, not a paper convention.
- **Distinctness:** overlap between subsets is allowed and expected. Reject identical selections or an impossible number of distinct subsets; do not silently retry seeds. Different labels/seeds do not guarantee useful model diversity, which requires later measurements.
- **Budget:** the API and CLI require label_fraction. Executed development examples use 0.5: 300 of 600 native training labels at N=1000, and 1,500 of 3,000 at N=5000. This is not comparable without qualification to the earlier full-pool source-only baseline. Final research budget remains open.
- **Persistence:** JSON stores exact memberships, seeds, per-class counts, requested fraction/rounding, source input fingerprint, native label-pool IDs/label hash, validation IDs, class schema, provenance and versions. Loading validates identity and membership, then uses stored IDs without resampling. JSON-compatible provenance is constructed upfront, avoiding tuple/list differences on reload. Never overwrite an existing manifest.
- **Integration:** subset views preserve source validation supervision and share the source graph read-only; fit_source clones that graph for execution. Existing checkpoint metadata includes selected train IDs and subset/manifest identity. No target graph/labels or source-test labels are needed for construction.
- **Verification:** 70 CPU tests passed, including 26 new cases, and small/paper-sized subset examples completed with reload verification. One test runs actual source fitting and checkpoint restoration from a subset. No new dependency, bank training, 18/6/6 allocation, or model-diversity conclusion.

Reference investigation: inspected the existing GPRGNN per-class random split utility and the separate PyG checkout's RandomNodeSplit implementation. Both illustrate class-aware splitting but rebuild validation/test masks in their relevant modes. We instead sample inside the existing permitted training pool. This is independently implemented with NumPy's maintained RNG API, not copied benchmark code or a claim to reproduce its label budget. Primary sources: [PyG dataset splitting](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/dataset_splitting.html), [NumPy independent random streams](https://numpy.org/doc/stable/reference/random/parallel.html).

## Implemented decision D4 — sequential source-only bank runner (2026-10-25)

Authority: owner requested the foundation commit and continuation to the next step after five source-label subsets. Foundation is committed locally as 30af4f9. The runner was initially uncommitted; the owner subsequently approved it as part of the separate Phase 1 completion commit.

- **Scope:** model-neutral sequential runner over the Cartesian product of saved label subsets and explicit model seeds. The GCN example enforces five subsets and six seeds, reconstructing only the CSBM source and validating graph/pool identity before training. No new dependency or change to data/training formulas.
- **Seeds:** defaults 31–36, reused across subsets. This deliberately pairs initialization/training streams for supervision comparisons; thirty model instances are not thirty IID independent initializations. Each job constructs a fresh model and optimizer, with its actual seed recorded. Tests verify identical weights/histories when job order changes.
- **Supervision/selection:** the runner receives only SourceTrainingData. Each job selects minimum source-validation CE using its existing early-stopping rule, then scores selected source train and source validation rows. No source-test/target scores, bank-wide score filtering, or target-driven selection.
- **Evidence:** each model gets a CPU inference checkpoint, JSON history/metrics, subset identity, and CPU reload prediction check (rtol=1e-4, atol=1e-5). The bank stores copied subsets, all planned IDs/pairs, configs, versions, device/threads, hashes and progress. The example records Git revision plus dirty status honestly; an uncommitted runner cannot be reconstructed from the foundation commit alone.
- **Failure policy:** existing output directories are rejected. Update only the new run's own bank.json using a same-directory temporary file and rename. A failed job preserves prior artifacts and marks failure when possible; abrupt termination can leave running status. No silent skipping, resumption, deletion, or overwriting of earlier runs. Do not treat a partial bank as complete.
- **Memory:** sequential fitting/reloading, shared read-only source view, execution clones, released per-job model references. No speculative parallel scheduling or new graph cache.
- **Verification:** 80 CPU tests passed, including ten new bank cases for membership/provenance/hashes, source-only metrics, job-order reproducibility, injected partial failure, invalid inputs, and no-overwrite behavior. Both small (1000×64, cap/patience 100/20) and paper-sized (5000×2000, 20/10) development banks completed all thirty jobs with CPU reloads verified; checkpoint/metric hashes were checked after execution. All paper-sized models selected epoch 20, so convergence is unestablished. Final generator grouping and diversity claims are not implemented.

Reference basis: the existing separate OGB PyG example resets model weights and constructs a fresh optimizer per run; we instead construct fresh seeded models through the shared factory. Its per-epoch source-test scoring and fixed-graph assumptions are not adopted. [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) supports state-dict restoration and distinguishes inference checkpoints from optimizer-containing resumption snapshots. No upstream code was copied.

## Implemented decision D5 — GCN/CSBM Phase 1 closure (2026-10-25)

Authority: owner explicitly requested the remaining task list and completion of Phase 1. The task list and protocol/partition recommendation were stated before implementation. Scope is roadmap steps 1–3 for GCN/CSBM, not teammates' architectures, adaptation, or final-paper completion. After reviewing the results, the owner explicitly approved the Phase 1 completion commit. No push or future commit authority is inferred.

- **Reference protocol:** configs/phase1_csbm_gcn.json freezes N=5000/F=2000, Matcha-code CSBM convention, source seeds 11/12/13, native 60/20/20, 50% of each training class, subset seeds 101–105, paired model seeds 31–36, H32/dropout0.5, Adam0.01/0.0005, cap200/patience30. The previously open budget is now fixed for this engineering phase; final-paper protocols/holdouts remain future work.
- **Artifact loading:** complete bank status and exact Cartesian membership, contained relative paths, artifact hashes, checkpoint tensor schema/finiteness, model spec/config, source/subset/train/validation identity, histories/earliest minimum selection, and reproduced source-validation metrics must agree. Loading is CPU/weights_only. Invalid/partial/mismatched banks fail visibly; no source-test/target labels are needed for the loader.
- **Partition choice:** hold out whole label-subset groups as 3/1/1 (18/6/6 models). Deterministic PCG64 seed401 assigns meta-train subsets [1,0,3], validation [4], test [2], without inspecting scores. Alternative random model-instance splitting can put the identical labeled selection into multiple roles. The chosen split tests new label selections on the same source graph, not new graphs or unseen initialization seeds. Individual node labels still overlap across those selections; do not claim disjoint labeled-data domains. Saved partitions bind the exact bank.json hash and require complete, disjoint model/group coverage.
- **Diversity probe:** all native source-training nodes in canonical ID order, labels unused. Measure raw/symmetric-relative weight L2, raw weight cosine, argmax disagreement, class-probability RMSE, and biased centered linear CKA on first-layer post-ReLU/pre-dropout representations. GCN.encode exposes that existing calculation without changing weights, state keys, or forward behavior (tested against the original formula with identical dropout RNG). The core accepts a representation callback for other architectures.
- **CKA basis:** independently implemented PyTorch feature-space formula from Kornblith et al. (ICML2019). Inspected the official Google Research README and Demo.ipynb, including feature_space_linear_cka, Gram comparison, biased/debiased distinction, and Apache2 terms. No notebook source was copied or installed; feature-space H×H products avoid dense N×N storage. Constant centered representations give undefined/None explicitly. Tests independently compare with centered Gram arithmetic and column permutation/scaling invariances.
- **Readiness:** require complete artifacts, observed probability variation, no exact parameter duplicates, and defined representation comparisons for the phase review. These are engineering diagnostics, not a universal scientific sufficiency threshold. Report pair categories (same subset, same seed across subsets, different both) and expose observed ranges. Useful adaptation diversity remains an experimental question; never insert arbitrary weight noise just to increase distance.
- **Training/evaluation:** preserved the earlier 20-epoch smoke bank and trained a fresh 200/30 bank. All thirty selected epochs24–98, stopping before the cap. Freeze/save the model partition before explicit source-test label access. Final source-only evaluation uses the native test complement and reports per-subset/per-seed descriptive spread, not thirty IID trials, confidence intervals, target selection, or adaptation results. This source graph/test split has earlier viewed development scores, so it is not a sealed final benchmark.
- **Provenance:** report configuration/partition/bank hashes and preserve an analysis-code snapshot separately from the training Git revision/dirty flag. Training took place in an uncommitted working tree; the analysis snapshot is not retroactively asserted to be the training implementation. No paper-level frozen-code claim.
- **Verification:** full-bank execution, complete artifact audit, 18/6/6 coverage, 435 diagnostic pairs, final source scoring, and 104 CPU tests. Test cases include resealed corrupted checkpoints, weight/metadata/selection mismatches, model-partition leakage, label-free diagnostic invariance, hidden permutation symmetry, exact forward compatibility, and the actual finalization CLI.

Primary CKA source: [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). Separate reference checkout: references/google-research at ba728e288c51126eac25e316b11dfee94d68c301, sparse representation_similarity/{README.md,Demo.ipynb} plus root LICENSE. Initial sandbox DNS failure during sparse fetching was resolved with an approved retry; the upstream worktree remains unmodified. This is a reference only, not a project dependency.

## Implemented decision D6 — shared GAT and GraphSAGE models (2026-10-25)

Authority: after the Phase 1 GCN commits were explicitly pushed, the owner requested GAT and GraphSAGE implementations too. After reviewing completed verification, the owner separately authorized this extension's commit and push. This expands the shared coding scope without silently changing team ownership. Later Git actions need separate authorization; CSBM and all label/visibility rules remain unchanged.

- **Interfaces:** all three models expose forward(x, edge_index) → raw [N,C] logits and encode → first-layer post-activation representations before hidden-feature dropout. Share dimension/dropout validation and a small explicit three-model factory. Existing two_layer_gcn tags, four config fields, state keys, direct GCNConfig imports and frozen GCN config remain compatible. The real historical 30-model GCN bank passed the new factory/audit, with unchanged 88.37% mean source accuracy.
- **GAT:** PyG GATConv, two layers, hidden ELU, default four heads with eight channels each for total width32, one output head/concat=False, layer-owned self-loops, slope0.2, bias, no added residual/input-dropout module. Feature dropout0.5 and attention-coefficient dropout0 are separate; the latter is configurable. Total hidden width must divide by the number of heads. This is classic GAT, not GATv2 or a reproduction of the paper's head counts/dropout/tuning.
- **GraphSAGE:** PyG SAGEConv(mean), separate root and neighbor transforms, ReLU, hidden dropout0.5, no L2 normalization, neighbor projection, added self-loops or sampled minibatches. Full-neighborhood/full-graph execution preserves the current protocol. This is the explicit PyG mean variant, not the original TensorFlow sampled/concatenated/normalized supervised pipeline. Larger graph sampling remains future work.
- **Fairness/parameterization:** width32, data/subsets, seed lists, optimizer and source-validation selection rules match. Model definitions retain their architecture-specific activations/aggregation. Actual parameter counts at F2000/H32/C2: GCN64,098, GAT64,166, GraphSAGE128,162. Same seed integers do not produce matched initial tensors across different architectures. These are initial defaults, not capacity-matched or separately tuned benchmark comparisons.
- **Integration:** --architecture gcn/gat/graphsage selects the source/bank model; --heads/--attention-dropout are GAT-only and rejected elsewhere. Checkpoint restoration uses the explicit architecture/config factory. The common finalization command accepts the corresponding phase1-csbm-{architecture}-v1 reference file, keeps whole-subset18/6/6 grouping, and rejects wrong reference/model combinations. GAT representations are post-ELU; the others are post-ReLU. No changes to the trainer, evaluator, data generator or bank selection math.
- **Verification:** 125 CPU tests passed (21 new model cases). Checks include manually calculated directed-edge uniform GAT attention/self-loops, GraphSAGE neighbor mean plus root term, total head width, configurable coefficient dropout/eval behavior, finite gradients, isolated/empty-edge graphs, node-permutation equivariance, source learning, held-out-label invariance, CPU checkpoint roundtrips, and actual architecture-specific bank/audit CLI execution. Small and paper-sized source training/reload runs completed for both new architectures. Both full 30-model reference banks and their audits completed with 18/6/6 groups, 435 pairs, no selected cap checkpoint and all CPU reloads verified. Initial descriptive source-test means: GAT87.91%, GraphSAGE88.69%; these are not a capacity-matched/tuned ranking or adaptation results.
- **Future adaptation boundary:** the output classifier is the second graph layer for each model, not a common separate linear head. GAT adds attention vectors; GraphSAGE has two output transforms. Head-only/low-rank scope must be explicitly defined in the later adaptation milestone. Reusing data/training code does not make weight tensors or generators interchangeable across architectures.

Reference investigation: separate official clones GAT at 5af87e7fce2b90ae1cbd621cd58059036a3c7436 and GraphSAGE at a0fdef95dca7b456dab01cb35034717c8b6dd017. Inspected GAT models/gat.py and utils/layers.py (ELU, head concatenation/output averaging and coefficient/feature dropout), and GraphSAGE graphsage/{aggregators,supervised_models,neigh_samplers}.py (root/neighbor maps, sampled supports, output normalization). Both MIT licenses inspected; neither TensorFlow environment was installed or source copied. Existing PyG reference GATConv/SAGEConv source and maintained APIs govern runtime semantics.

Primary sources: [GAT original repository](https://github.com/PetarV-/GAT), [GAT project explanation](https://petar-v.com/GAT/), [GraphSAGE original repository](https://github.com/williamleif/GraphSAGE), [GraphSAGE paper](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html), [PyG GATConv](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GATConv.html), and [PyG SAGEConv](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.SAGEConv.html).

## Established instructions

The owner approved investigation, separate reference clones, and Markdown project context in the initial request. Loader implementation and dependency changes require discussion and approval. Foundation/completion commits 30af4f9 and 3c7976e were explicitly approved and then pushed after a separate explicit request. The owner subsequently authorized committing and pushing the verified architecture extension. These approvals do not cover later changes. These are owner instructions, not new research decisions.

The detailed plan currently focuses on node classification, architecture-specific generators, labeled offline meta-training, and unlabeled deployment. It requires both controlled shifts and at least one natural-shift benchmark. The one-pager names GOOD-Cora/GOOD-Twitch/OGBN-Arxiv; the detailed plan suggests CSBM/Twitch-E/OGB-Arxiv. These are candidate lists, not equivalent dataset definitions or a finalized suite.

## Verified dataset distinctions

| Candidate | Verified identity and protocol | Implication |
| --- | --- | --- |
| CSBM | A family of stochastic models, not one downloadable benchmark with universal splits. Original CSBM, GPRGNN, and Matcha use different parameter/feature constructions. Matcha uses 5,000 nodes, 2,000 features, two balanced classes, controlled class-center/degree/homophily changes. | Specify equations, protocol version, dimensions, seeds, and split construction. A tiny development graph is not a paper-scale reproduction. |
| Twitch-E via EERM | Seven language graphs (`DE`, `ENGB`, `ES`, `FR`, `PTBR`, `RU`, `TW`), common 3,170 binary features. EERM uses DE for training, ENGB for validation, and the other five for testing; its Twitch metric is ROC-AUC. | Useful natural cross-domain candidate. This is an EERM experimental protocol, not a universal official Twitch split. Confirm raw artifact identity, node alignment, and edge representation. |
| PyG `Twitch` | Six named graphs (`DE`, `EN`, `ES`, `FR`, `PT`, `RU`), 128-dimensional embedded features; loader returns a PyG graph without train/validation/test masks. | Not a drop-in replacement for Twitch-E. Matching node counts do not establish equivalence of features or edges. |
| GOOD-Twitch | Code builds a disjoint union of the six PyG Twitch graphs and applies language-domain benchmark splits. Covariate/concept/no-shift variants; ROC-AUC; native binary targets shaped `[N,1]` as floats. | Preserve GOOD's five masks and domain information. The native binary task should not be assigned accuracy as its sole benchmark metric. |
| GOOD-Cora | Based on `CitationFull(name="Cora")`, not Planetoid Cora; domain options degree/word; native OOD masks; accuracy metric. | Do not substitute standard small Cora or regenerate GOOD splits. Covariate configuration fits the current candidate scope better than concept shift. |
| Standard `ogbn-arxiv` | Directed citation graph, 169,343 nodes, 1,166,243 directed edges, 128 features, 40 classes. Official time split: through 2017 train, 2018 validation, 2019 onward test; accuracy. Official PyG loader supplies `get_idx_split()`. | Wrap the OGB API and evaluator. Preserve node years and split indices. Full-graph transductive access and chronological graph access are different choices. |
| EERM/Matcha Arxiv | Constructs cumulative subgraphs by upper year bound; lower bound defines an evaluation mask, not node exclusion. Matcha code uses bounds `[1950,2011]` and `[2014,2020]` for source/target and adds label-aware edge deletion. | This is not the official OGB time split or three disjoint year-only graphs. Retain original node IDs and distinct protocol names. GOOD-Arxiv is another separate benchmark construction. |

Primary evidence: [Matcha dataset code](https://github.com/baowenxuan/Matcha/tree/master/src/dataset), [Matcha paper, Appendix D.1](https://arxiv.org/abs/2410.06976), [EERM multigraph pipeline](https://github.com/qitianwu/GraphOOD-EERM/blob/main/multigraph/main.py), [PyG Twitch source](https://pytorch-geometric.readthedocs.io/en/stable/_modules/torch_geometric/datasets/twitch.html), [GOOD-Twitch source](https://github.com/divelab/GOOD/blob/ec2f3fd13e69f1324f507e31ce3ce7252e9fb58e/GOOD/data/good_datasets/good_twitch.py), [GOOD-Cora source](https://github.com/divelab/GOOD/blob/ec2f3fd13e69f1324f507e31ce3ce7252e9fb58e/GOOD/data/good_datasets/good_cora.py), [OGB node datasets](https://ogb.stanford.edu/docs/nodeprop/). Local pinned snapshots are listed below; branch URLs can change.

## Research risks requiring explicit treatment

### CSBM degree and feature conventions

Verified: Matcha's `generate_csbm` sets `c_in=d*h`, `c_out=d*(1-h)` and samples edges with probabilities `c_in/n` or `c_out/n`. For even `n` and balanced classes, the expected degree is `(n/2-1)*d*h/n + (n/2)*d*(1-h)/n = d/2 - d*h/n`. Thus its code parameter `d` does not give the average degree described by the paper. This is a derivation from inspected source, not a measured experiment. Matcha's feature axis is fixed; GPRGNN's original-style generator samples a Gaussian latent direction and uses a different feature scaling.

Implemented as D1 below: explicit within/between probabilities with a named released-code conversion and fixed-direction Gaussian features. Do not silently multiply probabilities by two. A different expected-degree configuration must be a separately versioned methodology. No upstream source was copied.

Use local RNGs, separate graph-generation and split seeds, and stable label/feature semantics across environments. Independently sampled environment graphs should not accidentally rotate feature class meanings. Define whether controlled pairs share base features/noise or use independent realizations; shared lineage must not cross environment holdouts. Never assume matching node indices across independent graphs imply shared node identity.

The plan describes attribute-only shifts as preserving structure and structural-only shifts as preserving features. Independent source/target CSBM draws do not hold those realizations fixed, even when the unchanged component has the same distribution. For future CSBM development, independent graphs exercise the loader contract only. Before shift experiments, approve either paired interventions (reuse edges for attribute-only changes and features for structural-only changes) or an explicitly named independent-draw distributional protocol. Use separate edge/feature RNG streams where component independence is required.

### Label-aware benchmark construction versus unlabeled deployment

Verified: Matcha's real-graph `match_edge_homophily` consumes ground-truth labels to delete selected edges. This is a constructed shift on real data, not purely natural transfer. It must not be an automatic loader transform or label-free adaptation operation. True homophily requires labels; it cannot condition the deployment generator. Pseudo-label agreement can be studied separately and must be named accordingly.

Current scope is CSBM only. Natural-shift data remain part of the broader roadmap, not an approved current dataset task. If reproducing Matcha real-graph corruption in a later milestone, isolate it as an offline, explicitly label-aware benchmark-construction step, freeze artifacts before experiments, and keep true labels out of runtime inputs. It is not itself a supervised adaptation oracle; benchmark construction and model optimization use labels in different ways and must be reported separately.

### Graph visibility, holdouts, and model-bank splits

Native node masks do not determine which nodes/edges a model may observe. GOOD is a node-OOD benchmark with masks on a retained graph; cropping masks into induced subgraphs changes message passing. Standard OGB node splits likewise do not mean three separate graphs. Preserve connectivity at load time; define visibility explicitly at experiment setup.

The plan's five labeled-source splits conflict with blindly replacing official train masks. Recommendation (pending): keep native train/validation/test indices immutable, and store any approved source-label subsample separately within native training nodes. Decide the labeled fraction before creating the weight bank; using all training nodes cannot produce five different labeled subsets.

Model holdouts and environment holdouts are independent. Decide whether model-bank partitions test new initializations only or entirely new labeled splits. Hold out related graph realizations/base graphs together where needed. Never train the generator using official validation/test labels merely because they are available in the dataset bundle. Validation labels may support the approved selection protocol, not meta-training gradients. For same-graph corruption experiments, exclude held-out evaluation-node labels from meta-training even if the features/edges are visible.

The plan correctly requires a baseline trained on the same labeled shifted environments. Keep that control and compare generated versus optimized updates with equal parameterization. GraphTTA/GAPGC's inspected paper experiments concern graph-level molecular prediction; it is context, not a ready-made node-classification protocol. Matcha's main backbone is GPRGNN; its hop-parameter adapter is not directly equivalent to vanilla GCN low-rank weight adaptation. Novelty and backbone applicability need a later literature review, not an assumption from the one-pager.

Verified from Matcha section 3.2: its simplified analysis uses `Z = X + gamma D^(-1) A X`. Standard GCN/PyG GCNConv uses symmetric degree normalization with self-connections and a learned feature transformation. These operators are not identical; inspecting Matcha's data recipe does not commit us to its architecture or establish its theoretical guarantees for our proposed GCN weight updates. The handbook explains this distinction and the degree discrepancy without claiming a reproduced experiment.

### License and compatibility findings

- PyG and OGB code: MIT licenses inspected. OGB's site lists ogbn-arxiv data as ODC-BY. Library and dataset licenses must be tracked separately.
- GOOD code: GPLv3 inspected. Its README states datasets are MIT under a separately linked license; that linked artifact has not been independently inspected. Do not infer that code can be vendored under MIT.
- Twitch dataset author's repository and Non-Homophily-Benchmarks: MIT license files inspected. SNAP provides raw Twitch distribution and citation. Exact chosen archive/mirror terms and checksums still need verification before ingestion/redistribution.
- No repository-level license file found in the inspected Matcha, GPRGNN, or EERM checkouts. GPRGNN's `src/cSBM_dataset.py` has a file-level MIT statement. Do not assume that statement licenses every derived implementation elsewhere. No source was copied into this project.
- GOOD imports a broad package ecosystem and pins old dependencies; avoid installing its entire training stack for the first milestone. Its dataset classes also use legacy `torch.load` calls without explicit `weights_only`. PyTorch changed that default in 2.6; modern loading needs an actual compatibility check. No runtime failure was reproduced here.
- Minimal PyG usage does not require all compiled extension packages. Start with the minimal dependency set and add extensions only for demonstrated needs. See [official installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and [PyTorch 2.6 release](https://github.com/pytorch/pytorch/releases/tag/v2.6.0).

## Implemented milestone P1 — common interface and CSBM

Status: implemented and CPU-verified under the owner's 2026-10-25 authorization. At this milestone, the common contract, CSBM generation/loading, 31 passing tests, small/paper-scale examples, and project-local setup were complete. Source training was subsequently added in D2 above; adaptation remains unimplemented.

Rationale for CSBM first: small configurable graphs make edge/split/seed/leakage assertions practical and exercise separate source/target environments without archive dependencies. Disadvantage: we must specify and validate the sampling mathematics ourselves. Matcha is a reference from the owner's proposal/resource index, not an automatic methodological choice.

Current order: Phase 1 GCN/CSBM foundation through audited bank preparation is complete under D5. Phase 2 adaptation requires its own method/protocol discussion. The final-paper holdout protocol remains future work; the phase's reference label budget is fixed. Other datasets remain background only; the roadmap's eventual natural-shift evaluation needs a separate future discussion.

### Minimal implemented structure

```text
AGENTS.md
README.md
docs/PROJECT_STATE.md
docs/DECISIONS.md
.gitignore                        # ignored runtime artifacts
pyproject.toml                    # installed/tested dependency and test settings
src/amortized_gnn_tta/
    __init__.py
    data/
        __init__.py
        api.py                    # small records, load dispatch, validation
        csbm.py                   # versioned independent-draw generator/loader
    models/
        __init__.py
        gcn.py                    # two-layer source baseline
    training.py                   # shared source-only loop and checkpoints
    evaluation.py                 # label-free inference, explicit-label scoring
    source_subsets.py             # fixed-holdout source-label selections/manifests
    weight_bank.py                # sequential model-neutral bank execution/evidence
tests/test_data.py
tests/test_training.py
tests/test_source_subsets.py
tests/test_weight_bank.py
examples/inspect_data.py
examples/train_source.py
examples/prepare_source_subsets.py
examples/train_weight_bank.py
```

Use dataclasses and a short explicit dispatch, not a plugin registry or large base-class hierarchy. Only `csbm.py` is in the current dataset implementation scope. The source milestone adds one GCN module and shared training/evaluation; no adaptation-generator directory is needed. Direct runtime dependencies: torch, torch-geometric, NumPy; pytest as development dependency. Version constraints are in pyproject.toml and the executed stack is in README.md. Local .venv reuses the existing CPU torch/NumPy site packages; new dependencies are project-local. Do not adopt moving upstream HEAD as an experiment dependency.

### Proposed interface and ownership

Implemented interface; select returns cloned node indices and unlabeled_input returns an allowlisted cloned Data object:

```python
bundle = load_dataset(environments)  # sequence of CSBMEnvironment records; in memory
selection = bundle.select(graph_id="source_0", split="train")
target_input = bundle.unlabeled_input(graph_id="target_0")
```

- `DatasetBundle`: small mapping from stable graph IDs to records, plus task/provenance metadata. One graph for OGB/GOOD; multiple graphs for independently generated CSBM or language-specific Twitch-E. No replication of a shared graph for each mask.
- Graph record: CPU PyG `Data` with `x: float32[N,F]`, `edge_index: int64[2,E]`, explicit `num_nodes`, and supported optional edge attributes/weights. Labels stored separately as class indices `int64[N]` for the initial single-label task. Record native shape/dtype and any singleton-dimension conversion; do not silently renumber classes. Keep upstream cache intact.
- `splits`: named one-dimensional index tensors. Preserve native names (`valid` versus `val`, `id_val`, `id_test`) and exact membership, with an explicit experiment mapping. Do not force GOOD's five splits into three. Retain native node-level `env_id`/`domain_id` as benchmark metadata, including sentinel values; they are not our experiment phase labels.
- Experiment selections: graph ID, optional named node split, and phase (`source_train`, `meta_train`, `validation`, `target_test`). Use a simple selection manifest distinct from dataset facts. For M1, validate explicit source/target selections; defer episode generation. Record graph-context policy instead of silently constructing subgraphs.
- `unlabeled_input`: an allowlist-built PyG graph containing only permitted graph/features/edge attributes, without `y`, label aliases, or label-derived statistics. Adaptation accepts this restricted input, not the bundle. The evaluator alone receives the target label store. This boundary is tested against target-label changes and aliases; a wrapper around a label-bearing object would not suffice.
- Task metadata: single-label class schema, feature dimension, native metric and benchmark variant. `num_classes=2` is different from a binary one-logit output dimension; loss/head decisions belong to models/training. Preserve ROC-AUC where prescribed, with accuracy optional as a secondary metric.
- Class schema comes from the benchmark definition or permitted source data, not inferred target-label frequencies. Label-dependent diagnostic metadata stays evaluation-only and is excluded from deployment conditioning.
- Provenance: dataset/protocol version, upstream URL/revision/artifact hash, original node IDs when reindexing, label mapping, direction/edge-count convention, preprocessing ledger, RNG seeds and generation parameters. Missing provenance should be explicit rather than guessed.

Dataset loading stays on CPU and is model-neutral. GCN/GAT/GraphSAGE receive the same graph representation. GCN normalization/self-loops belong to `GCNConv`; GAT and SAGE have different aggregation semantics. The loader must not bake GCN-normalized adjacency into shared data. Avoid `GCNConv(cached=True)` across changing graph environments unless the cache is explicitly managed; OGB's fixed-graph example is not directly reusable for this scenario.

### Implemented preprocessing and reproducibility policy

No feature normalization, edge symmetrization, self-loop changes, or label-based filtering by default. Record what upstream loaders already do. Necessary format conversions must be explicit. Optional feature scaling is a separately approved transform with documented fitting scope; never fit it using target labels. Graph transforms should not mutate cached native data.

CSBM graphs: no self-loops, both directed entries for each undirected edge, stable class-feature semantics, explicit parameters and versioned local RNG. Use sparse edge storage; for a small initial generator, sample upper-triangle edges in bounded-memory chunks rather than allocate an `N×N` matrix. This can retain quadratic pair-scanning cost initially, which must be documented and measured before scaling. A faster sampler would require statistical-equivalence checks.

Keep generation, split, shift, and later model seeds independent. Cache identity must include protocol, all generation parameters, seeds, and preprocessing version; never reuse a graph solely because its dimensions or shift parameters match. A persistent cache is optional for the tiny M1 example and should be deferred unless useful.

CSBM has no native masks to preserve. Approved development policy: configurable class-stratified source train/validation/test indices (initial default 60/20/20), split independently from generation; independent target graphs with labels available only for offline evaluation. This is our development convention, not Matcha's full-source-label protocol or an official CSBM split. It does not apply over native real-benchmark splits. Validate per-class counts and reject infeasible requests. Final scientific label budgets and meta-environment allocation remain separate decisions.

### Verification requirements and current evidence

- Validate finite features, tensor dtypes/shapes, explicit node count, label schema, edge bounds, optional edge-attribute alignment, directionality, duplicates, self-loops, and isolated nodes. Empty-edge graphs must retain shape `[2,0]` and all nodes. Isolated nodes are a reported property, not automatically an error.
- Check index/mask conversions, in-range unique indices, required disjointness/coverage, and preservation of native membership. Do not impose disjointness on graph context or intentional cumulative-node overlap; enforce the protocol's supervision rules.
- Same configuration/seeds produce identical tensors in the tested environment. Changing only split seed preserves graph/features; changing a specified generation seed changes the appropriate realization. Do not promise bitwise equality across versions/hardware.
- Check the sampling equations on small fixed cases and empirical degree/feature moments across repeated draws with justified tolerances. Test the reference factor-of-two convention rather than enshrine the misleading parameter name.
- Test incompatible source/target feature/class schemas, graph lineage/role conflicts, and absence of target labels/label aliases in deployment input. Permuting held-out labels must not change the deployment input. Check source splitting cannot select target/evaluation labels.
- Run a small realistic CPU source/target load-and-inspect example and inspect counts, class balance, masks/indices, and fingerprints. When models are later approved, test GCN/GAT/SAGE forward compatibility through this same data contract.
- Before accepting real adapters, compare wrapper tensors and split membership against the official loader on actual downloaded data, verify archive checksums/license, and execute a real-data smoke test. Tiny fixtures alone cannot establish benchmark fidelity.
- Executed 31 automated checks and small/paper-scale layer smoke runs. Detailed counts and environment limitations are in PROJECT_STATE.md. Real-adapter, checkpoint, and backend checks above apply only to later relevant milestones.

## Reference checkouts and evidence

All paths below are relative to the repository's parent `references/` directory. They are investigation snapshots, not installed dependencies. Exact revisions permit links of the form `https://github.com/OWNER/REPO/blob/REVISION/PATH`.

| Checkout / upstream | Inspected revision | Relevant evidence |
| --- | --- | --- |
| [Matcha](https://github.com/baowenxuan/Matcha) | `f0e7104a84d118a8dabb7ad210915c43722b94af` | `README.md`, `src/dataset/{CSBM,NCDataset,create_dataset,utils}.py`, `src/csbm_gen.py`, `src/algo/ERM.py`, `script/csbm/pretrain.sh`; generation, source/target construction, corruption and training labels |
| [GPRGNN](https://github.com/jianhao2016/GPRGNN) | `4e0a7ee5435058b70eaec3c23c55fb96dc37f2d5` | `README.md`, `src/cSBM_dataset.py`; CSBM lineage and file-level license |
| [GraphOOD-EERM](https://github.com/qitianwu/GraphOOD-EERM) | `ffdc4a11161976fac7dd71e2aa1dcd72db6e44e9` | `README.md`, `multigraph/{dataset,load_data,main}.py`, `temp_arxiv/dataset.py`; Twitch domains/metric, raw features, cumulative years |
| [GOOD](https://github.com/divelab/GOOD) | `ec2f3fd13e69f1324f507e31ce3ce7252e9fb58e` | `README.md`, `LICENSE`, `setup.py`, `GOOD/__init__.py`, `GOOD/data/good_datasets/good_{cora,twitch,arxiv}.py`; masks, domains, metrics, imports and serialization |
| [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | `ea2c5f62d54c8b414dd3ea7139814a7c1ef83182` | `LICENSE`, `pyproject.toml`, `torch_geometric/datasets/twitch.py`, `torch_geometric/nn/conv/{gcn,gat,sage}_conv.py`; graph API and model defaults |
| [gcn](https://github.com/tkipf/gcn) | `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81` | Historical TensorFlow reference: README, `gcn/layers.py`, `gcn/models.py`, `gcn/metrics.py`; sparse aggregation and masked loss. Not installed or run; no repository-level LICENSE file found in this snapshot. No code copied. |
| [ogb](https://github.com/snap-stanford/ogb) | `61e9784ca76edeaa6e259ba0f836099608ff0586` | `LICENSE`, `setup.py`, `ogb/nodeproppred/dataset_pyg.py`, `examples/nodeproppred/arxiv/gnn.py`; native API, evaluator, explicit example symmetrization |
| [datasets](https://github.com/benedekrozemberczki/datasets) | `3b19135105432cbca65765c1113b91c395ed9d44` | Sparse checkout: README Twitch Social Networks section and MIT license; original distribution points to [SNAP](https://snap.stanford.edu/data/twitch-social-networks.html) |
| [Non-Homophily-Benchmarks](https://github.com/CUAI/Non-Homophily-Benchmarks) | `af14a88470d30b1dadd3803d911dfc1064bcf172` | Sparse checkout: README, LICENSE, `dataset.py`, `load_data.py`; original Twitch-E loader lineage, raw 3,170 feature vocabulary and FR duplicate handling |
| [Non-Homophily-Large-Scale](https://github.com/CUAI/Non-Homophily-Large-Scale) | `82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6` | Sparse checkout: README inspected; successor uses `twitch-gamer`, so it is not the chosen Twitch-E source |

Additional mathematical reference: [Contextual Stochastic Block Models, NeurIPS 2018](https://papers.nips.cc/paper/2018/file/08fc80de8121419136e443a70489c123-Paper.pdf), equations (1)–(3). The original paper's Gaussian latent-direction model should not be conflated with Matcha's fixed-direction variant. No separate original-author CSBM code repository was verified in this investigation.
