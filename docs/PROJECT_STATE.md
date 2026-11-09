# Project state

Last updated: 2026-11-09 IST. Current milestone: **real-data foundations and sequential Experiments 1–6**, in progress on `work/ogb-foundation`. Audited integration was merged into main at 80d2b92 and its integration branch deleted after explicit authorization. The current OGB implementation and concise verified evidence are included in the owner-authorized main delivery checkpoint; original training provenance remains unchanged. Existing main-checkout handbook formatting edits and the original repair checkout were preserved. CUDA component checks and all 270 formal GPU source fits are verified; full real-data adaptation experiments are in progress and MPS execution remains unverified.


## Current formal-bank and research execution milestone — 2026-11-08

**Latest verified state:** all 270 formal source checkpoints are downloaded and independently checked: 90 each for controlled Arxiv, temporal Arxiv and Pair-Align US-country. The formal GCN design job completed successfully. Its 541 output files were hash-verified, all 420 generated/control prediction cases passed numerical replay with zero class mismatches, and an additional local check confirmed exactly zero dense deltas in all 252 selected-fit artifacts. All six hypernetworks selected epoch zero at mean 2018 validation accuracy 62.89073299%. Inputs A/head were selected by the declared tie rules, not by a demonstrated advantage. The full three-architecture, three-seed comparison is now running privately as `deeptanshukumar/afml-controlled-extend-20261108/1`, using the same immutable fitting archive. Final test labels remain unscored. The owner confirmed Kaggle; Full Access is effective.

| Source setup | GCN | GAT | GraphSAGE | Verified total |
|---|---:|---:|---:|---:|
| Controlled Arxiv | 30/30 | 30/30 | 30/30 | 90/90 |
| Temporal source through 2014 | 30/30 | 30/30 | 30/30 | 90/90 |
| Pair-Align US country | 30/30 | 30/30 | 30/30 | 90/90 |

**Actual fit evidence:** the 30 controlled GCN fits total 569.47 seconds (18.98 seconds/model; range 18.61–19.91). Selected epochs span 198–200; convergence is not established. Maximum allocated CUDA memory was 1,196,469,248 bytes. Temporal GCN/GAT/GraphSAGE fit totals are 82.23/133.48/73.80 seconds; country totals are 336.07/614.08/351.48 seconds. These are measured GPU source-fit/checkpoint-verification times, not online adaptation speedups. Source runs used Tesla T4, Python 3.13.16 and Torch 2.11.0+cu130, with exact dependency/hardware records preserved. Dataset download/startup and audits add job overhead. The earlier one-thread CPU GCN pilot averaged 554.73 seconds/model, suggesting roughly 4.62 hours for 30 fits; its different label groups make this an approximate historical estimate, not a matched speed benchmark.

The 30-model controlled GCN diversity audit contains 435 pairs on an explicitly saved 1,024-node source-training probe (seed 901). Mean prediction disagreement is 7.483%; mean probability RMSE is 0.011867; mean hidden linear CKA is 0.987715. Variation exists, while representations remain very similar. Raw weight distance has permutation ambiguity; no sufficient-diversity threshold or convergence claim is invented.

**Verification:** final frozen study snapshot passed **456 tests, 7 expected skips in 229.62 seconds** (`runs/formal-study-kaggle-v1/pytest.log`). Source ZIP SHA-256 is `26a40737a7c86e6a0d6d117e4ac5b2ee5c8fc6b21b96c81a3ac0c14dd3fbea56`; it contains 114 allowlisted files. Earlier immutable remote regression passed 451 tests; 18 CUDA adaptation checks and six nonzero generated factor/delta/prediction cases passed. Independent local reconstruction verified all six captured factor products and saved predictions (maximum observed logit difference 4.7684e-7). A second private GPU job demonstrated recovery from actual saved versioned outputs before extending the bank. Earlier incomplete packaging and mixed-revision failures remain preserved as failed attempts.

**Implemented and tested:** formal groups/profiles, exact completed-model recovery, atomic checkpoint/index publication, exclusive Linux writer locking, source-only supervision/diversity audits and formal generator-use gates. Five overlapping groups retain exact per-class fit/stop budgets inside existing source ownership; natural diagnostics stay fixed and excluded. Six distinct initializations per group give global seeds 31–60 and whole-group 18/6/6 generator roles. Source graphs, native IDs/masks, labels, recipes, source hashes and dependency identities are recorded. An unfinished fit restarts from its seed; no optimizer-state continuation is claimed. Older pilots and checkouts are preserved.

The study adds actual CUDA allocation/reservation measurements outside primary timing, saved validation logits/factors/dense deltas, adaptation and generator parameter counts, and strict execution hardware/dependency/precision pins. A validation-only cyclic-donor diagnostic holds the actual target graph and source model fixed while supplying another validation graph's descriptor. It never revises design selection and is not mislabeled a retrained no-input baseline. Formal descriptors use chunk size 16,384 (an 8 MiB 128-channel float32 temporary); historical pilots retain 1,024. This changes batching of the same statistic, not labels, corruption, seeds, update sites or budgets. A real-GPU old/new numerical-agreement probe is required before the queued study fits any generator. The exact chunk is saved in study identity and descriptor checkpoints.

**Versioned evidence and operations:** concise source-bank/design results, CSVs, plots and verification summaries are tracked in [reports/real-data-2026-11-08](../reports/real-data-2026-11-08/README.md). Byte-identical snapshots of the local campaign helpers and restoration instructions are in [tools/kaggle](../tools/kaggle/README.md). Full raw datasets/checkpoints/predictions/bundles remain excluded and preserved in their original local/private locations. This owner-authorized Git delivery does not alter the immutable fitting ZIP or active workers.

### Verified GCN design-stage findings

All generator fits ran the declared five epochs; early stopping did not terminate training. Validation selected epoch zero afterward because trained epochs did not beat it. Each direct-CE fit performed 630 optimizer updates. The improving late LoRA curves do not establish convergence within this budget. Since every selected configuration tied at identity, A/head is the declared tie fallback, not evidence that those inputs/sites are optimal. Experiment 3 therefore compares generator families conditional on that recipe; no broader optimal-recipe or fully converged-method ranking is justified.

Each row below averages six held-out validation checkpoints × seven 2018 cases. These are validation results, not final estimates.

| Method | Mean accuracy | Gain over frozen (pp) |
|---|---:|---:|
| Frozen | 62.891% | 0.000 |
| Each of six selected A–D LoRA / A,D head hypernetworks | 62.891% | 0.000 |
| Practice-trained fixed model | 62.953% | +0.062 |
| Constant supervised-teacher update | 62.652% | -0.238 |
| GTrans feature-only variant | 61.153% | -1.737 |
| Iterative IM head | 60.674% | -2.217 |
| Iterative IM LoRA | 57.230% | -5.661 |

Both the practice-fixed model and constant-teacher construction use permitted offline practice labels; all six baseline methods use no online target labels. The frozen raw baseline flag only marks practice-fixed, so the reviewed derived report explicitly corrects the constant-teacher label annotation without changing scores or fitting. Earlier generated report files are preserved before that correction. Donor conditioning did not produce useful sensitivity from the selected zero-update checkpoints. These negative results remain in every report; they do not establish that all generator families or architectures fail.

The six generator fits took 2360.11 seconds (39m20s), the full input/site subprocess 2523.65 seconds, donor controls 517.91 seconds, baselines 1153.77 seconds, and independent replay 40.88 seconds. Whole-worker elapsed was 4443.42 seconds (74m03s); service publication and local downloading are additional. Do not double-count generator fitting inside the input/site subprocess or sum repeated references to the shared 158.00-second teacher collection as new work. The descriptor batching check passed its declared numerical tolerance; its component times do not establish an online-method speedup.

Reviewed report, CSVs, saved epoch histories and PNG/SVG figure: `../compute-tools/verified-design-20261108/`. Raw versioned outputs, `LOCAL_STUDY_VERIFICATION.json` and `LOCAL_ZERO_UPDATE_AUDIT.json` remain under the design job's `download-version-1/`. The durable continuation queue monitors the authorized sequential stages, verifies outputs before advancing, preserves failures and stops on failed checks or insufficient quota. Its status/log are `../compute-tools/RESEARCH_QUEUE.json` and `continuation-queue.log`. A process-scoped sleep/idle inhibitor supports unattended monitoring; desktop power defaults are unchanged. Final scientific interpretation still requires review after the remaining runs.

### Failure investigation — observations only, no changes authorized (2026-11-08)

The owner requested documentation and monitoring, explicitly keeping the code, experiment settings and existing execution queue unchanged. No normalization, learning-rate, update-scale, alignment or training-order fix has been applied. The findings below are diagnostic observations and hypotheses, not established causes or validated remedies.

**Measured conditioning-scale concern.** Saved A-head practice descriptors contain a maximum-degree/mean-degree scalar of approximately 962–965. Across the seven practice graphs this one field accounts for 99.45–99.73% of the squared magnitude of the 273-entry raw descriptor. The generator feeds this vector directly through Linear/ReLU/Linear layers, with no descriptor standardization. Its initial hidden activation has L2 norm approximately247 and L1 norm1635–1641. This supports a serious scale/conditioning concern; input-norm dominance does not measure feature importance or prove causality. A smaller learning rate or normalized descriptor has not been tested or approved.

**Measured optimization instability.** In the A-LoRA first epoch, average cross-entropy reached348.745 and average dense-update norm296.30; mean validation gain was−26.59pp. B/C/D LoRA also had large first-epoch updates and severe negative validation gains. Later epochs reduced losses/update norms but did not outperform epoch zero. These are epoch averages, not the first optimizer step, gradients or every individual update. The analytic first-Adam-step bound in the diagnostic JSON is an illustration with stated assumptions, not a measured first update.

**Measured A-conditioning ambiguity.** Within each practice graph the saved A descriptors are exactly identical across all18 source models, and raw source weights are explicitly disabled. Thus A supplies the same head correction across models. The supervised teacher corrections for those models differ: their average pairwise cosine is0.039–0.053, while the norm of the average correction is only30.4–32.5% of the average individual correction norm. Different source-model coordinates therefore deserve investigation. This does not prove that averaging caused the validation failure, and it does not explain B/C/D by itself because they contain model-dependent summaries. Neural-network hidden-unit permutation symmetry provides a known reason to examine coordinate alignment: [Git Re-Basin, primary paper](https://arxiv.org/abs/2209.04836).

**Further untested hypotheses.** Direct-CE meta-training uses a fixed environment/model order rather than shuffling the pairs, which could bias updates toward recent pairs. Practice labels are pre-2018 while validation labels are2018, adding temporal generalization beyond graph corruptions; the official split is documented by [OGB](https://ogb.stanford.edu/docs/nodeprop/). Neither order effects nor temporal drift have been isolated experimentally here. Other tested baselines also regress, while practice-fixed gains only0.062pp, so these observations do not justify attributing everything to a single generator bug or concluding that recovery is impossible.

Read-only evidence: `../compute-tools/failure-diagnosis-20261108/SAVED_SIGNAL_AUDIT.json`, its accompanying analysis script, the saved teacher file, and `../compute-tools/verified-design-20261108/TRAINING_HISTORY.csv`. Only saved practice descriptors/weights/corrections/training scores and existing histories were inspected; no final labels or new graph training were used. Fitting source and running jobs remain unchanged.

**Planning estimate, checked 2026-11-08 at18:24 UTC (23:54 IST):** the current E3 job has eight of its27 comparison fits available in live logs (one reused from E1/E2); the third GCN diffusion fit is in progress, with GAT/GraphSAGE still ahead. Allow roughly2.5–4 more hours including remaining fitting, baselines, replay, publication and local recovery, approximately02:30–04:00 IST on9November. This is a rough estimate from measured GCN phases and source-fit architecture timing proxies; GAT/GraphSAGE generator timings are not yet measured. Service delays can move it. This covers the current E3 job only.

The checklist below has six of11 major milestones verified complete, so an equal-milestone planning estimate is roughly55–60% overall now and approximately65% after E3 is complete and verified. Two of six experiment stages are verified; all270 source checkpoints are verified. These percentages describe deliverable progress, not consumed compute, validated research success or publication readiness. E4/E5/E6 and consolidated final auditing/write-up remain substantial work.

### Main-campaign scope and read-only shift/country confirmation — 2026-11-09 IST

The owner confirmed the approaching deadline and requested treating the existing six-experiment campaign as the final planned main run for the frozen method. No additional tuning sweep, descriptor fix or new research run is authorized by the diagnostic discussion. Existing queued stages remain unchanged. Negative findings must be reported honestly; a technical/data-integrity failure must be flagged as incomplete rather than automatically initiating a new campaign. Once all six stages and their numerical checks finish, planned experiment execution is complete; final consolidation/review/write-up/reproducibility packaging still remains. Rough overall readiness at that point is85–90%, a planning estimate rather than research success or guaranteed submission readiness.

A friend's question prompted a read-only check of the existing shift manifest, corruption implementation, prior real-data integration log and saved teacher descriptors. Training uses10%/30% feature-entry masking, uniform edge-pair deletion, and paired combined interventions. Validation uses20%/40% with different seeds. Labels are not passed into the corruption function; symmetric edge pairs share one deletion draw. The prior actual-data integration check passed the pairing/isolation contract. From existing saved practice descriptors, inferred erasure of initially nonzero features was10.0064%/29.9935%, and inferred edge removal was10.0085%/30.0067%, corroborating the intended Bernoulli rates. These are derived aggregate statistics, not newly reconstructed graph matrices or a fresh experiment. Frozen validation mean accuracy across six models was64.9703% unchanged,62.7555% under20% combined corruption and59.2951% under40% combined corruption. These checks support correct application of the declared interventions; they do not prove that this intervention family is equivalent to the earlier CSBM shifts or that information loss is recoverable. Trained adaptation candidates failed to beat their shifted frozen references; selected epoch-zero generators matched those references rather than adding harm.

Experiment5 explicitly covers BOTH strict temporal Arxiv and `pair-align-mag-country`. The saved country manifest assigns US to source-GNN pretraining, CN/DE to generator practice, FR to validation and JP/RU to final held-outs. All90 country source checkpoints are already independently verified. Country-generator training/final evaluation remains scheduled after controlled final evaluation; it has not yet completed. These are the published homogeneous Pair-Align MAG country graphs, not the native heterogeneous OGB ogbn-mag graph. Verification sources are the saved country `formal-bank/environment_manifest.json`, its `LOCAL_REPLAY_VERIFICATION.json`, and the existing `continue_research.py`/`research_stage_driver.py` code. No job, setting or executable code was changed for this confirmation.

### Completion checklist

- [x] Clean regression, real-loader checks, nonzero GPU factor/delta/prediction verification.
- [x] Private Kaggle authentication, quota, immutable jobs, downloaded evidence and versioned-output recovery.
- [x] Formal source grouping, exact recipe/ownership checks, atomic publication and diversity tooling.
- [x] All 270 source fits downloaded and audited.
- [x] Experiment 1: actual GCN A–D LoRA fitting, validation selection and conditioning-sensitivity evidence; negative result verified.
- [x] Experiment 2: head/LoRA selection, declared follow-up inputs, parameter counts and matched validation baselines.
- [ ] Experiment 3: all architectures, three seeds, direct/teacher hypernetwork and diffusion, fixed/constant/graph controls.
- [ ] Freeze design, then Experiment 4 held-out models/draws/severity/operations and sealed final scoring.
- [ ] Experiment 5 separate temporal/country generators and natural held-outs.
- [ ] Experiment 6 supervised development diagnostics, complete-pipeline timing, CUDA/RSS memory, harm and offline costs.
- [ ] Independent complete artifact replay, source/label/protocol audit and consolidated reports/handbook.

**Evidence locations:** ignored per-job source checkpoints/logs/manifests/bundles are under `../compute-tools/kaggle-*/download-version-1/`; `LOCAL_REPLAY_VERIFICATION.json` records successful downloads/checks. `../compute-tools/FORMAL_SOURCE_FITS.csv`, `FORMAL_BANK_SUMMARY.json` and `FORMAL_BANK_REPORT.md` consolidate unique checkpoint evidence without counting resumed fits twice. `ACTIVE_WORK_SESSION.json` records current jobs, immutable study pin and continuation instructions. Broad cleanup remains postponed; the owner authorized this main delivery, with no standing authorization for future pushes. No final target accuracy has been scored.

## Preceding handoff plan — Kaggle transition, 2026-11-08

The current formal-bank milestone above supersedes this handoff section's pending-authentication and pending-implementation statements. Unexecuted scientific gates still apply.

This section supersedes older statements that a Colab main job is running. A live Colab CLI check returned **no active sessions**. No main real-data GPU training completed. The latest remote regression gate stopped at 446 passed, two failed, two skipped; clean-process verification is still required after the exporter patch. Retrieved backend checks and failed-job logs are preserved under ignored `runs/colab-main-mirror/`.

**Bank terminology:** the historical 30-model CSBM banks are implemented and were executed. Current Arxiv work is a pilot: five GCN checkpoints finished locally, while formal 30-model real-data banks have not been constructed. The owner explicitly confirmed this distinction in the latest discussion. Do not present a pilot checkpoint collection as the formal bank.

The owner authorized Kaggle investigation/setup and completion of the remaining real-data work. Official Kaggle documentation currently describes 12-hour CPU/GPU notebook sessions and a commonly 30-hour **weekly** GPU quota; actual availability/remaining allowance must be checked on the account. Kaggle CLI 2.2.5 runs through an isolated uv tool environment with the missing six dependency supplied. No Kaggle authentication, private artifact upload or remote training has yet been verified. A hidden-entry helper exists outside Git at `../compute-tools/configure_kaggle_auth.py`; credentials must never enter snapshots, notebooks, logs or Git. Recommend rotating the token exposed in chat and entering its replacement only locally.

### Ordered deliverables and completion gates

| Step | Deliverable | Required evidence before proceeding |
|---|---|---|
| 1 | Clean revision and regression gate | Fresh process with one immutable code snapshot; reproduce/fix the two natural-runner failures; nonzero generated-factor export/replay checks; full regression and real-loader checks |
| 2 | Kaggle execution backend | Verified authentication/quota, private code/data distribution, exact code/dependency manifest, CUDA smoke test, bounded single-model run, downloaded outputs and demonstrated resume |
| 3 | Formal source-bank implementation | Thirty models per architecture/source protocol, five source supervision groups × six initializations, saved 18/6/6 whole-group roles, exact recipe/data hashes, resume and diversity audits |
| 4 | Formal controlled Arxiv banks | First GCN, then GraphSAGE/GAT; source-only stopping; audit checkpoint/schema/label boundaries; preserve existing five-model pilot separately |
| 5 | Experiment 1 | GCN both-layer LoRA, hypernetwork A–D inputs, identical source checkpoints/practice shifts/budgets, select using 2018 only; report identity/no-input/constant or shuffled-input controls to diagnose conditioning utility |
| 6 | Experiment 2 | Selected inputs; frozen, iterative/generated head and LoRA; parameter counts and matched objectives; A/D head check and conditional B/C follow-up if the ranking changes |
| 7 | Experiment 3 | Three architectures and three generator seeds; hypernetwork, comparable teacher-hypernetwork and diffusion; fixed practice-trained model and clearly labeled graph-specific baseline; frozen five-candidate MSE selection |
| 8 | Experiment 4 | Held-out bank models including unseen initialization seeds, new draws/severity/feature and edge operations; sealed final evaluation after validation choices freeze |
| 9 | Experiment 5 | Separate formal temporal and country banks/generators; temporal source <=2014, practice 2015–2017, validation 2018, final 2019–2020; country roles and separate training recorded |
| 10 | Experiment 6 | Permitted-label diagnostic head/LoRA/full fits; whole-pipeline latency, harm frequency, host RAM and actual CUDA peak memory; offline bank/teacher/generator costs and amortization separately |
| 11 | Scientific release checks | Independent factor/delta/prediction replay, seed variability, explicit failure reports, final protocol audit, readable tables/handbook/write-up; Git review and delivery only under applicable approval |

### Formal-bank design to implement and record before training

Preserve the fixed source-versus-generator label ownership. Do not split the source budget in half again or use validation/final labels to create a bank. For controlled Arxiv, create five independently seeded stratified train/stop subdivisions within the existing source-owned pool, preserving its 90/10 fit/stop budget. For natural source graphs preserve their declared source-only fit/stop/diagnostic allocations. Each group needs a recorded seed, native node IDs, graph visibility and supervision hashes. These are overlapping source subsamples, not five mutually disjoint source datasets.

Use six distinct initializations per group. A proposed explicit seed schedule is 31–36, 37–42, 43–48, 49–54 and 55–60, assigning whole groups 0–2 to generator training (18 models), group 3 to validation (six), and group 4 to final (six). Unlike the old paired-seed CSBM design, this also holds out initialization seeds; record this deliberate real-bank choice before execution. Save actual seed lists rather than infer roles from filenames. Formal implementation/tests are pending; this paragraph is a plan, not a completed artifact.

Thirty × three architectures = 90 models for one source protocol. Applying the same requirement to controlled Arxiv, temporal Arxiv and country-US source gives 270 source fits. Estimate throughput from one bounded GPU fit before committing the whole quota. Report measured time and completed counts rather than promising a finish time. Different graph visibility and label ownership prevent silently reusing controlled checkpoints as temporal checkpoints.

Resume only when source graph, supervision, model spec, optimizer/stopping recipe, code and artifact hashes match. Save each completed model independently. Audit supervision overlap, normalized weight distances with neuron-permutation caveats, prediction disagreement and representation similarity on allowed source nodes. Diversity is measured, not guaranteed by choosing seeds. Existing budget-limited source fits do not establish convergence.

### Kaggle job and evaluation discipline

Package one immutable snapshot per job; never replace modules while its tests/training run. Keep code, manifests, data references and outputs private. Exclude credentials, third-party checkouts, irrelevant runs and all unexpected files. Use official dataset distribution/checksums and verify internet/package availability. Checkpoint after each model/fit; split work into jobs comfortably below the verified runtime limit. Persist versioned outputs and verify downloads before the runtime expires. Do not assume an interrupted private notebook retains unsaved `/kaggle/working` files. Account for GPU quota across the whole study; two visible GPUs do not automatically accelerate full-graph PyG.

Long numerical work remains off the laptop. Small local unit checks and orchestration are permitted. Do not duplicate CPU/GPU matrices. No real final-test scoring has occurred, and labels remain sealed until design/protocol choices are frozen. Keep partial studies partial; never fabricate missing scores or claim all experiments finished merely because runners exist.


## Historical Colab attempt — stopped at verification gate

Owner requested stopping long local CPU work. Source-bank worker, generator worker and orchestration parent were interrupted and verified exited. Completed artifacts are preserved: five GCN source checkpoints and the completed A-input LoRA hypernetwork fit; B-input fitting was interrupted. A selected epoch 0 (the exact zero-update baseline), with mean 2018 validation accuracy 62.9527% across seven cases. Later trained A updates degraded validation; this is a failed-to-improve input-ablation result, not an adaptation gain. No real final-test scoring occurred. Raw interruption tracebacks are preserved and annotated as an intentional owner-requested stop.

GPU execution will be the main remaining study; no duplicate CPU/GPU experiment matrix is requested. Google Colab CLI 0.7.5 runs via isolated uv tools, but a read-only ADC session check confirmed missing credentials. The owner completed OAuth through their own terminal. The official Colab CLI provisioned session afml-research with a Tesla T4 (15,637,086,208 bytes), Python 3.13.16 and Torch 2.14.1+cu130. The explicit-allowlist code bundle (103 verified files) was uploaded. CUDA checks passed: 18 iterative combinations, 12 generator inference/checkpoint combinations and 18 generator training combinations across all three architectures, head/LoRA and teacher controls. These are random-graph software checks, not research accuracy results. The remote driver attempted regression and official-data checks, then stopped at the failed regression gate described below. No active Colab session remains.


Latest remote regression result: **446 passed, 2 failed, 2 skipped in 750.32 seconds**. The two natural-runner failures report an in-memory PipelineResult lacking factor_vector: an exporter update was uploaded while that test process was already running, mixing old imported modules with new runner code. This was an avoidable live-update error. A clean fresh-process verification of the corrected, matched files remains required. The driver stopped at this gate; no main real-data GPU source training or final scoring has started. Backend-check artifacts (18 iterative, 12 inference, 18 generator training checks) were retrieved locally under runs/colab-main-mirror/backend-checks.

Roadmap discrepancy found on reading the owner’s /home/deeptanshukumar/Downloads/AFML Research Project.txt: it explicitly requires 30 models per architecture/source protocol, five labeled subsets × six seeds, partitioned 18/6/6. The current JSON profile is a smaller development run with five checkpoints per architecture, one fixed source label pool and a 3/1/1 role split. Completing that profile would not complete the formal weight-bank requirement. Its status must remain developmental until this is reconciled. The later A–D experiment description and optional flow-matching statement supersede the file’s earlier broader input-grid/flow-matching suggestions. GPU memory measurement currently samples host RSS only; actual VRAM accounting remains outstanding.

## Real-data implementation and execution in progress

**Implemented:** optional OGB 1.3.6 integration; official Arxiv raw-reader/tensor cache; explicit partial supervision views; reproducible disjoint source/generator label pools; symmetric experiment views with label-free paired masking/edge removal; manifest-driven real environments; five-candidate diffusion agreement selection; shared teacher-target collection and teacher-regression hypernetwork control; source-bank and ordered study runners; separately gated generalization scoring; natural temporal/country loaders and bank dispatch; fixed practice-trained GNN and independently implemented GTrans feature-only controls; supervised development diagnostics and saved delta replay; complete-pipeline timing/RSS sampling and evidence-based report generation. Newly written final/natural runners still require broader integration verification; implementation alone is not experiment completion.

**Executed evidence:** 435 full-suite tests passed before the latest candidate/natural/runner extensions. Expanded focused data/runner tests passed; Complete tiny end-to-end fixture runs passed, including controlled design selection, partial-bank extension/pinning, final gates, supervised diagnostics, natural temporal/country generators and independent weight-delta replay. Fixture outputs are software checks, not Arxiv results. The subsequent full regression run passed 444 tests, with two opt-in integration tests skipped by default; both passed separately against actual artifacts. The latest ordered/natural/orchestration fixture checks passed after checkpoint-pin extensions.

Actual official Arxiv verified: 169,343 nodes, 128 features, 1,166,243 raw directed edges; 90,941/29,799/48,603 official train/validation/test nodes. Explicit symmetric view has 2,315,598 stored edge entries. Source fit/stop pools are 40,933/4,529 nodes; generator fit/stop/diagnostic pools are 36,417/4,531/4,531. Actual paired perturbation checks passed and the initial pilot score agrees with OGB's official evaluator. Three-epoch raw-directed engineering pilot achieved 16.9402% 2018 accuracy; it is not a converged result.

The full 15-model Arxiv source bank was intentionally interrupted after five completed GCN fits under `runs/arxiv-bank-research-v1`; its saved plan/source snapshot records dirty provenance. All five GCN source fits have completed: meta seeds 31/32/33 and validation seed 34 selected epoch 200; excluded seed 35 selected epoch 199. Observed fit wall times were approximately 531/547/593/553/550 seconds. These fits hit the configured budget; convergence/optimality is not established. Full-Arxiv GCN input/update design and independent source preparation were stopped at the owner’s request. The serial orchestration journal is runs/real-research-review-v1/JOURNAL.json. No real final-test score has been computed.

Frozen GCN development reference (checkpoint seed 34, official 2018 evaluation, verified against OGB evaluator): unchanged 64.8982%; at rate .2, feature/edge/combined accuracy 63.7907/64.2001/62.8645%; at rate .4, 62.1162/63.3645/59.4349%. These are unadapted validation scores, not generator or final-test results. Evidence: runs/arxiv-frozen-development-v1/{probe.py,RESULTS.json}.

All six published MAG country artifacts passed official MD5 checksums. Raw (nodes, stored directed edges): CN (101,952;285,991), DE (43,032;127,704), FR (29,262;79,182), JP (37,498;91,412), RU (32,833;68,294), US (132,558;702,482). All have 128 features and the published 20-class schema; final country class counts/accuracies were not used. No MAG training has run yet.

Approved temporal construction verified on real Arxiv: source through 2014 has 41,125 nodes; cumulative contexts through 2015/2016/2017/2018 have 53,160/69,499/90,941/120,740 nodes. Practice fit/stop/diagnostic counts are 9,612/1,189/1,234 (2015), 13,056/1,618/1,665 (2016), and 17,136/2,126/2,180 (2017). Validation covers exactly 29,799 2018 arrivals. Rare-class allocation is documented in D17. Temporal final scoring is gated; no temporal bank/generator has run yet.

**Remaining execution:** finish source bank, run ordered real generator studies and all baseline controls, verify frozen final protocol before scoring, execute held-out-checkpoint/severity/type tests, train separate natural temporal/country banks and generators, run development failure diagnostics and uncontended efficiency measurements, replay real artifacts and update final tables/handbook. Broad cleanup remains postponed. Do not claim all experiments complete from available foundations or launch long final studies before runner verification passes.

## Completed generated-LoRA comparison

Owner-approved scope: generated hypernetwork/diffusion LoRA for GCN/GAT/GraphSAGE, complete-event timing against existing head/LoRA entropy and IM, full-fine-tuning references, local milestone commits and write-up documentation. No push or merge. Implementation: `c2e883f`; matched full-reference reporter and completed-output protection: `600f34b`. Main numerical studies retain their clean c2e883f source snapshots; the later matched development diagnostic uses clean 600f34b. No core update mathematics changed between those revisions.

**Methods and data.** CSBM only: 5,000 nodes, 2,000 input features, two classes, two graph layers with total hidden width 32. Both generated update types receive the same 65-entry compact hidden-moment descriptor, **without** the historical separate raw-weight input. Three generator seeds (202801–202803), five epochs, three meta-training checkpoints with pretraining seed 31 from distinct saved source-label subset groups; twelve selected meta-training environments produce 36 training pairs. Main validation uses one excluded checkpoint and graph lineage per architecture, unchanged control plus eleven shifts at severity 0.5. These are development results, not final-target tests or independent pretraining-seed replications.

Generated LoRA predicts `[dA, B]` around local-seed-801 A0 and applies `(alpha/r) B(A0+dA)` at the existing linear-matrix sites. First/last ranks are 4/1 with alphas 4/1. Biases and GAT attention vectors remain fixed. GCN/GAT have 8,162 factor entries; GraphSAGE has 16,324 because both neighbor and root matrices are adapted. Direct head updates have 66/70/130 entries. The factor representation is explicit in checkpoints, with saved factors and authoritative final-minus-initial dense changes. A full functional output schema with zero frozen-site deltas is not full-model adaptation.

Hypernetwork training uses 180 graph-level Adam updates with permitted meta-training CE, dense-update L2 coefficient .001 and weight decay .0001. Diffusion uses 15 batched denoiser updates (batch 16), plus 1,080 supervised teacher steps (36 × 30) at teacher LR .01 and dense-update L2 .001. Generator LR is .001. The cosine diffusion schedule has 50 steps; inference uses ten DDIM steps, one candidate and sampling seed 42. Factor-target standard deviation has floor .001 and zero-centered normalization. These objectives/budgets differ; they do not establish an inherent generator-family advantage.

**Primary timing.** Canonical CPU graph plus resident models through required conditioning, generation/optimization, update application and usable prediction. No persistent target cache. Head reuses its frozen encoder within an episode; LoRA recomputes its changing encoder. One CPU thread; two warmups/nine repetitions per timed condition, interleaved with first-event samples stored separately. Timing averages medians over control/attribute/homophily/attribute+homophily and generator seeds; accuracy averages eleven shifted cases. A generated ratio divides its paired IM30 cost by generated cost. The separately measured baseline table has slightly different timings and is not the denominator for those ratios. Fit times below exclude canonical dataset preparation, bank creation and research search; diffusion teacher time is a subset, not an additional charge.

| Model | Update | Generator | Accuracy ± seed SD (%) | Δ frozen (pp) | Control Δ (pp) | Pipeline (ms) | Paired IM30 (ms) | Speed ratio | Generator fit (s) | Teacher subset (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCN | head | hypernetwork | 78.069 ± 0.020 | +0.815 | -0.427 | 13.50 | 52.39 | 3.88× | 8.31 | 0.00 |
| GCN | head | diffusion | 77.553 ± 0.028 | +0.299 | +0.027 | 14.74 | 52.62 | 3.57× | 6.00 | 2.63 |
| GCN | lora | hypernetwork | 78.861 ± 0.081 | +1.607 | +0.700 | 23.32 | 566.81 | 24.30× | 11.34 | 0.00 |
| GCN | lora | diffusion | 77.898 ± 0.587 | +0.644 | +0.427 | 28.32 | 569.70 | 20.11× | 25.59 | 21.52 |
| GAT | head | hypernetwork | 77.861 ± 0.126 | +0.414 | -0.453 | 14.97 | 75.12 | 5.02× | 9.13 | 0.00 |
| GAT | head | diffusion | 77.528 ± 0.042 | +0.081 | -0.233 | 16.64 | 76.75 | 4.61× | 7.29 | 3.56 |
| GAT | lora | hypernetwork | 78.138 ± 0.055 | +0.691 | +0.093 | 26.37 | 678.85 | 25.75× | 12.43 | 0.00 |
| GAT | lora | diffusion | 77.501 ± 0.091 | +0.053 | +0.067 | 31.76 | 690.74 | 21.75× | 31.03 | 26.63 |
| GraphSAGE | head | hypernetwork | 80.650 ± 0.039 | +1.334 | -3.073 | 83.08 | 122.91 | 1.48× | 49.71 | 0.00 |
| GraphSAGE | head | diffusion | 80.021 ± 0.068 | +0.705 | -0.733 | 85.22 | 123.14 | 1.44× | 29.43 | 10.35 |
| GraphSAGE | lora | hypernetwork | 83.845 ± 0.158 | +4.528 | +0.953 | 170.51 | 3085.00 | 18.09× | 58.78 | 0.00 |
| GraphSAGE | lora | diffusion | 79.585 ± 0.175 | +0.268 | +0.180 | 186.03 | 3198.29 | 17.19× | 144.88 | 122.96 |

Interpretation: hypernetwork LoRA beats iterative LoRA by 0.318pp for GCN and 1.928pp for GraphSAGE, while trailing it by 1.247pp for GAT. Diffusion LoRA recovers much less accuracy; GCN's seed SD is 0.587pp. GraphSAGE hyper-LoRA improves the control by 0.953pp but still loses about 3pp on the degree-increase case. Its head version loses 3.073pp on control. Keep both individual harms and mean gains. Broader generators also have more parameters: hyper-LoRA has 1,061,346/1,061,346/2,114,244 parameters for GCN/GAT/GraphSAGE, versus 16,962/17,478/25,218 for head-only. Equal inputs/epochs do not imply equal generator capacity.

**Full fine-tuning: separate matched development diagnostic.** Every row below starts from the identical first permitted meta-training checkpoint and uses the same meta-training graph lineage, control plus attribute/homophily/combined cases. Accuracy averages the three shifted cases on saved test-node masks. Supervised methods fit saved train-node labels and select by saved validation-node loss; all test-node scoring follows prediction. CSBM has no official splits: these are our recorded 60/20/20 development splits, not benchmark-provided masks. The checkpoints/environments participated in generator training, so this is **in-sample development evidence**, distinct from the held-out model/environment validation table above.

| Model | Frozen | IM-head | IM-LoRA | Hyper-head | Hyper-LoRA | Diff-head | Diff-LoRA | Supervised head | Supervised LoRA | Full fine-tuning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCN | 75.900 | 77.300 | 77.933 | 77.800 | 78.344 | 76.389 | 76.589 | 79.433 | 77.467 | 79.500 |
| GAT | 75.600 | 76.000 | 78.200 | 77.867 | 77.167 | 76.078 | 75.778 | 78.467 | 76.467 | 78.133 |
| GraphSAGE | 78.400 | 79.133 | 79.967 | 81.678 | 86.089 | 79.278 | 78.633 | 81.533 | 82.467 | 85.067 |

The generators received permitted labeled meta-training data offline; their deployment and IM calls receive no labels. Supervised references instead use labels per episode. Full fine-tuning is not a proven ceiling: its fixed LR/30-step budget and validation-selected step need not be optimal. GraphSAGE hyper-LoRA exceeding this particular full-fit reference does not demonstrate superiority to optimal full fine-tuning or held-out generalization.

Public supervised fitting includes native development-validation evaluation, best-step restoration, ordinary-model materialization and prediction. It executes all thirty optimizer steps here (patience=30), even when an earlier state is selected. These times are kept separate from the streamlined unlabeled fitter; no cross-objective ratio is headlined as matched unlabeled speedup.

| Model | Public supervised head (ms) | Public supervised LoRA (ms) | Public full fine-tuning (ms) |
| --- | ---: | ---: | ---: |
| GCN | 133.65 | 925.29 | 913.99 |
| GAT | 170.30 | 1072.96 | 1077.29 |
| GraphSAGE | 372.26 | 5969.32 | 5965.89 |

LoRA reduces trainable variables, but the existing adapter uses dense effective weights in functional GNN forwards. It does not guarantee faster graph computation than full fitting; the measured public LoRA/full times are nearly equal. Specialized low-rank branches or further graph-operation caching could change the baseline; current ratios describe the verified implementations, not an optimal lower bound.

**Verification.** 431 full-suite CPU tests passed for the generated-LoRA implementation, followed by three all-architecture real CLI/replay workflows covering the final matched reporter and existing-output preservation. Independent audits verified 432 generator cases, 144 iterative cases, 36 supervised artifacts and 168 additional matched-development cases: **780 saved update cases**, zero logit replay error and zero accuracy mismatches. This includes 288 independent generated-LoRA factor-to-dense reconstructions, 144 cross-report baseline comparisons, all reported aggregate rows and raw-timing medians. Small fixtures establish software behavior only; the paper-profile runs supply the numerical evidence. CUDA/MPS, clean installation and cross-hardware reproducibility remain unverified.

Evidence stays ignored under `runs/generated-adapter-matched-{head,lora}-v1`, `runs/generated-adapter-comparison-v1` and `runs/generated-adapter-full-matched-v1`. See their PLAN/source snapshots, complete checkpoints, per-case predictions/deltas, raw timings, independent audit JSON, CSV/Markdown tables and `shift_gains.png`. Configs are `configs/generator_matched_{head,lora}.json`; README has commands. A temporary launcher initially selected base Python and failed importing PyG before replay; corrected to the existing venv, with the failed launcher/log preserved and no experiment results replaced.

**Limits and next work.** No final-target labels were accessed. These results do not select formal Arxiv A–D inputs or demonstrate unfamiliar-initialization/dataset transfer. Real Arxiv/MAG, five-candidate selection, teacher-trained hypernetwork controls, fixed practice-shift/constant-update controls and broader held-out testing remain separate milestones. A generated update might learn a useful common correction; the current gains alone do not establish that target-specific conditioning is necessary. Fit-only break-even illustrations exclude additional bank creation/data/search costs: only genuinely shared costs cancel, and extra bank costs must be included for a full amortization claim.

## Completed integration checkpoint

`integration/generator-audit-fixes` starts at c137e28. Local commits `8c6203a` (audited repairs) and `d81bb02` (portable shared comparisons) preserve the original repair checkout and all ten contributor scripts/result files. No push or merge is authorized. The combined implementation passed **408 CPU tests**, including 24 focused portable workflow checks.

Portable commands now cover compact/structural/A-D and head/backbone/all direct scopes using saved partitions, complete checkpoints and validation-only review. Independent replay handles every scope. Standalone timing omits unmatched iterative ratios for broader scopes; review tables explicitly retain IM-head only as a reference. These direct broader updates are not generated LoRA.

The predeclared matched GCN study completed **30 configurations and 360 saved cases**: six new compact fits, 24 completed A-D checkpoints reused byte-for-byte after identity/recipe/exact-delta checks. All predictions/timings reran. The independent audit reconstructed every named target tensor and checked regenerated deltas, source/graph/checkpoint hashes, predictions, metrics and raw timing aggregates: **zero logit differences and zero accuracy mismatches**. Maximum requested-versus-effective float32 delta difference was 2.9802322387695312e-08. Source code was committed and clean at d81bb02 for the new compact fits and inference measurements; reused fits retain their original training provenance/costs.

All rows below use the same GCN checkpoint, head update sites (66 scalars), selected meta-training model/environment identities, five-epoch recipe and three generator seeds. Frozen shifted accuracy is **77.255%**; IM30 is **78.167%**. Accuracy excludes the unchanged control and averages eleven shifted validation conditions. Timing averages per-condition medians for control/attribute/homophily/combined; each uses 3 warmups and 15 repetitions, one local CPU thread. Ratios divide mean IM latency by mean generated latency within each row, not a shared grand-mean denominator.

| Inputs | Hyper accuracy ± seed SD | Hyper pipeline / ratio | Diffusion accuracy ± seed SD | Diffusion pipeline / ratio |
| --- | ---: | ---: | ---: | ---: |
| compact | 78.087 ± 0.026% | 13.35ms / 3.89× | 77.999 ± 0.064% | 15.40ms / 3.51× |
| A | 78.055 ± 0.002% | 94.12ms / 0.59× | 77.932 ± 0.147% | 101.18ms / 0.58× |
| B | 78.067 ± 0.004% | 93.64ms / 0.58× | 77.916 ± 0.133% | 104.58ms / 0.58× |
| C | 78.071 ± 0.005% | 96.48ms / 0.59× | 77.968 ± 0.092% | 102.83ms / 0.58× |
| D | 78.061 ± 0.004% | 89.82ms / 0.62× | 78.099 ± 0.011% | 99.82ms / 0.56× |

Compact gives +0.833pp / +0.744pp over frozen for hypernetwork/diffusion, while IM30 remains 0.080pp / 0.168pp higher in shifted accuracy. Both compact methods worsen 15/33 repeated shifted cases; unchanged-control mean losses are **0.653pp / 0.427pp**. D control losses are 0.307pp / 0.720pp. No method is universally safer. Compact hypernetwork has 25,410 generator parameters versus 522,818–538,434 for A-D; its input and model information differ, so this is not a one-field causal ablation.

Compact mean offline wall time is 8.16s hypernetwork and 6.21s diffusion (including 2.69s teacher subset); these are per-fit training costs, excluding common canonical dataset preparation and research search. Hypernetwork has 180 optimizer updates, diffusion 15 denoiser updates plus 1,080 teacher steps. Different objectives/budgets do not establish a generator-family ranking. A-D timings vary with machine noise; their saved fits/predictions agree with the prior review.

Interpretation: compact retains useful full-pipeline speedups on this binary-head workload. Expensive A-D remains slower despite similar mean accuracy. Keep both recipes explicit; do not select formal Arxiv inputs from this CSBM development study. One validation checkpoint and graph lineage per architecture are not an independent final benchmark. CPU only; no final-target scoring, CUDA/MPS execution or new real-data implementation.

Evidence: ignored `runs/generator-compact-matched-gcn-v1/{PLAN.json,code_snapshot.zip,seed_results.csv,AGGREGATED_RESULTS.csv,VERIFIED_REVIEW.md,INDEPENDENT_REPLAY.json,matched_tradeoffs.png}` and per-fit checkpoints/deltas/raw timings. Configuration: `configs/generator_compact_matched_gcn.json`; source: d81bb02. Reproduction commands are in README; rerun to a fresh directory. The figure plots actual verified values, seed SD and the grand-mean IM30 timing for orientation; exact per-group ratios are in CSV.

The c137e28 upload adds five experiment scripts and five result files, with no core changes. Independent arithmetic checks passed 2,205 relationships across 612 full-comparison records. The 44.49× GCN ratio is traceable, but generator timers omit adapted prediction and the compact diffusion timer also omits conditioning. The contributor’s 64-feature experiment and our 2,000-feature validation study are different workloads. Original files remain intact under `scratch/`, with a guide to portable equivalents.

At that earlier integration checkpoint, the roadmap listed ogbn-arxiv, generated LoRA, five-candidate selection and formal controls. Generated LoRA is now completed in the current section above; real-graph integration, five-candidate selection and the formal controls remain future work. Exact replay of the contributor’s small-profile run still needs its matching artifacts and seeds; no synthetic replacement is presented as that reproduction.

## Completed repair milestone

Implemented: atomic complete inference checkpoints and reload; numerical/schema guards; restored 57 regression cases; deterministic selection/statistic tests; caller-state/autograd preservation; saved partition/cap/seed handling; validation-default CLI with explicit final-evaluation gate; stable imports; short-linear-schedule restriction; versioned A-D descriptor definitions; bounded-memory edge statistics; shared encoder computation, within-event frozen-feature reuse for generated head predictions and frozen-head teacher reuse; explicit static D fingerprint caching; independent whole-pipeline benchmark and a predeclared three-seed development review.

Verification completed on local CPU: **392 automated tests passed** (including the restored 57 cases, all-architecture/scope/input training matrix, forced selection paths, numerical/schema guards, complete non-default checkpoints, actual CLI workflows and frozen-feature parity). Compileall and complete diff/whitespace review passed. The small complete-review/reuse workflow and independent reconstruction smoke also passed.

The predeclared paper-sized review completed **72 configurations** (3 architectures × A-D × 2 generators × 3 seeds), with **864 saved cases**. Thirty completed compatible training checkpoints were reused after exact generated-delta/configuration/identity checks; all prediction and timing checks reran. Partial training was never resumed. Independent manual named-weight reconstruction verified every source/graph/checkpoint identity, requested/effective delta, final head tensor, prediction, accuracy, per-case aggregate and raw-timing median. Maximum logit replay error and accuracy mismatch count were both zero. The largest requested-versus-effective float32 delta difference was 2.9802322387695312e-08, not an artifact error.

**Research verdict:** the repaired prototype runs correctly on tested inputs, but its A-D recipes do not demonstrate an amortized online speedup. Every A-D grouped speed ratio is below 1 against the improved cached IM30 baseline. Mean shifted accuracy improves; unchanged-control harm remains. Hypernetwork CE and diffusion teacher-MSE have different objectives/budgets; no definitive generator ranking or optimal accuracy claim follows.

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

One held-out GNN checkpoint and one validation graph lineage per architecture are used. The three training checkpoints share GNN initialization seed 31 across source-label subsets; only generator training seeds vary. Hypernetwork uses 180 optimizer updates; diffusion 15 denoiser updates plus 1,080 teacher steps. Both use five epochs, which is not equal compute. That repair review used head-coordinate updates and one diffusion candidate; the newer generated-LoRA study is described above. No final model/environment group was scored.

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


Colab real-study support is prepared in notebooks/colab_real_research.ipynb. Code cells compile and the existing explicit-allowlist exporter produced runs/real-research-colab-v2.zip with checksums. The v1 bundle is superseded by v2 (a notebook JSON-newline correction); use v2. No cloud files were uploaded and no CUDA run occurred. Browser access host is unavailable here; a GPU session cannot be provisioned autonomously from this environment. Local CPU runs are now stopped by explicit owner instruction. Free Colab resources are not guaranteed (https://research.google.com/colaboratory/faq.html).


Remote execution evidence: local driver copies/logs under /tmp/afml_colab_* and /tmp/afml-colab-*.log; durable remote status /content/afml-job/STATUS.json, hardware /content/afml-job/HARDWARE.json, bank /content/amortized-gnn-tta/runs/arxiv-main-bank-v1, study /content/amortized-gnn-tta/runs/research-main-v1. The CLI exec command may return zero despite a remote Python exception; completion is accepted only from explicit status files, result artifacts and logs. An editable-install import issue in the existing Jupyter kernel was resolved by adding the project src directory to its import path. A duplicate-graph fixture error in an extra CUDA probe was fixed in that probe; no graph-validation rule was relaxed.
