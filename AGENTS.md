# Project instructions

## Scope and approval

- Project: Amortized Weight Adaptation for Graph Neural Networks under Distribution Shift, an AFML course project with research-paper ambitions.
- Deeptanshu owns GCN implementation and experiments. Teammates own GAT and GraphSAGE. Share dataset, configuration, training, and evaluation protocols where practical.
- Read `README.md`, `docs/PROJECT_STATE.md`, and `docs/DECISIONS.md` before meaningful work. Inspect the branch, working tree, existing instructions, and relevant code first.
- The detailed research plan is a working roadmap, not an immutable specification. Explain discrepancies and proposed methodological changes; never silently resolve them.
- For significant tasks: understand, investigate, propose, discuss, implement after approval, verify, report. Routine work inside an approved milestone does not need repeated approval.
- Current dataset scope is CSBM only, explicitly clarified by the owner. The earlier real-benchmark-first interpretation was the assistant's mistake, not an approved decision. Do not implement or ask the owner to choose other datasets now.
- On 2026-10-25 the owner authorized the CSBM data milestone and then the small source-GCN training/evaluation milestone. Both are implemented and CPU-tested. The selected protocols/results are in docs/DECISIONS.md and docs/PROJECT_STATE.md. Do not re-request approval for work covered by that authorization. Adaptation, generators, and a weight bank remain separate milestones.
- The owner then requested remaining preparation one step at a time. The first bank-preparation step now implements explicit, stratified label subsets within the fixed native source-training pool. The CLI requires a label fraction; 50% is an executed development example, not a finalized scientific budget. Bank training, 18/6/6 grouping, and diversity diagnostics are subsequent steps; adaptation remains outside scope.
- On 2026-10-25 the owner explicitly authorized committing the completed foundation and then proceeding to the next implementation step. This authorization covers that checkpoint commit, not future commits or any push.
- Foundation checkpoint: local commit `30af4f9`. The subsequent sequential bank runner is implemented and CPU-tested; it is included in the later explicitly authorized Phase 1 completion commit. Keep the same six model seeds across subsets explicitly identified as a paired design, retain every job, and keep source-test/target supervision out of bank construction. Grouping/diversity were completed in the following milestone.
- The owner subsequently authorized completing Phase 1 after listing remaining tasks. Phase 1 now means the CSBM/GCN engineering foundation through audited pretraining-bank preparation (roadmap steps 1–3 for GCN), not the entire research project. Preserve configs/phase1_csbm_gcn.json: 50%-pool budget, five subsets, six paired seeds, 200/30 source-validation stopping, whole-subset 18/6/6 assignment with seed 401. Audit diagnostics use native training-pool features without labels; source-test labels are offline scoring only after partition/protocol freeze. This development source graph/split has prior viewed scores, so it is not a sealed paper benchmark. Do not start adaptation or make further commits/pushes implicitly.
- The owner explicitly authorized committing the completed Phase 1 code/config/tests/docs after reviewing the results. That authorization covers this completion checkpoint only. Runtime weights/reports stay excluded; no push or later commit is authorized implicitly. Preserve the historical dirty=true training provenance even after committing the implementation.
- Preserve the chosen versioned probability/feature convention, source full-graph visibility with label masking, separate labels, stratified 60/20/20 development splits, and independent source/target draws. Explain material methodological changes before changing these. Dependency changes stay project-local; never modify the global environment incidentally.
- Implement only the approved milestone. Follow the owner's stepwise bank-preparation scope; do not jump ahead to adaptation, generators, diffusion, or flow matching.

## Research references

- Local research materials live outside the repository in its parent directory: `Biased_Bozos_OnePager-3-1.pdf`, `edit.pdf`, and `AFML_Papers_and_Videos_Link_Sheet_Minimal.pdf`. They are research references, not runtime dependencies.
- Before implementing a research-specific component, consult original papers, official implementations, and maintained documentation. Ground research recommendations in web/Exa searches when relevant.
- Clone relevant references under the repository's parent `references/` directory; reuse existing clones. Never place upstream repositories inside this repository or modify them without an explicit decision.
- Inspect relevant code, dependencies, preprocessing, evaluation, and licenses before significant reuse. Prefer appropriate official APIs. Attribute adaptations and preserve required notices.
- Distinguish facts verified from documents/source, interpretations, recommendations, and unverified assumptions. Do not claim a novelty result from the proposal alone.

## Scientific correctness

- Preserve dataset identities and native benchmark splits. CSBM variants, Twitch-E, PyG Twitch, GOOD-Twitch, GOOD-Arxiv, and standard ogbn-arxiv are not interchangeable protocols.
- Keep source training, labeled meta-training, validation, held-out targets, deployment inputs, and evaluation labels distinct.
- Held-out target labels must not reach adaptation, conditioning, model selection, or label-free preprocessing. True-label homophily is an offline diagnostic, not a deployment statistic.
- Explicitly identify supervised oracles and any label-aware offline corruption. They are not unlabeled test-time methods.
- Record graph identity/lineage, configurations, preprocessing, versions, metrics, and independent generation/split/shift/model seeds as appropriate. Protect model and environment holdouts independently.
- Preserve official training masks when creating weight-bank diversity: any approved labeled subsampling must stay within the permitted source-training pool and have its own manifest.
- Never fabricate results, successful tests, or hardware support. Document nondeterminism and the limits of reproducibility.

## Software and portability

- Prefer Python, PyTorch, and PyTorch Geometric unless a change is justified and approved. Keep abstractions and dependencies minimal.
- Write readable modular code, practical type hints, and docstrings/comments explaining non-obvious assumptions.
- Use `pathlib`, configurable roots, and portable Python. Do not hardcode local absolute paths in application code.
- Keep graph connectivity sparse. Do not create dense adjacency matrices for normal graph processing.
- Do not silently normalize features, symmetrize edges, add self-loops, relabel, or regenerate splits. Record upstream processing and explicit transformations.
- Keep model-specific normalization and self-loop behavior out of the common loader. Check layer defaults to avoid duplicate preprocessing.
- Load/generate data on CPU; model execution should explicitly select supported CPU/CUDA/MPS devices. Test backend operations before claiming support; keep checkpoints portable.
- Discuss dependency changes first. Use an isolated environment, constrain important versions, and record configurations actually tested. Never modify the global environment as an incidental setup step.

## Verification and Git

- After implementation, review the diff, run relevant automated tests, execute a realistic small example, inspect output and edge cases, and report failures or limitations honestly.
- Loader tests must cover shapes, labels, edge validity, masks/indices, split integrity, reproducibility, and environment/label boundaries. Model/training tests must cover forward/backward behavior, checkpoint restoration, selection, and held-out label isolation.
- Never discard or overwrite existing work without approval. No destructive Git commands without explicit permission; never force-push.
- Do not commit until explicitly approved. Do not push until explicitly authorized. Do not stage unrelated files.
- Before proposing a commit, summarize files, purpose, verification, and unresolved concerns. Review the complete diff, including untracked files.
- Keep datasets, checkpoints, credentials, caches, temporary outputs, reference repositories, and large generated artifacts out of Git. Add appropriate ignore rules in the approved setup milestone before downloads/runs.

## Documentation

- `README.md`: objectives, actual setup/usage, concise verified progress.
- `docs/PROJECT_STATE.md`: completed/current/next work, implementation and test evidence, unresolved blockers.
- `docs/DECISIONS.md`: rationale, alternatives, assumptions, and pending proposals. Significant decisions need human approval; do not mark them accepted prematurely.
- `docs/learning/RESEARCH_ENGINEERING_NOTES.md`: maintain the beginner-friendly handbook. Define technical terms, explain data structure and flow, include only important project decisions/state, and provide worked examples, tables, viva/interview answers, and self-contained prompts suitable for a second-opinion ChatGPT conversation. Keep out operational logs and duplicated research inventories. Distinguish verified findings, derivations, recommendations, and unimplemented plans; never invent debugging experiences or results.
- Treat teaching as part of the work: explain important choices before hiding them behind code, invite a short explanation in the owner's own words, and correct gaps with concrete examples. Do not mark a topic understood merely because notes were written; keep checks lightweight and tied to the current milestone.
- Update these after meaningful milestones, preserve significant decisions, and avoid overlapping context files or logs of individual edits.
