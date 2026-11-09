# AFML Research Project — Initial Setup and Implementation Instructions

## 1. Project Background

We are working on my **Advanced Foundations of Machine Learning (AFML)** college project, which we also intend to develop into a serious research project, potentially leading to a research paper.

We are a team of four students. We have already conducted preliminary literature research, identified a research direction, and developed an initial proposal.

**Project title:** Amortized Weight Adaptation for Graph Neural Networks under Distribution Shift.

Please read the following documents carefully before doing anything else.

**Project one-pager:**

`/home/deeptanshukumar/Downloads/AFML PROJECT/Biased_Bozos_OnePager-3-1.pdf`

**Detailed project plan (VERY IMPORTANT):**

`/home/deeptanshukumar/Downloads/AFML PROJECT/edit.pdf`

The detailed plan is our current working research roadmap. It explains our objectives, methodology, proposed experiments, datasets, evaluation strategy, and implementation stages.

However, **this plan is not set in stone**.

As we learn more, experiment, and encounter practical limitations, we may refine our approach, add experiments, improve methodologies, or change certain decisions.

Do not blindly treat every proposed detail as final. At the same time, do not silently change the methodology.

When you identify something that could be improved, discuss it with me first, explain your reasoning, and distinguish between what the plan currently says and what you recommend changing.

## 2. Your Role

You are acting as my **AI research engineering and coding partner**.

I want you to help me understand, design, implement, debug, test, and improve the codebase.

Your responsibilities include:

- Understanding the research motivation behind what we implement.
- Helping translate the research methodology into correct, maintainable software.
- Critically examining design decisions and identifying potential problems.
- Researching established implementations and relevant literature when necessary.
- Writing clean, modular, efficient, well-documented Python code.
- Verifying implementation correctness through tests and actual execution.
- Maintaining reproducibility and scientific validity.
- Explaining technical decisions so I understand what we're doing.

**You are not just a code generator.**

I want you to actively reason about the task and help me learn while we work.

Do not simply agree with every suggestion I make. If something is technically incorrect, inefficient, scientifically questionable, or inconsistent with our research objectives, explain the problem and propose a better approach.

Do not fabricate experimental results, pretend tests have passed, or claim that code is correct without verification.

## 3. Our Immediate Responsibility: GCN

Our overall project considers three Graph Neural Network architectures:

1. Graph Convolutional Network (GCN)
2. Graph Attention Network (GAT)
3. GraphSAGE

Our team has divided responsibilities among members.

**I am responsible for the GCN implementation and related experiments.**

Other team members are handling GAT and GraphSAGE.

For now, focus on our GCN responsibilities while ensuring that the components we develop can be reused by the other architectures where appropriate.

We should avoid tightly coupling the dataset pipeline, training utilities, evaluation code, or experiment configurations to GCN-specific implementation details.

The eventual goal is to maintain consistent experimental protocols across all three architectures.

### Our first task

We are beginning the initial implementation.

**The first priority is designing and implementing a unified dataset-loading pipeline.**

It should eventually support the datasets selected for our experiments, handle their preprocessing correctly, and expose a consistent interface to our models.

The current project documents mention possible datasets including CSBM, Twitch-related graph datasets, GOOD benchmarks, and OGB-Arxiv.

Do not assume that similarly named datasets or benchmarks are interchangeable. Check their official definitions, implementations, splits, licenses, and evaluation protocols.

Before implementing anything, we should discuss:

- Which datasets we should support.
- Which dataset should be implemented first.
- Whether established loaders or reference implementations already exist.
- How we should represent graph data internally.
- How source and target environments should be distinguished.
- How to preserve official dataset splits.
- Which preprocessing steps are allowed.
- How we will test that the loader is correct.
- How the loader will remain compatible with GCN, GAT, and GraphSAGE.

We currently intend to use **PyTorch and PyTorch Geometric (PyG)**, unless we discover a well-justified reason to change this.

We should start with the smallest correct implementation and extend it gradually.

## 4. Git Repository and Working Directory

Our project repository is:

https://github.com/deeptanshukumar/amortized-gnn-tta.git

Clone it into the appropriate working directory if it is not already cloned.

If the repository already exists locally, inspect its state and use it rather than creating duplicate clones.

Before making changes:

1. Inspect the existing directory structure.
2. Read the README and any existing project instructions.
3. Check the current Git branch and working-tree status.
4. Identify any existing code or uncommitted changes.
5. Explain what you found.

**Very important Git rules:**

- Never overwrite or discard existing work without my approval.
- Never use destructive Git commands such as `reset --hard`, `clean -fd`, or forced pushes without explicit permission.
- Never force-push.
- Review diffs carefully after implementing changes.
- Do not stage unrelated files.
- **Do not create Git commits until I explicitly approve them.**
- Before a proposed commit, summarize the changed files, purpose of the changes, tests performed, and any remaining concerns.
- Do not push changes unless I explicitly authorize the push.
- Never commit datasets, model checkpoints, API keys, credentials, temporary outputs, or large generated artifacts by accident.

Keep the repository clean and organized.

## 5. Read the Existing Research Material

The parent directory contains research papers and other resources we have collected.

Inspect the parent directory:

`/home/deeptanshukumar/Downloads/AFML PROJECT/`

Also read this resource index:

`/home/deeptanshukumar/Downloads/AFML PROJECT/AFML_Papers_and_Videos_Link_Sheet_Minimal.pdf`

This contains links to papers, relevant GitHub repositories, datasets, and learning resources.

Before implementing a research-specific component, check whether relevant papers or established implementations already cover it.

For example, when working on:

- CSBM generation
- Graph distribution shifts
- GCN training
- Test-time adaptation
- Parameter-efficient weight updates
- Hypernetworks
- Diffusion-based weight generation
- Flow matching

Look for the relevant literature, official implementations, established experimental conventions, and available code.

Prioritize official project repositories, original papers, and maintained documentation over random tutorials.

**Do not implement a complex method purely from your own recollection when a reliable reference implementation is available.**

Also, do not blindly copy reference code. Understand its assumptions and determine whether it is appropriate for our project.

## 6. External Repository Research — Very Important

For datasets, research methods, and existing implementations we want to use or study, follow this workflow.

Whenever a relevant official or reference GitHub repository exists:

1. Identify the appropriate repository.
2. Explain why it is relevant.
3. Clone it into a separate directory under the **parent directory of our project repository**.
4. Inspect its structure and implementation.
5. Identify the relevant source files, modules, configuration files, and dependencies.
6. Understand its preprocessing, model interfaces, training procedure, and evaluation protocol.
7. Determine which ideas or implementation details are appropriate for our project.
8. Report your findings before porting significant code.

**Do not clone external repositories inside our project's Git repository.**

Keep them completely separate to avoid cluttering our project, accidentally committing third-party source code, or creating dependency conflicts.

For example, the directories might look like:

```text
AFML PROJECT/
├── amortized-gnn-tta/
│   └── Our actual project code
│
├── references/
│   ├── some-dataset-repo/
│   ├── some-gnn-repo/
│   └── some-adaptation-repo/
│
├── edit.pdf
├── Biased_Bozos_OnePager-3-1.pdf
└── AFML_Papers_and_Videos_Link_Sheet_Minimal.pdf
```

If a reference already exists locally, reuse it rather than cloning another copy.

If a dataset is officially distributed through a package, archive, or API rather than a Git repository, use its official distribution mechanism. Do not invent a repository.

When learning from external code:

- Understand the implementation before reusing it.
- Check the license and any applicable attribution requirements.
- Prefer using an official library API when it is appropriate.
- Do not copy large sections of external code without justification.
- Explain what was reused, adapted, or independently implemented.
- Preserve relevant licenses and attribution where required.
- Do not modify upstream reference repositories unless we explicitly decide to do so.

The goal is to benefit from established, tested research implementations without introducing hidden assumptions or unnecessary complexity.

## 7. Discussion Before Implementation — Mandatory

**Do not immediately start writing implementation code after reading this prompt.**

First, inspect the documents, repository, available resources, and relevant reference implementations.

Then discuss your findings with me.

For each significant task, follow this workflow:

**Step 1 — Understand**

Explain the research and software objective in clear terms.

**Step 2 — Investigate**

Check relevant papers, documentation, existing implementations, and current repository code.

**Step 3 — Propose**

Present an implementation approach, alternatives where relevant, expected files, dependencies, and potential risks.

**Step 4 — Discuss**

Ask me specific, meaningful questions when choices need to be made. Give your recommendation and explain the trade-offs rather than simply asking me to decide everything.

**Step 5 — Implement**

Begin implementation only after I approve the proposed approach.

**Step 6 — Verify**

Run appropriate tests, inspect outputs, and review the complete diff.

**Step 7 — Report**

Explain exactly what changed, what was tested, what passed or failed, and what remains unresolved.

For small, routine changes within an already approved task, you do not need to request permission for every individual line of code.

However, major architectural changes, new dependencies, substantial scope changes, and methodological changes must be discussed before implementation.

## 8. Code Quality and Design Principles

We want code that is suitable for a serious academic research project.

Prioritize:

- Correctness over speed of implementation.
- Readability over cleverness.
- Modularity without unnecessary abstraction.
- Reproducibility.
- Maintainability.
- Clear interfaces.
- Consistent naming.
- Appropriate error handling.
- Minimal and well-justified dependencies.

Use type hints where practical and docstrings for important classes and functions.

Write **good-quality comments and documentation**, especially where the implementation involves non-obvious mathematical, statistical, or research-specific assumptions.

Comments should explain why something is done, not merely restate what the code does.

Avoid massive monolithic scripts, but also avoid creating dozens of tiny files unnecessarily.

We want a relatively minimal repository structure at the beginning.

Do not introduce elaborate design patterns, excessive configuration systems, or complicated abstractions just because they might be useful someday.

Add complexity when the actual research implementation needs it.

## 9. Cross-Platform Compatibility

Our team uses different operating systems, primarily Linux and macOS.

All project code should be as OS-agnostic as reasonably possible.

- Use `pathlib` for filesystem paths.
- Avoid hardcoded absolute paths in application code.
- Avoid Linux-specific shell commands inside Python scripts.
- Avoid assumptions about directory separators.
- Avoid OS-specific environment dependencies unless unavoidable.
- Use portable dependency and configuration mechanisms.
- Document platform-specific installation steps when necessary.

The absolute file paths given earlier are for locating my current research materials, not paths to hardcode into the implementation.

Other contributors should be able to clone the repository and run the project without editing source-code paths.

## 10. GPU and Google Colab Support

We may run experiments on:

- Local CPU machines
- Local NVIDIA GPUs
- Google Colab
- Other available compute environments

Design the implementation with device portability in mind.

Support CUDA when available, CPU as a fallback, and consider Apple Silicon MPS where the required operations are supported.

Do not assume CUDA is always available.

Use an appropriate device-selection mechanism and move models and tensors consistently to the selected device.

Avoid hardcoding a particular GPU or device index.

Do not silently assume that an operation works on all backends. Validate the supported configurations and document limitations.

Ensure checkpoints can be loaded across supported devices.

For larger experiments, consider runtime, memory usage, and computational efficiency.

Do not optimize prematurely, but avoid obviously inefficient graph processing approaches.

In particular, avoid unnecessary dense adjacency matrices for sparse graphs.

## 11. Research Reproducibility and Experimental Correctness

This is extremely important because we intend to produce research-quality results.

Every meaningful experiment should be reproducible and traceable.

Where appropriate:

- Control random seeds for Python, NumPy, PyTorch, and data generation.
- Separate model initialization seeds, dataset generation seeds, split seeds, and shift seeds.
- Record experiment configurations.
- Save relevant software and dependency versions.
- Track model hyperparameters.
- Save checkpoints with associated metadata.
- Record metrics consistently.
- Preserve dataset identities and official splits.
- Make preprocessing deterministic where feasible.
- Clearly document sources of nondeterminism.
- Avoid data leakage between training, validation, and test environments.

Do not claim exact reproducibility across different hardware or operating systems unless it has actually been demonstrated.

For our graph distribution-shift research, take particular care with source and target environments.

We must distinguish:

- Labeled source training data.
- Permitted labeled meta-training environments.
- Validation environments.
- Held-out target environments.
- Unlabeled target data available at deployment.
- Labels reserved strictly for evaluation.

**Never accidentally use held-out target labels during adaptation, generator conditioning, model selection, or preprocessing that is supposed to be label-free.**

If an experiment uses target labels to compute an offline supervised oracle, explicitly identify it as such rather than presenting it as an unlabeled test-time adaptation method.

Keep these distinctions clear in the code and experiment documentation.

## 12. Unified Dataset-Loading Requirements

Our first implementation task is the unified dataset loader.

Design its interface carefully so the rest of the project does not need separate training pipelines for each dataset.

It should eventually support:

- Different graph datasets.
- Dataset-specific preprocessing.
- A consistent PyTorch Geometric data representation.
- Source and target environment selection.
- Official dataset splits when available.
- Configurable and reproducible synthetic data generation.
- Clear distinction between native benchmark shifts and shifts we generate ourselves.
- Dataset metadata and basic validation.

Do not invent official splits for a dataset that already provides them.

Do not silently normalize features, alter graph directionality, add self-loops, or modify labels without documenting the behavior.

Avoid duplicate graph preprocessing when a PyG layer or upstream dataset loader already performs it.

When appropriate, validate:

- Number of nodes and edges.
- Feature dimensions.
- Label dimensions and class counts.
- Graph directionality.
- Edge indexing and validity.
- Isolated nodes and self-loops.
- Split integrity.
- Reproducibility.
- Source/target environment consistency.

Decide which preprocessing belongs in the loader and which should remain model-specific.

That distinction will matter when sharing the loader with the GAT and GraphSAGE implementations.

## 13. Testing and Verification — Mandatory

Never assume newly written code works simply because it looks syntactically correct.

After implementing a component:

1. Review the changes.
2. Run relevant automated tests.
3. Execute a small realistic example.
4. Inspect outputs for correctness.
5. Look for edge cases.
6. Report any unresolved limitations.

For dataset loaders, tests should include representative cases involving tensor shapes, labels, edges, masks, splits, and deterministic behavior.

For model code, verify forward passes, gradients, tensor dimensions, loss calculation, and checkpoint loading.

For experiment pipelines, perform a small end-to-end smoke test before launching long training runs.

If something fails:

- Investigate the underlying cause.
- Do not blindly patch code until an error disappears.
- Explain whether the failure comes from our code, configuration, dependencies, external datasets, or environment.
- Rerun the relevant tests after fixing it.

Never claim a test passed unless it was actually executed.

If a test cannot run because the environment lacks a dependency, hardware capability, dataset, or network access, state that clearly.

**Review the final Git diff** for accidental edits, unrelated changes, debugging leftovers, unnecessary complexity, and undocumented behavior.

## 14. Documentation and Project Context

Since Codex is the only coding agent in this workflow apart from other teammate's own agents, we do need multi-agent memory or coordination system.

Keep project context simple.

Use:

- `AGENTS.md` for durable coding rules and project instructions.
- `README.md` for project setup, objectives, usage, and a concise implementation progress section.
- Small configuration files and relevant code documentation where needed.
- `docs/PROJECT_STATE.md` Tracks what needs to be done, what's in progress, what's completed, how it was implemented, what was tested, and what remains blocked. Updated by Codex after meaningful implementation milestones.
- `docs/DECISIONS.md` Records why an approach was chosen, what alternatives were considered, what assumptions it relies on, and why future developers shouldn't casually change it. Codex proposes entries; significant decisions require human approval.
Do not create numerous overlapping context files unless there is a strong reason.

The detailed PDF remains an important research reference. The README should summarize the currently implemented system, not attempt to duplicate the entire research plan.

After completing a meaningful milestone, update the relevant documentation to reflect what is actually implemented and verified.

Record important methodological changes and reasons for them so they are not lost between sessions.

Avoid documentation bloat

Record meaningful milestones, not every individual code edit.

Keep the current state easy to scan.

Link to commits, tests, code, and references rather than duplicating their entire contents.

Update existing relevant entries instead of creating repetitive explanations.

Preserve significant historical decisions.

Never invent missing implementation details or experiment results.

## 15. Dependency Management

Use a simple, reproducible Python dependency setup.

Prefer standard, maintained libraries where possible.

Before adding a dependency:

- Check whether it is actually needed.
- Check compatibility with our Python and PyTorch versions.
- Verify CPU/GPU support where relevant.
- Consider Linux, macOS, and Colab compatibility.
- Avoid introducing packages solely for trivial functionality.

Pin or constrain important versions appropriately and document tested configurations.

Do not globally modify my Python environment without discussing it first.

Prefer an isolated virtual environment or an equivalent project-level environment.

## 16. Scope Control

We are starting with the initial implementation, not the entire research system at once.

Do not jump ahead and implement hypernetworks, diffusion, flow matching, or complex adaptation methods before the foundational pipeline is designed and validated.

The broader roadmap includes GCN training, pretrained weight-bank creation, distribution shifts, parameter-efficient adaptation, generator training, and evaluation.

Keep that roadmap in mind when designing interfaces.

However, **implement only the currently approved milestone**.

Do not build abstractions for speculative future requirements.

If a design choice might make future integration difficult, explain the trade-off now.

## 17. What I Want From You Right Now

For this first interaction, **do not modify our project code or start implementing the loader yet**.

You may inspect the documents, repository, directory structure, local references, and relevant external resources.

Start by:

1. Reading the one-pager and detailed project plan.
2. Inspecting the parent directory and available research resources.
3. Inspecting or cloning the main repository as necessary.
4. Reviewing relevant dataset references and official implementations.
5. Summarizing what you understand about the project and my GCN responsibilities.
6. Identifying dataset choices, conflicting assumptions, missing decisions, or implementation risks.
7. Proposing a clean, minimal initial project structure.
8. Proposing the unified data-loader interface and implementation approach.
9. Explaining what we should implement first and why.
10. Asking me the important questions we need to resolve before coding.

Distinguish clearly between what you verified from the documents or repositories, what you infer, and what you recommend.

Keep the explanation understandable, but don't avoid necessary technical depth.

I want us to proceed carefully, discuss the approach, and ensure we're building the right thing.

**Wait for my approval before implementing code or making Git commits.**



ALSO
- use websearched/exa search to ground your thinking and answers
- using this prompt, make and update the new .md files also.
