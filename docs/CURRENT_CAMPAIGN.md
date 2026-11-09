# Current 100-epoch campaign and teammate handoff

The owner cancelled the earlier five-epoch E3 job and authorized a fresh run with up to100 epochs and early stopping, followed by E3–E6. This supersedes older no-method-change/five-epoch instructions for the new campaign only. Preserve old results and source archives.

## Exact submitted code

Private Kaggle job: https://www.kaggle.com/code/deeptanshukumar/afml-gcn-trained-patience-100-20261109

Source archive SHA-256: `5fa2a6b8dbfcc869eb50eddbb95db43a44d6550148676545dcd0c326a738a96a`.

Fresh server pull and embedded archive verification found no differences in85 scientific source/example/config files. See [launch audit](../reports/earlystop-100-2026-11-09/README.md). The draft's missing result fields/import were fixed before this archive was submitted. Focused tests:37 passed. Actual remote GPU component checks:9 passed. These checks do not establish successful adaptation.

The generator cap is100, patience10, minimum validation accuracy improvement0.0001 (0.01 percentage points). The monitored criterion is mean validation accuracy, including the initial zero-update candidate. Ten epochs without an improvement above that threshold can stop training even if a poor trained epoch is recovering. Best checkpoint and stop/early-stopped metadata are retained. Source GNN training is unchanged. Direct-CE hypernetworks use the same CE/update-L2 formula and coefficient as matched CSBM runs; diffusion has its own objective, consistently implemented across datasets. Data, class counts, descriptors and candidate-selection recipes still differ between studies.

## What runs

The current job reruns GCN E1 A–D LoRA and E2 head/LoRA design with follow-ups, conditioning controls, baselines and replay. All source banks are reused after verification; generator fits start fresh. E3 is subsequently all three architectures/three seeds/direct and teacher hypernetworks/diffusion. E4 is sealed controlled final evaluation. E5 includes strict temporal Arxiv and homogeneous Pair-Align MAG country graphs (US source, CN/DE practice, FR validation, JP/RU final), with separate generators. E6 contains supervised practice diagnostics and efficiency analysis. No final-result claim follows from submission.

## Queue and access

The active local queue is `workspace/compute-tools/earlystop_queue/continue_research.py`; inspect its `RESEARCH_QUEUE.json` and `queue.log`. It requires the host to stay running and connected. The separate source snapshot and restoration assumptions are documented in [tools/kaggle/earlystop_queue](../tools/kaggle/earlystop_queue/README.md). It verifies upstream artifacts before submission and stops on failure or insufficient quota. End-to-end progression awaits completed outputs. Do not run a second queue from another Codex session.

Git includes code, configs, helper source, tests, small source audits and earlier verified results/plots. Private Kaggle outputs, datasets, credentials, large ZIPs, full checkpoints and prediction tensors are not in Git. Teammates can review the source and tracked reports immediately; private artifact access must be arranged through the owner. Do not share authentication tokens. The original campaign helper snapshot remains separate for auditability.

## Latest stopping-policy correction

The owner explicitly selected patience10 against the best trained epoch only. Epoch zero remains eligible for checkpoint selection but never sets or resets the patience reference. Thus a trained epoch may improve on its predecessor and continue even while below frozen accuracy. A regression check demonstrates recovery below frozen followed by a plateau, stopping only after the trained-epoch patience is exhausted. The38 focused tests passed. The former baseline-referenced100-epoch job was cancelled by the owner before this corrected replacement was submitted. The exact new source audit is in reports/earlystop-100-2026-11-09/trained-patience/. This section supersedes earlier wording about epoch zero participating in stopping.

See [selective budget audit](BUDGET_AUDIT.md) for the later E6 diagnostic budget change; the running fitting snapshot is unchanged.
