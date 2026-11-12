# Final campaign evidence — 12 November 2026

These compact exports reproduce the reviewed report package in the parent AFML workspace (`reports/experiment-summary-20261112/data/`). They contain the final 100-epoch campaign results; earlier five-epoch and CSBM studies are separate.

| File | Contents |
|---|---|
| [final_method_summary.csv](final_method_summary.csv) | All nine methods, final accuracy, paired gain, harm fraction and generator-seed variability. |
| [validation_method_summary.csv](validation_method_summary.csv) | Validation performance of the three generator families. |
| [paired_speedup_summary.json](paired_speedup_summary.json) | Paired online speedup ratios against each architecture's corresponding iterative adapter. |

Accuracy and gain are fractions in the CSVs: multiply by 100 for percent accuracy or percentage-point gain. `seed_mean_std` describes variability across three generator-seed means, not a confidence interval. Final means weight declared source-checkpoint/environment cases equally; country means weight JP and RU equally. The README rounds accuracy to two decimal places.

Timing uses generator seed 202801, one warmup and three repeats, from a canonical CPU target graph and resident models through usable predictions. It includes required preprocessing, transfer, adaptation and prediction; offline training is excluded. GCN uses the iterative LoRA reference, while GAT/GraphSAGE use iterative head adaptation. The range in the root README spans the protocol/architecture ratios in the JSON, not an accuracy-matched speedup or full-cost amortization claim.

## Coverage and verification limits

- E1: GCN conditioning-input ablation complete.
- E2: update-site design selection complete for all architectures.
- E3: controlled validation comparison complete for all architectures.
- E4: all 2,430 GCN score observations are present. The first four models have recovered score/replay metadata; their original prediction tensors are not fully locally verified. The final two models have hash-verified output. GAT and GraphSAGE final shards are absent.
- E5: 270 temporal and 540 country final observations, covering all architectures; 648 recorded saved-update replays with zero class differences.
- E6: 72 supervised diagnostic observations and recorded replays with zero class differences. These diagnostics are separate from unlabeled adaptation results.

Verification here refers to the reviewed downloaded inventories, hashes and saved replay records. The matrices were not rerun for this documentation update. No settings were selected using final-test outcomes; these results do not establish statistical significance or broadly optimal recipes.

## Archived artifacts

The full report PDFs, case-level numerical exports and verification records remain in the parent workspace's report package. Raw Kaggle artifacts were removed locally only after recovery copies were verified; consult `compute-tools/cleanup-20261112/README.md` in that workspace before running historical replay tools.

The locally repaired E3 outputs are preserved in a [private, versioned Kaggle backup](https://www.kaggle.com/datasets/deeptanshukumar/afml-e3-final-evidence-20261112/versions/1). Its stored-object checksum was checked against the local archive before deletion. Other protocol outputs retain their own versioned Kaggle locations in the cleanup receipts. Private artifacts require the owner's account access.
