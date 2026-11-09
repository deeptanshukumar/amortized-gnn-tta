# Verified formal source banks

Verified unique checkpoints: **270/270**. Final target labels scored: **No**.

| Source protocol | Architecture | Verified / expected | Summed fit seconds | Mean fit seconds |
|---|---|---:|---:|---:|
| ogbn-arxiv | gcn | 30/30 | 569.47 | 18.98 |
| ogbn-arxiv | gat | 30/30 | 1048.46 | 34.95 |
| ogbn-arxiv | graphsage | 30/30 | 659.16 | 21.97 |
| ogbn-arxiv-temporal | gcn | 30/30 | 82.23 | 2.74 |
| ogbn-arxiv-temporal | gat | 30/30 | 133.48 | 4.45 |
| ogbn-arxiv-temporal | graphsage | 30/30 | 73.80 | 2.46 |
| pair-align-mag-country | gcn | 30/30 | 336.07 | 11.20 |
| pair-align-mag-country | gat | 30/30 | 614.08 | 20.47 |
| pair-align-mag-country | graphsage | 30/30 | 351.48 | 11.72 |

Times are recorded source-fit/checkpoint-validation wall times on the actual GPU backend, excluding dataset download, runtime startup and broader research search. Checkpoints reused by later jobs are counted once. Fixed budgets and late selected epochs do not establish convergence. Source-only diversity audits are separate evidence, not a universal sufficiency guarantee.
