# Historical contributor experiments

These five scripts and JSON results were published in `c137e28`. Keep their
original contents as the record of that experiment. They use private absolute
paths, do not save complete fitted generators, and have component timing
boundaries. In `run_full_reproduction.py`, diffusion also reuses conditioning
outside its timer. Their approximately 45x ratios are not complete prediction
pipeline speedups. The uploaded arithmetic was independently checked; exact
model replay still needs the matching artifact bundle and RNG/hardware settings.

Use the portable shared commands in `examples/` for new experiments:

| Historical experiment | Shared configuration or command |
| --- | --- |
| Full compact-head comparison | `run_generator_review.py`, `conditioning: ["compact"]`, `update_scope: "head"` |
| A-D input versions | Same runner, `conditioning: ["A", "B", "C", "D"]` |
| Moments versus structural | Same runner, `conditioning: ["compact", "structural"]` |
| Hypernetwork head/backbone/all | Same runner, `methods: ["hypernetwork"]`; separate plans with each `update_scope` |
| Diffusion head/backbone | Same runner, `methods: ["diffusion"]`; separate plans with each `update_scope` |

Start from `configs/generator_csbm_review.json`, supply an explicit artifact root,
and use a fresh output directory for every plan. All runs use saved model
partitions and validation environments. `replay_generator_review.py` independently
reconstructs named weights and checks saved predictions/metrics. Direct
backbone/all deltas are **not LoRA**. IM remains a head-only reference, so broader
generated updates have no matched speedup ratio. `benchmark_adapters.py` measures
broader generators against frozen inference without an unmatched optimizer ratio.

`compact` retains the original hidden moments plus the separate raw update-site
weights. A-D disable that extra raw-weight input. Thus compact is an additional
legacy-method baseline, not a fifth version in the nested A-D ablation.

Do not overwrite these historical results with reruns. New runtime artifacts
belong under ignored `runs/` directories, not in this archive or Git.
