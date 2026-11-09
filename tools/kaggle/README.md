# Kaggle campaign source snapshot

These are byte-for-byte source copies of the operational helpers used for the November 2026 research campaign. The active copies remain in the parent workspace's `compute-tools/` directory; no running worker or queue was changed for this Git delivery. `SOURCE_MANIFEST.json` records the copied bytes.

**Do not start a second queue while the existing queue is active.** These scripts are campaign-specific execution records, not a general-purpose Kaggle package. Driver files are templates with constants supplied by their builders and are not standalone local entry points.

## Working layout and recovery

The original helpers intentionally resolve paths in this layout:

```text
workspace/
  amortized-gnn-tta-generator-integration/  # this checkout (or an explicit equivalent symlink)
  compute-tools/                          # deployed helper copies and private job artifacts
```

For recovery, restore the helper files into `workspace/compute-tools/`, restore the per-job metadata/receipts/verified outputs, and recover the exact frozen source ZIP into `runs/formal-study-kaggle-v1/source.zip`. Its SHA-256 must be `26a40737a7c86e6a0d6d117e4ac5b2ee5c8fc6b21b96c81a3ac0c14dd3fbea56`. The code-only ZIP is preserved in the completed private design job's outputs; a fresh export of this later checkout has different documentation bytes and must not be silently substituted. Inspect existing job status and submission receipts before resuming; ambiguous submissions must never be blindly retried. Authentication is external through the official Kaggle CLI; no credential files are included.

Core reusable experiment CLIs live in `examples/` and `src/`. They accept explicit data/bank/output paths. These operational source copies retain the original workspace layout and job IDs for traceability.

## Files

- `prepare_kaggle_validation.py`, `prepare_kaggle_bank_job.py`: private immutable preflight/source-bank job builders.
- `gcn_study_driver.py`, `research_stage_driver.py`, `prepare_research_stage.py`: study templates and verified-stage builder.
- `continue_research.py`: sequential polling, verification and submission with exclusive locking and fail-closed gates.
- `verify_kaggle_outputs.py`, `verify_study_outputs.py`, `replay_real_study.py`: checkpoint/hash/inventory and saved-update replay checks.
- `evaluate_final_shard.py`, `diagnose_real_audit.py`: separately hashed audit/export runners using the frozen prediction/training library.
- `summarize_bank_progress.py`, `summarize_verified_design.py`, `merge_final_shards.py`, `plot_design_history.py`: evidence consolidation and plotting.
- `test_final_shard.py`, `test_continuation_queue.py`: small operational fixtures, tested in the original workspace layout; nine checks passed. The project test suite remains scoped to `tests/`.
- `configure_kaggle_auth.py`: hidden-entry local credential helper; preserves existing credential files and contains no credential value.

Source GNN training, generator training and numerical method settings are unchanged. The current campaign is the final planned main campaign; no additional tuning grid is authorized. Concise verified results are versioned in `reports/real-data-2026-11-08/`; datasets, raw prediction/checkpoint tensors, private job directories and large ZIP bundles remain outside Git.
