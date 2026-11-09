# Final-campaign budget audit and selective changes

The owner requested avoiding pilot-sized limits, then explicitly limited increases to stages where added optimization is plausibly useful. The current immutable Kaggle generator job is unchanged: up to100 epochs, patience10 relative to trained epochs, min-delta0.0001, with epoch zero still eligible for checkpoint selection. E3/E5 use this same generator policy. The prior five-epoch study remains a separate limited-budget observation.

- Source banks:200 epochs/patience30, unchanged.190 of270 selected epoch200; convergence is not established. This is a fixed source-model baseline budget, not a proof of optimized source models. No blanket retraining is scheduled.
- Teachers and online iterative baselines:30 matched steps, unchanged. These support the declared fixed-budget comparison; they are not optimized accuracy ceilings.
- E6 supervised head/LoRA/full diagnostic: up to200 steps with inner-validation patience30, replacing30 steps. All three supervised references get the same opportunity, with their actual selected step/history retained. This is a more informative recoverability diagnostic, still not a proven ceiling. Existing38 focused core checks and9 external export/queue fixtures passed; no new real-data experiment was launched for this audit.
- Timing: existing1 warmup/3 repeats retained. The briefly proposed increase was reverted under the selective-change instruction. Claims must remain descriptive; no precise uncertainty claim is justified from three within-case repeats alone.
- Diagnostic job allocation: increased from1 to2 hours to accommodate the higher E6 step cap. This is a runtime bound, not a training change elsewhere.

Only later-stage operational diagnostic settings changed. The running job/source archive, E3/E4/E5 scientific settings, source data, splits, loss functions, learning rates, and source checkpoints are unchanged. The local queue was restarted at its read-only waiting point to load the diagnostic quota allocation; no Kaggle job was cancelled or duplicated. The live operational helpers are under compute-tools/earlystop_queue in the parent workspace.

Successful completion and verification of all six stages completes the planned experimental campaign. Remaining submission work includes consolidated analysis/figures/tables, source/label/provenance review, report conclusions/limitations/references and reproducibility/demo packaging. This does not imply every method improved accuracy or that all fitted components converged. No automatic extra tuning campaign is planned.
