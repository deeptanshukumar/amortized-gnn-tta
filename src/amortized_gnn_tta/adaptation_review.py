"""Frozen baselines and PEFT experiments on explicitly permitted development data.

All selection lists/configs are written before scoring. This runner deliberately
rejects held-out MODELS and ENVIRONMENTS; it is not the final generator evaluator.
Native meta-environment test nodes are scored offline, never used for fitting.
"""

from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from .adaptation import (
    AdaptationConfig, PEFTConfig, development_supervision, fit_supervised_development, fit_unlabeled,
)
from .adaptation_artifacts import load_weight_delta, save_weight_delta, state_sha256
from .bank_artifacts import VerifiedBank, artifact_path, file_sha256, validate_model_partition
from .data import load_shift_environment, validate_environment_manifest
from .data.api import graph_fingerprint
from .evaluation import predict, score_logits
from .models import build_model
from .training import SourceTrainingData, load_checkpoint, software_versions, validate_source_training_data

OBJECTIVES = ("supervised_development", "entropy", "information_maximization")


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def prediction_diagnostics(logits: torch.Tensor) -> dict[str, Any]:
    log_p = logits.log_softmax(dim=1)
    p = log_p.exp()
    mean_log_p = torch.logsumexp(log_p, dim=0) - math.log(len(logits))
    counts = torch.bincount(logits.argmax(dim=1), minlength=logits.shape[1])
    return {"mean_prediction_entropy": float(-(p * log_p).sum(dim=1).mean()),
            "marginal_prediction_entropy": float(-(mean_log_p.exp() * mean_log_p).sum()),
            "predicted_class_counts": counts.tolist(), "majority_fraction": int(counts.max()) / len(logits)}


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["objective"], row["family"])].append(row)
    result = []
    for (method, objective, family), group in sorted(groups.items()):
        def mean(field):
            return sum(r[field] for r in group) / len(group)
        result.append({"method": method, "objective": objective, "family": family, "pairs": len(group),
                       "mean_before_accuracy": mean("before_accuracy"),
                       "mean_after_accuracy": mean("after_accuracy"),
                       "mean_gain_percentage_points": 100 * mean("accuracy_gain"),
                       "improved_pairs": sum(r["accuracy_gain"] > 0 for r in group),
                       "worsened_pairs": sum(r["accuracy_gain"] < 0 for r in group),
                       "mean_fit_seconds": mean("fit_seconds"),
                       "mean_delta_l2": mean("delta_l2")})
    return result


def run_development_review(
    bank: VerifiedBank, partition: dict[str, Any], environment_manifest: dict[str, Any],
    source: SourceTrainingData, *, model_ids: Sequence[str], environment_ids: Sequence[str],
    output_dir: Path, peft_configs: Sequence[PEFTConfig] = (PEFTConfig("head"), PEFTConfig("lora")),
    objectives: Sequence[str] = OBJECTIVES, config: AdaptationConfig = AdaptationConfig(),
    device: str = "cpu",
) -> dict[str, Any]:
    """Run fresh zero-delta episodes; save every result, including regressions.

    Set peft_configs/objectives to empty tuples for a frozen-only development
    sweep. No job is discarded based on improvement, convergence, or delta size.
    Supervised selection uses native meta train/val, evaluation uses native test.
    Unlabeled methods see all graph nodes without labels, as transductive TTA.
    """
    validate_model_partition(bank, partition)
    validate_environment_manifest(environment_manifest)
    validate_source_training_data(source)
    reference_config = json.loads(json.dumps(source.provenance["graph_metadata"]["config"]))
    if (environment_manifest["reference"] != reference_config
            or graph_fingerprint(source.graph) != bank.subsets["source"]["graph_fingerprint"]):
        raise ValueError("Environment reference or source graph does not match the source bank")
    model_ids, environment_ids = list(model_ids), list(environment_ids)
    if (not model_ids or len(set(model_ids)) != len(model_ids)
            or not set(model_ids) <= set(partition["model_ids"]["meta_train"])):
        raise ValueError("Choose distinct meta_train model IDs only; held-out models are reserved")
    all_environments = {e["id"]: e for e in environment_manifest["environments"]}
    if (not environment_ids or len(set(environment_ids)) != len(environment_ids)
            or any(e not in all_environments or all_environments[e]["role"] != "meta_train"
                   for e in environment_ids)):
        raise ValueError("Development review accepts distinct meta_train environments only")
    peft_configs, objectives = list(peft_configs), list(objectives)
    if (len(set(p.method for p in peft_configs)) != len(peft_configs)
            or len(set(objectives)) != len(objectives) or set(objectives) - set(OBJECTIVES)
            or bool(peft_configs) != bool(objectives)):
        raise ValueError("Choose distinct methods/objectives, or both empty for baseline only")
    full_diagnostic = any(p.method == "full" for p in peft_configs)
    if full_diagnostic and (len(peft_configs) != 1 or objectives != ["supervised_development"]):
        raise ValueError("Run full fine-tuning separately with supervised_development only")
    model_rows = {m["model_id"]: m for m in bank.manifest["models"]}
    if (source.graph.x.shape[1] != environment_manifest["reference"]["num_features"]
            or bank.manifest["model_spec"]["config"]["out_channels"] != 2):
        raise ValueError("Bank and CSBM feature/class schemas disagree")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "protocol": "full-finetuning-development-v1" if full_diagnostic else "peft-development-review-v1",
        "scope": "development only; no held-out model/environment use",
        "model_ids": model_ids, "environment_ids": environment_ids,
        "bank_sha256": bank.sha256, "environment_manifest_sha256": environment_manifest["manifest_sha256"],
        "peft_configs": [asdict(p) for p in peft_configs], "objectives": objectives,
        "adaptation_config": asdict(config), "model_spec": bank.manifest["model_spec"],
        "test_nodes": "native meta-environment test indices; scoring only",
        "unlabeled_visibility": "entire target graph; no labels or construction metadata",
        "versions": software_versions(), "device": device, "torch_num_threads": torch.get_num_threads(),
        "dropout_during_adaptation": False, "adaptation_weight_decay": 0.0,
    }
    _write_json(output / "plan.json", plan)
    _write_json(output / "environment_manifest.json", environment_manifest)
    _write_json(output / "model_partition.json", partition)
    rows, baselines = [], []
    planned = len(model_ids) * len(environment_ids) * len(peft_configs) * len(objectives)
    progress = {"status": "running", "planned_adaptations": planned, "completed_adaptations": 0,
                "planned_baselines": len(model_ids) * len(environment_ids), "completed_baselines": 0}
    def save_progress():
        (output / "progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
    save_progress()
    try:
        for environment_id in environment_ids:
            item = all_environments[environment_id]
            bundle = load_shift_environment(environment_manifest, environment_id)
            graph = bundle.unlabeled_input(environment_id)
            indices = bundle.select(environment_id, "test")
            # This explicit offline evaluator owns test labels. Neither adapter
            # fitting API accepts this tensor or its label-bearing bundle.
            labels = bundle.evaluation_labels(environment_id, "test")
            for model_id in model_ids:
                row = model_rows[model_id]
                checkpoint = artifact_path(bank.directory, row["checkpoint"])
                if file_sha256(checkpoint) != row["checkpoint_sha256"]:
                    raise ValueError("Source checkpoint changed after bank verification")
                model, payload = load_checkpoint(checkpoint, build_model)
                if payload["model_spec"] != bank.manifest["model_spec"]:
                    raise ValueError("Checkpoint model disagrees with bank")
                initial_hash = state_sha256(model.state_dict())
                before_logits = predict(model, graph, device=device)
                before = asdict(score_logits(before_logits, indices, labels))
                source_before = asdict(score_logits(predict(model, source.graph),
                                                    source.val_indices, source.val_labels))
                base_row = {"model_id": model_id, "environment_id": environment_id,
                            "family": item["family"], "shift": item["shift"], "severity": item["severity"],
                            "metrics": before, "source_validation_before": source_before,
                            "prediction_diagnostics": prediction_diagnostics(before_logits)}
                baselines.append(base_row)
                _write_json(output / f"baseline_{len(baselines):05d}.json", base_row)
                progress["completed_baselines"] += 1
                save_progress()
                for peft in peft_configs:
                    for objective in objectives:
                        start = time.perf_counter()
                        if objective == "supervised_development":
                            result = fit_supervised_development(
                                model, graph, development_supervision(bundle, environment_id), peft, config, device=device)
                        else:
                            result = fit_unlabeled(model, graph, peft, config, objective=objective, device=device)
                        fit_seconds = time.perf_counter() - start
                        after_logits = predict(result.model, graph)
                        after = asdict(score_logits(after_logits, indices, labels))
                        source_after = asdict(score_logits(predict(result.model, source.graph),
                                                           source.val_indices, source.val_labels))
                        if state_sha256(model.state_dict()) != initial_hash:
                            raise ValueError("Adaptation mutated the caller's source model")
                        job_dir = output / "jobs" / f"job_{len(rows) + 1:05d}"
                        context = {"model_id": model_id, "environment_id": environment_id,
                                   "model_role": "meta_train", "environment_role": "meta_train",
                                   "source_checkpoint_sha256": row["checkpoint_sha256"],
                                   "bank_sha256": bank.sha256,
                                   "environment_manifest_sha256": environment_manifest["manifest_sha256"],
                                   "graph_sha256": graph_fingerprint(graph),
                                   "evaluation_indices": indices.tolist(), "before": before, "after": after}
                        artifact = save_weight_delta(job_dir / "weight_delta.pt", result, payload["model_spec"],
                                                     graph, context=context)
                        restored, _ = load_weight_delta(job_dir / "weight_delta.pt")
                        torch.testing.assert_close(after_logits, predict(restored, graph), rtol=1e-4, atol=1e-5)
                        result_row = {
                            "model_id": model_id, "environment_id": environment_id, "family": item["family"],
                            "shift": item["shift"], "severity": item["severity"], "method": peft.method,
                            "objective": objective, "before_accuracy": before["accuracy"],
                            "after_accuracy": after["accuracy"], "accuracy_gain": after["accuracy"] - before["accuracy"],
                            "before": before, "after": after, "source_validation_before": source_before,
                            "source_validation_after": source_after,
                            "fit_seconds": fit_seconds, "selected_step": result.selected_step,
                            "steps_run": len([h for h in result.history if h["step"] > 0]),
                            "selection": result.selection, "delta_l2": artifact["delta_summary"]["delta_l2"],
                            "relative_delta_l2": artifact["delta_summary"]["relative_delta_l2"],
                            "trainable_parameters": artifact["trainable_parameters"],
                            "model_parameters": artifact["model_parameters"],
                            "prediction_diagnostics": prediction_diagnostics(after_logits),
                            "weight_delta": str((job_dir / "weight_delta.pt").relative_to(output)),
                            "weight_delta_sha256": file_sha256(job_dir / "weight_delta.pt"),
                            "cpu_reload_predictions_close": True,
                        }
                        _write_json(job_dir / "metrics.json", result_row)
                        rows.append(result_row)
                        progress["completed_adaptations"] = len(rows)
                        save_progress()
                        print(f"{model_id} | {environment_id} | {peft.method}/{objective}: "
                              f"{100 * before['accuracy']:.2f}% -> {100 * after['accuracy']:.2f}% "
                              f"delta={result_row['delta_l2']:.5g}", flush=True)
        report = {"status": "complete", "protocol": plan["protocol"], "plan_sha256": file_sha256(output / "plan.json"),
                  "baselines": baselines, "adaptations": rows, "summary": _summary(rows),
                  "limitations": ["Development meta models/environments only; no generator or final test result",
                                  "Related shifts, models and node samples are not independent trials",
                                  "Supervised reference uses labeled meta train/val; unlabeled baselines do not",
                                  "Update methods have different parameter counts/sites; no matched-capacity claim",
                                  "A supervised reference is not a proven optimization or accuracy upper bound",
                                  "Timing is local execution evidence, not a controlled speed benchmark"]}
        _write_json(output / "review.json", report)
        title = "Full fine-tuning development diagnostic" if full_diagnostic else "PEFT development review"
        lines = [f"# {title}", "", "Development models/environments only; no generator result.", "",
                 "| Update | Objective | Family | Pairs | Before | After | Gain (pp) |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
        for s in report["summary"]:
            lines.append(f"| {s['method']} | {s['objective']} | {s['family']} | {s['pairs']} | "
                         f"{100*s['mean_before_accuracy']:.2f}% | {100*s['mean_after_accuracy']:.2f}% | "
                         f"{s['mean_gain_percentage_points']:+.2f} |")
        lines += ["", "Every adaptation has initial/adapted/delta weights, factors, history and a CPU reload check.", ""]
        lines += [f"- {note}" for note in report["limitations"]]
        (output / "REVIEW.md").write_text("\n".join(lines) + "\n")
        progress["status"] = "complete"
        save_progress()
        return report
    except Exception as exc:
        progress.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_progress()
        raise
