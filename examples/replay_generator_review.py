"""Independently reconstruct ordinary GNNs from saved generator-review evidence.

Weight application deliberately does not use adapted_forward, apply_target_delta
or their schema resolver. It checks named tensors and loads an ordinary GNN.
Use after the review finishes; this is not part of deployment latency timing.
"""

import argparse
import json
import hashlib
import math
import statistics
import zipfile
import csv
from pathlib import Path

import torch

from train_adapter import file_hash, code_provenance
from amortized_gnn_tta.adaptation.checkpoints import load_generator_checkpoint, atomic_json
from amortized_gnn_tta.adaptation.experiments import review_run_name
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from amortized_gnn_tta.data import load_environment_manifest, load_shift_environment
from amortized_gnn_tta.data.api import graph_fingerprint
from amortized_gnn_tta.evaluation import score_logits
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint, software_versions


def write_verified_summary(review_dir, records, plan, provenance):
    """Aggregate verified per-seed evidence without selecting a winning input."""
    grouped = {}
    for row in records:
        grouped.setdefault((row["architecture"], row["method"], row["conditioning"]), []).append(row)
    rows = []
    for (architecture, method, conditioning), seeds in grouped.items():
        if {row["seed"] for row in seeds} != set(plan["generator_seeds"]):
            raise ValueError("Incomplete generator-seed group")
        mean = lambda field: statistics.mean(row[field] for row in seeds)
        pipeline, iterative = mean("pipeline_ms"), mean("iterative_ms")
        rows.append({"architecture": architecture, "method": method, "conditioning": conditioning,
            "training_seeds": len(seeds), "frozen_accuracy_percent": mean("frozen_accuracy_percent"),
            "iterative_accuracy_percent": mean("iterative_accuracy_percent"),
            "adapted_accuracy_percent": mean("adapted_accuracy_percent"),
            "adapted_accuracy_sd_percent": statistics.stdev(row["adapted_accuracy_percent"] for row in seeds) if len(seeds)>1 else 0.,
            "gain_pp": mean("gain_pp"), "control_gain_pp": mean("control_gain_pp"),
            "harm_cases": sum(row["harm_cases"] for row in seeds), "repeated_shift_cases": sum(row["shift_cases"] for row in seeds),
            "generated_pipeline_ms": pipeline, "iterative_pipeline_ms": iterative,
            "speedup_ratio_of_mean_latencies": iterative/pipeline if plan["update_scope"] == "head" else None,
            "update_scope": plan["update_scope"], "target_parameter_count": seeds[0].get("target_parameter_count"),
            "offline_wall_seconds": mean("offline_wall_seconds"), "teacher_wall_seconds": mean("teacher_wall_seconds"),
            "static_setup_seconds": mean("static_model_setup_seconds"),
            "completed_training_checkpoints_reused": sum(bool(row.get("training_reused_from")) for row in seeds)})
    with (review_dir / "AGGREGATED_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Verified CSBM generator development results", "",
        "All named initial/final/delta tensors, graph/source/checkpoint hashes, predictions, accuracies and raw-timing medians passed the independent replay audit.", "",
        f"This is validation evidence: one held-out GNN checkpoint and graph lineage per architecture. {len(plan['generator_seeds'])} generator seeds are repeated training runs, not independent graph samples. GNN seed 31 is shared across the three training source-label subsets. Controls participate in checkpoint selection; accuracy tables exclude controls and report their gains separately.", "",
        "Hypernetwork CE and diffusion teacher-MSE use different objectives/update budgets. These results do not select the formal Arxiv input design or establish a definitive generator ranking. Updates are direct coordinates at the recorded scope, with one diffusion candidate. IM is a head-only reference; broader updates have no matched speed ratio. Final target labels were not accessed.", "",
        f"Update scope: {plan['update_scope']}. Target parameter counts are recorded in AGGREGATED_RESULTS.csv.", "",
        "## Accuracy", "",
        f"| Model | Method | Input | Frozen (%) | IM-head{plan['iterative_steps']} (%) | Generated mean ± seed SD (%) | Gain (pp) | Harmed / repeated shifted cases | Control gain (pp) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['architecture']} | {row['method']} | {row['conditioning']} | {row['frozen_accuracy_percent']:.3f} | {row['iterative_accuracy_percent']:.3f} | {row['adapted_accuracy_percent']:.3f} ± {row['adapted_accuracy_sd_percent']:.3f} | {row['gain_pp']:+.3f} | {row['harm_cases']}/{row['repeated_shift_cases']} | {row['control_gain_pp']:+.3f} |")
    lines += ["", "## Deployment and offline cost", "",
        "Primary latency includes target processing, update work and predictions. Each seed's cost is the mean of the declared condition medians, each from the configured warmups/repetitions. The group speedup is the ratio of mean IM and generated latencies across those same conditions/seeds; >1 is faster, <1 is slower. This differs slightly from averaging per-seed ratios in the original incremental REVIEW.md.", "",
        f"| Model | Method | Input | Pipeline (ms) | IM-head{plan['iterative_steps']} (ms) | IM / generated | Offline wall (s) | Teacher subset (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        ratio = f"{row['speedup_ratio_of_mean_latencies']:.3f}×" if row["speedup_ratio_of_mean_latencies"] is not None else "not matched"
        lines.append(f"| {row['architecture']} | {row['method']} | {row['conditioning']} | {row['generated_pipeline_ms']:.3f} | {row['iterative_pipeline_ms']:.3f} | {ratio} | {row['offline_wall_seconds']:.2f} | {row['teacher_wall_seconds']:.2f} |")
    reused = sum(row["completed_training_checkpoints_reused"] for row in rows)
    lines += ["", f"{reused} completed training checkpoints were reused from the explicitly recorded parent run after identity checks and exact generated-delta agreement. Every prediction/timing check was rerun. Offline times for reused checkpoints are their original measured training costs, not new training runs. Partial training was never resumed.", "",
        "Static-model fingerprint setup and canonical dataset preparation are reported separately in seed/selection records. The per-seed break-even field is a training-plus-static-setup cost-only illustration; it excludes common dataset/bank preparation and research search costs. Accuracy and label-access differences remain relevant. If online savings are nonpositive, there is no positive amortization argument from repeated deployments.", "",
        "The iterative baseline is the documented existing frozen-first-layer-cache IM algorithm, with unnecessary export/extra-prediction work removed. Architecture-specific head aggregation caching is not implemented; these ratios are implementation comparisons, not an optimal computational lower bound.", "",
        "CPU RSS samples include resident review datasets/training models and are lower bounds at 5ms sampling, not minimal deployment or exact incremental allocation peaks. Memory profiling is separate from latency timing. CUDA/MPS execution is unverified.", "",
        f"Independent audit: {provenance['runs']} generator configurations, {provenance['pairs']} saved cases, maximum logit replay error {provenance['max_logit_error']}, accuracy mismatches {provenance['accuracy_mismatches']}.", ""]
    (review_dir / "VERIFIED_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def replay(review_dir):
    review_dir = Path(review_dir)
    plan_record = json.loads((review_dir / "PLAN.json").read_text())
    plan = plan_record["plan"]
    scope = plan["update_scope"]
    progress = json.loads((review_dir / "progress.json").read_text())
    if progress["status"] != "complete":
        raise ValueError("Only a completed review can receive a complete replay audit")
    output = review_dir / "INDEPENDENT_REPLAY.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(plan["threads"])
    artifact_root = Path(plan_record["artifact_root"])
    manifest_path = artifact_root / plan["environment_manifest"]
    manifest = load_environment_manifest(manifest_path)
    with zipfile.ZipFile(review_dir / "code_snapshot.zip") as archive:
        for name, digest in plan_record["code"]["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Captured code hash mismatch: {name}")
    metadata = {row["id"]: row for row in manifest["environments"]}
    bundles = {}
    verified_runs = []
    summary = {"status": "running", "pairs": 0, "runs": 0, "max_logit_error": 0.,
               "max_requested_vs_actual_delta_error": 0., "accuracy_mismatches": 0,
               "review_plan_sha256": file_hash(review_dir / "PLAN.json"),
               "code": code_provenance(), "software": software_versions(),
               "weight_application": "manual named source-plus-requested delta and ordinary load_state_dict; no adaptation application helper"}
    atomic_json(output, summary)
    try:
        for architecture in plan["architectures"]:
            selection = json.loads((review_dir / f"{architecture}_selection.json").read_text())
            if file_hash(manifest_path) != selection["environment_manifest_sha256"]:
                raise ValueError("Environment manifest changed")
            source_path = Path(selection["source_checkpoint"])
            if file_hash(source_path) != selection["source_checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed")
            source, source_payload = load_checkpoint(source_path, build_model)
            source.eval()
            initial = {name: value.detach().clone() for name, value in source.state_dict().items()}
            if state_sha256(initial) != selection["validation_state_sha256"]:
                raise ValueError("Source state identity mismatch")
            # Derive names directly, independently of adaptation schema helpers.
            named = list(source.conv2.named_parameters()) if scope == "head" else [
                (name, p) for name, p in source.named_parameters() if scope == "all" or name.startswith("conv1.")]
            expected_schema = [(name, tuple(parameter.shape)) for name, parameter in named]
            weights = torch.cat([parameter.detach().flatten() for _, parameter in named])
            for method in plan["methods"]:
                for conditioning in plan["conditioning"]:
                    for seed in plan["generator_seeds"]:
                        directory = review_dir / review_run_name(architecture, method, conditioning, seed, scope)
                        checkpoint_path = directory / "generator_best.pt"
                        generator, descriptor, payload = load_generator_checkpoint(checkpoint_path, model=source)
                        if payload["target_scope"] != scope:
                            raise ValueError("Checkpoint update scope differs from review")
                        checkpoint_sha = file_hash(checkpoint_path)
                        pair_rows = json.loads((directory / "pair_results.json").read_text())
                        files = sorted(directory.glob("pair_*.pt"))
                        if len(files) != len(pair_rows):
                            raise ValueError("Saved pair files and numeric records disagree")
                        for path, recorded in zip(files, pair_rows):
                            saved = torch.load(path, map_location="cpu", weights_only=True)
                            if saved["context"] != recorded or saved["generator_sha256"] != checkpoint_sha or saved["source_state_sha256"] != state_sha256(initial):
                                raise ValueError("Pair/source/generator identity mismatch")
                            env_id = recorded["environment_id"]
                            if metadata[env_id]["role"] != "validation":
                                raise ValueError("Replay attempted to score a non-validation environment")
                            if env_id not in bundles:
                                bundles[env_id] = load_shift_environment(manifest, env_id)
                            bundle = bundles[env_id]
                            graph = bundle.unlabeled_input(env_id)
                            if graph_fingerprint(graph) != saved["graph_sha256"]:
                                raise ValueError("Graph identity mismatch")
                            schema = [(name, tuple(shape)) for name, shape in saved["schema"]]
                            if schema != expected_schema:
                                raise ValueError("Saved schema is not this model's complete target scope")
                            with torch.no_grad():
                                generated_delta = generator(descriptor(source, graph), weights)
                            torch.testing.assert_close(generated_delta, saved["delta"], rtol=1e-5, atol=1e-7)
                            expected_target_names = {"conv2."+name if scope == "head" else name for name, _ in expected_schema}
                            target_state = saved.get("adapted_target_state", saved.get("adapted_head_state"))
                            target_delta = saved.get("actual_target_delta", saved.get("actual_head_delta"))
                            if set(target_state) != expected_target_names or set(target_delta) != expected_target_names:
                                raise ValueError("Saved tensors do not cover exactly the target scope")
                            delta = saved["delta"]
                            if delta.ndim != 1 or not torch.isfinite(delta).all():
                                raise ValueError("Invalid saved delta")
                            if not math.isclose(float(delta.norm()), recorded["delta_norm"], rel_tol=1e-6, abs_tol=1e-8):
                                raise ValueError("Saved delta norm differs from actual delta")
                            final = {name: value.clone() for name, value in initial.items()}
                            offset = 0
                            for name, shape in expected_schema:
                                full_name = "conv2." + name if scope == "head" else name
                                count = initial[full_name].numel()
                                requested = delta[offset:offset+count].reshape(shape)
                                final[full_name] = initial[full_name] + requested
                                if not torch.equal(final[full_name], target_state[full_name]):
                                    raise ValueError("Requested update does not produce saved final weights")
                                actual = final[full_name]-initial[full_name]
                                if not torch.equal(actual, target_delta[full_name]):
                                    raise ValueError("Saved effective delta is not final minus initial")
                                summary["max_requested_vs_actual_delta_error"] = max(summary["max_requested_vs_actual_delta_error"], float((actual-requested).abs().max()))
                                offset += count
                            if offset != delta.numel():
                                raise ValueError("Extra saved delta coordinates")
                            restored = build_model(source_payload["model_spec"])
                            restored.load_state_dict(final, strict=True)
                            restored.eval()
                            with torch.no_grad():
                                logits = restored(graph.x, graph.edge_index)
                                original_logits = source(graph.x, graph.edge_index)
                            torch.testing.assert_close(logits, saved["logits"], rtol=1e-5, atol=1e-6)
                            summary["max_logit_error"] = max(summary["max_logit_error"], float((logits-saved["logits"]).abs().max()))
                            # Ground-truth labels are consulted only now, for scoring.
                            indices, labels = bundle.select(env_id, "eval"), bundle.evaluation_labels(env_id, "eval")
                            accuracy = score_logits(logits, indices, labels).accuracy
                            baseline = score_logits(original_logits, indices, labels).accuracy
                            if accuracy != recorded["adapted_accuracy"] or baseline != recorded["frozen_accuracy"]:
                                summary["accuracy_mismatches"] += 1
                                raise ValueError("Replayed accuracy differs from saved numeric record")
                            if not math.isclose(100*(accuracy-baseline), recorded["gain_pp"], abs_tol=1e-10):
                                raise ValueError("Reported gain differs from replayed accuracies")
                            summary["pairs"] += 1
                        run_summary = json.loads((directory / "summary.json").read_text())
                        shifted = [row for row in pair_rows if row["family"] != "control"]
                        expected_summary = {
                            "frozen_accuracy_percent": 100*statistics.mean(row["frozen_accuracy"] for row in shifted),
                            "adapted_accuracy_percent": 100*statistics.mean(row["adapted_accuracy"] for row in shifted),
                            "gain_pp": statistics.mean(row["gain_pp"] for row in shifted),
                            "harm_cases": sum(row["gain_pp"] < 0 for row in shifted),
                            "shift_cases": len(shifted),
                            "control_gain_pp": statistics.mean(row["gain_pp"] for row in pair_rows if row["family"] == "control")}
                        for field, value in expected_summary.items():
                            if not math.isclose(run_summary[field], value, rel_tol=1e-10, abs_tol=1e-10):
                                raise ValueError(f"Run aggregate differs from per-case evidence: {field}")
                        timed_cases = json.loads((directory / "timing.json").read_text())
                        for case in timed_cases:
                            for measured in case["timing"].values():
                                samples = measured["samples_seconds"]
                                if len(samples) != plan["repeats"] or any(not math.isfinite(value) or value <= 0 for value in samples):
                                    raise ValueError("Invalid/incomplete timing repetitions")
                                if measured["median_seconds"] != statistics.median(samples):
                                    raise ValueError("Timing median differs from raw repetitions")
                        generated = 1000*statistics.mean(case["timing"]["generated"]["median_seconds"] for case in timed_cases)
                        iterative = 1000*statistics.mean(case["timing"]["iterative"]["median_seconds"] for case in timed_cases)
                        if not math.isclose(run_summary["pipeline_ms"], generated, rel_tol=1e-10) or not math.isclose(run_summary["iterative_ms"], iterative, rel_tol=1e-10):
                            raise ValueError("Reported latency differs from raw timing evidence")
                        if scope == "head":
                            if not math.isclose(run_summary["speedup"], iterative/generated, rel_tol=1e-10):
                                raise ValueError("Reported speedup differs from raw timing evidence")
                        elif run_summary["speedup"] is not None:
                            raise ValueError("Broader updates must not claim matched head-IM speedup")
                        verified_runs.append(run_summary)
                        summary["runs"] += 1
                        atomic_json(output, summary)
                        print(f"Replayed {directory.name}: {len(files)} saved cases", flush=True)
            if not all(torch.equal(initial[name], value) for name, value in source.state_dict().items()):
                raise ValueError("Replay mutated source weights")
        if summary["pairs"] != progress["replayed_pairs"] or summary["runs"] != progress["planned"]:
            raise ValueError("Audit did not cover the complete declared grid")
        summary["status"] = "complete"
        write_verified_summary(review_dir, verified_runs, plan, summary)
        atomic_json(output, summary)
        return summary
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(output, summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    print(json.dumps(replay(parser.parse_args().review_dir), indent=2))
