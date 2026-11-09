"""Run the predeclared CSBM generator repair review and replay its evidence.

Artifact root supplies existing verified banks/partitions/environments, typically
the original main checkout. This command never scores target_test roles. It is
a development study; hypernetwork CE and diffusion teacher-MSE are distinct
objectives and training budgets. No diffusion-superiority claim follows.
"""

import argparse
from dataclasses import asdict
import csv
import gc
import json
from pathlib import Path
import statistics
import shutil
from types import SimpleNamespace
import time

import torch

from train_adapter import load_experiment, file_hash, code_provenance
from amortized_gnn_tta.adaptation import (HeadAdapterGenerator, DiffusionHeadGenerator,
    extract_target_parameters, apply_target_delta, train_adaptation_generator,
    train_diffusion_generator)
from amortized_gnn_tta.adaptation.lora_generator import LoRAUpdateSpace, LoRAGenerator
from amortized_gnn_tta.adaptation.iterative import PEFTConfig
from amortized_gnn_tta.adaptation.experiments import make_descriptor, review_run_name, training_recipe, CONDITIONING_CHOICES
from amortized_gnn_tta.adaptation.benchmark import (measure_methods, generated_predictions, iterative_predictions,
    frozen_predictions, memory_observation, amortization_break_even)
from amortized_gnn_tta.adaptation.checkpoints import save_generator_checkpoint, load_generator_checkpoint, atomic_json
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from amortized_gnn_tta.data import load_shift_environment
from amortized_gnn_tta.data.api import graph_fingerprint
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.evaluation import score_logits
from amortized_gnn_tta.training import load_checkpoint, seed_model, software_versions


def summarize(output, records):
    if not records:
        return
    with (output / "seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# CSBM repaired-generator development review", "",
             "One diffusion candidate; explicit direct or LoRA-factor representation and scope are recorded in each row. Validation only. Hypernetwork CE and diffusion teacher-MSE are different training objectives/budgets.", "",
             f"Update scope: {records[0].get('update_scope', 'head')}. IM uses matched head/LoRA sites when available; broader direct-coordinate scopes keep head IM as an unmatched reference. Accuracy excludes unchanged controls; timing averages per-condition medians over control, attribute, homophily and combined cases. SD is across generator seeds.", "",
             "| Model | Method | Inputs | Frozen accuracy | IM accuracy | Adapted accuracy | Gain (pp) | Harmed shifted cases | Control gain (pp) | Pipeline (ms) | IM / generated |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    groups = {}
    for row in records:
        groups.setdefault((row["architecture"], row["method"], row["conditioning"]), []).append(row)
    for key, rows in groups.items():
        mean = lambda field: statistics.mean(row[field] for row in rows)
        ratio = f"{mean('speedup'):.3f}×" if rows[0].get("update_scope", "head") in {"head", "lora"} else "not matched"
        sd = statistics.stdev(row["adapted_accuracy_percent"] for row in rows) if len(rows) > 1 else 0.
        lines.append(f"| {' | '.join(key)} | {mean('frozen_accuracy_percent'):.3f} | {mean('iterative_accuracy_percent'):.3f} | {mean('adapted_accuracy_percent'):.3f} ± {sd:.3f} | {mean('gain_pp'):+.3f} | {sum(row['harm_cases'] for row in rows)}/{sum(row['shift_cases'] for row in rows)} | {mean('control_gain_pp'):+.3f} | {mean('pipeline_ms'):.3f} | {ratio} |")
    (output / "REVIEW.md").write_text("\n".join(lines) + "\n")


def run(args):
    plan = json.loads(args.config.read_text())
    scope = plan["update_scope"]
    is_lora = plan.get("update_representation", "direct") == "lora-offset-factors-v1"
    if plan.get("update_representation", "direct") not in {"direct", "lora-offset-factors-v1"}:
        raise ValueError("Unknown update representation")
    if is_lora and (scope != "all" or plan.get("include_raw_weights", True)):
        raise ValueError("Generated LoRA uses full functional schema and no raw-weight input")
    peft = PEFTConfig(**plan["lora_config"]) if is_lora else PEFTConfig("head")
    matched = scope == "head" or is_lora
    if plan["candidate_count"] != 1 or scope not in {"head", "backbone", "all"} or plan["device"] != "cpu":
        raise ValueError("This review supports CPU direct updates and one diffusion candidate")
    if not plan["conditioning"] or any(c not in CONDITIONING_CHOICES for c in plan["conditioning"]):
        raise ValueError("Unknown or empty conditioning grid")
    if plan["diffusion_batch_size"] != 16 or plan["target_std_floor"] != .001:
        raise ValueError("Review records the prototype fixed diffusion batch size 16 and target std floor 1e-3")
    torch.set_num_threads(plan["threads"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    reuse_root = getattr(args, "reuse_completed_from", None)
    if reuse_root is not None:
        original = json.loads((reuse_root / "PLAN.json").read_text())
        if training_recipe(original["plan"]) != training_recipe(plan):
            raise ValueError("Completed-checkpoint reuse requires an identical training recipe")
    code = code_provenance()
    provenance = {"plan": plan, "plan_sha256": file_hash(args.config), "code": code,
                  "software": software_versions(), "artifact_root": str(args.artifact_root.resolve()),
                  "torch_threads": torch.get_num_threads(), "completed_checkpoint_source": str(reuse_root.resolve()) if reuse_root else None}
    atomic_json(output / "PLAN.json", provenance)
    import zipfile
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output / "code_snapshot.zip", "x") as archive:
        for name in code["source_hashes"]:
            archive.write(root / name, name)
        archive.write(args.config, "configs/" + args.config.name)
    results, baselines, replay_count = [], [], 0
    progress = {"status": "running", "completed": 0, "planned": len(plan["architectures"])*len(plan["methods"])*len(plan["conditioning"])*len(plan["generator_seeds"])}
    atomic_json(output / "progress.json", progress)
    try:
        for architecture in plan["architectures"]:
            bank_name, partition_name = plan["banks"][architecture]
            manifest_path = args.artifact_root / plan["environment_manifest"]
            bank, partition, full, _ = load_experiment(args.artifact_root / bank_name, manifest_path,
                args.artifact_root / partition_name, max_models=1, load_models=False)
            # Explicitly span all meta-training source-subset groups, rather than
            # accidentally taking three initialization seeds from one subset.
            selected = []
            for subset in partition["groups"]["meta_train"]:
                selected.append(next(row for row in bank.manifest["models"] if row["subset_id"] == subset and row["model_seed"] == plan["model_seed"]))
            val_row = next(row for row in bank.manifest["models"] if row["model_id"] in partition["model_ids"]["validation"] and row["model_seed"] == plan["model_seed"])
            models = {row["model_id"]: load_checkpoint(bank.directory / row["checkpoint"], build_model)[0].eval() for row in selected}
            validation_model = load_checkpoint(bank.directory / val_row["checkpoint"], build_model)[0].eval()
            validation_models = {val_row["model_id"]: validation_model}
            training = [e for e in full["environments"] if e["role"] == "meta_train" and
                        (e["family"] == "control" or (e["severity"] == plan["severity"] and e["shift"] in plan["training_shifts"]))]
            validation = [e for e in full["environments"] if e["role"] == "validation" and
                          (e["family"] == "control" or e["severity"] == plan["severity"])]
            manifest = {"environments": training + validation}
            allowed = {e["id"] for e in manifest["environments"]}
            # This is a canonical dataset cache, not cached deployment summaries.
            # Store only the permitted 24 development graphs (~1GB paper profile).
            bundles = {}
            def loader(_manifest, env_id):
                if env_id not in allowed:
                    raise ValueError("Review attempted to load an unselected/final environment")
                if env_id not in bundles:
                    bundles[env_id] = load_shift_environment(full, env_id)
                return bundles[env_id]
            preparation_start = time.perf_counter()
            for env in manifest["environments"]:
                loader(manifest, env["id"])
            preparation_seconds = time.perf_counter() - preparation_start
            source_hashes = {name: state_sha256(model.state_dict()) for name, model in models.items()}
            validation_hash = state_sha256(validation_model.state_dict())
            context = {"architecture": architecture, "canonical_dataset_preparation_seconds": preparation_seconds,
                       "memory_context": "review process retains the permitted canonical dataset cache and training models; sampled RSS is not a minimal deployment footprint", "bank_manifest_sha256": bank.sha256,
                       "model_partition_sha256": file_hash(args.artifact_root / partition_name),
                       "environment_manifest_sha256": file_hash(manifest_path),
                       "training_model_ids": list(models), "validation_model_id": val_row["model_id"],
                       "selected_environment_ids": sorted(allowed), "plan_sha256": provenance["plan_sha256"],
                       "source_hashes": source_hashes, "validation_state_sha256": validation_hash,
                       "source_checkpoint": str(bank.directory / val_row["checkpoint"]),
                       "source_checkpoint_sha256": val_row["checkpoint_sha256"]}
            atomic_json(output / f"{architecture}_selection.json", context)
            print("BASELINES", architecture, flush=True)
            base_by_env = {}
            for env in validation:
                bundle = loader(manifest, env["id"])
                graph = bundle.unlabeled_input(env["id"])
                before = frozen_predictions(validation_model, graph)
                after = iterative_predictions(validation_model, graph, steps=plan["iterative_steps"], lr=plan["iterative_lr"], peft=peft, initialization_seed=plan.get("lora_initialization_seed", 801))
                indices, labels = bundle.select(env["id"], "eval"), bundle.evaluation_labels(env["id"], "eval")
                row = {"architecture": architecture, "environment_id": env["id"],
                       "frozen": asdict(score_logits(before.logits, indices, labels)),
                       "iterative": asdict(score_logits(after.logits, indices, labels))}
                base_by_env[env["id"]] = row
                baselines.append(row)
            atomic_json(output / "baselines.json", baselines)
            for method in plan["methods"]:
                for conditioning in plan["conditioning"]:
                    for seed in plan["generator_seeds"]:
                        name = review_run_name(architecture, method, conditioning, seed, scope)
                        directory = output / name
                        directory.mkdir()
                        print("START", name, flush=True)
                        seed_model(seed)
                        descriptor, include_weights = make_descriptor(validation_model, conditioning, scope=scope,
                            descriptor_version=plan["descriptor_version"], weight_scope=plan["weight_scope"], chunk_size=plan["chunk_size"])
                        if "include_raw_weights" in plan:
                            include_weights = plan["include_raw_weights"]
                        space = LoRAUpdateSpace.from_model(validation_model, peft, plan.get("lora_initialization_seed", 801)) if is_lora else None
                        flat, schema = extract_target_parameters(validation_model, scope)
                        kwargs = dict(descriptor_dim=descriptor.output_dim, head_dim=space.factor_dim if space is not None else len(flat), hidden_dim=plan["hidden_dim"], num_layers=plan["num_layers"], include_head_weights=include_weights)
                        generator = (HeadAdapterGenerator(**kwargs) if method == "hypernetwork" else
                            DiffusionHeadGenerator(**kwargs, diffusion_steps=plan["diffusion_steps"], sampling_steps=plan["sampling_steps"], default_seed=plan["sampling_seed"], time_dim=plan["time_dim"], schedule=plan["noise_schedule"]))
                        if space is not None:
                            generator = LoRAGenerator(generator, space)
                        run_context = {**context, "method": method, "conditioning": conditioning, "generator_seed": seed,
                                       "training_seed": seed+plan["training_seed_offset"], "code": code, "plan": plan}
                        def checkpoint(state, history):
                            save_generator_checkpoint(directory / "generator_best.pt", generator, descriptor, schema,
                                target_scope=scope, provenance=run_context, history=history, state_dict=state, overwrite=True)
                        seed_model(run_context["training_seed"])
                        common = dict(models=models, manifest=manifest, load_env_fn=loader, descriptor=descriptor,
                            generator=generator, epochs=plan["epochs"], lr=plan["learning_rate"], delta_l2_reg=plan["delta_l2"],
                            val_models=validation_models, verbose=False, checkpoint_callback=checkpoint, target_scope=scope,
                            early_stopping_patience=plan.get("generator_patience"),
                            early_stopping_min_delta=plan.get("generator_min_delta", 0.0))
                        old_directory = reuse_root / name if reuse_root is not None else None
                        reuse = old_directory is not None and (old_directory / "summary.json").is_file()
                        if reuse:
                            old_payload = torch.load(old_directory / "generator_best.pt", map_location="cpu", weights_only=True)
                            for field in ("bank_manifest_sha256", "model_partition_sha256", "environment_manifest_sha256", "source_hashes", "validation_state_sha256", "training_model_ids", "validation_model_id", "selected_environment_ids", "method", "conditioning", "generator_seed", "training_seed", "plan"):
                                actual, expected = old_payload["provenance"][field], run_context[field]
                                if field == "plan":
                                    actual, expected = training_recipe(actual), training_recipe(expected)
                                if actual != expected:
                                    raise ValueError(f"Completed checkpoint identity mismatch: {field}")
                            old_summary = json.loads((old_directory / "summary.json").read_text())
                            shutil.copyfile(old_directory / "generator_best.pt", directory / "generator_best.pt")
                            offline_seconds = old_summary["offline_wall_seconds"]
                            fit = SimpleNamespace(teacher_time_seconds=old_summary["teacher_wall_seconds"],
                                selected_epoch=old_summary["selected_epoch"], optimizer_steps=old_summary["optimizer_updates"])
                        else:
                            start = time.perf_counter()
                            fit = (train_adaptation_generator(**common, weight_decay=plan["hypernetwork_weight_decay"]) if method == "hypernetwork" else
                                   train_diffusion_generator(**common, oracle_steps=plan["teacher_steps"], oracle_lr=plan["teacher_lr"]))
                            offline_seconds = time.perf_counter() - start
                        generator, descriptor, _ = load_generator_checkpoint(directory / "generator_best.pt", model=validation_model)
                        checkpoint_sha = file_hash(directory / "generator_best.pt")
                        static_start = time.perf_counter()
                        descriptor.prepare_model(validation_model)
                        static_seconds = time.perf_counter() - static_start
                        pairs, timing = [], []
                        for index, env in enumerate(validation):
                            bundle = loader(manifest, env["id"])
                            graph = bundle.unlabeled_input(env["id"])
                            generated = generated_predictions(validation_model, graph, descriptor, generator, scope=scope)
                            if reuse:
                                old_pair = torch.load(old_directory / f"pair_{index:02d}.pt", map_location="cpu", weights_only=True)
                                if old_pair["context"]["environment_id"] != env["id"] or not torch.equal(old_pair["delta"], generated.delta):
                                    raise ValueError("Inference optimization changed a completed run's generated delta")
                                torch.testing.assert_close(generated.logits, old_pair["logits"], rtol=1e-5, atol=1e-6)
                            flat, schema = extract_target_parameters(validation_model, scope)
                            adapted = apply_target_delta(validation_model, generated.delta, schema).eval()
                            with torch.no_grad():
                                replay_logits = adapted(graph.x, graph.edge_index)
                            torch.testing.assert_close(replay_logits, generated.logits, rtol=1e-5, atol=1e-6)
                            indices, labels = bundle.select(env["id"], "eval"), bundle.evaluation_labels(env["id"], "eval")
                            score = score_logits(generated.logits, indices, labels)
                            assert score_logits(replay_logits, indices, labels).accuracy == score.accuracy
                            base = base_by_env[env["id"]]["frozen"]["accuracy"]
                            pair = {"environment_id": env["id"], "family": env["family"], "shift": env["shift"],
                                    "frozen_accuracy": base, "adapted_accuracy": score.accuracy,
                                    "gain_pp": 100*(score.accuracy-base), "delta_norm": float(generated.delta.norm())}
                            pairs.append(pair)
                            target_names = {"conv2."+name if scope == "head" else name for name, _ in schema}
                            payload = {"context": pair, "target_scope": scope, "graph_sha256": graph_fingerprint(graph), "generator_sha256": checkpoint_sha,
                                       "source_state_sha256": validation_hash, "schema": schema, "delta": generated.delta.cpu(),
                                       "adapted_target_state": {k: v.cpu() for k, v in adapted.state_dict().items() if k in target_names},
                                       "actual_target_delta": {k: (v-validation_model.state_dict()[k]).cpu() for k, v in adapted.state_dict().items() if k in target_names},
                                       "logits": generated.logits.cpu()}
                            if is_lora:
                                with torch.no_grad():
                                    payload["factor_offsets"] = generator.forward_factors(descriptor(validation_model, graph)).cpu()
                                payload["update_representation"] = generator.representation
                                payload["lora_space"] = generator.update_space.configuration()
                            torch.save(payload, directory / f"pair_{index:02d}.pt")
                            # Replay the saved bytes, not just the live tensors.
                            saved = torch.load(directory / f"pair_{index:02d}.pt", weights_only=True)
                            replay_model = apply_target_delta(validation_model, saved["delta"], saved["schema"]).eval()
                            with torch.no_grad():
                                torch.testing.assert_close(replay_model(graph.x, graph.edge_index), saved["logits"], rtol=1e-5, atol=1e-6)
                            assert all(torch.equal(v, replay_model.state_dict()[k]) for k, v in saved["adapted_target_state"].items())
                            replay_count += 1
                            if env["shift"] in plan["timed_shifts"] or env["family"] == "control":
                                methods = {"frozen": lambda: frozen_predictions(validation_model, graph),
                                    "iterative": lambda: iterative_predictions(validation_model, graph, steps=plan["iterative_steps"], lr=plan["iterative_lr"], peft=peft, initialization_seed=plan.get("lora_initialization_seed", 801)),
                                    "generated": lambda: generated_predictions(validation_model, graph, descriptor, generator, scope=scope)}
                                measured = measure_methods(methods, warmups=plan["warmups"], repeats=plan["repeats"])
                                profile = generated_predictions(validation_model, graph, descriptor, generator, scope=scope, profile=True).stages_seconds
                                iterative_profile = iterative_predictions(validation_model, graph, steps=plan["iterative_steps"], lr=plan["iterative_lr"], peft=peft, initialization_seed=plan.get("lora_initialization_seed", 801), profile=True).stages_seconds
                                with torch.no_grad():
                                    prepared = descriptor(validation_model, graph)
                                    weights, _ = extract_target_parameters(validation_model, scope)
                                    micro = measure_methods({"generator_only_prepared": lambda: generator(prepared, weights)}, warmups=plan["warmups"], repeats=plan["repeats"])
                                timing.append({"generator_only_microbenchmark": micro,"environment_id": env["id"], "timing": measured, "generated_profile": profile,
                                    "iterative_profile": iterative_profile,
                                    "generated_memory": memory_observation(methods["generated"]),
                                    "iterative_memory": memory_observation(methods["iterative"])})
                        assert all(state_sha256(model.state_dict()) == source_hashes[mid] for mid, model in models.items())
                        assert state_sha256(validation_model.state_dict()) == validation_hash
                        atomic_json(directory / "pair_results.json", pairs)
                        atomic_json(directory / "timing.json", timing)
                        shifted = [pair for pair in pairs if pair["family"] != "control"]
                        pipeline = statistics.mean(row["timing"]["generated"]["median_seconds"] for row in timing)
                        iterative = statistics.mean(row["timing"]["iterative"]["median_seconds"] for row in timing)
                        result = {"architecture": architecture, "method": method, "conditioning": conditioning, "seed": seed,
                            "update_scope": "lora" if is_lora else scope,
                            "update_representation": "lora-offset-factors-v1" if is_lora else "direct",
                            "factor_parameter_count": space.factor_dim if space is not None else None,
                            "target_parameter_count": sum(dict(validation_model.named_parameters())[n].numel() for n in space.targets) if space is not None else len(flat), "generator_parameter_count": sum(p.numel() for p in generator.parameters()),
                            "frozen_accuracy_percent": 100*statistics.mean(row["frozen_accuracy"] for row in shifted),
                            "iterative_accuracy_percent": 100*statistics.mean(base_by_env[row["environment_id"]]["iterative"]["accuracy"] for row in shifted),
                            "adapted_accuracy_percent": 100*statistics.mean(row["adapted_accuracy"] for row in shifted),
                            "gain_pp": statistics.mean(row["gain_pp"] for row in shifted),
                            "harm_cases": sum(row["gain_pp"] < 0 for row in shifted), "shift_cases": len(shifted),
                            "control_gain_pp": statistics.mean(row["gain_pp"] for row in pairs if row["family"] == "control"),
                            "pipeline_ms": 1000*pipeline, "iterative_ms": 1000*iterative, "speedup": iterative/pipeline if matched else None,
                            "training_reused_from": str(old_directory.resolve()) if reuse else None,
                            "offline_wall_seconds": offline_seconds, "teacher_wall_seconds": fit.teacher_time_seconds,
                            "selected_epoch": fit.selected_epoch, "static_model_setup_seconds": static_seconds,
                            "optimizer_updates": fit.optimizer_steps,
                            "break_even_events_cost_only": amortization_break_even(offline_generated=offline_seconds+static_seconds, offline_iterative=0., online_generated=pipeline, online_iterative=iterative) if matched else None}
                        atomic_json(directory / "summary.json", result)
                        results.append(result)
                        summarize(output, results)
                        progress.update(completed=len(results), replayed_pairs=replay_count, latest=name)
                        atomic_json(output / "progress.json", progress)
                        print("DONE", result, flush=True)
                        del generator, descriptor, fit
                        gc.collect()
            bundles.clear()
            del models, validation_model
            gc.collect()
        progress.update(status="complete", replayed_pairs=replay_count)
        atomic_json(output / "progress.json", progress)
    except BaseException as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "progress.json", progress)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/generator_csbm_review.json")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-completed-from", type=Path, help="Reuse only fully completed compatible training runs; rerun every inference/timing check. Never resumes partial training.")
    run(parser.parse_args())
