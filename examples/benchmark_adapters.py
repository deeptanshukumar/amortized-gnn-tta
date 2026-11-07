"""Reproducible validation-only whole-pipeline benchmark of saved generators.

Example: python examples/benchmark_adapters.py --checkpoint runs/adapter/generator_best.pt
  --bank runs/gcn-bank-phase1 --manifest runs/shifts/environment_manifest.json
  --model-partition runs/review/model_partition.json --output-dir runs/timing
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch

from train_adapter import load_experiment, file_hash, code_provenance
from amortized_gnn_tta.adaptation import subsample_environments, extract_target_parameters
from amortized_gnn_tta.adaptation.benchmark import generated_predictions, iterative_predictions, frozen_predictions, measure_methods, memory_observation
from amortized_gnn_tta.adaptation.checkpoints import load_generator_checkpoint, atomic_json
from amortized_gnn_tta.data import load_shift_environment
from amortized_gnn_tta.evaluation import score_logits, select_device
from amortized_gnn_tta.training import software_versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-partition", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-models", type=int, default=1)
    parser.add_argument("--max-environments", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from amortized_gnn_tta.adaptation._validation import positive_int
    for name in ("threads", "max_models", "max_environments", "steps", "repeats"):
        positive_int(getattr(args, name), name)
    device = str(select_device(args.device))
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "code": code_provenance(), "software": software_versions(), "device": device,
                  "torch_threads": torch.get_num_threads(), "scope": "validation; one diffusion candidate",
                  "checkpoint_sha256": file_hash(args.checkpoint),
                  "timer_boundary": "canonical CPU graph + resident models to usable device predictions; complete independent adaptation event",
                  "excluded": ["disk loading", "offline training", "benchmark construction", "label scoring", "report I/O"],
                  "cache_policy": "no persistent target cache; head IM builds frozen features inside every event; D static fingerprint prepared once per model and setup cost recorded"}
    atomic_json(args.output_dir / "run.json", {"status": "starting", **provenance})
    rows = []
    try:
        setup_start = time.perf_counter()
        _, _, manifest, grouped = load_experiment(args.bank, args.manifest, args.model_partition,
            device=device, max_models=args.max_models)
        sample = next(iter(grouped["validation"].values()))
        generator, descriptor, payload = load_generator_checkpoint(args.checkpoint, model=sample, device=device)
        scope = payload["target_scope"]
        provenance["target_scope"] = scope
        provenance["iterative_comparison"] = "matched head IM" if scope == "head" else "omitted: no matched broader direct-update optimizer in this benchmark"
        for key, path in (("bank_manifest_sha256", args.bank / "bank.json"),
                          ("model_partition_sha256", args.model_partition), ("environment_manifest_sha256", args.manifest)):
            expected = payload["provenance"].get(key)
            if expected and file_hash(path) != expected:
                raise ValueError(f"Checkpoint provenance mismatch: {key}")
        provenance["setup_wall_seconds"] = time.perf_counter() - setup_start
        import platform
        provenance.update(cpu=platform.processor(), platform=platform.platform(),
                          accelerator=torch.cuda.get_device_name(device) if device.startswith("cuda") else None)
        envs = subsample_environments([e for e in manifest["environments"] if e["role"] == "validation"], args.max_environments)
        provenance["selected_environment_ids"] = [e["id"] for e in envs]
        provenance["selected_model_ids"] = list(grouped["validation"])
        atomic_json(args.output_dir / "run.json", {"status": "running", **provenance})
        for env in envs:
            bundle = load_shift_environment(manifest, env["id"])
            graph = bundle.unlabeled_input(env["id"])
            for model_id, model in grouped["validation"].items():
                static_start = time.perf_counter()
                descriptor.prepare_model(model)
                static_setup_seconds = time.perf_counter() - static_start
                calls = {"frozen": lambda: frozen_predictions(model, graph, device=device),
                         "generated": lambda: generated_predictions(model, graph, descriptor, generator, scope=scope, device=device)}
                if scope == "head":
                    calls["iterative_im"] = lambda: iterative_predictions(model, graph, steps=args.steps, device=device)
                timing = measure_methods(calls, device=device, warmups=args.warmups, repeats=args.repeats)
                outputs = {name: call() for name, call in calls.items()}
                profile = generated_predictions(model, graph, descriptor, generator, scope=scope, device=device, profile=True).stages_seconds
                iterative_profile = (iterative_predictions(model, graph, steps=args.steps, device=device, profile=True).stages_seconds if scope == "head" else None)
                with torch.no_grad():
                    inputs = graph.clone().to(device)
                    prepared_descriptor = descriptor(model, inputs)
                    prepared_weights, _ = extract_target_parameters(model, scope)
                    micro = measure_methods({"generator_only_prepared": lambda: generator(prepared_descriptor, prepared_weights)}, device=device, warmups=args.warmups, repeats=args.repeats)
                memory = {name: memory_observation(call, device=device) for name, call in calls.items()}
                # No scoring labels are consulted until prediction/selection finishes.
                split = "eval" if "eval" in bundle.graphs[env["id"]].splits else "test"
                indices, labels = bundle.select(env["id"], split), bundle.evaluation_labels(env["id"], split)
                scores = {name: asdict(score_logits(value.logits, indices, labels)) for name, value in outputs.items()}
                row = {"model_id": model_id, "environment_id": env["id"], "family": env["family"],
                       "nodes": graph.num_nodes, "stored_edges": graph.num_edges, "features": graph.num_features,
                       "timing": timing, "scores": scores, "generator_only_microbenchmark": micro, "static_model_setup_seconds": static_setup_seconds, "generated_profile": profile, "iterative_profile": iterative_profile,
                       "profile_note": "separate instrumented pass; not a decomposition of the independently measured median",
                       "memory": memory, "target_scope": scope, "target_parameter_count": len(prepared_weights),
                       "speedup": timing["iterative_im"]["median_seconds"] / timing["generated"]["median_seconds"] if scope == "head" else None}
                rows.append(row)
                atomic_json(args.output_dir / "results.json", rows)
                ratio = f"{row['speedup']:.3f}x" if row["speedup"] is not None else "not compared (different update sites)"
                print(f"{model_id} {env['id']}: speedup {ratio}; gain {100*(scores['generated']['accuracy']-scores['frozen']['accuracy']):+.3f} pp", flush=True)
        atomic_json(args.output_dir / "run.json", {"status": "complete", **provenance})
    except BaseException as error:
        atomic_json(args.output_dir / "run.json", {"status": "failed", "error": str(error), **provenance})
        raise


if __name__ == "__main__":
    main()
