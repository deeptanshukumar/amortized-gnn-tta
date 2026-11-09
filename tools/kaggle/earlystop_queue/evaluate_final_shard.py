"""Explicitly gated final scoring and complete-pipeline latency measurements.

Run only after all design choices have been written to FROZEN_DESIGN.json.
Reports are descriptive across these fixed cases, not new model selection.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from run_arxiv_experiments import Study, lora_config
from train_adapter import file_hash, code_provenance
from amortized_gnn_tta.data.real_environments import ArxivEnvironments
from amortized_gnn_tta.data.generalization import arxiv_generalization_manifest
from amortized_gnn_tta.data.natural import TemporalEnvironments, CountryEnvironments
from amortized_gnn_tta.adaptation.checkpoints import (
    atomic_json,
    load_generator_checkpoint,
)
from amortized_gnn_tta.adaptation.benchmark import (
    generated_predictions,
    iterative_predictions,
    frozen_predictions,
    measure_methods,
    sample_process_memory,
    synchronize,
)
from amortized_gnn_tta.adaptation.candidates import agreement_predictions
from amortized_gnn_tta.adaptation.graph_transform import feature_gtrans_predictions
from amortized_gnn_tta.adaptation.practice import fit_practice_model
from amortized_gnn_tta.adaptation.iterative import PEFTConfig
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from amortized_gnn_tta.training import software_versions
from amortized_gnn_tta.adaptation.benchmark import execution_hardware


def check_replayed_logits(reference, replayed):
    """Numerical verification only; original predictions remain authoritative."""
    torch.testing.assert_close(replayed, reference, rtol=1e-4, atol=1e-5)
    return dict(max_abs_logit_difference=float((replayed-reference).abs().max()),
                prediction_class_differences=int((replayed.argmax(-1)!=reference.argmax(-1)).sum()))


def run(args):
    study = Study(args)
    frozen_path = args.output_dir / "FROZEN_DESIGN.json"
    if not args.final_evaluation or not frozen_path.exists():
        raise ValueError(
            "Explicit final flag and frozen validation design are required"
        )
    design = json.loads(frozen_path.read_text())
    if design["identity"] != study.identity:
        raise ValueError("Frozen source inputs changed")
    if any(
        study.pins.get(mid) != pin for mid, pin in design["fitting_source_pins"].items()
    ):
        raise ValueError("Frozen fitting checkpoints changed")
    if design["candidate_seeds"] != study.plan["candidate_seeds"]:
        raise ValueError("Frozen candidate seeds changed")
    original_architectures = list(study.plan["architectures"])
    shard = getattr(args, "architecture", None)
    if shard is not None:
        if shard not in original_architectures:
            raise ValueError("Architecture is absent from the frozen study")
        # Operational partition only. Frozen identity/selection dictionaries are
        # retained verbatim; every checkpoint, case and method for this arch runs.
        study.plan = {**study.plan, "architectures": [shard]}
    destination = args.output_dir / "FINAL"
    destination.mkdir(exist_ok=False)
    atomic_json(
        destination / "EVALUATION_PLAN.json",
        dict(
            design_sha256=file_hash(frozen_path),
            code=code_provenance(),
            audit_export_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            architecture_shard=shard,
            complete_architecture_list=original_architectures,
            replay_tolerances=dict(rtol=1e-4, atol=1e-5),
            replay_policy="Reload actual saved delta; compare original/replayed logits within declared floating tolerance; record every argmax difference; never regenerate conditioning",
            software=software_versions(),
            hardware=execution_hardware(study.device),
            timing_warmups=args.warmups,
            timing_repeats=args.repeats,
            label_access="Final labels accessed only after all predictions for a case; no configuration changes",
        ),
    )
    dataset = study.plan.get("dataset", "ogbn-arxiv")
    if dataset == "ogbn-arxiv":
        manifest = arxiv_generalization_manifest(study.manifest)
        loader = ArxivEnvironments(study.loader.dataset, manifest, allow_final=True)
    elif dataset == "ogbn-arxiv-temporal":
        manifest = study.manifest
        loader = TemporalEnvironments(study.loader.dataset, manifest, allow_final=True)
    else:
        manifest = study.manifest
        loader = CountryEnvironments(args.data_root, manifest, allow_final=True)
    atomic_json(destination / "ENVIRONMENTS.json", manifest)
    rows = []
    timings = []
    replay_count = 0
    replay_audit = []
    for arch in study.plan["architectures"]:
        selection = design["selections"][arch]
        conditioning = selection["conditioning"]
        update = selection["update"]
        scope = "all" if update == "lora" else "head"
        from amortized_gnn_tta.adaptation.benchmark import constant_delta_predictions

        constant = torch.load(
            args.output_dir / f"constant-{arch}.pt", weights_only=True
        )
        models = study.sources(arch, "test")
        model = next(iter(models.values()))
        generators = {}
        for seed in study.plan["generator_seeds"]:
            for method in ("hypernetwork", "teacher_hypernetwork", "diffusion"):
                name = f"{arch}-{conditioning}-{update}-{method}-{seed}"
                generator, descriptor, _ = load_generator_checkpoint(
                    args.output_dir / name / "generator.pt",
                    model=model,
                    device=study.device,
                )
                generators[f"{method}-{seed}"] = (generator, descriptor, method)
        for mid, source in models.items():
            initial_hash = state_sha256(source.state_dict())
            initial_parameters = {n:v.detach().cpu().clone() for n,v in source.named_parameters()}
            torch.manual_seed(study.plan["generator_seeds"][0])
            practice, practice_fit = fit_practice_model(
                source,
                study.manifest,
                study.loader,
                epochs=study.plan["generator_epochs"],
                lr=study.plan["generator_lr"],
                device=study.device,
                early_stopping_patience=study.plan.get("generator_patience"),
                early_stopping_min_delta=study.plan.get("generator_min_delta", 0.0),
            )
            torch.save(
                dict(
                    state_dict=practice.state_dict(),
                    fit=practice_fit,
                    source_sha256=initial_hash,
                ),
                destination / f"practice-{mid}.pt",
            )
            for env in manifest["environments"]:
                if env["role"] != "target_test":
                    continue
                eid = env["id"]
                bundle = loader(manifest, eid)
                graph = bundle.unlabeled_input(eid)
                indices = bundle.select(eid, "eval")
                methods = {
                    "frozen": lambda: frozen_predictions(
                        source, graph, device=study.device
                    ),
                    "constant_teacher": lambda: constant_delta_predictions(
                        source,
                        graph,
                        constant["delta"],
                        constant["schema"],
                        device=study.device,
                    ),
                    "practice_fixed": lambda: frozen_predictions(
                        practice, graph, device=study.device
                    ),
                    "gtrans_feature_only": lambda: feature_gtrans_predictions(
                        source,
                        graph,
                        indices=indices,
                        steps=study.plan["teacher_steps"],
                        device=study.device,
                    ),
                }
                for kind in ("head", "lora"):
                    peft = (
                        lora_config(study.plan)
                        if kind == "lora"
                        else PEFTConfig("head")
                    )
                    methods["iterative_im_" + kind] = (
                        lambda peft=peft: iterative_predictions(
                            source,
                            graph,
                            indices=indices,
                            steps=study.plan["teacher_steps"],
                            device=study.device,
                            lr=study.plan["teacher_lr"],
                            peft=peft,
                        )
                    )

                def generated(g, d, method):
                    if method == "diffusion":
                        return agreement_predictions(
                            source,
                            graph,
                            d,
                            g,
                            indices=indices,
                            seeds=study.plan["candidate_seeds"],
                            scope=scope,
                            device=study.device,
                        )[0]
                    return generated_predictions(
                        source, graph, d, g, scope=scope, device=study.device
                    )

                for key, (g, d, method) in generators.items():
                    methods[key] = lambda g=g, d=d, method=method: generated(
                        g, d, method
                    )
                predictions = {}
                identity_updates = {}
                for method, call in methods.items():
                    synchronize(study.device)
                    start = time.perf_counter()
                    output = call()
                    synchronize(study.device)
                    elapsed = time.perf_counter() - start
                    predictions[method] = output.logits[indices].argmax(-1).cpu()
                    artifact = destination / f"{mid}--{eid}--{method}.pt"
                    from amortized_gnn_tta.adaptation.generator import (
                        extract_target_parameters,
                    )

                    delta_schema = (
                        constant["schema"]
                        if method == "constant_teacher"
                        else output.delta_schema
                    )
                    artifact_delta = output.delta
                    if output.delta is not None and delta_schema is None:
                        _, delta_schema = extract_target_parameters(source, scope)
                    if method == "iterative_im_lora":
                        from amortized_gnn_tta.adaptation.generator import (
                            unflatten_head_delta,
                        )

                        changes = unflatten_head_delta(output.delta, delta_schema)
                        _, delta_schema = extract_target_parameters(source, "all")
                        artifact_delta = torch.cat(
                            [
                                changes.get(name, torch.zeros_like(value)).flatten()
                                for name, value in source.named_parameters()
                            ]
                        )
                    final_parameters = None
                    actual_parameter_deltas = None
                    effective_identity_update = None
                    if artifact_delta is not None:
                        from amortized_gnn_tta.adaptation.generator import unflatten_head_delta
                        increments = unflatten_head_delta(artifact_delta, delta_schema)
                        final_parameters = {
                            n:(v.detach()+increments[n].to(v)).cpu().clone() if n in increments else v.detach().cpu().clone()
                            for n,v in source.named_parameters()
                        }
                        actual_parameter_deltas = {n:final_parameters[n]-initial_parameters[n] for n in final_parameters}
                        effective_identity_update = all(not torch.count_nonzero(v) for v in actual_parameter_deltas.values())
                    torch.save(
                        dict(
                            final_parameters=final_parameters,
                            actual_parameter_deltas=actual_parameter_deltas,
                            effective_identity_update=effective_identity_update,
                            increment_definition="delta is the applied generated/optimized increment; actual_parameter_deltas are rounded final-minus-initial parameter differences",
                            source_id=mid,
                            source_sha256=initial_hash,
                            environment_id=eid,
                            method=method,
                            indices=indices,
                            predicted_classes=predictions[method],
                            delta=artifact_delta,
                            factor_vector=output.factor_vector.detach().cpu()
                            if output.factor_vector is not None
                            else None,
                            selection=output.selection,
                            scope=scope
                            if method in generators
                            else constant["scope"]
                            if method == "constant_teacher"
                            else ("all" if method == "iterative_im_lora" else "head"),
                            delta_schema=delta_schema,
                            delta_kind="features_not_saved"
                            if method == "gtrans_feature_only"
                            else "weights",
                        ),
                        artifact,
                    )
                    identity_updates[method] = effective_identity_update
                    if output.delta is not None:
                        from amortized_gnn_tta.adaptation.generator import (
                            adapted_forward,
                            extract_target_parameters,
                        )

                        saved = torch.load(artifact, map_location=study.device, weights_only=True)
                        if saved.get("factor_vector") is not None:
                            product = generators[method][0].update_space.dense_delta(saved["factor_vector"])
                            torch.testing.assert_close(product, saved["delta"], rtol=0, atol=0)
                        with torch.no_grad():
                            replayed_logits = adapted_forward(
                                source, graph.clone().to(study.device),
                                saved["delta"], saved["delta_schema"],
                            )[indices.to(study.device)]
                        numerical = check_replayed_logits(
                            output.logits[indices.to(study.device)], replayed_logits
                        )
                        replay_audit.append(dict(source_id=mid, environment_id=eid,
                                                 method=method, **numerical))
                        atomic_json(destination / "REPLAY_AUDIT.json", dict(
                            rows=replay_audit, rtol=1e-4, atol=1e-5,
                            original_saved_predictions_authoritative=True,
                        ))
                        replay_count += 1
                    if state_sha256(source.state_dict()) != initial_hash:
                        raise ValueError("Source changed")
                    print("FINAL PREDICTED", arch, eid, method, elapsed, flush=True)
                # No final labels were used to compute any prediction above.
                labels = bundle.evaluation_labels(eid, "eval")
                base = int((predictions["frozen"] == labels).sum()) / len(labels)
                for method, prediction in predictions.items():
                    accuracy = int((prediction == labels).sum()) / len(labels)
                    rows.append(
                        dict(
                            architecture=arch,
                            model_id=mid,
                            environment_id=eid,
                            method=method,
                            group=env.get("generalization_group", env["family"]),
                            accuracy=accuracy,
                            effective_identity_update=identity_updates[method],
                            gain=accuracy - base,
                            baseline=base,
                        )
                    )
                atomic_json(
                    destination / "SCORES.json",
                    dict(
                        rows=rows,
                        independent_delta_replays=replay_count,
            prediction_class_differences=sum(r["prediction_class_differences"] for r in replay_audit),
            architecture_shard=shard,
            complete_architecture_list=original_architectures,
            audit_export_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        final_test_scored=True,
                    ),
                )
                # Time one predeclared case per family, first generator seed only.
                timed_case = dataset != "ogbn-arxiv" or eid in {
                    "test-control",
                    "test-new_draw-combined-0-0",
                    "test-stronger-combined-1-0",
                    "test-gaussian_features",
                    "test-degree_biased_removal",
                }
                if timed_case:
                    first = study.plan["generator_seeds"][0]
                    timing_methods = {
                        k: v
                        for k, v in methods.items()
                        if k not in generators or k.endswith("-" + str(first))
                    }
                    timing = measure_methods(
                        timing_methods,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        device=study.device,
                    )
                    memory = {}
                    if args.measure_memory:
                        for name, call in timing_methods.items():
                            memory[name] = sample_process_memory(call, device=study.device)
                    timings.append(
                        dict(
                            architecture=arch,
                            environment_id=eid,
                            methods=timing,
                            memory=memory,
                        )
                    )
                    atomic_json(
                        destination / "TIMINGS.json",
                        dict(
                            rows=timings,
                            boundary="Canonical CPU graph and resident models to usable predictions; includes required target preprocessing and all five candidates; excludes scoring/disk/offline training",
                            device=study.device,
                        ),
                    )
    summary = []
    for arch in study.plan["architectures"]:
        for method in sorted({r["method"] for r in rows}):
            selected = [
                r for r in rows if r["architecture"] == arch and r["method"] == method
            ]
            if selected:
                summary.append(
                    dict(
                        architecture=arch,
                        method=method,
                        cases=len(selected),
                        mean_accuracy=statistics.mean(r["accuracy"] for r in selected),
                        mean_gain=statistics.mean(r["gain"] for r in selected),
                        harm_fraction=statistics.mean(r["gain"] < 0 for r in selected),
                        effective_identity_cases=sum(r["effective_identity_update"] is True for r in selected),
                    )
                )
    atomic_json(
        destination / "COMPLETE.json",
        dict(
            summary=summary,
            independent_delta_replays=replay_count,
            prediction_class_differences=sum(r["prediction_class_differences"] for r in replay_audit),
            architecture_shard=shard,
            complete_architecture_list=original_architectures,
            audit_export_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            final_test_scored=True,
            design_sha256=file_hash(frozen_path),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=("gcn", "gat", "graphsage"), required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--final-evaluation", action="store_true")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--measure-memory", action="store_true")
    run(p.parse_args())
