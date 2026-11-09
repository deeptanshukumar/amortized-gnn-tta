"""Development-only supervised references: never a deployment-label protocol."""

import argparse
import json
from pathlib import Path
import time
import torch
from run_arxiv_experiments import Study, lora_config
from amortized_gnn_tta.adaptation.iterative import (
    development_supervision,
    fit_supervised_development,
    AdaptationConfig,
    PEFTConfig,
)
from amortized_gnn_tta.adaptation.benchmark import (
    frozen_predictions,
    generated_predictions,
)
from amortized_gnn_tta.adaptation.candidates import agreement_predictions
from amortized_gnn_tta.adaptation.checkpoints import (
    atomic_json,
    load_generator_checkpoint,
)
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from amortized_gnn_tta.evaluation import score_logits


def run(args):
    study = Study(args)
    frozen = json.loads((args.output_dir / "FROZEN_DESIGN.json").read_text())
    destination = args.output_dir / "DIAGNOSTIC"
    destination.mkdir(exist_ok=False)
    rows = []
    for arch in study.plan["architectures"]:
        mid, source = next(iter(study.sources(arch, "meta_train").items()))
        initial = {k: v.detach().clone() for k, v in source.state_dict().items()}
        selected = frozen["selections"][arch]
        scope = "all" if selected["update"] == "lora" else "head"
        cases = []
        for family in ("control", "attribute", "structural", "combined"):
            match = next(
                (
                    e
                    for e in study.manifest["environments"]
                    if e["role"] == "meta_train" and e["family"] == family
                ),
                None,
            )
            if match:
                cases.append(match)
        for env in cases:
            eid = env["id"]
            bundle = study.loader(study.manifest, eid)
            graph = bundle.unlabeled_input(eid)
            indices = bundle.select(eid, "test")
            labels = bundle.evaluation_labels(eid, "test")
            baseline = score_logits(
                frozen_predictions(source, graph, device=study.device).logits,
                indices,
                labels,
            ).accuracy
            supervision = development_supervision(bundle, eid)
            for update in ("head", "lora", "full"):
                peft = (
                    lora_config(study.plan) if update == "lora" else PEFTConfig(update)
                )
                start = time.perf_counter()
                result = fit_supervised_development(
                    source,
                    graph,
                    supervision,
                    peft,
                    AdaptationConfig(
                        steps=study.plan["teacher_steps"],
                        patience=study.plan["teacher_steps"],
                        learning_rate=study.plan["teacher_lr"],
                    ),
                    device=study.device,
                )
                elapsed = time.perf_counter() - start
                with torch.no_grad():
                    logits = result.model(
                        graph.x.to(study.device), graph.edge_index.to(study.device)
                    )
                accuracy = score_logits(logits, indices, labels).accuracy
                final = {
                    k: v.detach().cpu().clone()
                    for k, v in result.model.state_dict().items()
                }
                delta = {
                    k: final[k] - initial[k].cpu()
                    for k in final
                    if final[k].is_floating_point()
                }
                torch.save(
                    dict(
                        initial_state_sha256=state_sha256(initial),
                        final_state=final,
                        deltas=delta,
                        predictions=logits[indices].argmax(-1),
                        indices=indices,
                        method="supervised_" + update,
                        environment_id=eid,
                        selected_step=result.selected_step,
                        history=result.history,
                    ),
                    destination / f"{arch}-{eid}-{update}.pt",
                )
                rows.append(
                    dict(
                        architecture=arch,
                        environment_id=eid,
                        method="supervised_" + update,
                        accuracy=accuracy,
                        gain=accuracy - baseline,
                        fit_seconds=elapsed,
                        selected_step=result.selected_step,
                        evaluation="seen meta-training checkpoint/environment, disjoint native diagnostic nodes",
                        labels="practice train + inner validation only; diagnostic labels score after fit",
                    )
                )
            for method in ("hypernetwork", "teacher_hypernetwork", "diffusion"):
                seed = study.plan["generator_seeds"][0]
                name = f"{arch}-{selected['conditioning']}-{selected['update']}-{method}-{seed}"
                generator, descriptor, _ = load_generator_checkpoint(
                    args.output_dir / name / "generator.pt",
                    model=source,
                    device=study.device,
                )
                if method == "diffusion":
                    output, _ = agreement_predictions(
                        source,
                        graph,
                        descriptor,
                        generator,
                        indices=indices,
                        scope=scope,
                        seeds=study.plan["candidate_seeds"],
                        device=study.device,
                    )
                else:
                    output = generated_predictions(
                        source,
                        graph,
                        descriptor,
                        generator,
                        scope=scope,
                        device=study.device,
                    )
                accuracy = score_logits(output.logits, indices, labels).accuracy
                rows.append(
                    dict(
                        architecture=arch,
                        environment_id=eid,
                        method=method,
                        accuracy=accuracy,
                        gain=accuracy - baseline,
                        evaluation="seen meta-training checkpoint/environment, disjoint native diagnostic nodes",
                        labels="no diagnostic labels in prediction",
                    )
                )
            if state_sha256(source.state_dict()) != state_sha256(initial):
                raise ValueError("Diagnostic mutated source")
            atomic_json(
                destination / "RESULTS.json",
                dict(
                    rows=rows,
                    final_test_scored=False,
                    interpretation="Fixed-budget supervised references are diagnostic, not proven accuracy ceilings or unlabeled methods",
                ),
            )
            print("DIAGNOSTIC DONE", arch, eid, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())
