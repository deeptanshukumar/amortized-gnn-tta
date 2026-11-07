"""Combine independently replayed head/LoRA studies with label-free baselines.

Runs actual entropy/IM optimization, records all predictions/dense updates,
independently reconstructs ordinary models, and benchmarks complete events.
Only the selected validation checkpoint and validation environments are loaded.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

import torch

from train_adapter import code_provenance, file_hash
from amortized_gnn_tta.adaptation.benchmark import (
    frozen_predictions, iterative_predictions, measure_methods,
)
from amortized_gnn_tta.adaptation.checkpoints import atomic_json
from amortized_gnn_tta.adaptation.iterative import (PEFTConfig, AdaptationConfig, adaptation_targets,
    development_supervision, fit_supervised_development)
from amortized_gnn_tta.adaptation_artifacts import state_sha256, save_weight_delta, load_weight_delta
from amortized_gnn_tta.data import load_environment_manifest, load_shift_environment
from amortized_gnn_tta.data.api import graph_fingerprint
from amortized_gnn_tta.evaluation import score_logits
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint, software_versions


def run(head_dir, lora_dir, output):
    roots = {"head": Path(head_dir), "lora": Path(lora_dir)}
    plans, studies = {}, {}
    for scope, root in roots.items():
        if json.loads((root/"INDEPENDENT_REPLAY.json").read_text())["status"] != "complete":
            raise ValueError("Both generator studies require a successful complete independent audit")
        plans[scope] = json.loads((root/"PLAN.json").read_text())
        with (root/"AGGREGATED_RESULTS.csv").open() as handle:
            studies[scope] = list(csv.DictReader(handle))
    plan = plans["lora"]["plan"]
    for field in ("architectures", "generator_seeds", "model_seed", "severity", "training_shifts", "conditioning", "include_raw_weights", "iterative_steps", "iterative_lr", "timed_shifts", "warmups", "repeats", "threads", "device", "banks", "environment_manifest", "epochs", "learning_rate", "delta_l2", "descriptor_version", "chunk_size", "hidden_dim", "num_layers", "teacher_steps", "teacher_lr", "diffusion_steps", "sampling_steps", "sampling_seed"):
        if plans["head"]["plan"].get(field) != plan.get(field):
            raise ValueError(f"Studies do not share the comparison protocol: {field}")
    if plan["device"] != "cpu":
        raise ValueError("This saved-study comparison has only been validated on CPU")
    torch.set_num_threads(plan["threads"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifact_root = Path(plans["lora"]["artifact_root"])
    manifest = load_environment_manifest(artifact_root/plan["environment_manifest"])
    metadata = {e["id"]:e for e in manifest["environments"]}
    provenance = {"head_study":str(roots["head"].resolve()), "lora_study":str(roots["lora"].resolve()),
        "plan":plan, "code":code_provenance(), "software":software_versions(),
        "timer":"canonical CPU graph and resident model to usable predictions; no graph-dependent cache across events",
        "excluded":["disk I/O", "canonical benchmark generation", "offline generator fitting", "label scoring", "reporting"],
        "evidence_hashes":{scope:{name:file_hash(root/name) for name in ("PLAN.json", "INDEPENDENT_REPLAY.json", "AGGREGATED_RESULTS.csv")} for scope,root in roots.items()}}
    atomic_json(output/"PLAN.json", provenance)
    import zipfile
    repo = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output/"code_snapshot.zip", "x") as archive:
        for name in provenance["code"]["source_hashes"]:
            archive.write(repo/name, name)
    rows, all_cases, all_timings, replayed = [], [], [], 0
    supervised = []
    atomic_json(output/"progress.json", {"status":"running"})
    for architecture in plan["architectures"]:
        selections = {scope:json.loads((root/f"{architecture}_selection.json").read_text()) for scope,root in roots.items()}
        selected = selections["head"]
        for field in ("source_checkpoint_sha256", "validation_state_sha256", "selected_environment_ids", "training_model_ids", "validation_model_id"):
            if selected[field] != selections["lora"][field]:
                raise ValueError(f"Studies selected different data/models: {field}")
        if file_hash(selected["source_checkpoint"]) != selected["source_checkpoint_sha256"]:
            raise ValueError("Source checkpoint changed")
        model, payload = load_checkpoint(selected["source_checkpoint"], build_model)
        model.eval()
        source = {n:v.clone() for n,v in model.state_dict().items()}
        pefts = {"head":PEFTConfig("head"), "lora":PEFTConfig(**plan["lora_config"])}
        cases, timings = [], []
        for env_id in selected["selected_environment_ids"]:
            if metadata[env_id]["role"] != "validation":
                continue
            env = metadata[env_id]
            bundle = load_shift_environment(manifest, env_id)
            graph = bundle.unlabeled_input(env_id)
            calls = {"frozen":lambda:frozen_predictions(model,graph)}
            for scope in pefts:
                for short, objective in (("entropy","entropy"), ("IM","information_maximization")):
                    calls[f"{scope}_{short}"] = lambda scope=scope,objective=objective:iterative_predictions(
                        model, graph, peft=pefts[scope], objective=objective, steps=plan["iterative_steps"],
                        lr=plan["iterative_lr"], initialization_seed=plan["lora_initialization_seed"])
            outputs = {name:call() for name,call in calls.items()}
            for name,result in outputs.items():
                if name == "frozen":
                    continue
                scope = name.split("_")[0]
                names = adaptation_targets(model, pefts[scope].method)
                final, offset = {n:v.clone() for n,v in source.items()}, 0
                for n in names:
                    count = source[n].numel()
                    final[n] = source[n]+result.delta[offset:offset+count].reshape_as(source[n])
                    offset += count
                if offset != result.delta.numel():
                    raise ValueError("Extra iterative update coordinates")
                restored = build_model(payload["model_spec"])
                restored.load_state_dict(final); restored.eval()
                with torch.no_grad():
                    replay = restored(graph.x, graph.edge_index)
                torch.testing.assert_close(replay, result.logits, rtol=1e-5, atol=1e-6)
                torch.save({"architecture":architecture,"method":name,"environment_id":env_id,
                    "graph_sha256":graph_fingerprint(graph),"source_sha256":state_sha256(source),
                    "target_names":list(names),"delta":result.delta,"final_state":final,
                    "actual_delta":{n:final[n]-source[n] for n in names},"logits":result.logits.detach()},
                    output/f"{architecture}-{name}-{env_id}.pt")
                saved = torch.load(output/f"{architecture}-{name}-{env_id}.pt", weights_only=True)
                if saved["source_sha256"] != state_sha256(source) or saved["graph_sha256"] != graph_fingerprint(graph):
                    raise ValueError("Saved baseline identity mismatch")
                offset = 0
                for n in names:
                    count = source[n].numel()
                    expected = source[n]+saved["delta"][offset:offset+count].reshape_as(source[n])
                    assert torch.equal(expected,saved["final_state"][n])
                    assert torch.equal(expected-source[n],saved["actual_delta"][n])
                    offset += count
                assert all(torch.equal(source[n],v) for n,v in saved["final_state"].items() if n not in names)
                restored.load_state_dict(saved["final_state"])
                with torch.no_grad():
                    torch.testing.assert_close(restored(graph.x,graph.edge_index),saved["logits"],rtol=1e-5,atol=1e-6)
                replayed += 1
            # Deployment computations above never receive a label object.
            indices, labels = bundle.select(env_id,"eval"), bundle.evaluation_labels(env_id,"eval")
            scores = {name:score_logits(result.logits,indices,labels).accuracy for name,result in outputs.items()}
            case = {"architecture":architecture,"environment_id":env_id,"family":env["family"],"shift":env["shift"],"scores":scores}
            cases.append(case); all_cases.append(case)
            if env["family"] == "control" or env["shift"] in plan["timed_shifts"]:
                timing = measure_methods(calls,warmups=plan["warmups"],repeats=plan["repeats"])
                profiles = {name:iterative_predictions(model,graph,peft=pefts[name.split("_")[0]],
                    objective="entropy" if name.endswith("entropy") else "information_maximization",
                    steps=plan["iterative_steps"],lr=plan["iterative_lr"],initialization_seed=plan["lora_initialization_seed"],profile=True).stages_seconds
                    for name in calls if name != "frozen"}
                timed = {"architecture":architecture,"environment_id":env_id,"timing":timing,"profiles":profiles}
                timings.append(timed); all_timings.append(timed)
            atomic_json(output/"cases.json",all_cases); atomic_json(output/"timings.json",all_timings)
        shifted = [r for r in cases if r["family"] != "control"]
        control = [r for r in cases if r["family"] == "control"]
        latencies = {name:1000*statistics.mean(r["timing"][name]["median_seconds"] for r in timings) for name in calls}
        for name in calls:
            scope = "none" if name == "frozen" else name.split("_")[0]
            accuracy = 100*statistics.mean(r["scores"][name] for r in shifted)
            row = dict(architecture=architecture,method=name,update_scope=scope,
                updated_variables=0 if scope=="none" else sum(source[n].numel() for n in adaptation_targets(model,"head")) if scope=="head" else int(next(r for r in studies["lora"] if r["architecture"]==architecture)["factor_parameter_count"]),
                accuracy_percent=accuracy,seed_sd_percent=None,gain_pp=statistics.mean(100*(r["scores"][name]-r["scores"]["frozen"]) for r in shifted),
                control_gain_pp=statistics.mean(100*(r["scores"][name]-r["scores"]["frozen"]) for r in control),
                harmed=sum(r["scores"][name]<r["scores"]["frozen"] for r in shifted),cases=len(shifted),
                pipeline_ms=latencies[name], matched_IM_ms=None if scope=="none" else latencies[scope+"_IM"],
                matched_speedup=None if scope=="none" else latencies[scope+"_IM"]/latencies[name],offline_seconds=0.,teacher_seconds=0.)
            rows.append(row)
        for scope,study in studies.items():
            for r in study:
                if r["architecture"] != architecture: continue
                rows.append(dict(architecture=architecture,method=r["method"],update_scope=scope,
                    updated_variables=int(r["target_parameter_count"] if scope=="head" else r["factor_parameter_count"]),
                    accuracy_percent=float(r["adapted_accuracy_percent"]),seed_sd_percent=float(r["adapted_accuracy_sd_percent"]),
                    gain_pp=float(r["gain_pp"]),control_gain_pp=float(r["control_gain_pp"]),harmed=int(r["harm_cases"]),cases=int(r["repeated_shift_cases"]),
                    pipeline_ms=float(r["generated_pipeline_ms"]),matched_IM_ms=float(r["iterative_pipeline_ms"]),
                    matched_speedup=float(r["speedup_ratio_of_mean_latencies"]),offline_seconds=float(r["offline_wall_seconds"]),teacher_seconds=float(r["teacher_wall_seconds"])))
        assert state_sha256(source) == state_sha256(model.state_dict())
        print("COMPARED", architecture, "validation cases",len(cases),flush=True)
        write_table(output,rows,plan)
        supervised.extend(supervised_references(architecture, selected, plan, artifact_root, manifest, metadata, output))
        atomic_json(output/"supervised_references.json",supervised)
        write_supervised_table(output,supervised)
    atomic_json(output/"progress.json",dict(status="complete",iterative_saved_replays=replayed,rows=len(rows),supervised_cases=len(supervised)))



def supervised_references(architecture, selected, plan, artifact_root, manifest, metadata, output):
    """Existing labeled references, deliberately outside the deployment table.

    Timer uses the existing public supervised fitting interface including its
    validation/selection/materialization. It is not an optimized unlabeled
    counterpart and has a different permitted graph/checkpoint/label pool.
    """
    bank_path = artifact_root/plan["banks"][architecture][0]
    bank = json.loads((bank_path/"bank.json").read_text())
    model_id = selected["training_model_ids"][0]
    entry = next(row for row in bank["models"] if row["model_id"] == model_id)
    if file_hash(bank_path/entry["checkpoint"]) != entry["checkpoint_sha256"]:
        raise ValueError("Supervised reference source checkpoint changed")
    model,payload = load_checkpoint(bank_path/entry["checkpoint"],build_model)
    model.eval()
    initial_hash = state_sha256(model.state_dict())
    if initial_hash != selected["source_hashes"][model_id]:
        raise ValueError("Unapproved supervised source model")
    # One meta-training lineage; same four conditions for each labeled method.
    chosen = []
    for env_id in selected["selected_environment_ids"]:
        env = metadata[env_id]
        if env["role"] == "meta_train" and (env["family"] == "control" or env["shift"] in plan["timed_shifts"]):
            if not any(row["shift"] == env["shift"] for row in chosen): chosen.append(env)
    results=[]
    for env in chosen:
        env_id=env["id"]
        bundle=load_shift_environment(manifest,env_id)
        graph=bundle.unlabeled_input(env_id)
        supervision=development_supervision(bundle,env_id)
        config=AdaptationConfig(steps=plan["iterative_steps"],learning_rate=plan["iterative_lr"],
            patience=plan["iterative_steps"],initialization_seed=plan["lora_initialization_seed"])
        pefts={"head":PEFTConfig("head"),"lora":PEFTConfig(**plan["lora_config"]),"full":PEFTConfig("full")}
        def fit(peft):
            result=fit_supervised_development(model,graph,supervision,peft,config)
            with torch.no_grad(): logits=result.model(graph.x,graph.edge_index)
            return result,logits
        calls={scope:lambda peft=peft:fit(peft) for scope,peft in pefts.items()}
        # Three repetitions are diagnostic only, retained raw and separate.
        timing=measure_methods({name:lambda call=call:call()[1] for name,call in calls.items()},warmups=1,repeats=3)
        outputs={name:call() for name,call in calls.items()}
        baseline=frozen_predictions(model,graph)
        indices,labels=bundle.select(env_id,"test"),bundle.evaluation_labels(env_id,"test")
        frozen=score_logits(baseline.logits,indices,labels).accuracy
        for scope,(fit_result,logits) in outputs.items():
            path=output/f"supervised-{architecture}-{scope}-{env_id}.pt"
            artifact=save_weight_delta(path,fit_result,payload["model_spec"],graph,
                context={"protocol":"supervised-meta-training-diagnostic", "model_id":model_id,"environment_id":env_id,"role":"meta_train"})
            restored, saved = load_weight_delta(path)
            with torch.no_grad():
                torch.testing.assert_close(restored(graph.x,graph.edge_index),logits,rtol=1e-5,atol=1e-6)
            accuracy=score_logits(logits,indices,labels).accuracy
            results.append(dict(architecture=architecture,method=scope,environment_id=env_id,family=env["family"],shift=env["shift"],
                frozen_accuracy_percent=100*frozen,accuracy_percent=100*accuracy,gain_pp=100*(accuracy-frozen),
                updated_variables=artifact["trainable_parameters"],pipeline_ms=1000*timing[scope]["median_seconds"],timing=timing[scope],selected_step=fit_result.selected_step,
                budget_steps=plan["iterative_steps"],source_model_id=model_id,source_sha256=initial_hash,
                timer="public supervised fitting including validation/selection/materialization plus usable predictions; artifact export and scoring excluded"))
        if state_sha256(model.state_dict()) != initial_hash: raise ValueError("Supervised reference mutated source")
    return results



def write_supervised_table(output,records):
    grouped={}
    for row in records: grouped.setdefault((row["architecture"],row["method"]),[]).append(row)
    rows=[]
    for (architecture,method),cases in grouped.items():
        shifted=[r for r in cases if r["family"] != "control"]
        controls=[r for r in cases if r["family"] == "control"]
        rows.append(dict(architecture=architecture,method=method,updated_variables=cases[0]["updated_variables"],
            frozen_accuracy_percent=statistics.mean(r["frozen_accuracy_percent"] for r in shifted),
            accuracy_percent=statistics.mean(r["accuracy_percent"] for r in shifted),
            gain_pp=statistics.mean(r["gain_pp"] for r in shifted),
            control_gain_pp=statistics.mean(r["gain_pp"] for r in controls),
            pipeline_ms=statistics.mean(r["pipeline_ms"] for r in cases),
            selected_steps=[r["selected_step"] for r in cases],budget_steps=cases[0]["budget_steps"]))
    for row in rows:
        full=next(r["pipeline_ms"] for r in rows if r["architecture"]==row["architecture"] and r["method"]=="full")
        row["full_fit_cost_ratio"]=full/row["pipeline_ms"]
    with (output/"SUPERVISED_REFERENCES.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# Separate supervised development references", "",
        "One permitted meta-training checkpoint and one meta-training graph lineage per architecture; control plus attribute, homophily and combined cases. Uses train labels for fitting and validation labels for step selection; native test labels are offline diagnostics. These are not unlabeled validation/test deployment results. Accuracy excludes control; timing includes all four cases.", "",
        "The timer runs the existing public supervised fitter including validation, step selection, model materialization and usable prediction. Raw samples use one warmup/three repetitions. This is a different implementation and label-access boundary from the streamlined unlabeled optimizer. Full-fit cost ratios compare only these labeled references, not generator speedup or equal utility.", "",
        "| Model | Supervised scope | Trainable variables | Frozen (%) | Adapted (%) | Gain (pp) | Control gain (pp) | Complete public fit (ms) | Full / method cost | Selected steps (control/shift order saved in JSON) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['architecture']} | {r['method']} | {r['updated_variables']:,} | {r['frozen_accuracy_percent']:.3f} | {r['accuracy_percent']:.3f} | {r['gain_pp']:+.3f} | {r['control_gain_pp']:+.3f} | {r['pipeline_ms']:.2f} | {r['full_fit_cost_ratio']:.2f}× | {r['selected_steps']} |")
    lines += ["", "Thirty steps is a fixed development budget, not an optimum or proven recovery ceiling. Step zero is an allowed selection. These teachers differ from diffusion's fixed-step CE-plus-dense-L2 training teachers. Every saved reference delta was validated and reloaded with the existing factor/state artifact checker."]
    (output/"SUPERVISED_REFERENCES.md").write_text("\n".join(lines)+"\n")


def write_table(output,rows,plan):
    with (output/"COMPARISON.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# Matched head and generated-LoRA development comparison","",
        f"CSBM validation only; one held-out checkpoint/graph lineage per architecture. Accuracy averages the selected shifted cases; controls are separate. Generators use {len(plan['generator_seeds'])} training seeds, compact moments without raw weights, {plan['epochs']} epochs. Diffusion uses one candidate and {plan['sampling_steps']} DDIM steps. Final target labels were not accessed.","",
        "Updated variables count direct head scalars or A/B factors; frozen original weights are not trainable. Both methods use the same rank-4 first-layer/rank-1 second-layer LoRA sites/scaling. Biases/attention stay fixed for LoRA. Generator training uses permitted labeled meta-training graphs; iterative entropy/IM and deployment generator inputs use no target labels.","",
        "| Model | Method | Scope | Updated variables | Accuracy ± seed SD (%) | Gain (pp) | Control gain (pp) | Harmed / shifted cases | Pipeline (ms) | Matched IM (ms) | IM / method | Offline fit (s) | Teacher subset (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        acc=f"{r['accuracy_percent']:.3f}"+(f" ± {r['seed_sd_percent']:.3f}" if r['seed_sd_percent'] is not None else "")
        ratio=f"{r['matched_speedup']:.2f}×" if r['matched_speedup'] is not None else "—"
        reference=f"{r['matched_IM_ms']:.2f}" if r['matched_IM_ms'] is not None else "—"
        lines.append(f"| {r['architecture']} | {r['method']} | {r['update_scope']} | {r['updated_variables']:,} | {acc} | {r['gain_pp']:+.3f} | {r['control_gain_pp']:+.3f} | {r['harmed']}/{r['cases']} | {r['pipeline_ms']:.2f} | {reference} | {ratio} | {r['offline_seconds']:.2f} | {r['teacher_seconds']:.2f} |")
    lines += ["", "Ratios >1 mean faster. Each generated ratio uses the IM measurement paired with that fit, scope and timing conditions. Baseline rows use a separate interleaved timing pass; small denominator differences are ordinary machine variation. Primary latency includes conditioning, generation/optimization, applying the update and required prediction, with no persistent target cache. Disk I/O, graph construction, label scoring and offline fitting are excluded. Raw repetitions, first-event timings and separate stage profiles are preserved.","",
        "Head can reuse its frozen encoder within an event. LoRA changes both graph layers and recomputes the adapted encoder. Comparing their speedups does not establish equal accuracy or equal difficulty. Entropy and IM are distinct objectives; neither guarantees improved accuracy. A selected identity generator can be fast yet provide no adaptation gain.","",
        "Hypernetwork CE and diffusion teacher-factor MSE differ in objective and optimizer-step budgets. Offline diffusion time includes supervised teacher fitting; it is not free. Three generator seeds do not provide independent graph/checkpoint replication. Seed SD measures generator-training variation only. No optimality or definitive generator ranking follows.","",
        "Full-model/head/LoRA supervised references are measured separately in supervised_references.json on permitted meta-training graphs; their validation-aware public fitting includes selection/materialization. They have no comparable unlabeled validation row. Broader direct-coordinate generation, historical compact-plus-head-weights and expensive A–D inputs are separate recipes; old component-only 45× ratios must not be mixed with these complete-event timings. This study does not implement real graphs or five-candidate diffusion selection.",""]
    (output/"COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-study",type=Path,required=True)
    parser.add_argument("--lora-study",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a fresh path")
    try:
        run(args.head_study,args.lora_study,args.output_dir)
    except BaseException as error:
        if args.output_dir.exists() and (args.output_dir/"progress.json").exists():
            atomic_json(args.output_dir/"progress.json", {"status":"failed", "error":f"{type(error).__name__}: {error}"})
        raise
