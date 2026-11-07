"""Matched development diagnostic including labeled full fine-tuning.

Reuses the completed supervised references and evaluates frozen/IM/generator
predictions on exactly their source checkpoint and meta-training graph cases.
These checkpoint/environment pairs participated in generator fitting: this is
an in-sample development diagnostic, not additional held-out evidence.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

import torch

from train_adapter import code_provenance, file_hash
from amortized_gnn_tta.adaptation.benchmark import (
    frozen_predictions, generated_predictions, iterative_predictions, measure_methods,
)
from amortized_gnn_tta.adaptation.checkpoints import atomic_json, load_generator_checkpoint
from amortized_gnn_tta.adaptation.iterative import PEFTConfig, adaptation_targets
from amortized_gnn_tta.adaptation_artifacts import state_sha256, load_weight_delta
from amortized_gnn_tta.data import load_environment_manifest, load_shift_environment
from amortized_gnn_tta.data.api import graph_fingerprint
from amortized_gnn_tta.evaluation import score_logits
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import software_versions


def run(comparison, output):
    comparison, output = Path(comparison), Path(output)
    if json.loads((comparison/"progress.json").read_text())["status"] != "complete":
        raise ValueError("Requires a complete parent comparison")
    parent = json.loads((comparison/"PLAN.json").read_text())
    plan = parent["plan"]
    roots = {scope:Path(parent[scope+"_study"]) for scope in ("head","lora")}
    root_plan = json.loads((roots["lora"]/"PLAN.json").read_text())
    artifact_root = Path(root_plan["artifact_root"])
    manifest_path = artifact_root/plan["environment_manifest"]
    manifest = load_environment_manifest(manifest_path)
    metadata = {e["id"]:e for e in manifest["environments"]}
    references = json.loads((comparison/"supervised_references.json").read_text())
    torch.set_num_threads(plan["threads"])
    output.mkdir(parents=True,exist_ok=False)
    provenance = dict(parent_comparison=str(comparison.resolve()),
        parent_plan_sha256=file_hash(comparison/"PLAN.json"), references_sha256=file_hash(comparison/"supervised_references.json"),
        code=code_provenance(),software=software_versions(),plan=plan,warmups=1,repeats=3,
        protocol="matched-meta-training-diagnostic-v1",label_access="Supervised references fit native train labels and select by native validation labels. All scoring uses native test nodes after predictions. Generators/IM receive unlabeled inputs only.",
        limitation="Models and graph environments participated in generator fitting. This is not a held-out graph/checkpoint test. Public supervised fitter includes validation/selection/materialization; do not headline a ratio against it as matched unlabeled speedup.")
    atomic_json(output/"PLAN.json", provenance)
    import zipfile
    repo = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output/"code_snapshot.zip", "x") as archive:
        for name in provenance["code"]["source_hashes"]:
            archive.write(repo/name, name)
    atomic_json(output/"progress.json",dict(status="running"))
    all_cases, timing_rows, aggregates = [], [], []
    replayed = 0
    for architecture in plan["architectures"]:
        refs = [r for r in references if r["architecture"]==architecture]
        if not refs: raise ValueError("Missing supervised architecture")
        first = refs[0]
        _, initial_payload = load_weight_delta(comparison/f"supervised-{architecture}-{first['method']}-{first['environment_id']}.pt")
        source = build_model(initial_payload["model_spec"])
        source.load_state_dict(initial_payload["initial_state"]);source.eval()
        initial = {name:value.clone() for name,value in source.state_dict().items()}
        source_hash = state_sha256(initial)
        selection = json.loads((roots["lora"]/f"{architecture}_selection.json").read_text())
        if file_hash(manifest_path)!=selection["environment_manifest_sha256"]:
            raise ValueError("Environment manifest changed")
        if source_hash != first["source_sha256"] or first["source_model_id"] not in selection["training_model_ids"]:
            raise ValueError("Unapproved development checkpoint")
        generators, checkpoint_hashes = {}, {}
        for scope in roots:
            for method in ("hypernetwork","diffusion"):
                for seed in plan["generator_seeds"]:
                    checkpoint=roots[scope]/f"{architecture}-{method}-compact{'-all' if scope=='lora' else ''}-{seed}"/"generator_best.pt"
                    generator,descriptor,payload=load_generator_checkpoint(checkpoint,model=source)
                    descriptor.prepare_model(source)
                    name=f"{method}_{scope}_{seed}"
                    generators[name]=(generator,descriptor,payload)
                    checkpoint_hashes[name]=dict(path=str(checkpoint.resolve()),sha256=file_hash(checkpoint))
        atomic_json(output/f"{architecture}_selection.json",dict(source_sha256=source_hash,
            source_reference=str((comparison/f"supervised-{architecture}-{first['method']}-{first['environment_id']}.pt").resolve()),
            source_model_id=first["source_model_id"],generator_checkpoints=checkpoint_hashes,
            environment_manifest_sha256=file_hash(manifest_path)))
        arch_cases,arch_timings=[],[]
        for env_id in dict.fromkeys(r["environment_id"] for r in refs):
            env=metadata[env_id]
            if env["role"]!="meta_train":raise ValueError("Labeled diagnostic requires a permitted meta-training graph")
            bundle=load_shift_environment(manifest,env_id);graph=bundle.unlabeled_input(env_id)
            calls={"frozen":lambda:frozen_predictions(source,graph)}
            for scope in ("head","lora"):
                peft=PEFTConfig("head") if scope=="head" else PEFTConfig(**plan["lora_config"])
                calls[f"IM_{scope}"]=lambda peft=peft:iterative_predictions(source,graph,steps=plan["iterative_steps"],lr=plan["iterative_lr"],peft=peft,initialization_seed=plan["lora_initialization_seed"])
            for name,(generator,descriptor,payload) in generators.items():
                calls[name]=lambda g=generator,d=descriptor,p=payload:generated_predictions(source,graph,d,g,scope=p["target_scope"])
            timing=measure_methods(calls,warmups=1,repeats=3)
            outputs={name:call() for name,call in calls.items()}
            for name,result in outputs.items():
                if name=="frozen":continue
                if name in generators:
                    payload=generators[name][2]
                    schema=[("conv2."+n if payload["target_scope"]=="head" else n,tuple(s)) for n,s in payload["schema"]]
                else:
                    scope=name.split("_")[1]
                    schema=[(n,tuple(dict(source.named_parameters())[n].shape)) for n in adaptation_targets(source,scope)]
                final,offset={n:v.clone() for n,v in initial.items()},0
                for n,shape in schema:
                    count=initial[n].numel();final[n]=initial[n]+result.delta[offset:offset+count].reshape(shape);offset+=count
                if offset!=result.delta.numel():raise ValueError("Extra update coordinates")
                restored=build_model(initial_payload["model_spec"]);restored.load_state_dict(final);restored.eval()
                with torch.no_grad():torch.testing.assert_close(restored(graph.x,graph.edge_index),result.logits,rtol=1e-5,atol=1e-6)
                path=output/f"{architecture}-{name}-{env_id}.pt"
                evidence=dict(model_spec=initial_payload["model_spec"],source_state_sha256=source_hash,source_model_id=first["source_model_id"],environment_id=env_id,
                    graph_sha256=graph_fingerprint(graph),method=name,schema=schema,delta=result.delta,final_state=final,logits=result.logits)
                if name in generators:
                    evidence["generator_checkpoint_sha256"]=checkpoint_hashes[name]["sha256"]
                    generator,descriptor,payload=generators[name]
                    if "update_representation" in payload:
                        with torch.no_grad():evidence["factor_offsets"]=generator.forward_factors(descriptor(source,graph))
                        evidence["lora_space"]=payload["lora_space"]
                torch.save(evidence,path)
                saved=torch.load(path,weights_only=True);restored.load_state_dict(saved["final_state"])
                with torch.no_grad():torch.testing.assert_close(restored(graph.x,graph.edge_index),saved["logits"],rtol=1e-5,atol=1e-6)
                replayed+=1
            # No diagnostic scoring labels entered any deployment call.
            indices,labels=bundle.select(env_id,"test"),bundle.evaluation_labels(env_id,"test")
            scores={name:100*score_logits(result.logits,indices,labels).accuracy for name,result in outputs.items()}
            for reference in (r for r in refs if r["environment_id"]==env_id):
                if reference["source_sha256"]!=source_hash:raise ValueError("Supervised source mismatch")
                if abs(reference["frozen_accuracy_percent"]-scores["frozen"])>1e-10:raise ValueError("Supervised graph/checkpoint mismatch")
                scores["supervised_"+reference["method"]]=reference["accuracy_percent"]
                timing["supervised_"+reference["method"]]=reference["timing"]
            case=dict(architecture=architecture,environment_id=env_id,family=env["family"],shift=env["shift"],source_sha256=source_hash,scores=scores)
            arch_cases.append(case);all_cases.append(case)
            timed=dict(architecture=architecture,environment_id=env_id,timing=timing)
            arch_timings.append(timed);timing_rows.append(timed)
            atomic_json(output/"cases.json",all_cases);atomic_json(output/"timings.json",timing_rows)
        method_groups={name:[name] for name in ("frozen","IM_head","IM_lora","supervised_head","supervised_lora","supervised_full")}
        for scope in roots:
            for method in ("hypernetwork","diffusion"):
                method_groups[method+"_"+scope]=[f"{method}_{scope}_{seed}" for seed in plan["generator_seeds"]]
        shifted=[r for r in arch_cases if r["family"]!="control"]
        control=[r for r in arch_cases if r["family"]=="control"]
        for method,names in method_groups.items():
            seed_means=[statistics.mean(r["scores"][n] for r in shifted) for n in names]
            scope=method.rsplit("_",1)[-1]
            count=0 if method=="frozen" else next(r["updated_variables"] for r in refs if r["method"]==scope)
            aggregates.append(dict(architecture=architecture,method=method,updated_variables=count,
                accuracy_percent=statistics.mean(seed_means),seed_sd_percent=statistics.stdev(seed_means) if len(seed_means)>1 else None,
                frozen_accuracy_percent=statistics.mean(r["scores"]["frozen"] for r in shifted),
                gain_pp=statistics.mean(r["scores"][n]-r["scores"]["frozen"] for r in shifted for n in names),
                control_gain_pp=statistics.mean(r["scores"][n]-r["scores"]["frozen"] for r in control for n in names),
                pipeline_ms=1000*statistics.mean(r["timing"][n]["median_seconds"] for r in arch_timings for n in names),
                label_access="native train/validation labels" if method.startswith("supervised") else "none at deployment"))
        if state_sha256(source.state_dict())!=source_hash:raise ValueError("Diagnostic mutated source")
        write_table(output,aggregates)
        print("MATCHED DEVELOPMENT",architecture,flush=True)
    atomic_json(output/"progress.json",dict(status="complete",rows=len(aggregates),saved_replays=replayed))


def write_table(output,rows):
    with (output/"MATCHED_FULL_COMPARISON.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# Matched full-fine-tuning development diagnostic","",
        "Every method uses the identical starting checkpoint and the same permitted meta-training graphs. Native test nodes are scored after predictions; supervised methods use native train labels and native validation labels for fitting/step selection. These models/environments participated in generator training, so this is an in-sample development diagnostic, not a held-out result. Accuracy averages three shifted cases; controls are separate. Generator SD is across training seeds.","",
        "Timing uses complete prediction events. The supervised rows retain the parent public-fitter measurements, including validation/selection/materialization; generator/IM timing is a fresh interleaved one-warmup/three-repetition pass. Required computations/objectives and label access differ; do not headline cross-method ratios here as matched unlabeled speedup. Thirty adaptation steps is a budget, not a guaranteed optimum or ceiling.","",
        "| Model | Method | Updated variables | Label access | Accuracy ± seed SD (%) | Frozen (%) | Gain (pp) | Control gain (pp) | Pipeline (ms) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        accuracy=f"{r['accuracy_percent']:.3f}"+(f" ± {r['seed_sd_percent']:.3f}" if r['seed_sd_percent'] is not None else "")
        lines.append(f"| {r['architecture']} | {r['method']} | {r['updated_variables']:,} | {r['label_access']} | {accuracy} | {r['frozen_accuracy_percent']:.3f} | {r['gain_pp']:+.3f} | {r['control_gain_pp']:+.3f} | {r['pipeline_ms']:.2f} |")
    (output/"MATCHED_FULL_COMPARISON.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error("Output directory already exists; choose a fresh path")
    try:run(args.comparison_dir,args.output_dir)
    except BaseException as error:
        if (args.output_dir/"progress.json").exists():atomic_json(args.output_dir/"progress.json",dict(status="failed",error=f"{type(error).__name__}: {error}"))
        raise
