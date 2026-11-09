"""Sequential Arxiv design studies, with held-out labels sealed by default.

Requires a completed, hash-checked source bank. Every run has an exclusive output
folder; completed outputs may be resumed only with the same recipe and source
bank. A failure leaves explicit evidence and must not become a completed result.
"""

import argparse
from dataclasses import asdict
import json
import hashlib
import zipfile
from pathlib import Path
import time
import torch

from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint, software_versions
from amortized_gnn_tta.evaluation import score_logits, select_device
from amortized_gnn_tta.adaptation.experiments import make_descriptor
from amortized_gnn_tta.adaptation.generator import (
    HeadAdapterGenerator,
    extract_target_parameters,
)
from amortized_gnn_tta.adaptation.diffusion import (
    DiffusionHeadGenerator,
    fit_oracle_deltas,
    train_diffusion_generator,
    OracleTargetRecord,
)
from amortized_gnn_tta.adaptation.meta_trainer import train_adaptation_generator
from amortized_gnn_tta.adaptation.distillation import train_teacher_hypernetwork
from amortized_gnn_tta.adaptation.lora_generator import (
    LoRAGenerator,
    LoRAUpdateSpace,
)
from amortized_gnn_tta.adaptation.iterative import PEFTConfig
from amortized_gnn_tta.adaptation.checkpoints import (
    atomic_json,
    save_generator_checkpoint,
    load_generator_checkpoint,
)
from amortized_gnn_tta.adaptation.benchmark import (
    generated_predictions,
    iterative_predictions,
    frozen_predictions,
    execution_hardware,
)
from amortized_gnn_tta.adaptation.candidates import agreement_predictions
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from train_adapter import file_hash, code_provenance


def lora_config(plan):
    rank = plan["lora_rank"]
    return PEFTConfig(
        "lora",
        backbone_rank=rank,
        head_rank=rank,
        backbone_alpha=float(rank),
        head_alpha=float(rank),
    )


def descriptor_chunk_size(plan):
    # Engineering memory/launch budget, not a different summary statistic.
    # A 16,384-by-128 float32 temporary is 8 MiB. Historical pilots retain 1024.
    return plan.get("descriptor_chunk_size", 16384 if plan.get("bank_kind") == "formal-real-v1" else 1024)


def create_generator(model, conditioning, update, method, plan):
    scope = "all" if update == "lora" else "head"
    descriptor, _ = make_descriptor(model, conditioning, scope=scope,
                                    chunk_size=descriptor_chunk_size(plan))
    weights, schema = extract_target_parameters(model, scope)
    space = (
        LoRAUpdateSpace.from_model(model, lora_config(plan), 801)
        if update == "lora"
        else None
    )
    arguments = dict(
        descriptor_dim=descriptor.output_dim,
        head_dim=space.factor_dim if space else len(weights),
        include_head_weights=False,
    )
    if method == "diffusion":
        core = DiffusionHeadGenerator(
            **arguments,
            diffusion_steps=plan["diffusion_steps"],
            sampling_steps=plan["sampling_steps"],
        )
    else:
        core = HeadAdapterGenerator(**arguments)
    return descriptor, LoRAGenerator(core, space) if space else core, schema, scope


class Study:
    def __init__(self, args):
        self.args = args
        self.root = args.output_dir
        self.root.mkdir(parents=True, exist_ok=True)
        bank = json.loads((args.bank / "bank.json").read_text())
        if bank["status"] != "complete" and not (
            bank["status"] in {"running", "partial"} and getattr(args, "stop_after", None) == 2
        ):
            raise ValueError(
                "A partial bank is allowed only for the GCN input/update design stage"
            )
        self.plan = bank["plan"]
        self.formal_audit = None
        if self.plan.get("bank_kind") == "formal-real-v1":
            audit_path = args.bank / "DIVERSITY.json"
            if not audit_path.exists():
                raise ValueError("Formal banks require a source-only diversity audit before generator use")
            self.formal_audit = json.loads(audit_path.read_text())
            if self.formal_audit["bank_sha256"] != file_hash(args.bank / "bank.json"):
                raise ValueError("Formal diversity audit does not match the current bank")
        self.device = str(select_device(self.plan.get("device", "cpu")))
        torch.set_num_threads(self.plan["threads"])
        self.manifest = json.loads(
            (args.bank / "environment_manifest.json").read_text()
        )
        from amortized_gnn_tta.data.loaders import make_environment_loader

        self.loader = make_environment_loader(
            self.plan.get("dataset", "ogbn-arxiv"), args.data_root, self.manifest
        )
        self.models = {}
        self.rows = bank["models"]
        for row in self.rows:
            path = args.bank / row["path"]
            if file_hash(path) != row["checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed")
        self.identity = dict(
            protocol="real-study-inputs-v2",
            bank_plan_sha256=hashlib.sha256(
                json.dumps(self.plan, sort_keys=True).encode()
            ).hexdigest(),
            manifest_sha256=file_hash(args.bank / "environment_manifest.json"),
            plan=self.plan,
            descriptor_chunk_size=descriptor_chunk_size(self.plan),
        )
        pin_path = self.root / "SOURCE_PINS.json"
        self.pins = json.loads(pin_path.read_text()) if pin_path.exists() else {}
        indexed = {r["id"]: r for r in self.rows}
        for mid, pin in self.pins.items():
            if mid not in indexed or indexed[mid] != pin:
                raise ValueError("Previously used source checkpoint identity changed")
        identity_path = self.root / "INPUTS.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != self.identity:
                raise ValueError("Resume inputs changed")
        else:
            atomic_json(identity_path, self.identity)
            provenance = code_provenance()
            atomic_json(
                self.root / "CODE.json",
                dict(
                    code=provenance,
                    software=software_versions(),
                    hardware=execution_hardware(self.device),
                    final_test_scored=False,
                ),
            )
            source_root = Path(__file__).resolve().parents[1]
            with zipfile.ZipFile(self.root / "code_snapshot.zip", "x") as archive:
                for name, digest in provenance["source_hashes"].items():
                    if file_hash(source_root / name) != digest:
                        raise ValueError("Code changed during snapshot")
                    archive.write(source_root / name, name)
        execution = json.loads((self.root / "CODE.json").read_text())
        recorded = execution["code"]["source_hashes"]
        if recorded != code_provenance()["source_hashes"]:
            raise ValueError(
                "Study source changed; preserve this run and use an explicitly reviewed new study"
            )
        if execution["software"] != software_versions() or execution.get("hardware") != execution_hardware(self.device):
            raise ValueError("Study hardware/dependencies/precision changed; refusing mixed-environment resume")
        self.validation = [
            e for e in self.manifest["environments"] if e["role"] == "validation"
        ]

    def sources(self, architecture, role):
        key = (architecture, role)
        if key not in self.models:
            if self.formal_audit is not None:
                audit = self.formal_audit['architectures'].get(architecture)
                if audit is None or not any(p['probability_rmse'] > 0 for p in audit['pairs']):
                    raise ValueError('Formal bank has no observed functional variation; review construction')
            required_seeds = set(self.plan.get("checkpoint_seeds", {}).get(role, []))
            available = [
                r
                for r in self.rows
                if r["architecture"] == architecture and r["role"] == role
            ]
            if required_seeds and {r["seed"] for r in available} != required_seeds:
                raise ValueError("Requested source architecture/role is incomplete")
            loaded = {}
            for row in self.rows:
                if row["architecture"] == architecture and row["role"] == role:
                    model, _ = load_checkpoint(
                        self.args.bank / row["path"], build_model, device=self.device
                    )
                    if state_sha256(model.state_dict()) != row["state_sha256"]:
                        raise ValueError("Source state mismatch")
                    loaded[row["id"]] = model
                    self.pins[row["id"]] = row
                    atomic_json(self.root / "SOURCE_PINS.json", self.pins)
            if not loaded:
                raise ValueError("Missing checkpoint role")
            self.models[key] = loaded
        return self.models[key]

    def teachers(
        self, architecture, conditioning, update, descriptor, generator, scope
    ):
        path = self.root / f"teachers-{architecture}-{conditioning}-{update}.pt"
        if path.exists():
            saved = torch.load(path, weights_only=True, map_location="cpu")
            if saved["identity"] != self.identity:
                raise ValueError("Teacher cache recipe changed")
            return [OracleTargetRecord(**row) for row in saved["records"]], saved[
                "seconds"
            ]
        start = time.perf_counter()
        targets = fit_oracle_deltas(
            self.sources(architecture, "meta_train"),
            self.manifest,
            self.loader,
            descriptor,
            steps=self.plan["teacher_steps"],
            lr=self.plan["teacher_lr"],
            delta_l2_reg=self.plan["delta_l2"],
            target_scope=scope,
            device=self.device,
            update_space=getattr(generator, "update_space", None),
        )
        elapsed = time.perf_counter() - start
        torch.save(
            dict(
                identity=self.identity,
                records=[asdict(r) for r in targets],
                seconds=elapsed,
            ),
            path,
        )
        return targets, elapsed

    def fit(self, architecture, conditioning, update, method, seed):
        name = f"{architecture}-{conditioning}-{update}-{method}-{seed}"
        folder = self.root / name
        done = folder / "RESULT.json"
        if done.exists():
            return json.loads(done.read_text())
        if folder.exists():
            raise FileExistsError(
                f"Incomplete run {folder}; preserve it and diagnose before resuming"
            )
        folder.mkdir()
        torch.manual_seed(seed)
        print("GENERATOR START", name, flush=True)
        model = next(iter(self.sources(architecture, "meta_train").values()))
        descriptor, generator, schema, scope = create_generator(
            model, conditioning, update, method, self.plan
        )
        common = dict(
            models=self.sources(architecture, "meta_train"),
            manifest=self.manifest,
            load_env_fn=self.loader,
            descriptor=descriptor,
            generator=generator,
            val_models=self.sources(architecture, "validation"),
            epochs=self.plan["generator_epochs"],
            lr=self.plan["generator_lr"],
            target_scope=scope,
            device=self.device,
        )
        teacher_seconds = 0.0
        if method in {"diffusion", "teacher_hypernetwork"}:
            teachers, teacher_seconds = self.teachers(
                architecture, conditioning, update, descriptor, generator, scope
            )
            if method == "diffusion":
                result = train_diffusion_generator(
                    **common,
                    oracle_steps=self.plan["teacher_steps"],
                    oracle_lr=self.plan["teacher_lr"],
                    delta_l2_reg=self.plan["delta_l2"],
                    candidate_seeds=self.plan["candidate_seeds"],
                    teacher_targets=teachers,
                )
            else:
                result = train_teacher_hypernetwork(**common, targets=teachers)
        else:
            result = train_adaptation_generator(
                **common, delta_l2_reg=self.plan["delta_l2"]
            )
        save_generator_checkpoint(
            folder / "generator.pt",
            generator,
            descriptor,
            schema,
            target_scope=scope,
            provenance=dict(
                identity=self.identity,
                architecture=architecture,
                conditioning=conditioning,
                update=update,
                method=method,
                seed=seed,
                candidate_seeds=self.plan["candidate_seeds"]
                if method == "diffusion"
                else None,
            ),
            history=result.history,
            state_dict=result.best_state_dict,
        )
        generator, descriptor, _ = load_generator_checkpoint(
            folder / "generator.pt", model=model, device=self.device
        )
        records = self.evaluate(
            architecture, generator, descriptor, scope, method, folder
        )
        row = dict(
            name=name,
            architecture=architecture,
            conditioning=conditioning,
            update=update,
            method=method,
            seed=seed,
            mean_validation_accuracy=sum(r["adapted_accuracy"] for r in records)
            / len(records),
            records=records,
            selected_epoch=result.selected_epoch,
            optimizer_steps=result.optimizer_steps,
            adaptation_parameter_count=getattr(generator, "latent_dim", generator.head_dim),
            dense_delta_schema_entries=generator.head_dim,
            source_parameter_count=sum(p.numel() for p in model.parameters()),
            generator_parameter_count=sum(p.numel() for p in generator.parameters()),
            generator_fit_seconds=result.total_time_seconds,
            shared_teacher_collection_seconds=teacher_seconds,
            final_test_scored=False,
        )
        atomic_json(done, row)
        print("GENERATOR DONE", name, row["mean_validation_accuracy"], flush=True)
        return row

    def evaluate(self, architecture, generator, descriptor, scope, method, folder):
        records = []
        for env in self.validation:
            eid = env["id"]
            bundle = self.loader(self.manifest, eid)
            graph = bundle.unlabeled_input(eid)
            indices = bundle.select(eid, "eval")
            for mid, model in self.sources(architecture, "validation").items():
                source_hash = state_sha256(model.state_dict())
                baseline = frozen_predictions(model, graph, device=self.device)
                if method == "diffusion":
                    output, selection = agreement_predictions(
                        model,
                        graph,
                        descriptor,
                        generator,
                        indices=indices,
                        scope=scope,
                        seeds=self.plan["candidate_seeds"],
                        device=self.device,
                    )
                else:
                    output = generated_predictions(
                        model,
                        graph,
                        descriptor,
                        generator,
                        scope=scope,
                        device=self.device,
                    )
                    selection = None
                # Labels become accessible only after prediction/selection finishes.
                labels = bundle.evaluation_labels(eid, "eval")
                base = score_logits(baseline.logits, indices, labels).accuracy
                adapted = score_logits(output.logits, indices, labels).accuracy
                if state_sha256(model.state_dict()) != source_hash:
                    raise ValueError("Adaptation mutated source")
                factors = (
                    output.factor_vector.detach().cpu()
                    if output.factor_vector is not None
                    else None
                )
                if factors is not None:
                    torch.testing.assert_close(
                        generator.update_space.dense_delta(factors.to(self.device)),
                        output.delta,
                        rtol=0,
                        atol=0,
                    )
                artifact = folder / f"{mid}--{eid}.pt"
                torch.save(
                    dict(
                        source_id=mid,
                        source_state_sha256=source_hash,
                        environment_id=eid,
                        indices=indices,
                        delta=output.delta.detach().cpu(),
                        factor_vector=factors,
                        predicted_classes=output.logits[indices].argmax(-1).cpu(),
                        prediction_logits=output.logits[indices].detach().cpu(),
                        selection=selection,
                        accuracy=adapted,
                        baseline_accuracy=base,
                        scope=scope,
                    ),
                    artifact,
                )
                records.append(
                    dict(
                        model_id=mid,
                        environment_id=eid,
                        family=env["family"],
                        severity=env["severity"],
                        baseline_accuracy=base,
                        adapted_accuracy=adapted,
                        gain=adapted - base,
                        delta_norm=float(output.delta.norm()),
                        selection=selection,
                        artifact=artifact.name,
                    )
                )
        return records

    def baselines(self, architecture, selected):
        """Validation baselines, with objective and label access kept explicit."""
        destination = self.root / f"BASELINES-{architecture}.json"
        if destination.exists():
            return json.loads(destination.read_text())
        from amortized_gnn_tta.adaptation.graph_transform import (
            feature_gtrans_predictions,
        )
        from amortized_gnn_tta.adaptation.practice import fit_practice_model

        rows = []
        practice_cost = {}
        from amortized_gnn_tta.adaptation.benchmark import constant_delta_predictions

        example = next(iter(self.sources(architecture, "meta_train").values()))
        desc, gen, schema, scope = create_generator(
            example,
            selected["conditioning"],
            selected["update"],
            "hypernetwork",
            self.plan,
        )
        teachers, teacher_seconds = self.teachers(
            architecture, selected["conditioning"], selected["update"], desc, gen, scope
        )
        mean_coordinates = torch.stack([r.target_delta for r in teachers]).mean(0)
        constant_delta = (
            gen.update_space.dense_delta(mean_coordinates)
            if isinstance(gen, LoRAGenerator)
            else mean_coordinates
        )
        torch.save(
            dict(
                delta=constant_delta,
                schema=schema,
                scope=scope,
                teacher_seconds=teacher_seconds,
                definition="mean teacher factor coordinates for LoRA; mean direct deltas for head",
            ),
            self.root / f"constant-{architecture}.pt",
        )
        for mid, model in self.sources(architecture, "validation").items():
            torch.manual_seed(self.plan["generator_seeds"][0])
            robust, fit = fit_practice_model(
                model,
                self.manifest,
                self.loader,
                epochs=self.plan["generator_epochs"],
                lr=self.plan["generator_lr"],
                device=self.device,
            )
            practice_cost[mid] = fit
            torch.save(
                dict(
                    state_dict=robust.state_dict(),
                    initial_state_sha256=state_sha256(model.state_dict()),
                    fit=fit,
                ),
                self.root / f"practice-{mid}.pt",
            )
            for env in self.validation:
                eid = env["id"]
                bundle = self.loader(self.manifest, eid)
                graph = bundle.unlabeled_input(eid)
                indices = bundle.select(eid, "eval")
                methods = {
                    "frozen": lambda: frozen_predictions(
                        model, graph, device=self.device
                    ),
                    "constant_teacher": lambda: constant_delta_predictions(
                        model, graph, constant_delta, schema, device=self.device
                    ),
                    "practice_fixed": lambda: frozen_predictions(
                        robust, graph, device=self.device
                    ),
                    "gtrans_feature_only": lambda: feature_gtrans_predictions(
                        model,
                        graph,
                        indices=indices,
                        steps=self.plan["teacher_steps"],
                        device=self.device,
                    ),
                }
                for update in ("head", "lora"):
                    peft = (
                        lora_config(self.plan)
                        if update == "lora"
                        else PEFTConfig("head")
                    )
                    methods["iterative_im_" + update] = (
                        lambda peft=peft: iterative_predictions(
                            model,
                            graph,
                            indices=indices,
                            steps=self.plan["teacher_steps"],
                            lr=self.plan["teacher_lr"],
                            peft=peft,
                            device=self.device,
                        )
                    )
                for method, call in methods.items():
                    output = call()
                    accuracy = score_logits(
                        output.logits, indices, bundle.evaluation_labels(eid, "eval")
                    ).accuracy
                    rows.append(
                        dict(
                            model_id=mid,
                            environment_id=eid,
                            method=method,
                            accuracy=accuracy,
                            online_labels=False,
                            offline_practice_labels=method == "practice_fixed",
                        )
                    )
                    print("BASELINE", architecture, eid, method, accuracy, flush=True)
        result = dict(
            rows=rows, practice_offline_cost=practice_cost, final_test_scored=False
        )
        atomic_json(destination, result)
        return result

    def run_natural(self, design_path):
        design = json.loads(Path(design_path).read_text())
        rows = []
        for arch in self.plan["architectures"]:
            selected = design["selections"][arch]
            for seed in self.plan["generator_seeds"]:
                for method in ("hypernetwork", "teacher_hypernetwork", "diffusion"):
                    rows.append(
                        self.fit(
                            arch,
                            selected["conditioning"],
                            selected["update"],
                            method,
                            seed,
                        )
                    )
            self.baselines(arch, design["selections"][arch])
        atomic_json(
            self.root / "EXPERIMENT_5_VALIDATION.json",
            dict(
                rows=rows,
                source_design_sha256=file_hash(design_path),
                final_test_scored=False,
            ),
        )
        atomic_json(
            self.root / "FROZEN_DESIGN.json",
            dict(
                identity=self.identity,
                fitting_source_pins=dict(self.pins),
                selections=design["selections"],
                candidate_seeds=self.plan["candidate_seeds"],
                distance="probability MSE",
                validation_selection_only=True,
                final_test_scored=False,
            ),
        )

    def run(self):
        if (self.root / "FINAL").exists():
            raise ValueError("Cannot revise design after final evaluation started")
        seed = self.plan["generator_seeds"][0]
        # E1: predeclared single design seed; final selected fits use all three.
        e1 = [self.fit("gcn", c, "lora", "hypernetwork", seed) for c in "ABCD"]
        best = max(e1, key=lambda r: r["mean_validation_accuracy"])
        conditioning = best["conditioning"]
        atomic_json(
            self.root / "EXPERIMENT_1.json",
            dict(
                rows=e1,
                selected_inputs=conditioning,
                tie_break="A before B before C before D",
                design_seed=seed,
            ),
        )
        head = self.fit("gcn", conditioning, "head", "hypernetwork", seed)
        e2 = [head, best]
        selected = (
            "head"
            if head["mean_validation_accuracy"] >= best["mean_validation_accuracy"]
            else "lora"
        )
        if selected == "head":
            checks = [self.fit("gcn", c, "head", "hypernetwork", seed) for c in "AD"]
            # If head A/D ordering differs, evaluate B/C before selecting inputs.
            lora_ad = [next(r for r in e1 if r["conditioning"] == c) for c in "AD"]
            changed = (
                checks[0]["mean_validation_accuracy"]
                >= checks[1]["mean_validation_accuracy"]
            ) != (
                lora_ad[0]["mean_validation_accuracy"]
                >= lora_ad[1]["mean_validation_accuracy"]
            )
            if changed:
                checks += [
                    self.fit("gcn", c, "head", "hypernetwork", seed) for c in "BC"
                ]
            conditioning = max(
                [head] + checks, key=lambda r: r["mean_validation_accuracy"]
            )["conditioning"]
        # Input checks can revise the recipe; the definitive site comparison
        # must then use that same revised input on both update types.
        if conditioning != best["conditioning"]:
            e2 = [
                self.fit("gcn", conditioning, u, "hypernetwork", seed)
                for u in ("head", "lora")
            ]
            selected = max(e2, key=lambda r: r["mean_validation_accuracy"])["update"]
        atomic_json(
            self.root / "EXPERIMENT_2.json",
            dict(rows=e2, selected_update=selected, selected_inputs=conditioning),
        )
        if getattr(self.args, "stop_after", None) == 2:
            return
        selections = {}
        e3 = []
        for arch in self.plan["architectures"]:
            candidates = [
                self.fit(arch, conditioning, u, "hypernetwork", seed)
                for u in ("head", "lora")
            ]
            update = max(candidates, key=lambda r: r["mean_validation_accuracy"])[
                "update"
            ]
            selections[arch] = dict(conditioning=conditioning, update=update)
            for fit_seed in self.plan["generator_seeds"]:
                for method in ("hypernetwork", "teacher_hypernetwork", "diffusion"):
                    e3.append(self.fit(arch, conditioning, update, method, fit_seed))
        for arch in self.plan["architectures"]:
            self.baselines(arch, selections[arch])
        atomic_json(
            self.root / "EXPERIMENT_3.json",
            dict(rows=e3, selections=selections, final_test_scored=False),
        )
        atomic_json(
            self.root / "FROZEN_DESIGN.json",
            dict(
                identity=self.identity,
                fitting_source_pins=dict(self.pins),
                selections=selections,
                candidate_seeds=self.plan["candidate_seeds"],
                distance="probability MSE",
                tie_break="lowest candidate index",
                validation_selection_only=True,
                final_test_scored=False,
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--natural-design",
        type=Path,
        help="Frozen Arxiv design for separately trained natural-shift generators",
    )
    p.add_argument(
        "--stop-after",
        type=int,
        choices=[2],
        help="Finish GCN design while independent source-bank preparation continues",
    )
    args = p.parse_args()
    study = Study(args)
    if args.natural_design:
        study.run_natural(args.natural_design)
    else:
        study.run()
