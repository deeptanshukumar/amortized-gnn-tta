"""Comparative benchmark: Diffusion Generator with Head vs. Backbone Adaptation Scopes.

Evaluates Diffusion generators (oracle target distillation + DDIM sampling) across:
1. 'head': Classification head only (conv2.*, original baseline)
2. 'backbone': Feature encoder only (conv1.*, new backbone adaptation)

Across all three architectures (GCN, GAT, GraphSAGE) on the full unconstrained dataset:
- Meta-Training: 18 models x 69 environments = 1,242 pairs
- Meta-Validation: 6 models x 23 environments = 138 pairs
- Test: 6 models x 34 environments = 204 pairs per condition
"""

import json
from pathlib import Path
import time

import numpy as np
import torch

from amortized_gnn_tta.adaptation import (
    DiffusionHeadGenerator,
    GraphShiftDescriptor,
    adapted_forward,
    extract_target_parameters,
    train_diffusion_generator,
)
from amortized_gnn_tta.bank_artifacts import (
    artifact_path,
    create_model_partition,
    load_verified_bank,
)
from amortized_gnn_tta.data import (
    CSBMConfig,
    CSBMEnvironment,
    load_dataset,
    load_environment_manifest,
    load_shift_environment,
)
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint, source_training_data


def load_full_bank_and_partition(bank_dir: Path, manifest_path: Path, device: str = "cpu"):
    subsets_file = bank_dir / "source_subsets.json"
    subset_manifest = json.loads(subsets_file.read_text(encoding="utf-8"))
    source_cfg = dict(subset_manifest["source_provenance"]["graph_metadata"]["config"])
    source_cfg["split_ratios"] = tuple(source_cfg["split_ratios"])
    source_csbm = CSBMConfig(**source_cfg) if "p_in" in source_cfg else CSBMConfig.from_matcha(**source_cfg)
    bundle = load_dataset([CSBMEnvironment("source", "source_train", source_csbm)])
    source_data = source_training_data(bundle, "source")

    bank = load_verified_bank(bank_dir, source_data, build_model)
    partition = create_model_partition(bank, seed=401)
    manifest = load_environment_manifest(manifest_path)

    roles = ("meta_train", "validation", "test")
    models: dict[str, dict[str, torch.nn.Module]] = {role: {} for role in roles}
    for role in roles:
        for mid in partition["model_ids"].get(role, []):
            row = next(r for r in bank.manifest["models"] if r["model_id"] == mid)
            ckpt = artifact_path(bank.directory, row["checkpoint"])
            m, _ = load_checkpoint(ckpt, build_model, device=device)
            m.eval()
            models[role][mid] = m

    return bank, partition, manifest, models


def run_diffusion_scope_comparison():
    root = Path("/home/atharva/programs/ml/afmlproj/amortized-gnn-tta")
    manifest_path = root / "runs" / "matched-small-shifts" / "environment_manifest.json"
    device = "cpu"

    architectures = [
        ("gcn", root / "runs" / "bank-gcn"),
        ("gat", root / "runs" / "bank-gat"),
        ("graphsage", root / "runs" / "bank-graphsage"),
    ]

    scopes = ["head", "backbone"]
    results = {}

    print("=" * 80)
    print("DIFFUSION GENERATOR BENCHMARK: HEAD vs. BACKBONE ADAPTATION SCOPES")
    print("Scope: Full unconstrained dataset (18 train models x 69 envs = 1,242 pairs;")
    print("       6 test models x 34 envs = 204 pairs per condition)")
    print("=" * 80)

    for arch_name, bank_dir in architectures:
        print(f"\n{'#' * 80}")
        print(f"ARCHITECTURE: {arch_name.upper()}")
        print(f"{'#' * 80}")

        bank, partition, manifest, models = load_full_bank_and_partition(bank_dir, manifest_path, device=device)
        test_envs = [e for e in manifest["environments"] if e["role"] == "target_test"]
        test_bundles = {e["id"]: (e, load_shift_environment(manifest, e["id"])) for e in test_envs}

        sample_model = next(iter(models["meta_train"].values()))
        hidden_dim = sample_model.config.hidden_channels

        descriptor = GraphShiftDescriptor(hidden_channels=hidden_dim, include_degree=True, descriptor_type="moments")
        arch_results = {}

        for scope in scopes:
            print(f"\n--- Training Diffusion Generator with scope='{scope}' on {arch_name.upper()} ---")
            sample_flat, schema = extract_target_parameters(sample_model, scope=scope)
            target_dim = len(sample_flat)
            print(f"Target parameter count: {target_dim} ({len(schema)} tensors: {[s[0] for s in schema]})")

            gen = DiffusionHeadGenerator(
                descriptor_dim=descriptor.output_dim,
                head_dim=target_dim,
                hidden_dim=128,
                diffusion_steps=100,
                sampling_steps=10,
            )

            t0 = time.perf_counter()
            train_res = train_diffusion_generator(
                models=models["meta_train"],
                manifest=manifest,
                load_env_fn=load_shift_environment,
                descriptor=descriptor,
                generator=gen,
                epochs=5,
                lr=1e-3,
                oracle_steps=50,
                delta_l2_reg=1e-3,
                val_models=models["validation"],
                device=device,
                verbose=True,
                target_scope=scope,
            )
            train_time = time.perf_counter() - t0
            gen.load_state_dict(train_res.best_state_dict)
            gen.eval()

            print(f"\nEvaluating on {len(models['test'])} test models x 34 test environments (204 pairs) ...")
            records = []
            for model_id, model in models["test"].items():
                target_flat, test_schema = extract_target_parameters(model, scope=scope)
                for env_id, (env_meta, bundle) in test_bundles.items():
                    unlabeled = bundle.unlabeled_input(env_id)
                    split_name = "eval" if "eval" in bundle.graphs[env_id].splits else "test"
                    eval_idx = bundle.select(env_id, split_name)
                    eval_y = bundle.evaluation_labels(env_id, split_name)

                    with torch.no_grad():
                        base_logits = model(unlabeled.x, unlabeled.edge_index)
                        base_acc = float((base_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                        t_inf = time.perf_counter()
                        desc = descriptor(model, unlabeled)
                        delta = gen(desc, target_flat)
                        lat_ms = (time.perf_counter() - t_inf) * 1000.0

                        adapt_logits = adapted_forward(model, unlabeled, delta, test_schema)
                        adapt_acc = float((adapt_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                    gain = adapt_acc - base_acc
                    delta_norm = float(torch.norm(delta).item())

                    records.append({
                        "model_id": model_id,
                        "env_id": env_id,
                        "family": env_meta.get("family") or "control",
                        "shift": env_meta.get("shift") or "none",
                        "severity": env_meta.get("severity", 0.0),
                        "baseline_acc": base_acc,
                        "adapted_acc": adapt_acc,
                        "gain": gain,
                        "delta_norm": delta_norm,
                        "latency_ms": lat_ms,
                    })

            base_accs = [r["baseline_acc"] for r in records]
            adapt_accs = [r["adapted_acc"] for r in records]
            gains = [r["gain"] for r in records]
            norms = [r["delta_norm"] for r in records]
            lats = [r["latency_ms"] for r in records]

            # Breakdown by family
            family_breakdown = {}
            for fam in sorted(set(r["family"] for r in records)):
                fam_records = [r for r in records if r["family"] == fam]
                family_breakdown[fam] = {
                    "count": len(fam_records),
                    "baseline_acc": float(np.mean([r["baseline_acc"] for r in fam_records])),
                    "adapted_acc": float(np.mean([r["adapted_acc"] for r in fam_records])),
                    "gain": float(np.mean([r["gain"] for r in fam_records])),
                    "delta_norm": float(np.mean([r["delta_norm"] for r in fam_records])),
                }

            # Breakdown by shift
            shift_breakdown = {}
            for sh in sorted(set(r["shift"] for r in records), key=str):
                sh_records = [r for r in records if r["shift"] == sh]
                shift_breakdown[sh] = {
                    "count": len(sh_records),
                    "baseline_acc": float(np.mean([r["baseline_acc"] for r in sh_records])),
                    "adapted_acc": float(np.mean([r["adapted_acc"] for r in sh_records])),
                    "gain": float(np.mean([r["gain"] for r in sh_records])),
                    "delta_norm": float(np.mean([r["delta_norm"] for r in sh_records])),
                }

            arch_results[scope] = {
                "train_time_sec": train_time,
                "target_dim": target_dim,
                "total_pairs": len(records),
                "baseline_mean_acc": float(np.mean(base_accs)),
                "adapted_mean_acc": float(np.mean(adapt_accs)),
                "mean_gain": float(np.mean(gains)),
                "improved_pairs": sum(1 for g in gains if g > 1e-4),
                "degraded_pairs": sum(1 for g in gains if g < -1e-4),
                "unchanged_pairs": sum(1 for g in gains if abs(g) <= 1e-4),
                "mean_delta_norm": float(np.mean(norms)),
                "mean_latency_ms": float(np.mean(lats)),
                "family_breakdown": family_breakdown,
                "shift_breakdown": shift_breakdown,
            }

            print(f"Results for scope='{scope}' on {arch_name.upper()}:")
            print(f"  Baseline Acc:    {np.mean(base_accs)*100:.2f}%")
            print(f"  Adapted Acc:     {np.mean(adapt_accs)*100:.2f}%")
            print(f"  Mean Gain:       {np.mean(gains)*100:+.2f} pp")
            print(f"  Improved/Degr:   {arch_results[scope]['improved_pairs']}/{arch_results[scope]['degraded_pairs']}")
            print(f"  Mean Delta Norm: {np.mean(norms):.4f}")
            print(f"  Training Time:   {train_time:.1f}s")

        results[arch_name] = arch_results

    out_file = root / "scratch" / "diffusion_scope_comparison_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll Diffusion scope benchmark results saved to: {out_file}")


if __name__ == "__main__":
    run_diffusion_scope_comparison()
