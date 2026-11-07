"""Comprehensive comparative benchmark: Moments Descriptor vs. Structural Descriptor.

Evaluates whether the new structural descriptor (Dirichlet energy, edge distance dispersion,
cluster dispersion, degree statistics) improves generator performance over the baseline
moments descriptor across:
1. GCN: Hypernetwork (Moments vs. Structural) & Diffusion (Moments vs. Structural)
2. GAT: Hypernetwork (Moments vs. Structural)
3. GraphSAGE: Hypernetwork (Moments vs. Structural)

Evaluated on the full unconstrained dataset:
- Meta-Training: 18 models x 69 environments = 1,242 pairs
- Meta-Validation: 6 models x 23 environments = 138 pairs
- Test: 6 models x 34 environments = 204 pairs per architecture (612 pairs total)
"""

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from amortized_gnn_tta.adaptation import (
    DiffusionHeadGenerator,
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    extract_head_parameters,
    train_adaptation_generator,
    train_diffusion_generator,
)
from amortized_gnn_tta.adaptation.generator import adapted_forward
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


def run_benchmark():
    root = Path("/home/atharva/programs/ml/afmlproj/amortized-gnn-tta")
    manifest_path = root / "runs" / "matched-small-shifts" / "environment_manifest.json"
    device = "cpu"

    experiments = [
        ("gcn", root / "runs" / "bank-gcn", "hypernetwork"),
        ("gcn", root / "runs" / "bank-gcn", "diffusion"),
        ("gat", root / "runs" / "bank-gat", "hypernetwork"),
        ("graphsage", root / "runs" / "bank-graphsage", "hypernetwork"),
    ]

    benchmark_results = {}

    print("=" * 80)
    print("EMPIRICAL COMPARISON: MOMENTS vs. STRUCTURAL DESCRIPTOR")
    print("Scope: Full unconstrained dataset (18 train models x 69 envs = 1,242 pairs;")
    print("       6 test models x 34 envs = 204 pairs per condition)")
    print("=" * 80)

    for arch_name, bank_dir, gen_type in experiments:
        exp_key = f"{arch_name}_{gen_type}"
        print(f"\n{'#' * 80}")
        print(f"EXPERIMENT: Architecture={arch_name.upper()}, Generator={gen_type.upper()}")
        print(f"{'#' * 80}")

        bank, partition, manifest, models = load_full_bank_and_partition(bank_dir, manifest_path, device=device)

        sample_model = next(iter(models["meta_train"].values()))
        sample_flat, schema = extract_head_parameters(sample_model)
        head_dim = len(sample_flat)
        hidden_dim = sample_model.config.hidden_channels

        # We will test both descriptor types on the exact same setup
        descriptor_types = ["moments", "structural"]
        exp_results = {}

        for desc_type in descriptor_types:
            print(f"\n--- Training {gen_type.upper()} with descriptor='{desc_type}' ---")
            descriptor = GraphShiftDescriptor(
                hidden_channels=hidden_dim,
                include_degree=True,
                descriptor_type=desc_type,
            )
            print(f"Descriptor dim: {descriptor.output_dim}, Head dim: {head_dim}")

            t_start = time.perf_counter()
            if gen_type == "diffusion":
                gen = DiffusionHeadGenerator(
                    descriptor_dim=descriptor.output_dim,
                    head_dim=head_dim,
                    hidden_dim=128,
                    diffusion_steps=100,
                    sampling_steps=10,
                )
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
                )
            else:
                gen = HeadAdapterGenerator(
                    descriptor_dim=descriptor.output_dim,
                    head_dim=head_dim,
                    hidden_dim=128,
                )
                train_res = train_adaptation_generator(
                    models=models["meta_train"],
                    manifest=manifest,
                    load_env_fn=load_shift_environment,
                    descriptor=descriptor,
                    generator=gen,
                    epochs=5,
                    lr=1e-3,
                    delta_l2_reg=1e-3,
                    val_models=models["validation"],
                    device=device,
                    verbose=True,
                )
            train_time = time.perf_counter() - t_start
            gen.load_state_dict(train_res.best_state_dict)
            gen.eval()

            # Evaluate on held-out test environments
            print(f"\nEvaluating on {len(models['test'])} test models x 34 test environments (204 pairs) ...")
            test_envs = [e for e in manifest["environments"] if e["role"] == "target_test"]
            test_bundles = {e["id"]: (e, load_shift_environment(manifest, e["id"])) for e in test_envs}
            records = []

            for model_id, model in models["test"].items():
                for env_id, (env_meta, bundle) in test_bundles.items():
                    unlabeled = bundle.unlabeled_input(env_id)
                    split_name = "eval" if "eval" in bundle.graphs[env_id].splits else "test"
                    eval_idx = bundle.select(env_id, split_name)
                    eval_y = bundle.evaluation_labels(env_id, split_name)

                    with torch.no_grad():
                        base_logits = model(unlabeled.x, unlabeled.edge_index)
                        base_acc = float((base_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                        t0 = time.perf_counter()
                        desc = descriptor(model, unlabeled)
                        flat_w, _ = extract_head_parameters(model)
                        delta = gen(desc, flat_w)
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                        adapt_logits = adapted_forward(model, unlabeled, delta, schema)
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
                        "latency_ms": latency_ms,
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

            exp_results[desc_type] = {
                "train_time_sec": train_time,
                "descriptor_dim": descriptor.output_dim,
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

            print(f"Results for {desc_type}:")
            print(f"  Baseline Acc:    {np.mean(base_accs)*100:.2f}%")
            print(f"  Adapted Acc:     {np.mean(adapt_accs)*100:.2f}%")
            print(f"  Mean Gain:       {np.mean(gains)*100:+.2f} pp")
            print(f"  Improved/Degr:   {exp_results[desc_type]['improved_pairs']}/{exp_results[desc_type]['degraded_pairs']}")
            print(f"  Mean Delta Norm: {np.mean(norms):.4f}")
            print(f"  Training Time:   {train_time:.1f}s")

        benchmark_results[exp_key] = exp_results

    out_json = root / "scratch" / "descriptor_comparison_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    print(f"\nAll benchmark results saved to: {out_json}")


if __name__ == "__main__":
    run_benchmark()
