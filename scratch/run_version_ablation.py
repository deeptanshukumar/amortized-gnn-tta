"""Conditioning Version Ablation Benchmark on Full Unconstrained Dataset.

Evaluates the 4 Conditioning Versions:
- Version A: Target graph only (summaries of features and connections)
- Version B: Target graph + hidden representation (A + latent moments/Dirichlet)
- Version C: Target graph + hidden + predictive uncertainty (B + entropy/confidence/margin/prior)
- Version D: Target graph + hidden + predictive uncertainty + compact pretrained weights (C + SVD/Frobenius)

Across:
1. Iterative Fine-Tuning (Test-Time Optimization anchor)
2. Hypernetwork Generator
3. Diffusion Generator (DDIM 10 steps)

On the full unconstrained CSBM distribution-shift benchmark (34 test environments x 6 models = 204 pairs).
"""

from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from amortized_gnn_tta.adaptation import (
    AdaptationConfig,
    DiffusionHeadGenerator,
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    PEFTConfig,
    adapted_forward,
    extract_target_parameters,
    fit_unlabeled,
    train_adaptation_generator,
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


def run_version_ablation():
    root = Path("/home/atharva/programs/ml/afmlproj/amortized-gnn-tta")
    manifest_path = root / "runs" / "matched-small-shifts" / "environment_manifest.json"
    device = "cpu"

    architectures = [
        ("gcn", root / "runs" / "bank-gcn"),
        ("gat", root / "runs" / "bank-gat"),
        ("graphsage", root / "runs" / "bank-graphsage"),
    ]

    versions = ["A", "B", "C", "D"]
    all_results = {}

    print("=" * 80)
    print("CONDITIONING VERSION ABLATION BENCHMARK (A vs B vs C vs D)")
    print("=" * 80)
    print("Full unconstrained dataset: 18 meta-train models x 69 envs = 1,242 train pairs")
    print("Test evaluation: 6 models x 34 environments = 204 test pairs per architecture")
    print(f"Conditioning versions: {versions}\n")

    for arch_name, bank_dir in architectures:
        print(f"\n{'#' * 80}")
        print(f"EVALUATING ARCHITECTURE: {arch_name.upper()}")
        print(f"{'#' * 80}")

        bank, partition, manifest, models = load_full_bank_and_partition(bank_dir, manifest_path, device=device)
        meta_train_models = models["meta_train"]
        val_models = models["validation"]
        test_models = models["test"]

        sample_model = next(iter(meta_train_models.values()))
        flat_w, schema = extract_target_parameters(sample_model, scope="head")
        target_dim = len(flat_w)
        in_channels = sample_model.config.in_channels
        hidden_channels = sample_model.config.hidden_channels
        out_channels = sample_model.config.out_channels

        test_envs = [e for e in manifest["environments"] if e["role"] == "target_test"]
        print(f"Loaded {arch_name.upper()}: {len(meta_train_models)} train models, {len(test_models)} test models, {len(test_envs)} test envs")

        arch_results = {}

        # ----------------------------------------------------------------------
        # 1. First, run Regular Fine-Tuning reference (Test-Time Optimization)
        # ----------------------------------------------------------------------
        print(f"\n[1/3] Running Reference Test-Time Fine-Tuning (30 steps IM) on 204 pairs ...")
        ft_gains, ft_accs, ft_lats, ft_norms, base_accs = [], [], [], [], []
        peft_cfg = PEFTConfig("head")
        adapt_cfg = AdaptationConfig(steps=30, learning_rate=0.01)

        # Pre-load graphs for test environments
        env_graphs = {}
        for env_meta in test_envs:
            env_id = env_meta["id"]
            bundle = load_shift_environment(manifest, env_id)
            unlabeled = bundle.unlabeled_input(env_id).to(device)
            split_name = "eval" if "eval" in bundle.graphs[env_id].splits else "test"
            eval_idx = bundle.select(env_id, split_name).to(device)
            eval_y = bundle.evaluation_labels(env_id, split_name).to(device)
            env_graphs[env_id] = (unlabeled, eval_idx, eval_y, env_meta)

        for env_id, (unlabeled, eval_idx, eval_y, env_meta) in env_graphs.items():
            for model_id, model in test_models.items():
                model.eval()
                with torch.no_grad():
                    b_logits = model(unlabeled.x, unlabeled.edge_index)
                    b_acc = float((b_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())
                base_accs.append(b_acc)

                t0 = time.perf_counter()
                ft_res = fit_unlabeled(
                    model=model,
                    graph=unlabeled,
                    peft=peft_cfg,
                    config=adapt_cfg,
                    objective="information_maximization",
                    device=device,
                )
                t_ft = (time.perf_counter() - t0) * 1000.0
                with torch.no_grad():
                    ft_logits = ft_res.model(unlabeled.x, unlabeled.edge_index)
                    ft_acc = float((ft_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())
                ft_norm = float(torch.sqrt(sum(torch.sum(d**2) for d in ft_res.adapter.updates().values())).item())

                ft_accs.append(ft_acc)
                ft_gains.append(ft_acc - b_acc)
                ft_lats.append(t_ft)
                ft_norms.append(ft_norm)

        baseline_mean = float(np.mean(base_accs))
        ft_mean_lat = float(np.mean(ft_lats))
        arch_results["baseline"] = {"mean_acc": baseline_mean}
        arch_results["fine_tuning"] = {
            "mean_acc": float(np.mean(ft_accs)),
            "mean_gain": float(np.mean(ft_gains)),
            "mean_delta_norm": float(np.mean(ft_norms)),
            "mean_latency_ms": ft_mean_lat,
            "win_rate": float(sum(1 for g in ft_gains if g > 1e-4) / len(ft_gains)),
        }
        print(f"      Fine-Tuning: Base {baseline_mean*100:.2f}% -> Adapted {np.mean(ft_accs)*100:.2f}% (Gain: {np.mean(ft_gains)*100:+.2f} pp) | Latency: {ft_mean_lat:.2f} ms")

        # ----------------------------------------------------------------------
        # 2. Evaluate Versions A, B, C, D for Hypernetwork
        # ----------------------------------------------------------------------
        print(f"\n[2/3] Evaluating Hypernetworks across Conditioning Versions A, B, C, D ...")
        hyp_version_results = {}

        for ver in versions:
            print(f"   --- Version {ver} (Hypernetwork) ---")
            desc_mod = GraphShiftDescriptor(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                conditioning=ver,
            )
            hyp_gen = HeadAdapterGenerator(
                descriptor_dim=desc_mod.output_dim,
                head_dim=target_dim,
                hidden_dim=128,
                include_head_weights=False,  # Condition vector contains all required information
            )

            # Train generator
            t_train_start = time.time()
            hyp_res = train_adaptation_generator(
                models=meta_train_models,
                manifest=manifest,
                load_env_fn=load_shift_environment,
                descriptor=desc_mod,
                generator=hyp_gen,
                epochs=5,
                lr=1e-3,
                val_models=val_models,
                device=device,
                verbose=False,
                target_scope="head",
            )
            hyp_gen.load_state_dict(hyp_res.best_state_dict)
            hyp_gen.eval()
            train_time = time.time() - t_train_start

            # Evaluate on 204 test pairs
            h_accs, h_gains, h_norms, desc_lats, gen_lats = [], [], [], [], []
            for env_id, (unlabeled, eval_idx, eval_y, env_meta) in env_graphs.items():
                for model_id, model in test_models.items():
                    model.eval()
                    with torch.no_grad():
                        b_logits = model(unlabeled.x, unlabeled.edge_index)
                        b_acc = float((b_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                    # Time descriptor
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        desc_vec = desc_mod(model, unlabeled)
                    t_desc = (time.perf_counter() - t0) * 1000.0

                    # Time generator forward
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        delta = hyp_gen(desc_vec)
                    t_gen = (time.perf_counter() - t0) * 1000.0

                    # Forward adapted model
                    with torch.no_grad():
                        ad_logits = adapted_forward(model, unlabeled, delta, schema)
                        ad_acc = float((ad_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                    gain = ad_acc - b_acc
                    h_accs.append(ad_acc)
                    h_gains.append(gain)
                    h_norms.append(float(torch.norm(delta).item()))
                    desc_lats.append(t_desc)
                    gen_lats.append(t_gen)

            total_lat = float(np.mean(desc_lats)) + float(np.mean(gen_lats))
            speedup = ft_mean_lat / max(total_lat, 1e-4)

            hyp_version_results[ver] = {
                "conditioning": ver,
                "descriptor_dim": desc_mod.output_dim,
                "train_time_sec": train_time,
                "mean_acc": float(np.mean(h_accs)),
                "mean_gain": float(np.mean(h_gains)),
                "win_rate": float(sum(1 for g in h_gains if g > 1e-4) / len(h_gains)),
                "mean_delta_norm": float(np.mean(h_norms)),
                "desc_latency_ms": float(np.mean(desc_lats)),
                "gen_latency_ms": float(np.mean(gen_lats)),
                "total_latency_ms": total_lat,
                "speedup_vs_ft": speedup,
            }
            print(f"      Version {ver} (Dim {desc_mod.output_dim}): Acc {np.mean(h_accs)*100:.2f}% (Gain: {np.mean(h_gains)*100:+.2f} pp) | TTA Lat: {total_lat:.2f} ms | Speedup: {speedup:.1f}x")

        arch_results["hypernetwork"] = hyp_version_results

        # ----------------------------------------------------------------------
        # 3. Evaluate Versions A, B, C, D for Diffusion
        # ----------------------------------------------------------------------
        print(f"\n[3/3] Evaluating Diffusion Generators across Conditioning Versions A, B, C, D ...")
        diff_version_results = {}

        for ver in versions:
            print(f"   --- Version {ver} (Diffusion) ---")
            desc_mod = GraphShiftDescriptor(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                conditioning=ver,
            )
            diff_gen = DiffusionHeadGenerator(
                descriptor_dim=desc_mod.output_dim,
                head_dim=target_dim,
                hidden_dim=128,
                diffusion_steps=50,
                sampling_steps=10,
                include_head_weights=False,  # Condition vector contains all required information
            )

            # Train generator
            t_train_start = time.time()
            diff_res = train_diffusion_generator(
                models=meta_train_models,
                manifest=manifest,
                load_env_fn=load_shift_environment,
                descriptor=desc_mod,
                generator=diff_gen,
                epochs=5,
                lr=1e-3,
                oracle_steps=30,
                val_models=val_models,
                device=device,
                verbose=False,
                target_scope="head",
            )
            diff_gen.load_state_dict(diff_res.best_state_dict)
            diff_gen.eval()
            train_time = time.time() - t_train_start

            # Evaluate on 204 test pairs
            d_accs, d_gains, d_norms, desc_lats, gen_lats = [], [], [], [], []
            for env_id, (unlabeled, eval_idx, eval_y, env_meta) in env_graphs.items():
                for model_id, model in test_models.items():
                    model.eval()
                    with torch.no_grad():
                        b_logits = model(unlabeled.x, unlabeled.edge_index)
                        b_acc = float((b_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                    # Time descriptor
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        desc_vec = desc_mod(model, unlabeled)
                    t_desc = (time.perf_counter() - t0) * 1000.0

                    # Time diffusion DDIM sampling
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        delta = diff_gen(desc_vec)
                    t_gen = (time.perf_counter() - t0) * 1000.0

                    # Forward adapted model
                    with torch.no_grad():
                        ad_logits = adapted_forward(model, unlabeled, delta, schema)
                        ad_acc = float((ad_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                    gain = ad_acc - b_acc
                    d_accs.append(ad_acc)
                    d_gains.append(gain)
                    d_norms.append(float(torch.norm(delta).item()))
                    desc_lats.append(t_desc)
                    gen_lats.append(t_gen)

            total_lat = float(np.mean(desc_lats)) + float(np.mean(gen_lats))
            speedup = ft_mean_lat / max(total_lat, 1e-4)

            diff_version_results[ver] = {
                "conditioning": ver,
                "descriptor_dim": desc_mod.output_dim,
                "train_time_sec": train_time,
                "mean_acc": float(np.mean(d_accs)),
                "mean_gain": float(np.mean(d_gains)),
                "win_rate": float(sum(1 for g in d_gains if g > 1e-4) / len(d_gains)),
                "mean_delta_norm": float(np.mean(d_norms)),
                "desc_latency_ms": float(np.mean(desc_lats)),
                "gen_latency_ms": float(np.mean(gen_lats)),
                "total_latency_ms": total_lat,
                "speedup_vs_ft": speedup,
            }
            print(f"      Version {ver} (Dim {desc_mod.output_dim}): Acc {np.mean(d_accs)*100:.2f}% (Gain: {np.mean(d_gains)*100:+.2f} pp) | TTA Lat: {total_lat:.2f} ms | Speedup: {speedup:.1f}x")

        arch_results["diffusion"] = diff_version_results
        all_results[arch_name] = arch_results

    out_file = root / "scratch" / "version_ablation_results.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll version ablation benchmark results saved to: {out_file}")


if __name__ == "__main__":
    run_version_ablation()
