"""Full unconstrained comparative benchmark: Hypernetwork vs. Diffusion vs. Regular Fine-Tuning.

Compares:
1. Hypernetwork Generator (amortized single-pass weight generation)
2. Diffusion Generator (amortized DDIM generative weight sampling)
3. Regular Fine-Tuning (online test-time optimization via fit_unlabeled with information maximization)

Across all three architectures (GCN, GAT, GraphSAGE) on the full unconstrained test set.
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
    AdaptationConfig,
    DiffusionHeadGenerator,
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    PEFTConfig,
    extract_head_parameters,
    fit_unlabeled,
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


def run_full_comparison():
    root = Path("/home/atharva/programs/ml/afmlproj/amortized-gnn-tta")
    manifest_path = root / "runs" / "matched-small-shifts" / "environment_manifest.json"
    device = "cpu"

    architectures = [
        ("gcn", root / "runs" / "bank-gcn"),
        ("gat", root / "runs" / "bank-gat"),
        ("graphsage", root / "runs" / "bank-graphsage"),
    ]

    all_results = {}

    print("=" * 80)
    print("FULL UNCONSTRAINED BENCHMARK: HYPERNETWORK vs. DIFFUSION vs. REGULAR FINE-TUNING")
    print("=" * 80)
    print("Scope: Full 30-model banks (18 meta-train, 6 validation, 6 test)")
    print("       Full environment manifest (69 meta-train, 23 validation, 34 target-test)")
    print("       Architectures: GCN, GAT, GraphSAGE\n")

    for arch_name, bank_dir in architectures:
        print(f"\n{'#' * 80}")
        print(f"STARTING EVALUATION FOR ARCHITECTURE: {arch_name.upper()}")
        print(f"{'#' * 80}")

        bank, partition, manifest, models = load_full_bank_and_partition(bank_dir, manifest_path, device=device)
        meta_train_models = models["meta_train"]
        val_models = models["validation"]
        test_models = models["test"]

        sample_model = next(iter(meta_train_models.values()))
        head_flat, schema = extract_head_parameters(sample_model)
        hidden_dim = sample_model.config.hidden_channels
        head_dim = len(head_flat)
        descriptor = GraphShiftDescriptor(hidden_channels=hidden_dim, include_degree=True)

        print(f"Loaded {arch_name.upper()}:")
        print(f"  • Meta-Train Models: {len(meta_train_models)}")
        print(f"  • Validation Models: {len(val_models)}")
        print(f"  • Test Models:       {len(test_models)}")
        print(f"  • Head Parameters:   {head_dim} ({[s[0] for s in schema]})")

        # ----------------------------------------------------------------------
        # 1. Train Hypernetwork Generator
        # ----------------------------------------------------------------------
        print(f"\n[1/4] Training Hypernetwork Generator ({arch_name.upper()}) on all meta-train pairs ...")
        hyp_gen = HeadAdapterGenerator(
            descriptor_dim=descriptor.output_dim,
            head_dim=head_dim,
            hidden_dim=128,
        )
        t_hyp_start = time.time()
        hyp_res = train_adaptation_generator(
            models=meta_train_models,
            manifest=manifest,
            load_env_fn=load_shift_environment,
            descriptor=descriptor,
            generator=hyp_gen,
            epochs=5,
            lr=1e-3,
            val_models=val_models,
            device=device,
            verbose=False,
        )
        hyp_gen.load_state_dict(hyp_res.best_state_dict)
        hyp_train_time = time.time() - t_hyp_start
        print(f"      Hypernetwork trained in {hyp_train_time:.2f}s.")

        # ----------------------------------------------------------------------
        # 2. Train Diffusion Generator
        # ----------------------------------------------------------------------
        print(f"\n[2/4] Training Diffusion Generator ({arch_name.upper()}) on all meta-train pairs ...")
        diff_gen = DiffusionHeadGenerator(
            descriptor_dim=descriptor.output_dim,
            head_dim=head_dim,
            hidden_dim=128,
            diffusion_steps=50,
            sampling_steps=10,
        )
        t_diff_start = time.time()
        diff_res = train_diffusion_generator(
            models=meta_train_models,
            manifest=manifest,
            load_env_fn=load_shift_environment,
            descriptor=descriptor,
            generator=diff_gen,
            epochs=5,
            lr=1e-3,
            oracle_steps=30,
            val_models=val_models,
            device=device,
            verbose=False,
        )
        diff_gen.load_state_dict(diff_res.best_state_dict)
        diff_train_time = time.time() - t_diff_start
        print(f"      Diffusion Generator trained in {diff_train_time:.2f}s.")

        # ----------------------------------------------------------------------
        # 3. Comprehensive Evaluation on ALL 34 Target Test Environments (204 Pairs)
        # ----------------------------------------------------------------------
        test_envs = [e for e in manifest["environments"] if e["role"] == "target_test"]
        print(f"\n[3/4] Evaluating Test-Time Adaptation across {len(test_models)} Models × {len(test_envs)} Environments = {len(test_models)*len(test_envs)} Pairs ...")

        arch_records = []
        pair_count = 0
        total_pairs = len(test_models) * len(test_envs)

        for env_meta in test_envs:
            env_id = env_meta["id"]
            bundle = load_shift_environment(manifest, env_id)
            unlabeled = bundle.unlabeled_input(env_id).to(device)
            split_name = "eval" if "eval" in bundle.graphs[env_id].splits else "test"
            eval_idx = bundle.select(env_id, split_name).to(device)
            eval_y = bundle.evaluation_labels(env_id, split_name).to(device)

            for model_id, model in test_models.items():
                pair_count += 1
                model.eval()

                # Baseline (unadapted frozen model)
                with torch.no_grad():
                    base_logits = model(unlabeled.x, unlabeled.edge_index)
                    base_acc = float((base_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())

                # A. Hypernetwork TTA
                t0 = time.perf_counter()
                with torch.no_grad():
                    desc = descriptor(model, unlabeled)
                    flat_w, _ = extract_head_parameters(model)
                    hyp_delta = hyp_gen(desc, flat_w)
                t_hyp_gen = (time.perf_counter() - t0) * 1000.0  # ms
                with torch.no_grad():
                    hyp_logits = adapted_forward(model, unlabeled, hyp_delta, schema)
                    hyp_acc = float((hyp_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())
                hyp_gain = hyp_acc - base_acc
                hyp_norm = float(torch.norm(hyp_delta).item())

                # B. Diffusion TTA (DDIM 10 steps)
                t0 = time.perf_counter()
                with torch.no_grad():
                    diff_delta = diff_gen(desc, flat_w)
                t_diff_gen = (time.perf_counter() - t0) * 1000.0  # ms
                with torch.no_grad():
                    diff_logits = adapted_forward(model, unlabeled, diff_delta, schema)
                    diff_acc = float((diff_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())
                diff_gain = diff_acc - base_acc
                diff_norm = float(torch.norm(diff_delta).item())

                # C. Regular Fine-Tuning TTA (fit_unlabeled with information maximization)
                peft_cfg = PEFTConfig("head")
                adapt_cfg = AdaptationConfig(steps=30, learning_rate=0.01)
                t0 = time.perf_counter()
                ft_res = fit_unlabeled(
                    model=model,
                    graph=unlabeled,
                    peft=peft_cfg,
                    config=adapt_cfg,
                    objective="information_maximization",
                    device=device,
                )
                t_ft_adapt = (time.perf_counter() - t0) * 1000.0  # ms
                with torch.no_grad():
                    ft_logits = ft_res.model(unlabeled.x, unlabeled.edge_index)
                    ft_acc = float((ft_logits.argmax(-1)[eval_idx] == eval_y).float().mean().item())
                ft_gain = ft_acc - base_acc
                ft_norm = float(torch.sqrt(sum(torch.sum(d**2) for d in ft_res.adapter.updates().values())).item())

                record = {
                    "model_id": model_id,
                    "env_id": env_id,
                    "family": env_meta["family"],
                    "severity": env_meta["severity"],
                    "baseline_acc": base_acc,
                    "hypernetwork": {
                        "adapted_acc": hyp_acc,
                        "gain": hyp_gain,
                        "delta_norm": hyp_norm,
                        "latency_ms": t_hyp_gen,
                    },
                    "diffusion": {
                        "adapted_acc": diff_acc,
                        "gain": diff_gain,
                        "delta_norm": diff_norm,
                        "latency_ms": t_diff_gen,
                    },
                    "fine_tuning": {
                        "adapted_acc": ft_acc,
                        "gain": ft_gain,
                        "delta_norm": ft_norm,
                        "latency_ms": t_ft_adapt,
                    },
                }
                arch_records.append(record)

                if pair_count % 34 == 0 or pair_count == total_pairs:
                    print(f"      Progress: {pair_count}/{total_pairs} pairs evaluated ({pair_count/total_pairs*100:.0f}%) ...")

        # ----------------------------------------------------------------------
        # 4. Compute Aggregate Statistics for Architecture
        # ----------------------------------------------------------------------
        print(f"\n[4/4] Computing Summary for {arch_name.upper()} ({len(arch_records)} total pairs) ...")
        mean_base = float(np.mean([r["baseline_acc"] for r in arch_records]))

        hyp_accs = [r["hypernetwork"]["adapted_acc"] for r in arch_records]
        hyp_gains = [r["hypernetwork"]["gain"] for r in arch_records]
        hyp_norms = [r["hypernetwork"]["delta_norm"] for r in arch_records]
        hyp_lats = [r["hypernetwork"]["latency_ms"] for r in arch_records]

        diff_accs = [r["diffusion"]["adapted_acc"] for r in arch_records]
        diff_gains = [r["diffusion"]["gain"] for r in arch_records]
        diff_norms = [r["diffusion"]["delta_norm"] for r in arch_records]
        diff_lats = [r["diffusion"]["latency_ms"] for r in arch_records]

        ft_accs = [r["fine_tuning"]["adapted_acc"] for r in arch_records]
        ft_gains = [r["fine_tuning"]["gain"] for r in arch_records]
        ft_norms = [r["fine_tuning"]["delta_norm"] for r in arch_records]
        ft_lats = [r["fine_tuning"]["latency_ms"] for r in arch_records]

        summary = {
            "num_pairs": len(arch_records),
            "baseline_mean_acc": mean_base,
            "hypernetwork": {
                "mean_acc": float(np.mean(hyp_accs)),
                "mean_gain": float(np.mean(hyp_gains)),
                "mean_delta_norm": float(np.mean(hyp_norms)),
                "mean_latency_ms": float(np.mean(hyp_lats)),
                "train_time_sec": hyp_train_time,
            },
            "diffusion": {
                "mean_acc": float(np.mean(diff_accs)),
                "mean_gain": float(np.mean(diff_gains)),
                "mean_delta_norm": float(np.mean(diff_norms)),
                "mean_latency_ms": float(np.mean(diff_lats)),
                "train_time_sec": diff_train_time,
            },
            "fine_tuning": {
                "mean_acc": float(np.mean(ft_accs)),
                "mean_gain": float(np.mean(ft_gains)),
                "mean_delta_norm": float(np.mean(ft_norms)),
                "mean_latency_ms": float(np.mean(ft_lats)),
            },
        }

        # Breakdown by family
        family_breakdown = {}
        families = sorted(list(set(r["family"] for r in arch_records)))
        for f in families:
            f_recs = [r for r in arch_records if r["family"] == f]
            family_breakdown[f] = {
                "count": len(f_recs),
                "baseline_acc": float(np.mean([r["baseline_acc"] for r in f_recs])),
                "hypernetwork_gain": float(np.mean([r["hypernetwork"]["gain"] for r in f_recs])),
                "diffusion_gain": float(np.mean([r["diffusion"]["gain"] for r in f_recs])),
                "fine_tuning_gain": float(np.mean([r["fine_tuning"]["gain"] for r in f_recs])),
            }
        summary["family_breakdown"] = family_breakdown

        all_results[arch_name] = {
            "summary": summary,
            "records": arch_records,
        }

        print(f"\nSummary for {arch_name.upper()} across {len(arch_records)} pairs:")
        print(f"  • Baseline Mean Acc:         {mean_base*100:.2f}%")
        print(f"  • Hypernetwork:  Acc {np.mean(hyp_accs)*100:.2f}% | Gain {np.mean(hyp_gains)*100:+.2f}% | ||Δ|| {np.mean(hyp_norms):.4f} | TTA Latency: {np.mean(hyp_lats):.2f} ms")
        print(f"  • Diffusion:     Acc {np.mean(diff_accs)*100:.2f}% | Gain {np.mean(diff_gains)*100:+.2f}% | ||Δ|| {np.mean(diff_norms):.4f} | TTA Latency: {np.mean(diff_lats):.2f} ms")
        print(f"  • Fine-Tuning:   Acc {np.mean(ft_accs)*100:.2f}% | Gain {np.mean(ft_gains)*100:+.2f}% | ||Δ|| {np.mean(ft_norms):.4f} | TTA Latency: {np.mean(ft_lats):.2f} ms")

    out_file = Path("scratch/full_comparison_reproduced.json")
    out_file.write_text(json.dumps(all_results, indent=2))
    brain_out = Path("/home/atharva/.gemini/antigravity-cli/brain/f34f64af-a0f4-4247-a591-6c1468e3f6a1/scratch/full_comparison_reproduced.json")
    brain_out.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'=' * 80}")
    print(f"FULL BENCHMARK COMPLETE! Results written to: {out_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    run_full_comparison()
