"""Train and evaluate the Amortized Weight Adaptation Generator on GNNs under distribution shift.

Example usage:
    uv run python examples/train_adapter.py \\
        --bank runs/phase1-bank \\
        --manifest runs/controlled-shifts-small/environment_manifest.json \\
        --output-dir runs/adapter-small \\
        --epochs 5 \\
        --max-environments 10 \\
        --max-models 3

This example runs a fast test run with user-controlled limits, evaluates baseline vs
adapted models, and outputs a formatted Markdown report and checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from amortized_gnn_tta.adaptation import (
    DiffusionHeadGenerator,
    GraphShiftDescriptor,
    HeadAdapterGenerator,
    evaluate_adaptation,
    extract_head_parameters,
    extract_target_parameters,
    train_adaptation_generator,
    train_diffusion_generator,
)
from amortized_gnn_tta.bank_artifacts import (
    artifact_path,
    create_model_partition,
    load_model_partition,
    load_verified_bank,
)
from amortized_gnn_tta.data import (
    CSBMConfig,
    CSBMEnvironment,
    load_dataset,
    load_environment_manifest,
    load_shift_environment,
)
from amortized_gnn_tta.models import ARCHITECTURES, build_model, make_model_spec
from amortized_gnn_tta.training import load_checkpoint, source_training_data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]

    parser.add_argument("--bank", type=Path, default=root / "runs" / "phase1-bank",
                        help="Directory containing the completed 30-model source weight bank")
    parser.add_argument("--manifest", type=Path, default=root / "runs" / "controlled-shifts-small" / "environment_manifest.json",
                        help="Path to environment manifest JSON (from prepare_shift_environments.py)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save the trained generator and report (optional; if omitted, results are dumped to terminal only)")
    parser.add_argument("--conditioning", type=str, default="D", choices=["A", "B", "C", "D"],
                        help="Conditioning version: 'A' (target graph only), 'B' (+hidden rep), 'C' (+predictive uncertainty), 'D' (+compact weights)")
    parser.add_argument("--generator", type=str, default="hypernetwork", choices=["hypernetwork", "diffusion"],
                        help="Generator architecture: 'hypernetwork' (direct MLP) or 'diffusion' (conditional DDPM/DDIM)")
    parser.add_argument("--scope", type=str, default="head", choices=["head", "backbone", "all"],
                        help="Parameter adaptation scope: 'head' (conv2 only, default), 'backbone' (conv1 only), or 'all' (conv1 + conv2)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of meta-training sweeps over (model, graph) pairs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate for the generator")
    parser.add_argument("--delta-l2", type=float, default=1e-3,
                        help="L2 regularization penalty on the predicted weight delta")
    parser.add_argument("--max-environments", type=int, default=None,
                        help="Limit the number of environments per role (for quick runs)")
    parser.add_argument("--max-models", type=int, default=None,
                        help="Limit the number of models per role (for quick runs)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run adaptation on ('cpu' or 'cuda')")
    parser.add_argument("--diffusion-steps", type=int, default=50,
                        help="Number of diffusion timesteps T (diffusion generator only)")
    parser.add_argument("--sampling-steps", type=int, default=10,
                        help="Number of DDIM sampling steps during inference (diffusion generator only)")
    parser.add_argument("--oracle-steps", type=int, default=30,
                        help="Steps to fit teacher oracle deltas (diffusion generator only)")
    return parser.parse_args()


def load_partitioned_models(bank, partition, model_factory, device="cpu"):
    """Load model instances grouped by their partition role (meta_train, validation, test)."""
    roles = ("meta_train", "validation", "test")
    loaded: dict[str, dict[str, torch.nn.Module]] = {role: {} for role in roles}

    for role in roles:
        model_ids = partition["model_ids"].get(role, [])
        for model_id in model_ids:
            row = next(r for r in bank.manifest["models"] if r["model_id"] == model_id)
            ckpt_path = artifact_path(bank.directory, row["checkpoint"])
            model, _ = load_checkpoint(ckpt_path, model_factory, device=device)
            model.eval()
            loaded[role][model_id] = model

    return loaded


def main():
    args = parse_args()

    if args.output_dir is not None and args.output_dir.exists():
        sys.exit(f"Error: Output directory '{args.output_dir}' already exists. Use a new directory.")

    print("=" * 80)
    print("AMORTIZED GNN TEST-TIME ADAPTATION: GENERATOR TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Load Environment Manifest
    print(f"\n[1/5] Loading Environment Manifest from '{args.manifest}' ...")
    if not args.manifest.exists():
        sys.exit(f"Error: Environment manifest not found at '{args.manifest}'.\n"
                 f"Generate it first via: python examples/prepare_shift_environments.py --output-dir {args.manifest.parent}")

    manifest = load_environment_manifest(args.manifest)
    total_envs = len(manifest["environments"])
    print(f"      Manifest loaded successfully ({total_envs} environments total).")

    # 2. Load Pretrained Bank and Partition
    print(f"\n[2/5] Loading Pretrained Source Bank from '{args.bank}' ...")
    if not args.bank.exists():
        sys.exit(f"Error: Pretrained bank directory not found at '{args.bank}'.\n"
                 f"Train it first via: python examples/train_weight_bank.py --output-dir {args.bank}")

    subsets_file = args.bank / "source_subsets.json"
    if subsets_file.exists():
        subset_manifest = json.loads(subsets_file.read_text(encoding="utf-8"))
        source_cfg = dict(subset_manifest["source_provenance"]["graph_metadata"]["config"])
    else:
        protocol = json.loads(args.config.read_text(encoding="utf-8"))
        source_cfg = dict(protocol["source"])
        if "from_matcha" in source_cfg:
            del source_cfg["from_matcha"]

    source_cfg["split_ratios"] = tuple(source_cfg["split_ratios"])
    source_csbm = CSBMConfig(**source_cfg) if "p_in" in source_cfg else CSBMConfig.from_matcha(**source_cfg)
    bundle = load_dataset([CSBMEnvironment("source", "source_train", source_csbm)])
    source_data = source_training_data(bundle, "source")

    bank = load_verified_bank(args.bank, source_data, build_model)
    partition = create_model_partition(bank, seed=401)

    print(f"      Bank verified (30 models). Partition: "
          f"{len(partition['model_ids']['meta_train'])} meta-train, "
          f"{len(partition['model_ids']['validation'])} validation, "
          f"{len(partition['model_ids']['test'])} test.")

    # 3. Materialize Models in Memory
    print("\n[3/5] Instantiating GNN Models ...")
    grouped_models = load_partitioned_models(bank, partition, build_model, device=args.device)

    # Inspect target parameters on sample model to configure generator
    sample_model = next(iter(grouped_models["meta_train"].values()))
    target_flat, schema = extract_target_parameters(sample_model, scope=args.scope)
    target_dim = len(target_flat)
    hidden_dim = sample_model.config.hidden_channels

    in_channels = sample_model.config.in_channels
    out_channels = sample_model.config.out_channels
    descriptor = GraphShiftDescriptor(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_channels,
        conditioning=args.conditioning,
        target_scope=args.scope,
    )
    include_head_weights = (args.conditioning is None)
    if args.generator == "diffusion":
        generator = DiffusionHeadGenerator(
            descriptor_dim=descriptor.output_dim,
            head_dim=target_dim,
            hidden_dim=128,
            diffusion_steps=args.diffusion_steps,
            sampling_steps=args.sampling_steps,
            include_head_weights=include_head_weights,
        )
    else:
        generator = HeadAdapterGenerator(
            descriptor_dim=descriptor.output_dim,
            head_dim=target_dim,
            hidden_dim=128,
            include_head_weights=include_head_weights,
        )
    print(f"      Generator Architecture: {args.generator.upper()}")
    print(f"      Adaptation Scope: {args.scope.upper()}")
    print(f"      Conditioning Version: {args.conditioning.upper()}")
    print(f"      Descriptor Vector Size: {descriptor.output_dim}")
    print(f"      Target Parameter Count: {target_dim} ({len(schema)} tensors: {[s[0] for s in schema]})")
    print(f"      Generator Total Trainable Weights: {sum(p.numel() for p in generator.parameters())}")

    # 4. Meta-Training
    print("\n[4/5] Executing Generator Meta-Training ...")
    if args.generator == "diffusion":
        train_res = train_diffusion_generator(
            models=grouped_models["meta_train"],
            manifest=manifest,
            load_env_fn=load_shift_environment,
            descriptor=descriptor,
            generator=generator,
            epochs=args.epochs,
            lr=args.lr,
            oracle_steps=args.oracle_steps,
            delta_l2_reg=args.delta_l2,
            max_environments=args.max_environments,
            max_models=args.max_models,
            val_models=grouped_models["validation"],
            device=args.device,
            verbose=True,
            target_scope=args.scope,
        )
    else:
        train_res = train_adaptation_generator(
            models=grouped_models["meta_train"],
            manifest=manifest,
            load_env_fn=load_shift_environment,
            descriptor=descriptor,
            generator=generator,
            epochs=args.epochs,
            lr=args.lr,
            delta_l2_reg=args.delta_l2,
            max_environments=args.max_environments,
            max_models=args.max_models,
            val_models=grouped_models["validation"],
            device=args.device,
            verbose=True,
            target_scope=args.scope,
        )

    # Load best validation weights into generator
    generator.load_state_dict(train_res.best_state_dict)

    # 5. Final Evaluation on Held-Out Test Set
    print("\n[5/5] Evaluating Amortized Adaptation on Test Environments ...")
    test_models = grouped_models["test"]
    if args.max_models is not None:
        test_models = dict(list(test_models.items())[:args.max_models])

    records, report = evaluate_adaptation(
        models=test_models,
        manifest=manifest,
        load_env_fn=load_shift_environment,
        descriptor=descriptor,
        generator=generator,
        role="target_test",
        max_environments=args.max_environments,
        device=args.device,
        target_scope=args.scope,
    )

    # Print the user-readable report
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)

    # Save artifacts if an output directory was provided
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_file = args.output_dir / "ADAPTATION_REPORT.md"
        report_file.write_text(report, encoding="utf-8")

        generator_ckpt = args.output_dir / "generator_best.pt"
        torch.save({
            "generator_type": args.generator,
            "scope": args.scope,
            "descriptor_type": args.descriptor,
            "generator_state_dict": train_res.best_state_dict,
            "descriptor_dim": descriptor.output_dim,
            "head_dim": target_dim,
            "schema": schema,
            "history": train_res.history,
        }, generator_ckpt)

        print(f"\nAll outputs saved successfully:")
        print(f"  • Human-readable report: {report_file}")
        print(f"  • Generator checkpoint:  {generator_ckpt}\n")
    else:
        print("\n[Notice] No --output-dir provided. Results dumped to terminal only; no artifacts saved to disk.\n")


if __name__ == "__main__":
    main()
