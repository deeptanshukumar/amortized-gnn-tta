"""Train a CSBM direct-delta generator; validation-only unless explicitly requested.

Every saved run includes a complete inference recipe. This prototype does not
implement generated LoRA, five-candidate selection or real-graph experiments.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from amortized_gnn_tta.adaptation import (
    DiffusionHeadGenerator, HeadAdapterGenerator,
    evaluate_adaptation, extract_target_parameters, train_adaptation_generator, subsample_environments,
    train_diffusion_generator,
)
from amortized_gnn_tta.adaptation.experiments import CONDITIONING_CHOICES, make_descriptor
from amortized_gnn_tta.adaptation.checkpoints import atomic_json, save_generator_checkpoint, load_generator_checkpoint
from amortized_gnn_tta.adaptation._validation import training_options
from amortized_gnn_tta.bank_artifacts import artifact_path, load_model_partition, load_verified_bank
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset, load_environment_manifest, load_shift_environment
from amortized_gnn_tta.evaluation import select_device
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint, source_training_data, seed_model, software_versions
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]

    parser.add_argument("--bank", type=Path, default=root / "runs" / "phase1-bank",
                        help="Directory containing the completed 30-model source weight bank")
    parser.add_argument("--manifest", type=Path, default=root / "runs" / "controlled-shifts-small" / "environment_manifest.json",
                        help="Path to environment manifest JSON (from prepare_shift_environments.py)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for durable checkpoints, configuration and results; must not exist")
    parser.add_argument("--conditioning", type=str, default="D", choices=CONDITIONING_CHOICES,
                        help="compact/structural baselines or conditioning version: 'A' (target graph only), 'B' (+hidden rep), 'C' (+predictive uncertainty), 'D' (+compact weights)")
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
    parser.add_argument("--threads", type=int, default=1, help="Recorded PyTorch CPU thread count")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run adaptation on ('cpu' or 'cuda')")
    parser.add_argument("--diffusion-steps", type=int, default=50,
                        help="Number of diffusion timesteps T (diffusion generator only)")
    parser.add_argument("--sampling-steps", type=int, default=10,
                        help="Number of DDIM sampling steps during inference (diffusion generator only)")
    parser.add_argument("--oracle-steps", type=int, default=30,
                        help="Steps to fit teacher oracle deltas (diffusion generator only)")
    parser.add_argument("--model-partition", type=Path, required=True, help="Verified saved model partition; never regenerated implicitly")
    parser.add_argument("--seed", type=int, default=202701, help="Generator initialization/training seed")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=None, help="Training noise/order seed; default initialization seed + 10000")
    parser.add_argument("--descriptor-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--final-evaluation", action="store_true", help="Explicitly score held-out test roles; default is validation only")
    return parser.parse_args()


def load_experiment(bank_dir, manifest_path, partition_path, *, device="cpu", max_models=None, final_evaluation=False, load_models=True):
    """Verify inputs before constructing only the requested model roles/IDs."""
    bank_dir = Path(bank_dir)
    manifest = load_environment_manifest(Path(manifest_path))
    subsets_path = bank_dir / "source_subsets.json"
    if not subsets_path.is_file():
        raise ValueError("Bank is incomplete: source_subsets.json is required")
    subsets = json.loads(subsets_path.read_text())
    config = dict(subsets["source_provenance"]["graph_metadata"]["config"])
    config["split_ratios"] = tuple(config["split_ratios"])
    bundle = load_dataset([CSBMEnvironment("source", "source_train", CSBMConfig(**config))])
    bank = load_verified_bank(bank_dir, source_training_data(bundle, "source"), build_model)
    partition = load_model_partition(Path(partition_path), bank)
    if not load_models:
        return bank, partition, manifest, {}
    roles = ["meta_train", "validation"] + (["test"] if final_evaluation else [])
    grouped = {}
    for role in roles:
        ids = partition["model_ids"][role][:max_models]
        grouped[role] = {}
        for model_id in ids:
            row = next(row for row in bank.manifest["models"] if row["model_id"] == model_id)
            model, _ = load_checkpoint(artifact_path(bank.directory, row["checkpoint"]), build_model, device=device)
            grouped[role][model_id] = model.eval()
        if not grouped[role]:
            raise ValueError(f"No models selected for {role}")
    return bank, partition, manifest, grouped


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    # Hash the actual source, including uncommitted fixes and new modules.
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "examples").glob("*.py"))
    try:
        revision, dirty, git_status = git("rev-parse", "HEAD"), bool(git("status", "--porcelain")), "available"
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, dirty, git_status = None, None, "unavailable: exported code or git executable absent"
    return {"revision": revision, "dirty": dirty, "git_status": git_status,
            "source_hashes": {str(p.relative_to(root)): file_hash(p) for p in files}}


def run(args):
    from amortized_gnn_tta.adaptation._validation import positive_int
    positive_int(args.threads, "threads")
    torch.set_num_threads(args.threads)
    training_options(args.epochs, args.lr, args.delta_l2, args.max_models, args.max_environments)
    device = str(select_device(args.device))
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError("Generator seed must be in [0,2**32)")
    if args.training_seed is None:
        args.training_seed = (args.seed + 10000) % (2**32)
    seed_model(args.seed)
    bank, partition, manifest, grouped = load_experiment(args.bank, args.manifest, args.model_partition,
        device=device, max_models=args.max_models, final_evaluation=args.final_evaluation)
    sample = next(iter(grouped["meta_train"].values()))
    flat, schema = extract_target_parameters(sample, args.scope)
    descriptor, include_weights = make_descriptor(sample, args.conditioning, scope=args.scope,
        descriptor_version=args.descriptor_version, weight_scope="all", chunk_size=args.chunk_size)
    # Bank reconstruction consumes RNG for temporary models; generator
    # initialization must depend on its own seed, not on caps or bank size.
    seed_model(args.seed)
    kwargs = dict(descriptor_dim=descriptor.output_dim, head_dim=len(flat), hidden_dim=128, include_head_weights=include_weights)
    if args.generator == "diffusion":
        generator = DiffusionHeadGenerator(**kwargs, diffusion_steps=args.diffusion_steps,
            sampling_steps=args.sampling_steps, default_seed=args.sampling_seed)
    else:
        generator = HeadAdapterGenerator(**kwargs)
    provenance = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code": code_provenance(), "software": software_versions(), "torch_threads": torch.get_num_threads(),
        "bank_manifest_sha256": file_hash(args.bank / "bank.json"),
        "model_partition_sha256": file_hash(args.model_partition),
        "environment_manifest_sha256": file_hash(args.manifest), "model_partition": partition,
        "selected_model_ids": {role: list(models) for role, models in grouped.items()},
        "selected_environment_ids": {role: [e["id"] for e in subsample_environments([e for e in manifest["environments"] if e["role"] == role], args.max_environments)] for role in ["meta_train", "validation"] + (["target_test"] if args.final_evaluation else [])},
        "effective_training": {"hypernetwork_weight_decay": 1e-4, "diffusion_batch_size": 16, "diffusion_target_std_floor": 1e-3, "oracle_lr": .01, "optimizer": "Adam"},
        "selection": "mean validation accuracy; strict improvement, otherwise initial candidate",
        "evaluation_role": "target_test" if args.final_evaluation else "validation"}
    print(f"Verified {len(bank.manifest['models'])} bank models; selected " + str(provenance["selected_model_ids"]))
    if args.output_dir:
        atomic_json(args.output_dir / "run.json", {"status": "running", **provenance})
        # Capture the actual code as well as its hashes for dirty-run replay.
        import zipfile
        root = Path(__file__).resolve().parents[1]
        with zipfile.ZipFile(args.output_dir / "code_snapshot.zip", "x") as archive:
            for name in provenance["code"]["source_hashes"]:
                archive.write(root / name, name)
    def checkpoint(state, history):
        if args.output_dir:
            save_generator_checkpoint(args.output_dir / "generator_best.pt", generator, descriptor, schema,
                target_scope=args.scope, provenance=provenance, history=history, state_dict=state, overwrite=True)
    training_seed = args.training_seed if args.training_seed is not None else (args.seed + 10000) % (2**32)
    if type(training_seed) is not int or not 0 <= training_seed < 2**32:
        raise ValueError("Training seed must be in [0,2**32)")
    seed_model(training_seed)
    common = dict(models=grouped["meta_train"], manifest=manifest, load_env_fn=load_shift_environment,
        descriptor=descriptor, generator=generator, epochs=args.epochs, lr=args.lr,
        delta_l2_reg=args.delta_l2, max_environments=args.max_environments, max_models=args.max_models,
        val_models=grouped["validation"], device=device, target_scope=args.scope, checkpoint_callback=checkpoint)
    result = (train_diffusion_generator(**common, oracle_steps=args.oracle_steps)
              if args.generator == "diffusion" else train_adaptation_generator(**common))
    generator.load_state_dict(result.best_state_dict)
    if args.output_dir:
        restored, restored_descriptor, _ = load_generator_checkpoint(args.output_dir / "generator_best.pt", model=sample, device=device)
        graph = next(row for row in manifest["environments"] if row["role"] == "validation")
        inputs = load_shift_environment(manifest, graph["id"]).unlabeled_input(graph["id"]).to(device)
        with torch.no_grad():
            torch.testing.assert_close(restored(restored_descriptor(sample, inputs), flat), generator(descriptor(sample, inputs), flat), rtol=0, atol=0)
    role = "target_test" if args.final_evaluation else "validation"
    models = grouped["test" if args.final_evaluation else "validation"]
    records, report = evaluate_adaptation(models, manifest, load_shift_environment, descriptor, generator,
        role=role, max_environments=args.max_environments, device=device, target_scope=args.scope,
        allow_final_evaluation=args.final_evaluation)
    print(report)
    if args.output_dir:
        atomic_json(args.output_dir / "pair_results.json", [asdict(r) for r in records])
        (args.output_dir / "ADAPTATION_REPORT.md").write_text(report, encoding="utf-8")
        atomic_json(args.output_dir / "run.json", {"status": "complete", "training_wall_seconds": result.total_time_seconds, "teacher_wall_seconds": result.teacher_time_seconds, "optimizer_updates": result.optimizer_steps, "selected_epoch": result.selected_epoch, **provenance})
    return result


def main():
    args = parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output_dir / "run.json", {"status": "starting", "argv": sys.argv})
    try:
        run(args)
    except BaseException as error:
        if args.output_dir:
            path = args.output_dir / "run.json"
            status = json.loads(path.read_text())
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(path, status)
        raise


if __name__ == "__main__":
    main()
