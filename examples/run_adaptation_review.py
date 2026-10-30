"""Run frozen baselines and head/LoRA adaptation on permitted development pairs.

Requires an existing audited bank, its model partition, and a controlled-shift
manifest. Defaults choose one model from EACH meta-training source subset and
all controls/shift variants at severity .5 in the first meta-training lineage.
No held-out model or environment is selected. Use --all-meta-lineages for all
three development base draws, --models-per-subset for repeated source seeds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from amortized_gnn_tta.adaptation import AdaptationConfig, PEFTConfig
from amortized_gnn_tta.adaptation_review import OBJECTIVES, run_development_review
from amortized_gnn_tta.bank_artifacts import load_model_partition, load_verified_bank
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset, load_environment_manifest
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import source_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--model-partition", type=Path, required=True)
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--all-meta-lineages", action="store_true")
    parser.add_argument("--models-per-subset", type=int, default=1)
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--shift-names", nargs="+", help="Optional exact shift names; control is always included")
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=list(OBJECTIVES))
    parser.add_argument("--methods", nargs="+", choices=("head", "lora"), default=["head", "lora"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--initialization-seed", type=int, default=801)
    parser.add_argument("--backbone-rank", type=int, default=4)
    parser.add_argument("--head-rank", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or args.models_per_subset < 1 or args.output_dir.exists():
        parser.error("Use positive threads/model count and a new output directory")
    torch.set_num_threads(args.threads)
    raw = json.loads((args.bank / "source_subsets.json").read_text(encoding="utf-8"))
    values = dict(raw["source_provenance"]["graph_metadata"]["config"])
    values["split_ratios"] = tuple(values["split_ratios"])
    source_id = raw["source"]["graph_id"]
    source = source_training_data(load_dataset([
        CSBMEnvironment(source_id, "source_train", CSBMConfig(**values))]), source_id)
    bank = load_verified_bank(args.bank, source, build_model)
    partition = load_model_partition(args.model_partition, bank)
    environments = load_environment_manifest(args.environments)
    model_ids = []
    for subset in partition["groups"]["meta_train"]:
        members = sorted((r for r in bank.manifest["models"] if r["subset_id"] == subset),
                         key=lambda r: r["model_seed"])
        if args.models_per_subset > len(members):
            parser.error("Requested more models per subset than the bank contains")
        model_ids.extend(row["model_id"] for row in members[:args.models_per_subset])
    lineages = [row["id"] for row in environments["lineages"] if row["role"] == "meta_train"]
    if not args.all_meta_lineages:
        lineages = lineages[:1]
    if args.severity not in environments["train_severities"]:
        parser.error("Choose a declared meta-training severity; no automatic test-severity reuse")
    names = {s["name"] for s in environments["shifts"]}
    if args.shift_names and not set(args.shift_names) <= names:
        parser.error("Unknown shift name")
    env_ids = [row["id"] for row in environments["environments"] if row["lineage_id"] in lineages
               and (row["family"] == "control" or (row["severity"] == args.severity
                    and (not args.shift_names or row["shift"] in args.shift_names)))]
    peft = [PEFTConfig(m, backbone_rank=args.backbone_rank, head_rank=args.head_rank,
                       backbone_alpha=float(args.backbone_rank), head_alpha=float(args.head_rank))
            for m in args.methods]
    config = AdaptationConfig(args.steps, args.learning_rate, args.patience, args.initialization_seed)
    # Capture BEFORE execution, so editing files while a long run is executing
    # cannot silently substitute later code into its provenance snapshot.
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml", Path(__file__)]
    captured = {str(p.resolve().relative_to(root)): p.read_bytes() for p in files}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                  capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                capture_output=True, text=True).stdout
        git = {"revision": revision, "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        git = {"revision": None, "dirty": None}
    try:
        report = run_development_review(
            bank, partition, environments, source, model_ids=model_ids, environment_ids=env_ids,
            output_dir=args.output_dir, peft_configs=() if args.baseline_only else peft,
            objectives=() if args.baseline_only else args.objectives, config=config, device=args.device,
        )
    finally:
        if args.output_dir.exists():
            snapshots = {}
            for relative, content in captured.items():
                target = args.output_dir / "adaptation_code_snapshot" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                snapshots[relative] = hashlib.sha256(content).hexdigest()
            (args.output_dir / "implementation.json").write_text(json.dumps(
                {"git": git, "snapshot_captured": "before execution",
                 "adaptation_code_snapshot_sha256": snapshots}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "baselines": len(report["baselines"]),
                      "adaptations": len(report["adaptations"]), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
