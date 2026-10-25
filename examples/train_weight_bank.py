"""Train five saved source-label subsets × six model seeds, sequentially.

    python examples/train_weight_bank.py --subsets runs/source-subsets-small.json --output-dir runs/gcn-bank

Reconstructs only the source from the CSBM subset provenance. No target graph or
source-test labels are read. All outputs are development artifacts ignored by Git.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import subprocess

import torch

from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.source_subsets import load_source_subsets
from amortized_gnn_tta.training import TrainingConfig, source_training_data
from amortized_gnn_tta.weight_bank import train_weight_bank


def git_state() -> dict:
    """Record the current checkout honestly, including uncommitted changes."""
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                  capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                capture_output=True, text=True).stdout
        return {"revision": revision, "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "status": [], "note": "Git state unavailable"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seeds", type=int, nargs=6, default=[31, 32, 33, 34, 35, 36])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a new path")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = json.loads(args.subsets.read_text(encoding="utf-8"))
    if raw.get("protocol") != "source-label-subsets-v1" or len(raw.get("subsets", [])) != 5:
        parser.error("Expected a five-subset source-label manifest")
    provenance = raw["source_provenance"]
    if provenance["dataset_metadata"].get("dataset") != "csbm":
        parser.error("This example reconstructs CSBM only")
    config_values = dict(provenance["graph_metadata"]["config"])
    config_values["split_ratios"] = tuple(config_values["split_ratios"])
    bundle = load_dataset([CSBMEnvironment(raw["source"]["graph_id"], "source_train", CSBMConfig(**config_values))])
    source = source_training_data(bundle, raw["source"]["graph_id"])
    subsets = load_source_subsets(args.subsets, source)  # Validate actual regenerated graph/pool identity.
    model_config = GCNConfig(source.graph.x.shape[1], len(source.class_names),
                             args.hidden_channels, args.dropout)
    training = TrainingConfig(args.epochs, args.patience, args.learning_rate, args.weight_decay)
    manifest = train_weight_bank(
        lambda: GCN(model_config), source, subsets, training,
        model_spec={"architecture": "two_layer_gcn", "config": asdict(model_config)},
        model_seeds=args.model_seeds, output_dir=args.output_dir, device=args.device,
        run_metadata={"git": git_state(), "subset_input_sha256": hashlib.sha256(args.subsets.read_bytes()).hexdigest()},
    )
    print(json.dumps({"status": manifest["status"], "models": manifest["completed_models"],
                      "source_nodes": source.graph.num_nodes, "output_dir": str(args.output_dir),
                      "all_cpu_reloads_verified": all(row["cpu_reload_predictions_close"] for row in manifest["models"])},
                     indent=2))


if __name__ == "__main__":
    main()
