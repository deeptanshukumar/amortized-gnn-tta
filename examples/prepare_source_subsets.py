"""Prepare five source-label subsets without changing the graph or held-out splits.

    python examples/prepare_source_subsets.py --label-fraction 0.5 --output runs/source-subsets.json

The required fraction is a development choice, not an official CSBM label budget.
No target graph or model bank is created by this preparation step.
"""

import argparse
import json
from pathlib import Path

import torch

from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.source_subsets import create_source_subsets, load_source_subsets, save_source_subsets
from amortized_gnn_tta.training import source_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-fraction", type=float, required=True,
                        help="Fraction of each class in the existing training pool, rounded down")
    parser.add_argument("--subset-seeds", type=int, nargs=5, default=[101, 102, 103, 104, 105])
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    torch.set_num_threads(1)
    n, f = (5000, 2000) if args.paper_scale else (1000, 64)
    bundle = load_dataset([CSBMEnvironment("source", "source_train", CSBMConfig.from_matcha(
        num_nodes=n, num_features=f, degree_scale=5, homophily_parameter=0.8,
        edge_seed=11, feature_seed=12, split_seed=13))])
    source = source_training_data(bundle, "source")
    manifest = create_source_subsets(source, label_fraction=args.label_fraction, seeds=args.subset_seeds)
    save_source_subsets(args.output, manifest, source)
    assert load_source_subsets(args.output, source) == manifest
    print(json.dumps({
        "output": str(args.output), "protocol": manifest["protocol"],
        "source_nodes": n, "native_training_pool": len(source.train_indices),
        "fixed_validation_nodes": len(source.val_indices),
        "fixed_source_test_nodes": len(bundle.select("source", "test")),
        "label_fraction_of_training_pool": args.label_fraction,
        "labels_per_subset": sum(manifest["subset_class_counts"]),
        "class_counts_per_subset": manifest["subset_class_counts"],
        "subset_ids": [item["id"] for item in manifest["subsets"]],
        "reload_verified": True,
    }, indent=2))


if __name__ == "__main__":
    main()
