"""Generate/validate source and target CSBM graphs and exercise a PyG layer.

This is a data/interface smoke test, not a trained model or adaptation result.
Use --paper-scale to inspect Matcha-sized graphs. Optional JSON is an offline
report containing label-derived diagnostics, never deployment conditioning.
"""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch
from torch_geometric.nn import GCNConv

from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, graph_diagnostics, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-scale", action="store_true", help="Use 5000 nodes and 2000 features per graph")
    parser.add_argument("--output", type=Path, help="Optional offline JSON report path")
    args = parser.parse_args()
    # Keep a small CPU smoke test fast without assuming an accelerator exists.
    torch.set_num_threads(1)
    n, f = (5000, 2000) if args.paper_scale else (300, 32)
    environments = [
        CSBMEnvironment("source", "source_train", CSBMConfig.from_matcha(
            num_nodes=n, num_features=f, degree_scale=5, homophily_parameter=0.8,
            edge_seed=11, feature_seed=12, split_seed=13)),
        CSBMEnvironment("target", "target_test", CSBMConfig.from_matcha(
            num_nodes=n, num_features=f, degree_scale=5, homophily_parameter=0.2,
            edge_seed=21, feature_seed=22, split_seed=23,
            negative_mean=-0.01, positive_mean=0.05)),
    ]
    start = time.perf_counter()
    bundle = load_dataset(environments)
    generation_seconds = time.perf_counter() - start
    torch.manual_seed(31)  # Separate smoke-model initialization from data RNGs.
    model = GCNConv(f, 2, cached=False)
    source = bundle.unlabeled_input("source")
    logits = model(source.x, source.edge_index)
    loss = torch.nn.functional.cross_entropy(logits[bundle.select("source", "train")],
                                             bundle.training_labels("source"))
    loss.backward()
    if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("PyG forward/backward smoke check failed")
    with torch.no_grad():
        target = bundle.unlabeled_input("target")
        target_scores = model(target.x, target.edge_index)
    report = {
        "purpose": "offline data inspection; random GCN layer, no training/adaptation result",
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ("torch", "torch-geometric", "numpy")}},
        "device": "cpu", "generation_and_validation_seconds": generation_seconds,
        "protocol": bundle.metadata,
        "graphs": {key: {**graph_diagnostics(record), "provenance": record.metadata}
                   for key, record in bundle.graphs.items()},
        "pyg_smoke": {"source_logits_shape": list(logits.shape),
                      "target_logits_shape": list(target_scores.shape),
                      "source_train_loss": float(loss.detach()), "finite_gradients": True,
                      "target_input_keys": sorted(target.keys())},
        "environment_configs": [asdict(env) for env in environments],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
