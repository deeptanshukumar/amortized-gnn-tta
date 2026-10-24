"""Train one source GCN, restore source-val selection, then evaluate once.

    python examples/train_source.py --output-dir runs/source-gcn

Creates an inference checkpoint and JSON audit report in a new run directory.
Target scores are baseline transfer results: no adaptation or target tuning.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset
from amortized_gnn_tta.evaluation import predict, score_logits
from amortized_gnn_tta.models import GCN, GCNConfig
from amortized_gnn_tta.training import (
    TrainingConfig, fit_source, load_checkpoint, save_checkpoint,
    software_versions, source_training_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/source-gcn"))
    parser.add_argument("--paper-scale", action="store_true", help="5000 nodes × 2000 features per graph")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--model-seed", type=int, default=31)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or explicit mps (unverified)")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; recorded for traceability")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    n, f = (5000, 2000) if args.paper_scale else (1000, 64)
    model_config = GCNConfig(f, 2, args.hidden_channels, args.dropout)
    train_config = TrainingConfig(args.epochs, args.patience, args.learning_rate,
                                  args.weight_decay, args.model_seed)
    # Reject existing runs before training; never overwrite earlier evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environments = [
        CSBMEnvironment("source", "source_train", CSBMConfig.from_matcha(
            num_nodes=n, num_features=f, degree_scale=5, homophily_parameter=0.8,
            edge_seed=11, feature_seed=12, split_seed=13)),
        CSBMEnvironment("target", "target_test", CSBMConfig.from_matcha(
            num_nodes=n, num_features=f, degree_scale=5, homophily_parameter=0.2,
            negative_mean=-0.01, positive_mean=0.05,
            edge_seed=21, feature_seed=22, split_seed=23)),
    ]
    bundle = load_dataset(environments)
    view = source_training_data(bundle, "source")
    start = time.perf_counter()
    result = fit_source(lambda: GCN(model_config), view, train_config, device=args.device)
    training_seconds = time.perf_counter() - start
    model_spec = {"architecture": "two_layer_gcn", "config": asdict(model_config)}
    checkpoint_path = args.output_dir / "best_model.pt"
    save_checkpoint(checkpoint_path, result, model_spec)

    def factory(spec):
        if spec["architecture"] != "two_layer_gcn":
            raise ValueError("Example only restores two_layer_gcn")
        return GCN(GCNConfig(**spec["config"]))

    # CPU restoration is deliberate: validate portability of the saved tensors.
    restored, _ = load_checkpoint(checkpoint_path, factory, device="cpu")
    source_logits = predict(result.model, bundle.unlabeled_input("source"))
    restored_logits = predict(restored, bundle.unlabeled_input("source"))
    torch.testing.assert_close(source_logits, restored_logits, rtol=1e-4, atol=1e-5)
    source_metrics = {
        split: asdict(score_logits(source_logits, bundle.select("source", split),
                                    bundle.evaluation_labels("source", split)))
        for split in ("train", "val", "test")
    }
    # Neither target inference nor target-label scoring occurs during fitting.
    target_logits = predict(result.model, bundle.unlabeled_input("target"))
    target_metrics = asdict(score_logits(target_logits, bundle.select("target", "eval"),
                                        bundle.evaluation_labels("target", "eval")))
    report = {
        "purpose": "single-run source-only GCN baseline; no adaptation or target selection",
        "versions": software_versions(), "model": model_spec,
        "training": asdict(train_config), "device": result.device, "cpu_threads": args.threads,
        "training_seconds": training_seconds, "epochs_run": len(result.history),
        "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
        "selection": "minimum source-validation CE; source-test and target scored afterward",
        "source_metrics": source_metrics, "target_metrics": target_metrics,
        "checkpoint": "best_model.pt", "cpu_reload_predictions_close": True,
        "environment_configs": [asdict(env) for env in environments],
        "data_provenance": {key: record.metadata for key, record in bundle.graphs.items()},
        "source_splits": {key: values.tolist() for key, values in bundle.graphs["source"].splits.items()},
        "history": result.history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "purpose", "device", "epochs_run", "best_epoch", "source_metrics", "target_metrics",
        "cpu_reload_predictions_close", "training_seconds")}, indent=2))
    print(f"Saved checkpoint and audit report in {args.output_dir}")


if __name__ == "__main__":
    main()
