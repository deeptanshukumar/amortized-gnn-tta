"""Full-graph source pilot: inner source stopping, 2018 scoring, final labels sealed."""

import argparse
from pathlib import Path
import torch
from amortized_gnn_tta.data.ogb import load_arxiv, partition_training_labels
from amortized_gnn_tta.training import (
    SourceTrainingData,
    TrainingConfig,
    fit_source,
    save_checkpoint,
    load_checkpoint,
)
from amortized_gnn_tta.models import build_model, make_model_spec
from amortized_gnn_tta.evaluation import predict, score_logits
from amortized_gnn_tta.adaptation.checkpoints import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source-fraction", type=float, required=True)
    p.add_argument("--pool-seed", type=int, default=20281108)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.output_dir.exists():
        p.error("Choose a fresh output directory")
    torch.set_num_threads(1)
    d = load_arxiv(a.data_root, download=a.download)
    pools = partition_training_labels(
        d, source_fraction=a.source_fraction, seed=a.pool_seed
    )
    a.output_dir.mkdir(parents=True)
    atomic_json(
        a.output_dir / "protocol.json",
        dict(
            dataset=d.metadata,
            source_fraction=a.source_fraction,
            pool_seed=a.pool_seed,
            source_stopping_fraction=0.1,
            direction="native-directed",
            epochs=a.epochs,
            label_pools={k: v.tolist() for k, v in pools.items()},
            final_test_scored=False,
        ),
    )
    view = SourceTrainingData(
        d.graph,
        pools["source_train"],
        d.labels[pools["source_train"]],
        pools["source_stop"],
        d.labels[pools["source_stop"]],
        tuple(map(str, range(40))),
        d.metadata,
    )
    spec = make_model_spec("gcn", 128, 40, hidden_channels=32)
    result = fit_source(
        lambda: build_model(spec),
        view,
        TrainingConfig(max_epochs=a.epochs, patience=a.epochs),
        device=a.device,
    )
    save_checkpoint(a.output_dir / "source.pt", result, spec)
    model, _ = load_checkpoint(a.output_dir / "source.pt", build_model)
    logits = predict(model, d.graph)
    torch.testing.assert_close(
        logits, predict(result.model, d.graph), rtol=1e-4, atol=1e-5
    )
    indices = d.official_splits["valid"]
    score = score_logits(logits, indices, d.labels[indices])
    atomic_json(
        a.output_dir / "result.json",
        dict(
            status="complete",
            validation_accuracy=score.accuracy,
            final_test_scored=False,
            checkpoint_replay=True,
            interpretation="Short engineering pilot, not a converged benchmark",
        ),
    )
    print("2018 validation accuracy", score.accuracy, flush=True)


if __name__ == "__main__":
    main()
